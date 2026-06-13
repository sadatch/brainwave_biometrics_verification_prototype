"""Feature extraction: MAEEG / GMAEEG deep encoder + handcrafted fallback.

Intended deployment
-------------------
A self-supervised **MAEEG** (Masked Auto-Encoder for EEG) encoder, *pretrained* on a
large unlabelled EEG corpus, is used **frozen** as a generic feature extractor:
EEG time-series → embedding vector. ``GMAEEG`` adds a learnable channel-graph mixing
stage to exploit the spatial montage.

Honest design decision (read me)
--------------------------------
We do **not** ship pretrained weights here, and a *randomly-initialised* transformer
is **not** identity-discriminative. So that the end-to-end demo is meaningful, the
factory :func:`make_encoder` defaults to a :class:`HandcraftedSpectralEncoder`
(band-power + Hjorth + spectral-edge), which genuinely separates subjects on the
demo data. The deep :class:`MAEEGEncoder` is still provided and runnable (the demo
performs a real forward pass through it); loading real weights via
``MAEEGEncoder.load_pretrained(path)`` — or passing ``prefer="deep"`` — promotes it to
the scoring encoder. This keeps the architecture truthful without faking learned
discriminability.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from .data import EEGTrial
    from .dsp import EEG_BANDS, band_powers, hjorth_parameters, spectral_edge_frequency
except ImportError:
    from data import EEGTrial
    from dsp import EEG_BANDS, band_powers, hjorth_parameters, spectral_edge_frequency

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# Handcrafted encoder (default, no heavy dependencies, genuinely discriminative)
# --------------------------------------------------------------------------- #
class HandcraftedSpectralEncoder:
    """Deterministic spectral/temporal embedding used as the default extractor.

    Per channel (10 features): 5 relative band powers, 3 Hjorth parameters,
    spectral-edge frequency, and log-variance. The embedding is the per-channel
    features concatenated over the selected channels, so its dimensionality is
    fixed once the channel set is fixed.
    """

    name = "HandcraftedSpectral"
    n_per_channel = 10

    def __init__(self) -> None:
        self.embed_dim: Optional[int] = None

    def embed(self, trial: EEGTrial, channel_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return a 1-D embedding for one trial (optionally restricted to channels)."""
        idx = list(channel_idx) if channel_idx is not None else list(range(trial.n_channels))
        feats: List[float] = []
        for ci in idx:
            x = trial.data[ci]
            bp = band_powers(x, trial.sfreq, relative=True)
            act, mob, comp = hjorth_parameters(x)
            sef = spectral_edge_frequency(x, trial.sfreq)
            logvar = float(np.log(np.var(x) + 1e-8))
            feats.extend(bp[b] for b in EEG_BANDS)
            feats.extend([act, mob, comp, sef, logvar])
        v = np.asarray(feats, dtype=float)
        self.embed_dim = int(v.size)
        return v

    def embed_many(self, trials: Sequence[EEGTrial], channel_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        return np.vstack([self.embed(t, channel_idx) for t in trials])


# --------------------------------------------------------------------------- #
# Deep encoders (optional; require PyTorch)
# --------------------------------------------------------------------------- #
if _HAVE_TORCH:

    def _num_groups(channels: int) -> int:
        """Largest GroupNorm group count (≤8) dividing ``channels``."""
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    class _ConvFrontend(nn.Module):
        """6-layer strided 1-D conv tokeniser (Conv → GroupNorm → GELU → Dropout).

        Mirrors the MAEEG convolutional encoder (Chien et al.): the raw multichannel
        waveform is encoded into a sequence of ``token_dim``-dimensional tokens, with
        temporal down-sampling via three stride-2 layers.
        """

        def __init__(self, n_channels: int, token_dim: int = 64, dropout: float = 0.1) -> None:
            super().__init__()
            chans = [n_channels, 32, 32, 64, 64, 64, token_dim]
            kernels = [7, 3, 5, 3, 3, 3]
            strides = [2, 1, 2, 1, 2, 1]
            blocks: list = []
            for i in range(6):
                blocks += [
                    nn.Conv1d(chans[i], chans[i + 1], kernel_size=kernels[i],
                              stride=strides[i], padding=kernels[i] // 2),
                    nn.GroupNorm(_num_groups(chans[i + 1]), chans[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            self.net = nn.Sequential(*blocks)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)  # (B, token_dim, T')

    class _DynamicGraphConv(nn.Module):
        """Learnable dynamic adjacency + graph convolution across electrodes (GMAEEG).

        Implements Fu et al.'s update rule for the dynamic adjacency exactly:

            A_updated = ρ( δ( W₂ · ξ( W₁ · Ã_init ) ) )

        with ξ = ELU, δ = tanh, ρ = ReLU, where ``Ã_init = E·Eᵀ`` is an initial
        adjacency from learnable node embeddings. The result is given self-loops and
        row-normalised for a stable spatial graph convolution ``X' = ELU(Â · X)``
        applied across the channel dimension.
        """

        def __init__(self, n_channels: int, emb: int = 16, hidden: int = 16) -> None:
            super().__init__()
            self.node_emb = nn.Parameter(torch.randn(n_channels, emb) * 0.1)  # → Ã_init
            self.W1 = nn.Parameter(torch.randn(hidden, n_channels) * 0.1)     # W₁
            self.W2 = nn.Parameter(torch.randn(n_channels, hidden) * 0.1)     # W₂
            self.elu = nn.ELU()

        def adjacency(self) -> "torch.Tensor":
            a_init = self.node_emb @ self.node_emb.t()            # Ã_init  (C, C)
            h = self.elu(self.W1 @ a_init)                        # ξ = ELU (hidden, C)
            a = torch.relu(torch.tanh(self.W2 @ h))               # ρ(δ(·)) (C, C), ≥ 0
            a = a + torch.eye(a.shape[0], device=a.device)        # self-loops
            return a / (a.sum(dim=1, keepdim=True) + 1e-6)        # row-normalise

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.elu(torch.einsum("cd,bdt->bct", self.adjacency(), x))

    class MAEEGEncoder(nn.Module):
        """MAEEG encoder (paper-faithful dims): 6-conv frontend → 8-layer Transformer.

        Flow ``(B, C, T)`` → conv tokens (``token_dim``=64) → linear lift to
        ``model_dim``=192 → CLS + Transformer(depth=8) → 192-d context → ``embed_dim``=64
        L2-normalised embedding. Used **frozen** as a feature extractor.
        """

        def __init__(
            self,
            n_channels: int,
            input_len: int,
            embed_dim: int = 64,
            token_dim: int = 64,
            model_dim: int = 192,
            depth: int = 8,
            n_heads: int = 6,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.n_channels = int(n_channels)
            self.input_len = max(32, int(input_len))
            self.embed_dim = int(embed_dim)
            self.token_dim = int(token_dim)
            self.model_dim = int(model_dim)
            self.frontend = _ConvFrontend(n_channels, token_dim, dropout)
            with torch.no_grad():  # size positional embeddings from the real token count
                n_tokens = self.frontend(torch.zeros(1, n_channels, self.input_len)).shape[-1]
            self.input_proj = nn.Linear(token_dim, model_dim)
            self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
            self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, model_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.norm = nn.LayerNorm(model_dim)
            self.proj = nn.Linear(model_dim, embed_dim)  # 192 → 64 context mapping
            self.pretrained = False
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def encode_tokens(self, x: "torch.Tensor") -> "torch.Tensor":
            """Raw signal → projected token sequence ``(B, n_tokens, model_dim)``."""
            tok = self.frontend(x).transpose(1, 2)
            return self.input_proj(tok)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.encode_tokens(x)
            cls = self.cls.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1) + self.pos[:, : h.shape[1] + 1]
            h = self.encoder(h)
            emb = self.proj(self.norm(h[:, 0]))  # CLS context → embedding
            return F.normalize(emb, dim=-1)

        def freeze(self) -> "MAEEGEncoder":
            """Switch to eval mode and disable grads (frozen feature-extractor use)."""
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            return self

        def load_pretrained(self, path: str, map_location: str = "cpu") -> "MAEEGEncoder":
            """Load encoder weights from ``path`` (accepts a raw or wrapped state-dict)."""
            state = torch.load(path, map_location=map_location)
            state = state.get("model", state) if isinstance(state, dict) else state
            self.load_state_dict(state)
            self.pretrained = True
            return self

    class GMAEEGEncoder(MAEEGEncoder):
        """Graph-MAEEG: a learnable dynamic-adjacency graph conv across electrodes,
        applied (residually) before the temporal MAEEG encoder."""

        def __init__(self, *args, graph_emb: int = 16, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.graph = _DynamicGraphConv(self.n_channels, graph_emb)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.graph(x)  # residual spatial graph mixing
            return super().forward(x)

        def learned_adjacency(self) -> np.ndarray:
            """Return the learned channel adjacency (an interpretable identity signature)."""
            with torch.no_grad():
                return self.graph.adjacency().cpu().numpy()

    class MaskedReconstructionPretrainer:
        """Illustrative MAEEG pretraining: Gaussian-noise masking + cosine recon loss.

        Per Chien et al., a fraction of the input is corrupted with Gaussian noise and
        the model is trained to restore the clean context, minimising a cosine-
        similarity reconstruction loss. Simplified (reconstructs the pooled token
        context rather than every token) and NOT run in the demo — it documents how
        the frozen weights would be produced.
        """

        def __init__(self, encoder: "MAEEGEncoder", mask_ratio: float = 0.5,
                     noise_std: float = 1.0, lr: float = 1e-3) -> None:
            self.encoder = encoder
            self.mask_ratio = float(mask_ratio)
            self.noise_std = float(noise_std)
            self.decoder = nn.Sequential(
                nn.Linear(encoder.embed_dim, encoder.model_dim), nn.GELU(),
                nn.Linear(encoder.model_dim, encoder.model_dim),
            )
            params = list(encoder.parameters()) + list(self.decoder.parameters())
            self.opt = torch.optim.Adam(params, lr=lr)

        def pretrain_step(self, batch: "torch.Tensor") -> float:
            """One step on a ``(B, C, T)`` batch; returns the cosine reconstruction loss."""
            self.encoder.train()
            with torch.no_grad():
                target = self.encoder.encode_tokens(batch).mean(dim=1)  # clean context
            mask = (torch.rand(batch.shape[0], 1, batch.shape[2], device=batch.device)
                    < self.mask_ratio).float()
            noisy = batch * (1 - mask) + torch.randn_like(batch) * self.noise_std * mask
            recon = self.decoder(self.encoder(noisy))
            loss = (1.0 - F.cosine_similarity(recon, target, dim=-1)).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return float(loss.item())

else:  # PyTorch unavailable — expose names so imports never break.
    MAEEGEncoder = None          # type: ignore
    GMAEEGEncoder = None         # type: ignore
    MaskedReconstructionPretrainer = None  # type: ignore


class MAEEGAdapter:
    """Wrap a torch MAEEG/GMAEEG module behind the encoder ``embed`` interface."""

    def __init__(self, module, input_len: int, device: str = "cpu",
                 channel_idx: Optional[Sequence[int]] = None, name: str = "MAEEG") -> None:
        self.module = module.to(device).eval()
        self.input_len = int(input_len)
        self.device = device
        self.channel_idx = list(channel_idx) if channel_idx is not None else None
        self.embed_dim = int(module.embed_dim)
        self._name = name

    @property
    def name(self) -> str:
        tag = "pretrained" if getattr(self.module, "pretrained", False) else "random-init"
        return f"{self._name}({tag})"

    def _prep(self, trial: EEGTrial, channel_idx: Optional[Sequence[int]]) -> np.ndarray:
        idx = list(channel_idx) if channel_idx is not None else (
            self.channel_idx if self.channel_idx is not None else list(range(trial.n_channels)))
        data = trial.data[idx].astype(float)
        t = data.shape[1]
        if t < self.input_len:
            data = np.pad(data, ((0, 0), (0, self.input_len - t)))
        else:
            data = data[:, : self.input_len]
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
        return data

    def embed(self, trial: EEGTrial, channel_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        x = self._prep(trial, channel_idx)
        with torch.no_grad():
            tensor = torch.tensor(x[None], dtype=torch.float32, device=self.device)
            emb = self.module(tensor).cpu().numpy()[0]
        return emb

    def embed_many(self, trials: Sequence[EEGTrial], channel_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        return np.vstack([self.embed(t, channel_idx) for t in trials])


def make_encoder(
    prefer: str = "auto",
    n_channels: Optional[int] = None,
    input_len: Optional[int] = None,
    embed_dim: int = 64,
    variant: str = "maeeg",
    pretrained_path: Optional[str] = None,
    device: str = "cpu",
    seed: int = 0,
):
    """Return an embedding encoder.

    Defaults to :class:`HandcraftedSpectralEncoder`. Returns a frozen
    :class:`MAEEGEncoder`/:class:`GMAEEGEncoder` (wrapped in :class:`MAEEGAdapter`)
    only when a deep encoder is explicitly requested (``prefer`` in
    {"deep","maeeg","gmaeeg","torch"} or a ``pretrained_path`` is given) *and*
    PyTorch plus the required shapes are available.
    """
    want_deep = prefer in ("deep", "maeeg", "gmaeeg", "torch") or pretrained_path is not None
    if want_deep and _HAVE_TORCH and n_channels and input_len:
        torch.manual_seed(seed)
        cls = GMAEEGEncoder if (variant == "gmaeeg" or prefer == "gmaeeg") else MAEEGEncoder
        module = cls(n_channels=n_channels, input_len=input_len, embed_dim=embed_dim)
        if pretrained_path:
            try:
                module.load_pretrained(pretrained_path)
            except Exception as exc:
                print(f"[make_encoder] could not load weights ({exc}); using random init.")
        module.freeze()
        return MAEEGAdapter(module, input_len=module.input_len, device=device, name=cls.__name__)
    if want_deep and not _HAVE_TORCH:
        print("[make_encoder] deep encoder requested but PyTorch unavailable; "
              "using handcrafted encoder.")
    return HandcraftedSpectralEncoder()
