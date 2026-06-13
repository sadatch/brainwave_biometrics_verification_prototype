"""Adversarial augmentation & presentation-attack simulation (defensive red-team).

Survey hypothesis 4: a generative model serves a **dual defensive** role —

1. **Data augmentation** — expand the handful of enrollment trials so the recogniser
   generalises from few samples (cf. EEG-GAN / ATGAN).
2. **Attack simulation** — synthesise "spoof" EEG and confirm that our own
   liveness + open-set defenses *reject* it. This is penetration testing of the
   defender, in the spirit of ISO/IEC 30107 PAD evaluation.

Scope guardrails
----------------
This module only **generates signals and scores them against our own pipeline**. It
contains no signal-injection capability, no hardware/DAC interface, and targets no
real individual — it operates solely on public/synthetic enrollment data. Its purpose
is to *measure and harden* the rejection of synthetic inputs.

Backends
--------
* ``EEGGAN`` — a small GAN (PyTorch) trained briefly on enrollment trials.
* ``SurrogateEEGGenerator`` — a NumPy phase-randomised surrogate that preserves each
  channel's power spectrum (runs with no deep-learning dependency). This is the
  default in the demo for determinism and speed.

Key defensive insight: a generator can mimic *resting* EEG statistics, but it does not
produce a *bona fide on-cue blink* synchronised to a random challenge — so the
liveness stage rejects these spoofs regardless of how realistic the background looks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from .data import EEGTrial, make_blink_waveform
except ImportError:
    from data import EEGTrial, make_blink_waveform

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# NumPy surrogate generator (default; no deep-learning dependency)
# --------------------------------------------------------------------------- #
class SurrogateEEGGenerator:
    """Phase-randomised surrogate generator.

    For each synthetic trial, a real template is chosen and every channel is
    phase-randomised in the Fourier domain. This preserves the per-channel power
    spectrum (so the signal looks spectrally realistic) while destroying the exact
    waveform — a classic surrogate-data construction, here repurposed to probe the
    biometric and liveness defenses.
    """

    name = "PhaseRandomizedSurrogate"

    def __init__(self, jitter: float = 0.05, seed: int = 0) -> None:
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.templates: List[np.ndarray] = []
        self.channels: Optional[List[str]] = None
        self.sfreq: Optional[float] = None

    def fit(self, trials: Sequence[EEGTrial]) -> "SurrogateEEGGenerator":
        self.templates = [np.asarray(t.data, dtype=float) for t in trials]
        self.channels = list(trials[0].channels)
        self.sfreq = float(trials[0].sfreq)
        return self

    def _surrogate(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x)
        mag = np.abs(spec)
        phases = np.exp(1j * self.rng.uniform(0.0, 2 * np.pi, size=spec.shape))
        phases[0] = 1.0  # keep DC real
        return np.fft.irfft(mag * phases, n=len(x))

    def generate(self, n: int) -> List[EEGTrial]:
        if not self.templates:
            raise RuntimeError("generator is not fitted")
        out: List[EEGTrial] = []
        for _ in range(n):
            tmpl = self.templates[int(self.rng.integers(0, len(self.templates)))]
            data = np.empty_like(tmpl)
            for c in range(tmpl.shape[0]):
                data[c] = self._surrogate(tmpl[c])
            data += self.rng.normal(0.0, self.jitter * (np.std(data) + 1e-9), size=data.shape)
            out.append(EEGTrial(data, list(self.channels), float(self.sfreq), subject="SPOOF"))
        return out


# --------------------------------------------------------------------------- #
# GAN generator (optional; requires PyTorch)
# --------------------------------------------------------------------------- #
if _HAVE_TORCH:

    class _Generator(nn.Module):
        def __init__(self, latent_dim: int, n_channels: int, n_times: int, hidden: int = 128) -> None:
            super().__init__()
            self.n_channels, self.n_times = n_channels, n_times
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden * 2, n_channels * n_times),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z).view(-1, self.n_channels, self.n_times)

    class _Discriminator(nn.Module):
        def __init__(self, n_channels: int, n_times: int, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_channels * n_times, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
                nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

else:  # pragma: no cover
    _Generator = None  # type: ignore
    _Discriminator = None  # type: ignore


class EEGGAN:
    """A compact EEG-GAN trained briefly on enrollment trials (skeleton).

    Standardises the training trials, fits a small MLP generator/discriminator with
    the non-saturating BCE objective, and samples de-standardised synthetic trials.
    Intended as a runnable architectural placeholder, not a state-of-the-art EEG-GAN.
    """

    name = "EEG-GAN"

    def __init__(self, latent_dim: int = 32, epochs: int = 60, batch: int = 16,
                 lr: float = 2e-4, seed: int = 0) -> None:
        if not _HAVE_TORCH:
            raise RuntimeError("EEGGAN requires PyTorch")
        self.latent_dim = int(latent_dim)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.lr = float(lr)
        self.seed = int(seed)
        self.G = None
        self.D = None

    def fit(self, trials: Sequence[EEGTrial]) -> "EEGGAN":
        torch.manual_seed(self.seed)
        X = np.stack([np.asarray(t.data, dtype=np.float32) for t in trials])  # (N, C, T)
        self.channels = list(trials[0].channels)
        self.sfreq = float(trials[0].sfreq)
        n, c, t = X.shape
        self.C, self.T = c, t
        self.mean, self.std = float(X.mean()), float(X.std()) + 1e-6
        data = torch.tensor((X - self.mean) / self.std)
        self.G = _Generator(self.latent_dim, c, t)
        self.D = _Discriminator(c, t)
        optG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                real = data[perm[i:i + self.batch]]
                b = real.shape[0]
                if b < 2:  # BatchNorm needs ≥2 samples
                    continue
                z = torch.randn(b, self.latent_dim)
                fake = self.G(z)
                optD.zero_grad()
                loss_d = bce(self.D(real), torch.ones(b, 1)) + bce(self.D(fake.detach()), torch.zeros(b, 1))
                loss_d.backward()
                optD.step()
                optG.zero_grad()
                loss_g = bce(self.D(fake), torch.ones(b, 1))
                loss_g.backward()
                optG.step()
        self.G.eval()
        return self

    def generate(self, n: int) -> List[EEGTrial]:
        if self.G is None:
            raise RuntimeError("generator is not fitted")
        with torch.no_grad():
            xn = self.G(torch.randn(n, self.latent_dim)).cpu().numpy()
        x = xn * self.std + self.mean
        return [EEGTrial(x[i], list(self.channels), self.sfreq, subject="SPOOF") for i in range(n)]


def make_generator(prefer: str = "surrogate", seed: int = 0, epochs: int = 60):
    """Return a generator backend.

    ``"surrogate"`` (default) → NumPy phase-randomised surrogate (deterministic, fast).
    ``"gan"``/``"torch"``/``"auto"`` → ``EEGGAN`` when PyTorch is available, else surrogate.
    """
    if prefer == "surrogate":
        return SurrogateEEGGenerator(seed=seed)
    want_gan = prefer in ("gan", "torch", "deep") or (prefer == "auto" and _HAVE_TORCH)
    if want_gan and _HAVE_TORCH:
        return EEGGAN(seed=seed, epochs=epochs)
    return SurrogateEEGGenerator(seed=seed)


# --------------------------------------------------------------------------- #
# Defensive simulator
# --------------------------------------------------------------------------- #
class PresentationAttackSimulator:
    """Generate spoof EEG and (a) augment enrollment, (b) red-team the defenses.

    Defensive use only: it scores synthetic inputs against our own pipeline to verify
    they are rejected; it never injects signals or targets a real person.
    """

    def __init__(self, generator=None, seed: int = 0) -> None:
        self.generator = generator or SurrogateEEGGenerator(seed=seed)

    def fit(self, genuine_trials: Sequence[EEGTrial]) -> "PresentationAttackSimulator":
        self.generator.fit(genuine_trials)
        return self

    def augment(self, genuine_trials: Sequence[EEGTrial], n_synthetic: int) -> List[EEGTrial]:
        """Return genuine trials plus ``n_synthetic`` generator samples (augmentation)."""
        return list(genuine_trials) + self.generator.generate(n_synthetic)

    def synthesize_spoofs(
        self, n: int, inject_blink: bool = False, blink_times: Optional[Sequence[float]] = None,
    ) -> List[EEGTrial]:
        """Generate ``n`` spoof trials; optionally overlay a blink on frontal channels.

        With ``inject_blink=False`` (default) the spoof lacks an on-cue blink and is
        expected to fail liveness. With a blink injected, it should still fail the
        biometric stage unless the identity statistics also match.
        """
        spoofs = self.generator.generate(n)
        if inject_blink and blink_times:
            for tr in spoofs:
                bw = make_blink_waveform(tr.n_times, tr.sfreq, blink_times)
                for name in ("Fp1", "Fp2"):
                    idx = tr.channel_index(name)
                    if idx is not None:
                        tr.data[idx] = tr.data[idx] + bw
                tr.has_blink, tr.blink_times = True, tuple(blink_times)
        return spoofs

    def evaluate_rejection(
        self, pipeline, claimed_id: str, challenge, n: int = 20, inject_blink: bool = False,
    ) -> Dict[str, object]:
        """Run ``n`` spoofs through ``pipeline.verify`` and report the rejection rate.

        Returns ``rejection_rate`` (defensive success; APCER ≈ 1 − rejection_rate) and
        a per-stage breakdown of where each spoof was caught.
        """
        cue = [challenge.blink_cue_time] if inject_blink else None
        spoofs = self.synthesize_spoofs(n, inject_blink=inject_blink, blink_times=cue)
        results = [pipeline.verify(claimed_id, s, challenge) for s in spoofs]
        rejected = sum(1 for r in results if not r.decision)
        by_stage: Dict[str, int] = {}
        for r in results:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
        return {
            "n": n,
            "rejected": rejected,
            "rejection_rate": rejected / max(n, 1),
            "apcer": 1.0 - rejected / max(n, 1),
            "by_stage": by_stage,
            "generator": getattr(self.generator, "name", type(self.generator).__name__),
        }
