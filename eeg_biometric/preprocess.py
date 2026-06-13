"""ATAR — Automatic and Tunable Artifact Removal (wavelet, single-channel, low-latency).

Reference idea: Bajaj et al., *"Automatic and Tunable Algorithm for EEG Artifact
Removal Using Wavelet Decomposition with Applications in Predictive Modeling during
Auditory Tasks"* (and the ``spkit`` implementation). This is a faithful,
self-contained re-implementation tuned for **real-time, per-channel** use rather than
a byte-exact copy of the reference library.

Why ATAR over ICA here
----------------------
ICA needs the full multi-channel block and a (relatively expensive) unmixing
estimate — awkward for low-latency, channel-streaming inference. ATAR instead works
on **one channel at a time** in short overlapping windows, which suits the
ESP32→server streaming setting and keeps latency to roughly one window.

The counter-intuitive core
--------------------------
Unlike classic wavelet *denoising* (which zeroes the *small* coefficients assumed to
be noise), ATAR assumes the artifacts (blinks, EOG, muscle) are the **high-amplitude
transients** — i.e. the *large* wavelet coefficients — and suppresses *those* while
preserving the lower-amplitude neural rhythm. A single tunable knob ``beta`` sets the
operating point from gentle (β→0) to aggressive (β→1).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

try:
    from .data import EEGTrial
    from .dsp import robust_zscore
except ImportError:  # allow running as a loose script
    from data import EEGTrial
    from dsp import robust_zscore

try:
    import pywt
    _HAVE_PYWT = True
except Exception:  # pragma: no cover - exercised only without PyWavelets
    _HAVE_PYWT = False


class ATARPreprocessor:
    """Wavelet artifact remover with a single tunable operating point.

    Parameters
    ----------
    wavelet : mother wavelet (e.g. ``"db4"``).
    mode : ``"soft"`` (sigmoidal suppression), ``"linatten"`` (magnitude clip to θ),
        or ``"elim"`` (hard removal of supra-threshold coefficients).
    beta : tunability in ``[0, 1]``; higher → lower threshold → more aggressive.
    kappa_min, kappa_max : threshold factors (in robust-σ units) mapped by ``beta``.
        ``θ = κ·σ`` with ``κ = κ_max·(1−β) + κ_min·β``.
    win_seconds : analysis window length (low latency ⇒ keep ~0.5–1.0 s).
    overlap : fractional overlap between consecutive windows for overlap-add.
    level : decomposition level; ``None`` picks a sensible per-window maximum (≤5).
    decomposition : ``"wpd"`` (Wavelet Packet Decomposition — the variant the survey
        specifies) or ``"dwt"`` (lighter multi-level DWT).
    """

    def __init__(
        self,
        wavelet: str = "db4",
        mode: str = "soft",
        beta: float = 0.5,
        kappa_min: float = 1.5,
        kappa_max: float = 8.0,
        win_seconds: float = 1.0,
        overlap: float = 0.5,
        level: Optional[int] = None,
        decomposition: str = "wpd",
    ) -> None:
        if mode not in ("soft", "linatten", "elim"):
            raise ValueError(f"unknown mode: {mode!r}")
        if decomposition not in ("wpd", "dwt"):
            raise ValueError(f"unknown decomposition: {decomposition!r}")
        self.wavelet = wavelet
        self.mode = mode
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.win_seconds = float(win_seconds)
        self.overlap = float(np.clip(overlap, 0.0, 0.95))
        self.level = level
        self.decomposition = decomposition

    @property
    def backend(self) -> str:
        """Active backend label, e.g. ``"pywt-wpd"`` or ``"robust-fallback"``."""
        return f"pywt-{self.decomposition}" if _HAVE_PYWT else "robust-fallback"

    @property
    def _kappa(self) -> float:
        return self.kappa_max * (1.0 - self.beta) + self.kappa_min * self.beta

    # --------------------------------------------------------------- public
    def transform(self, trial: EEGTrial) -> EEGTrial:
        """Clean every channel of ``trial`` independently; return a new trial."""
        cleaned = np.empty_like(trial.data, dtype=float)
        for c in range(trial.n_channels):
            cleaned[c] = self.clean_signal(trial.data[c], trial.sfreq)
        return trial.copy_with(cleaned)

    def transform_many(self, trials: Iterable[EEGTrial]) -> List[EEGTrial]:
        """Vectorised convenience over an iterable of trials."""
        return [self.transform(t) for t in trials]

    def clean_signal(self, x: np.ndarray, sfreq: float) -> np.ndarray:
        """Clean a single channel via overlap-add windowed wavelet thresholding.

        This is the streaming-friendly entry point: it processes one 1-D channel
        in short Hann-weighted windows, so it can be driven sample-block by
        sample-block in a live setting.
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        w = int(round(self.win_seconds * sfreq))
        w = max(16, min(w, n))
        if n <= w:
            return self._atar_window(x)
        hop = max(1, int(round(w * (1.0 - self.overlap))))
        win = np.hanning(w)
        out = np.zeros(n)
        norm = np.zeros(n)
        starts = list(range(0, n - w + 1, hop))
        if starts[-1] != n - w:
            starts.append(n - w)
        for s in starts:
            seg = x[s:s + w]
            cleaned = self._atar_window(seg)
            out[s:s + w] += cleaned * win
            norm[s:s + w] += win
        norm[norm < 1e-8] = 1.0
        return out / norm

    # -------------------------------------------------------------- internals
    def _safe_level(self, n: int) -> int:
        """Pick a decomposition level that is valid for a window of length ``n``."""
        try:
            max_lvl = pywt.dwt_max_level(n, pywt.Wavelet(self.wavelet).dec_len)
        except Exception:
            max_lvl = 4
        return self.level if self.level is not None else max(1, min(5, max_lvl))

    def _atar_window(self, seg: np.ndarray) -> np.ndarray:
        """Apply ATAR thresholding to a single window (WPD or DWT, with fallback)."""
        seg = np.asarray(seg, dtype=float)
        if not _HAVE_PYWT:
            return self._robust_attenuate(seg)
        if self.decomposition == "wpd":
            try:
                return self._atar_window_wpd(seg)
            except Exception:
                return self._atar_window_dwt(seg)
        return self._atar_window_dwt(seg)

    def _atar_window_wpd(self, seg: np.ndarray) -> np.ndarray:
        """Wavelet-Packet-Decomposition ATAR (the variant the survey specifies).

        WPD splits *both* approximation and detail branches, giving uniform frequency
        resolution across the packet tree; ATAR then suppresses the high-amplitude
        (artifact) coefficients in every leaf node — including the low-frequency leaf
        where blink slow-waves concentrate.
        """
        n = seg.size
        level = self._safe_level(n)
        if level < 1:
            return self._robust_attenuate(seg)
        wp = pywt.WaveletPacket(data=seg, wavelet=self.wavelet, mode="periodization", maxlevel=level)
        kappa = self._kappa
        for node in wp.get_level(level, order="natural"):
            d = np.asarray(node.data, dtype=float)
            med = np.median(d)
            sigma = 1.4826 * np.median(np.abs(d - med)) + 1e-12
            node.data = self._apply_mode(d, kappa * sigma)
        rec = np.asarray(wp.reconstruct(update=True), dtype=float)
        if rec.size >= n:
            return rec[:n]
        out = np.zeros(n)
        out[: rec.size] = rec
        return out

    def _atar_window_dwt(self, seg: np.ndarray) -> np.ndarray:
        """Multi-level DWT ATAR (lighter alternative; approximation preserved)."""
        n = seg.size
        level = self._safe_level(n)
        if level < 1:
            return self._robust_attenuate(seg)
        coeffs = pywt.wavedec(seg, self.wavelet, level=level, mode="periodization")
        kappa = self._kappa
        new_coeffs = [coeffs[0]]  # keep approximation (slow EEG baseline)
        for d in coeffs[1:]:
            d = np.asarray(d, dtype=float)
            med = np.median(d)
            sigma = 1.4826 * np.median(np.abs(d - med)) + 1e-12  # robust scale per level
            new_coeffs.append(self._apply_mode(d, kappa * sigma))
        rec = np.asarray(pywt.waverec(new_coeffs, self.wavelet, mode="periodization"), dtype=float)
        if rec.size >= n:
            return rec[:n]
        out = np.zeros(n)
        out[: rec.size] = rec
        return out

    def _apply_mode(self, d: np.ndarray, theta: float) -> np.ndarray:
        """Suppress supra-threshold (artifact) coefficients per the operating mode."""
        a = np.abs(d)
        if self.mode == "elim":
            return np.where(a > theta, 0.0, d)
        if self.mode == "linatten":
            return np.sign(d) * np.minimum(a, theta)
        # "soft": smooth sigmoidal suppression of large coefficients.
        s = theta / 4.0 + 1e-12
        suppression = 1.0 / (1.0 + np.exp((a - theta) / s))
        return d * suppression

    def _robust_attenuate(self, x: np.ndarray) -> np.ndarray:
        """No-wavelet fallback: sigmoidally pull high-|z| excursions toward the median.

        Degraded but still removes blink-scale spikes when PyWavelets is absent.
        """
        x = np.asarray(x, dtype=float)
        z = robust_zscore(x)
        kappa = self._kappa
        a = np.abs(z)
        s = kappa / 4.0 + 1e-12
        suppression = 1.0 / (1.0 + np.exp((a - kappa) / s))
        med = np.median(x)
        return med + (x - med) * suppression
