"""Shared digital-signal-processing helpers.

Every routine degrades gracefully when SciPy is unavailable, falling back to a
NumPy-only implementation, so the pipeline remains runnable in minimal
environments (e.g. a fresh server without the scientific stack fully installed).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:  # SciPy is preferred but optional.
    from scipy.signal import butter, filtfilt, find_peaks, welch
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only without SciPy
    _HAVE_SCIPY = False


# Canonical EEG frequency bands (Hz).
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def have_scipy() -> bool:
    """Return True if SciPy signal routines are available."""
    return _HAVE_SCIPY


def power_spectral_density(
    x: Sequence[float], sfreq: float, nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD of a 1-D signal.

    Uses Welch's method via SciPy when available, else a single-segment
    Hann-windowed periodogram. Returns ``(freqs, psd)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    if nperseg is None:
        nperseg = int(min(n, max(64, n // 2)))
    if _HAVE_SCIPY:
        f, p = welch(x, fs=sfreq, nperseg=min(nperseg, n))
        return np.asarray(f), np.asarray(p)
    w = np.hanning(n)
    xw = (x - x.mean()) * w
    fft = np.fft.rfft(xw)
    p = (np.abs(fft) ** 2) / (sfreq * np.sum(w ** 2) + 1e-12)
    if p.size > 2:
        p[1:-1] *= 2.0
    f = np.fft.rfftfreq(n, d=1.0 / sfreq)
    return f, p


def bandpower(
    x: Sequence[float],
    sfreq: float,
    band: Tuple[float, float],
    psd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """Absolute power within ``band`` (trapezoidal integration of the PSD)."""
    f, p = psd if psd is not None else power_spectral_density(x, sfreq)
    lo, hi = band
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(p[idx], f[idx]))


def band_powers(
    x: Sequence[float],
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = True,
) -> Dict[str, float]:
    """Return per-band power (relative to total by default)."""
    bands = bands or EEG_BANDS
    f, p = power_spectral_density(x, sfreq)
    total = float(np.trapz(p, f)) + 1e-12
    out: Dict[str, float] = {}
    for name, b in bands.items():
        bp = bandpower(x, sfreq, b, psd=(f, p))
        out[name] = bp / total if relative else bp
    return out


def hjorth_parameters(x: Sequence[float]) -> Tuple[float, float, float]:
    """Hjorth activity, mobility and complexity of a 1-D signal."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = float(np.var(x)) + 1e-12
    var_dx = float(np.var(dx)) + 1e-12
    var_ddx = float(np.var(ddx)) + 1e-12
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def spectral_edge_frequency(x: Sequence[float], sfreq: float, edge: float = 0.95) -> float:
    """Frequency below which ``edge`` fraction of spectral power lies."""
    f, p = power_spectral_density(x, sfreq)
    c = np.cumsum(p)
    if c[-1] <= 0:
        return 0.0
    c = c / c[-1]
    idx = int(np.searchsorted(c, edge))
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def robust_zscore(x: Sequence[float]) -> np.ndarray:
    """Median/MAD-based z-score, resistant to artifact outliers."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + 1e-12
    return (x - med) / scale


def bandpass_filter(x: Sequence[float], sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase band-pass (SciPy Butterworth, else FFT mask fallback)."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * sfreq
    if _HAVE_SCIPY and x.size > 3 * (order + 1):
        b, a = butter(order, [max(1e-6, low / nyq), min(0.999, high / nyq)], btype="band")
        return filtfilt(b, a, x)
    return _fft_band(x, sfreq, low, high)


def lowpass_filter(x: Sequence[float], sfreq: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass (SciPy Butterworth, else FFT mask fallback)."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * sfreq
    if _HAVE_SCIPY and x.size > 3 * (order + 1):
        b, a = butter(order, min(0.999, cutoff / nyq), btype="low")
        return filtfilt(b, a, x)
    return _fft_band(x, sfreq, 0.0, cutoff)


def _fft_band(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    """FFT brick-wall band selection used when SciPy is unavailable."""
    n = x.size
    if n < 4:
        return x.astype(float)
    mean = float(np.mean(x))
    f = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spec = np.fft.rfft(x - mean)
    mask = np.ones_like(f, dtype=bool)
    if low > 0:
        mask &= f >= low
    if high > 0:
        mask &= f <= high
    y = np.fft.irfft(spec * mask, n=n)
    return y + (mean if low <= 0 else 0.0)


def detect_peaks(
    x: Sequence[float], height: Optional[float] = None, distance: int = 1
) -> np.ndarray:
    """Return indices of local maxima above ``height`` with a minimum spacing."""
    x = np.asarray(x, dtype=float)
    if _HAVE_SCIPY:
        idx, _ = find_peaks(x, height=height, distance=max(1, distance))
        return np.asarray(idx, dtype=int)
    out = []
    last = -np.inf
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            if (height is None or x[i] >= height) and (i - last) >= distance:
                out.append(i)
                last = i
    return np.asarray(out, dtype=int)
