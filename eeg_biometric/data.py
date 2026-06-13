"""EEG data containers and data sources.

``EEGDataSource`` resolves to one of two backends:

* ``mne``       — PhysioNet EEGBCI motor-movement/imagery recordings via MNE-Python
                  (many subjects → ideal for genuine/impostor 1:1 verification).
* ``synthetic`` — NumPy waveforms with per-subject spectral/spatial *signatures*,
                  so the downstream classifiers have something real to discriminate.

With ``source="auto"`` the MNE path is attempted first and any failure (package
missing, no network, download error) transparently falls back to synthetic data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 10–20 subset used by the synthetic generator (frontal channels first).
DEFAULT_MONTAGE: List[str] = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T7", "T8",
    "P3", "P4", "Pz", "O1", "O2",
]
# Channel names treated as frontal for blink/EOG-based liveness checks.
FRONTAL_CHANNELS: Tuple[str, ...] = ("Fp1", "Fp2", "Fpz", "AFz", "AF3", "AF4")


def _stable_int(value: object) -> int:
    """Process-independent integer hash (unlike built-in ``hash`` for strings)."""
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def make_blink_waveform(
    n_times: int,
    sfreq: float,
    onset_times: Sequence[float],
    amplitude: float = 120.0,
    width_s: float = 0.22,
) -> np.ndarray:
    """Synthesize a frontal blink/EOG artifact train.

    Each blink is a fast-rise / slow-decay monophasic bump (~120 µV, ~220 ms),
    matching the high-amplitude low-frequency deflection blinks produce at Fp1/Fp2.
    """
    t = np.arange(n_times) / sfreq
    sig = np.zeros(n_times)
    rise = max(width_s * 0.30, 1e-3)
    fall = max(width_s * 0.60, 1e-3)
    for onset in onset_times:
        left = np.exp(-0.5 * ((t - onset) / rise) ** 2)
        right = np.exp(-0.5 * ((t - onset) / fall) ** 2)
        sig += amplitude * np.where(t < onset, left, right)
    return sig


@dataclass
class EEGTrial:
    """A single multi-channel EEG segment.

    Attributes
    ----------
    data : np.ndarray, shape ``(n_channels, n_times)``, microvolts.
    channels : channel names aligned with ``data`` rows.
    sfreq : sampling frequency (Hz).
    subject : optional identity label (bookkeeping only).
    has_blink : whether an on-cue blink/EOG response is present (demo metadata).
    blink_times : onset times (s) of injected blinks, if any.
    echoed_nonce : challenge nonce echoed back by the capture device (anti-replay).
        The device should embed the nonce it received with the challenge so the
        liveness verifier can bind the response to that specific challenge.
    """

    data: np.ndarray
    channels: List[str]
    sfreq: float
    subject: Optional[str] = None
    has_blink: bool = False
    blink_times: Tuple[float, ...] = ()
    echoed_nonce: Optional[str] = None

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_times / self.sfreq

    def channel_index(self, name: str) -> Optional[int]:
        """Row index for ``name`` (case- and trailing-dot-insensitive), else None."""
        key = name.strip(". ").lower()
        for i, ch in enumerate(self.channels):
            if ch.strip(". ").lower() == key:
                return i
        return None

    def copy_with(self, data: np.ndarray) -> "EEGTrial":
        """Return a copy carrying new ``data`` but identical metadata."""
        return EEGTrial(
            data=data,
            channels=list(self.channels),
            sfreq=self.sfreq,
            subject=self.subject,
            has_blink=self.has_blink,
            blink_times=self.blink_times,
            echoed_nonce=self.echoed_nonce,
        )


class EEGDataSource:
    """Yield :class:`EEGTrial` objects from MNE sample data or synthetic waveforms.

    Parameters
    ----------
    source : {"auto", "mne", "synthetic"}
        ``auto`` tries MNE then falls back to synthetic on any error.
    sfreq : sampling rate for synthetic data (Hz).
    trial_seconds : length of each emitted trial (s).
    montage : channel names for synthetic data.
    seed : RNG seed for reproducibility.
    """

    def __init__(
        self,
        source: str = "auto",
        sfreq: float = 160.0,
        trial_seconds: float = 3.0,
        montage: Optional[Sequence[str]] = None,
        seed: int = 7,
    ) -> None:
        self.requested_source = source
        self.sfreq = float(sfreq)
        self.trial_seconds = float(trial_seconds)
        self.montage: List[str] = list(montage) if montage else list(DEFAULT_MONTAGE)
        self.seed = int(seed)
        self.active_source: Optional[str] = None
        self._mne_cache: Dict[str, object] = {}
        self._resolve_source()

    # ------------------------------------------------------------------ props
    @property
    def n_times(self) -> int:
        return int(round(self.sfreq * self.trial_seconds))

    @property
    def channels(self) -> List[str]:
        return list(self.montage)

    # ------------------------------------------------------------- resolution
    def _resolve_source(self) -> None:
        if self.requested_source == "synthetic":
            self.active_source = "synthetic"
            return
        try:
            import mne  # noqa: F401

            self.active_source = "mne"
        except Exception:
            if self.requested_source == "mne":
                print("[EEGDataSource] MNE requested but unavailable; using synthetic.")
            self.active_source = "synthetic"

    # ---------------------------------------------------------------- public
    def get_subject_trials(
        self,
        subject_id: object,
        n_trials: int = 10,
        with_blink: bool = False,
        blink_times: Optional[Sequence[float]] = None,
        base_seed: int = 0,
    ) -> List[EEGTrial]:
        """Return ``n_trials`` trials for ``subject_id``.

        Synthetic trials are reproducible given ``(subject_id, base_seed)``. For the
        MNE backend, recorded epochs are returned; when ``with_blink`` is set a
        synthetic blink is overlaid on frontal channels to emulate an on-cue
        liveness response (a documented simulation — the public recordings do not
        contain blinks aligned to our challenge windows).
        """
        if self.active_source == "mne":
            try:
                return self._mne_trials(subject_id, n_trials, with_blink, blink_times)
            except Exception as exc:  # robust fallback at call time, too
                print(f"[EEGDataSource] MNE load failed ({exc}); using synthetic.")
                self.active_source = "synthetic"
        return self._synthetic_trials(subject_id, n_trials, with_blink, blink_times, base_seed)

    @staticmethod
    def frontal_indices(trial: EEGTrial, frontal: Sequence[str] = FRONTAL_CHANNELS) -> List[int]:
        """Indices of frontal channels present in ``trial`` (for liveness checks)."""
        idxs = []
        for name in frontal:
            i = trial.channel_index(name)
            if i is not None:
                idxs.append(i)
        return idxs

    # ----------------------------------------------------------- synthetic
    def _subject_signature(self, subject_id: object) -> Dict[str, np.ndarray]:
        """Deterministic per-subject spectral + spatial signature.

        The spatial *mixing* matrix simulates volume conduction: each scalp
        channel is a smooth mixture of nearby sources, which deliberately induces
        inter-channel multicollinearity — exactly the structure the Elastic-Net
        channel selector is designed to cope with.
        """
        rng = np.random.default_rng((self.seed * 1000003 + _stable_int(subject_id)) % (2 ** 32))
        n = len(self.montage)
        alpha_freq = rng.uniform(9.0, 12.0)
        theta_freq = rng.uniform(4.5, 7.0)
        chan_gain = rng.uniform(0.6, 1.4, size=n)
        band_profile = rng.uniform(0.5, 1.5, size=5)  # delta..gamma weighting
        spread = rng.uniform(0.05, 0.25, size=(n, n))
        spread = (spread + spread.T) / 2.0
        np.fill_diagonal(spread, 0.0)
        mixing = np.eye(n) + 0.3 * spread
        mixing /= mixing.sum(axis=1, keepdims=True)
        phase = rng.uniform(0.0, 2 * np.pi, size=n)
        return dict(
            alpha_freq=alpha_freq, theta_freq=theta_freq, chan_gain=chan_gain,
            band_profile=band_profile, mixing=mixing, phase=phase,
        )

    @staticmethod
    def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
        """Unit-variance 1/f (pink) background noise."""
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n)
        if f.size > 1:
            f[0] = f[1]
        else:
            f[0] = 1.0
        y = np.fft.irfft(spec / np.sqrt(f), n=n)
        return (y - y.mean()) / (y.std() + 1e-12)

    def _synth_trial(
        self,
        subject_id: object,
        trial_rng: np.random.Generator,
        with_blink: bool,
        blink_times: Sequence[float],
    ) -> EEGTrial:
        sig = self._subject_signature(subject_id)
        n, N, sf = len(self.montage), self.n_times, self.sfreq
        t = np.arange(N) / sf
        bp = sig["band_profile"]
        X = np.empty((n, N))
        for c in range(n):
            X[c] = self._pink_noise(N, trial_rng)
            X[c] += sig["chan_gain"][c] * bp[2] * 8.0 * np.sin(2 * np.pi * sig["alpha_freq"] * t + sig["phase"][c])
            X[c] += sig["chan_gain"][c] * bp[1] * 5.0 * np.sin(2 * np.pi * sig["theta_freq"] * t + 0.7 * sig["phase"][c])
        X *= trial_rng.normal(1.0, 0.05, size=(n, 1))
        X = sig["mixing"] @ X           # volume-conduction mixing
        X *= 6.0                         # scale toward EEG µV range
        blink_on = bool(with_blink and len(blink_times))
        if blink_on:
            blink = make_blink_waveform(N, sf, blink_times)
            for name, scale in (("Fp1", 1.0), ("Fp2", 1.0), ("F7", 0.5), ("F8", 0.5),
                                ("F3", 0.4), ("F4", 0.4), ("Fz", 0.45)):
                if name in self.montage:
                    X[self.montage.index(name)] += scale * blink
        return EEGTrial(
            data=X, channels=list(self.montage), sfreq=sf, subject=str(subject_id),
            has_blink=blink_on, blink_times=tuple(blink_times) if blink_on else (),
        )

    def _synthetic_trials(
        self,
        subject_id: object,
        n_trials: int,
        with_blink: bool,
        blink_times: Optional[Sequence[float]],
        base_seed: int,
    ) -> List[EEGTrial]:
        seed = (self.seed * 7919 + _stable_int(subject_id) * 31 + base_seed) % (2 ** 32)
        rng = np.random.default_rng(seed)
        bt = list(blink_times) if (with_blink and blink_times) else []
        trials = []
        for _ in range(n_trials):
            trng = np.random.default_rng(int(rng.integers(0, 2 ** 32)))
            trials.append(self._synth_trial(subject_id, trng, with_blink, bt))
        return trials

    # ----------------------------------------------------------------- MNE
    def _mne_trials(
        self,
        subject_id: object,
        n_trials: int,
        with_blink: bool,
        blink_times: Optional[Sequence[float]],
    ) -> List[EEGTrial]:
        import mne
        from mne.datasets import eegbci

        mne.set_log_level("ERROR")
        if str(subject_id).isdigit():
            subj = int(subject_id)
        else:
            subj = (abs(_stable_int(subject_id)) % 109) + 1
        key = f"s{subj}"
        if key not in self._mne_cache:
            fnames = eegbci.load_data(subj, runs=[1, 2], update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in fnames]
            raw = mne.concatenate_raws(raws)
            eegbci.standardize(raw)          # rename to standard 10–05 labels
            raw.pick("eeg")
            raw.filter(1.0, 45.0, verbose="ERROR")
            self._mne_cache[key] = raw
        raw = self._mne_cache[key]
        sf = float(raw.info["sfreq"])
        data = raw.get_data() * 1e6           # volts → microvolts
        ch_names = [c.strip(".") for c in raw.ch_names]
        # keep the source's notion of channels/sfreq aligned with the MNE backend
        self.montage, self.sfreq = ch_names, sf
        seg = int(round(sf * self.trial_seconds))
        n_use = min(n_trials, max(1, data.shape[1] // seg))
        bt = list(blink_times) if (with_blink and blink_times) else []
        trials = []
        for k in range(n_use):
            chunk = data[:, k * seg:(k + 1) * seg].copy()
            on = False
            if bt:
                bw = make_blink_waveform(seg, sf, bt)
                for name in ("Fp1", "Fp2"):
                    if name in ch_names:
                        chunk[ch_names.index(name)] += bw
                on = True
            trials.append(
                EEGTrial(chunk, ch_names, sf, subject=str(subject_id),
                         has_blink=on, blink_times=tuple(bt) if on else ())
            )
        return trials
