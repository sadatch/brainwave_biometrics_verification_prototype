"""Active liveness detection — Presentation Attack Detection (PAD).

Standards context
-----------------
ISO/IEC 30107 frames PAD as a standard, *defensive* biometric component. This module
implements the **active challenge–response** variant: the system issues a
time-stamped, nonce-bearing challenge ("blink once inside this window") and then
checks that the characteristic blink/EOG signature actually appears **in that
window** on the frontal channels (Fp1/Fp2). A static replay or a purely synthetic
EEG stream cannot produce a *bona fide* on-cue blink, so it is rejected.

Critical ordering note
----------------------
This detector must run on the **raw, pre-ATAR** signal. ATAR's whole job is to
*remove* blink/EOG artifacts, so running liveness after preprocessing would erase
the very evidence it needs. The pipeline therefore taps the raw stream for liveness
and the cleaned stream for the biometric path.

The three checks that defeat replays
------------------------------------
1. **Presence & count** — the expected number of blinks appear inside the window.
2. **Timing** — they fall *inside* the challenge window (a mistimed/old recording fails).
3. **Pre-prompt cleanliness** — no blink appears *before* the prompt, which resists a
   spliced clip that simply contains a blink somewhere.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from .data import EEGTrial, FRONTAL_CHANNELS
    from .dsp import detect_peaks, lowpass_filter, robust_zscore
except ImportError:
    from data import EEGTrial, FRONTAL_CHANNELS
    from dsp import detect_peaks, lowpass_filter, robust_zscore


@dataclass
class Challenge:
    """An active liveness challenge.

    Attributes
    ----------
    nonce : single-use token binding the response to this challenge (anti-replay).
    prompt_time : time (s, from trial start) at which the user is prompted to blink.
    window : ``(start, end)`` seconds in which the blink response is expected.
    expected_blinks : number of blinks requested.
    tolerance : allowed deviation in observed blink count.
    issued_at : wall-clock issue time (epoch seconds).
    """

    nonce: str
    prompt_time: float
    window: Tuple[float, float]
    expected_blinks: int = 1
    tolerance: int = 0
    issued_at: float = field(default_factory=time.time)

    @property
    def blink_cue_time(self) -> float:
        """Convenience: a sensible blink onset (window midpoint) for simulation."""
        return 0.5 * (self.window[0] + self.window[1])


@dataclass
class LivenessResult:
    """Outcome of a liveness check."""

    passed: bool
    observed_in_window: int
    observed_pre_prompt: int
    expected: int
    score: float
    latency: Optional[float]
    reasons: List[str] = field(default_factory=list)


class LivenessDetector:
    """Challenge–response PAD using blink/EOG detection on frontal channels.

    Parameters
    ----------
    frontal_channels : channel names treated as frontal EOG-bearing sites.
    amp_z_thresh : robust-z amplitude threshold for a blink candidate.
    lowpass_hz : low-pass cutoff isolating the slow blink waveform.
    min_blink_ms, max_blink_ms : admissible blink duration band.
    refractory_ms : minimum spacing between distinct blinks.
    require_clean_pre_prompt : if True, any blink before the prompt fails the check.
    """

    def __init__(
        self,
        frontal_channels: Sequence[str] = FRONTAL_CHANNELS,
        amp_z_thresh: float = 4.0,
        lowpass_hz: float = 5.0,
        min_blink_ms: float = 80.0,
        max_blink_ms: float = 600.0,
        refractory_ms: float = 250.0,
        require_clean_pre_prompt: bool = True,
    ) -> None:
        self.frontal_channels = tuple(frontal_channels)
        self.amp_z_thresh = float(amp_z_thresh)
        self.lowpass_hz = float(lowpass_hz)
        self.min_blink_ms = float(min_blink_ms)
        self.max_blink_ms = float(max_blink_ms)
        self.refractory_ms = float(refractory_ms)
        self.require_clean_pre_prompt = bool(require_clean_pre_prompt)

    # ----------------------------------------------------------- challenge
    def make_challenge(
        self,
        trial_duration: float,
        n_blinks: int = 1,
        window_width: float = 1.2,
        rng: Optional[np.random.Generator] = None,
    ) -> Challenge:
        """Issue a challenge with a randomly-placed window (timing is part of the secret)."""
        rng = rng or np.random.default_rng()
        latest_start = max(0.5, trial_duration - window_width - 0.3)
        start = float(rng.uniform(0.5, latest_start)) if latest_start > 0.5 else 0.5
        end = min(trial_duration - 0.05, start + window_width)
        return Challenge(
            nonce=uuid.uuid4().hex,
            prompt_time=start,
            window=(start, end),
            expected_blinks=int(n_blinks),
            tolerance=0,
        )

    # -------------------------------------------------------------- verify
    def verify(self, raw_trial: EEGTrial, challenge: Challenge) -> LivenessResult:
        """Check a raw (pre-ATAR) trial against ``challenge``."""
        reasons: List[str] = []
        idxs = [raw_trial.channel_index(n) for n in self.frontal_channels]
        idxs = [i for i in idxs if i is not None]
        if not idxs:  # fail-closed: cannot assess liveness without frontal channels
            return LivenessResult(False, 0, 0, challenge.expected_blinks, 0.0, None,
                                  ["no_frontal_channels_available"])

        frontal = raw_trial.data[idxs].mean(axis=0)
        sf = raw_trial.sfreq
        slow = lowpass_filter(frontal, sf, self.lowpass_hz)
        zabs = np.abs(robust_zscore(slow))

        distance = max(1, int(self.refractory_ms * 1e-3 * sf))
        peaks = detect_peaks(zabs, height=self.amp_z_thresh, distance=distance)

        # Keep peaks whose duration is blink-like.
        half_level = self.amp_z_thresh * 0.5
        blink_times: List[float] = []
        for p in peaks:
            width_ms = self._width_ms(zabs, int(p), sf, half_level)
            if self.min_blink_ms <= width_ms <= self.max_blink_ms:
                blink_times.append(p / sf)

        w0, w1 = challenge.window
        in_window = [t for t in blink_times if w0 <= t <= w1]
        pre_prompt = [t for t in blink_times if t < challenge.prompt_time]

        count_ok = abs(len(in_window) - challenge.expected_blinks) <= challenge.tolerance
        presence_ok = len(in_window) >= 1
        prepane_ok = (not self.require_clean_pre_prompt) or (len(pre_prompt) == 0)

        if not presence_ok:
            reasons.append("no_blink_in_challenge_window")
        if presence_ok and not count_ok:
            reasons.append(f"blink_count_mismatch(observed={len(in_window)},"
                           f"expected={challenge.expected_blinks})")
        if not prepane_ok:
            reasons.append("blink_present_before_prompt(possible_replay)")

        passed = bool(presence_ok and count_ok and prepane_ok)
        latency = (min(in_window) - challenge.prompt_time) if in_window else None
        score = self._confidence(len(in_window), challenge.expected_blinks, prepane_ok)
        if passed and not reasons:
            reasons.append("bona_fide_on_cue_blink_detected")
        return LivenessResult(passed, len(in_window), len(pre_prompt),
                              challenge.expected_blinks, score, latency, reasons)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _width_ms(sig: np.ndarray, peak: int, sfreq: float, level: float) -> float:
        """Full width (ms) of the excursion around ``peak`` above ``level``."""
        n = len(sig)
        left = right = peak
        while left > 0 and sig[left] > level:
            left -= 1
        while right < n - 1 and sig[right] > level:
            right += 1
        return (right - left) / sfreq * 1000.0

    @staticmethod
    def _confidence(observed: int, expected: int, prepane_ok: bool) -> float:
        """Soft liveness confidence in ``[0, 1]``."""
        if observed == 0 or not prepane_ok:
            return 0.0
        return float(np.exp(-abs(observed - expected)))
