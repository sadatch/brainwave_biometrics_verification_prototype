"""Smoke tests — exercise the end-to-end path so regressions like the NumPy 2.0
``np.trapz`` removal or a broken enroll/verify are caught immediately in CI.

Run with::

    python -m pytest -q tests
    # or directly:
    python tests/test_smoke.py
"""
import os
import sys

import numpy as np

# リポジトリルートを import パスに追加（pytest / 直接実行の双方で eeg_biometric を解決）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_biometric.data import EEGDataSource, EEGTrial  # noqa: E402
from eeg_biometric.pipeline import EEGBiometricPipeline, PipelineConfig  # noqa: E402

CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
SF = 128.0
TS = 2.0


def _enrolled():
    cfg = PipelineConfig(data_source="synthetic", sfreq=SF, trial_seconds=TS,
                         n_bootstrap=4, max_channels=5, random_state=0)
    pipe = EEGBiometricPipeline(cfg)
    src = EEGDataSource(source="synthetic", sfreq=SF, trial_seconds=TS, montage=CH, seed=0)
    cue = [TS / 2.0]
    genuine = src.get_subject_trials("S001", 12, with_blink=True, blink_times=cue, base_seed=1)
    background = []
    for k, bid in enumerate(["B1", "B2", "B3"]):
        background += src.get_subject_trials(bid, 5, with_blink=True, blink_times=cue, base_seed=100 + 10 * k)
    calib_imp = src.get_subject_trials("C1", 5, with_blink=True, blink_times=cue, base_seed=200)
    pipe.enroll("S001", genuine[:8], background, calib_genuine=genuine[8:], calib_impostor=calib_imp)
    return pipe, src


def test_dsp_bandpower_runs():
    """① NumPy 2.0 で削除された np.trapz への依存が残っていないこと。"""
    from eeg_biometric.dsp import band_powers

    x = np.random.default_rng(0).standard_normal(int(SF * TS))
    bp = band_powers(x, SF)
    assert {"delta", "alpha", "beta"} <= set(bp)
    assert all(np.isfinite(v) for v in bp.values())


def test_enroll_and_verify_runs():
    """②⑤ enroll → verify が例外なく一気通貫で動くこと。"""
    pipe, src = _enrolled()
    assert "S001" in pipe.enrollments

    ch = pipe.liveness.make_challenge(trial_duration=TS)
    trial = src.get_subject_trials("S001", 1, with_blink=True,
                                   blink_times=[ch.blink_cue_time], base_seed=900)[0]
    res = pipe.verify("S001", trial, ch)
    assert res.stage in {"accept", "recognition_reject", "liveness_reject", "montage_error"}

    # 瞬目なし → liveness 不成立で必ず REJECT。
    ch2 = pipe.liveness.make_challenge(trial_duration=TS)
    no_blink = src.get_subject_trials("S001", 1, with_blink=False, base_seed=901)[0]
    assert pipe.verify("S001", no_blink, ch2).decision is False


def test_metrics_runs():
    pipe, src = _enrolled()
    cue = [TS / 2.0]
    g = src.get_subject_trials("S001", 8, with_blink=True, blink_times=cue, base_seed=300)
    i = src.get_subject_trials("S002", 8, with_blink=True, blink_times=cue, base_seed=400)
    m = pipe.biometric_metrics("S001", g, i)
    for key in ("FAR", "FRR", "ACC"):
        assert 0.0 <= m[key] <= 1.0


def test_montage_guard():
    """⑦ enroll と異なる montage は例外でなく明示的に弾くこと。"""
    pipe, _ = _enrolled()
    ch = pipe.liveness.make_challenge(trial_duration=TS)
    bad = EEGTrial(np.zeros((3, int(SF * TS))), ["X1", "X2", "X3"], SF, echoed_nonce=ch.nonce)
    res = pipe.verify("S001", bad, ch)
    assert res.decision is False
    assert res.stage in {"montage_error", "liveness_reject"}


if __name__ == "__main__":
    test_dsp_bandpower_runs()
    test_enroll_and_verify_runs()
    test_metrics_runs()
    test_montage_guard()
    print("smoke OK")
