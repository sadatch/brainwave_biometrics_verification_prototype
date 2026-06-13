"""Smoke tests — exercise the end-to-end path so regressions like the NumPy 2.0
``np.trapz`` removal, a broken enroll/verify, a calibration that lets FAR blow up,
or a missing anti-replay check are caught immediately in CI.

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
CUE = [TS / 2.0]


def _new():
    cfg = PipelineConfig(data_source="synthetic", sfreq=SF, trial_seconds=TS,
                         n_bootstrap=4, max_channels=5, random_state=0)
    return EEGBiometricPipeline(cfg), EEGDataSource(
        source="synthetic", sfreq=SF, trial_seconds=TS, montage=CH, seed=0)


def _fetch(src, sid, n, seed):
    return src.get_subject_trials(sid, n, with_blink=True, blink_times=CUE, base_seed=seed)


def _enroll(pipe, src, calib_ids):
    g = _fetch(src, "S001", 12, 1)
    bg = []
    for k, b in enumerate(["B1", "B2", "B3"]):
        bg += _fetch(src, b, 5, 100 + 10 * k)
    ci = []
    for k, c in enumerate(calib_ids):
        ci += _fetch(src, c, 5, 500 + 10 * k)
    pipe.enroll("S001", g[:8], bg, calib_genuine=g[8:], calib_impostor=ci)
    return pipe


def _enrolled(calib_ids=("C1", "C2", "C3", "C4", "C5")):
    pipe, src = _new()
    return _enroll(pipe, src, list(calib_ids)), src


def _far_with_calib(calib_ids):
    pipe, src = _new()
    _enroll(pipe, src, list(calib_ids))
    eg = _fetch(src, "S001", 8, 300)
    ei = []
    for k, s in enumerate(["S002", "S003", "S004"]):
        ei += _fetch(src, s, 6, 700 + 10 * k)
    return pipe.biometric_metrics("S001", eg, ei)["FAR"]


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

    ch2 = pipe.liveness.make_challenge(trial_duration=TS)
    no_blink = src.get_subject_trials("S001", 1, with_blink=False, base_seed=901)[0]
    assert pipe.verify("S001", no_blink, ch2).decision is False


def test_metrics_runs():
    pipe, src = _enrolled()
    g = _fetch(src, "S001", 8, 300)
    i = _fetch(src, "S002", 8, 400)
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


def test_anti_replay_rejects_used_nonce():
    """③ 既定で nonce 追跡が有効。同じ nonce の再提出はリプレイとして拒否されること。"""
    pipe, src = _enrolled()
    assert pipe.liveness.track_nonce is True            # 既定 ON
    ch = pipe.liveness.make_challenge(trial_duration=TS)
    trial = src.get_subject_trials("S001", 1, with_blink=True,
                                   blink_times=[ch.blink_cue_time], base_seed=900)[0]
    pipe.verify("S001", trial, ch)                      # 1 回目: nonce 消費
    r2 = pipe.verify("S001", trial, ch)                 # 2 回目: 同じ nonce
    assert r2.decision is False
    assert r2.liveness is not None
    assert "nonce_already_used(replay)" in (r2.liveness.reasons or [])


def test_calibration_multiple_impostors_lowers_far():
    """①② 較正 impostor を複数にすると未知 impostor への FAR が下がる（汎化する）こと。"""
    far_single = _far_with_calib(["C1"])
    far_multi = _far_with_calib(["C1", "C2", "C3", "C4", "C5"])
    assert far_multi <= far_single + 1e-9     # 単一より悪化しない
    assert far_multi <= 0.34                  # 回帰に対する緩い上限


if __name__ == "__main__":
    test_dsp_bandpower_runs()
    test_enroll_and_verify_runs()
    test_metrics_runs()
    test_montage_guard()
    test_anti_replay_rejects_used_nonce()
    test_calibration_multiple_impostors_lowers_far()
    print("smoke OK")
