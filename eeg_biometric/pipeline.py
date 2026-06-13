"""End-to-end integration: :class:`EEGBiometricPipeline` + runnable ``main()``.

Data flow per verification
--------------------------
                         ┌─────────────── RAW trial ───────────────┐
                         │                                          │
                    (frontal taps)                          (all channels)
                         │                                          │
                 ┌───────▼────────┐                        ┌────────▼────────┐
                 │ LivenessDetector│  fail → REJECT         │ ATAR preprocess │
                 │  (pre-ATAR!)    │───────────────►        │ (blink removed) │
                 └───────┬─────────┘                        └────────┬────────┘
                         │ pass                                      │
                         │                               ┌───────────▼───────────┐
                         │                               │ Elastic-Net channels  │
                         │                               └───────────┬───────────┘
                         │                               ┌───────────▼───────────┐
                         │                               │ MAEEG / Handcrafted    │
                         │                               │ embedding              │
                         │                               └───────────┬───────────┘
                         │                               ┌───────────▼───────────┐
                         │                               │ OC-SVM ⊕ LightGBM      │
                         │                               └───────────┬───────────┘
                         └────────────── AND ────────────────────────┘
                                          │
                                   ACCEPT / REJECT

Liveness sees the raw signal (so the blink/EOG evidence survives); the biometric
path sees the ATAR-cleaned signal (so identity features are not dominated by that
same blink). Final acceptance requires **both** to pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from . import dsp, features as _features, recognition as _rec
    from .data import EEGDataSource, EEGTrial
    from .preprocess import ATARPreprocessor
    from .channels import ElasticNetChannelSelector, PerChannelFeatureExtractor, build_selection_dataset
    from .features import make_encoder
    from .recognition import OpenSetRecognizer
    from .liveness import Challenge, LivenessDetector, LivenessResult
except ImportError:  # loose-script execution
    import dsp, features as _features, recognition as _rec
    from data import EEGDataSource, EEGTrial
    from preprocess import ATARPreprocessor
    from channels import ElasticNetChannelSelector, PerChannelFeatureExtractor, build_selection_dataset
    from features import make_encoder
    from recognition import OpenSetRecognizer
    from liveness import Challenge, LivenessDetector, LivenessResult


# --------------------------------------------------------------------------- #
# Configuration & result containers
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Tunable knobs for the whole pipeline."""

    # data
    data_source: str = "auto"
    sfreq: float = 160.0
    trial_seconds: float = 3.0
    # ATAR
    atar_wavelet: str = "db4"
    atar_mode: str = "soft"
    atar_beta: float = 0.5
    atar_win_seconds: float = 1.0
    atar_decomposition: str = "wpd"   # "wpd" (survey-specified) | "dwt"
    # channel selection
    l1_ratio: float = 0.5
    select_C: float = 1.0
    n_bootstrap: int = 15
    selection_threshold: float = 0.6
    max_channels: Optional[int] = 8
    # encoder
    encoder_prefer: str = "auto"      # "auto"|"deep"|"maeeg"|"gmaeeg"
    encoder_variant: str = "maeeg"
    embed_dim: int = 64
    pretrained_path: Optional[str] = None
    # recognizer
    nu: float = 0.1
    fusion_weight: float = 0.5
    decision_threshold: float = 0.5
    recognizer_mode: str = "fusion"   # "fusion"|"and"
    target_far: float = 0.05
    # liveness
    liveness_amp_z: float = 4.0
    # GAN augmentation / red-team (defensive)
    use_gan_augmentation: bool = False
    gan_backend: str = "surrogate"    # "surrogate" | "gan" | "auto"
    n_synthetic_augment: int = 12
    # misc
    random_state: int = 0


@dataclass
class Enrollment:
    """Per-identity enrolled template/models."""

    subject_id: str
    selector: ElasticNetChannelSelector
    encoder: object
    recognizer: OpenSetRecognizer
    selected_channels: List[str]
    threshold: float
    n_genuine: int
    n_background: int


@dataclass
class AuthResult:
    """Final verification verdict with full provenance."""

    claimed_id: str
    decision: bool
    stage: str
    liveness: Optional[LivenessResult]
    recognition: Optional[dict]
    reason: str = ""


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class EEGBiometricPipeline:
    """Composable EEG 1:1 verification pipeline with active liveness."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        c = self.config
        self.atar = ATARPreprocessor(
            wavelet=c.atar_wavelet, mode=c.atar_mode, beta=c.atar_beta,
            win_seconds=c.atar_win_seconds, decomposition=c.atar_decomposition,
        )
        self.feature_extractor = PerChannelFeatureExtractor()
        self.liveness = LivenessDetector(amp_z_thresh=c.liveness_amp_z)
        self.enrollments: Dict[str, Enrollment] = {}

    # ------------------------------------------------------------- helpers
    def _embed_trials(self, selector, encoder, trials: Sequence[EEGTrial]) -> np.ndarray:
        """ATAR-clean is assumed already applied; embed each trial on selected channels."""
        rows = [encoder.embed(t, channel_idx=selector.selected_indices_for(t)) for t in trials]
        return np.vstack(rows)

    # -------------------------------------------------------------- enroll
    def enroll(
        self,
        subject_id: str,
        genuine_trials: Sequence[EEGTrial],
        background_trials: Sequence[EEGTrial],
        calib_genuine: Optional[Sequence[EEGTrial]] = None,
        calib_impostor: Optional[Sequence[EEGTrial]] = None,
    ) -> Dict[str, object]:
        """Enroll ``subject_id`` from genuine + background cohort trials.

        Steps: ATAR-clean → Elastic-Net channel selection (genuine vs background) →
        embed selected channels → fit OC-SVM ⊕ LightGBM → optional FAR-targeted
        threshold calibration. Returns a summary dict and stores the template.
        """
        c = self.config
        # Optional GAN/surrogate augmentation of the (few) enrollment trials.
        if c.use_gan_augmentation and len(genuine_trials) > 0:
            try:
                from .adversarial import PresentationAttackSimulator, make_generator
            except ImportError:
                from adversarial import PresentationAttackSimulator, make_generator
            sim = PresentationAttackSimulator(
                make_generator(c.gan_backend, seed=c.random_state)).fit(genuine_trials)
            genuine_trials = sim.augment(genuine_trials, c.n_synthetic_augment)

        gen = self.atar.transform_many(genuine_trials)
        bg = self.atar.transform_many(background_trials)

        # 1) Stable channel/feature selection.
        X, y, ch_of_feat, ch_names = build_selection_dataset(gen, bg, self.feature_extractor)
        selector = ElasticNetChannelSelector(
            l1_ratio=c.l1_ratio, C=c.select_C, n_bootstrap=c.n_bootstrap,
            selection_threshold=c.selection_threshold, max_channels=c.max_channels,
            random_state=c.random_state,
        ).fit(X, y, ch_of_feat, ch_names)

        # 2) Encoder (deep if requested+available, else handcrafted).
        n_sel = len(selector.selected_channels) or len(ch_names)
        input_len = gen[0].n_times
        encoder = make_encoder(
            prefer=c.encoder_prefer, variant=c.encoder_variant, n_channels=n_sel,
            input_len=input_len, embed_dim=c.embed_dim, pretrained_path=c.pretrained_path,
            seed=c.random_state,
        )

        # 3) Embed and fit the open-set recognizer.
        g_emb = self._embed_trials(selector, encoder, gen)
        b_emb = self._embed_trials(selector, encoder, bg)
        recognizer = OpenSetRecognizer(
            nu=c.nu, fusion_weight=c.fusion_weight, threshold=c.decision_threshold,
            mode=c.recognizer_mode, random_state=c.random_state,
        ).fit(g_emb, b_emb)

        # 4) Optional threshold calibration to a target FAR.
        if calib_genuine and calib_impostor:
            cg = self.atar.transform_many(calib_genuine)
            ci = self.atar.transform_many(calib_impostor)
            cg_emb = self._embed_trials(selector, encoder, cg)
            ci_emb = self._embed_trials(selector, encoder, ci)
            recognizer.calibrate_threshold(cg_emb, ci_emb, target_far=c.target_far)

        enr = Enrollment(
            subject_id=subject_id, selector=selector, encoder=encoder, recognizer=recognizer,
            selected_channels=selector.selected_channels, threshold=recognizer.threshold,
            n_genuine=len(gen), n_background=len(bg),
        )
        self.enrollments[subject_id] = enr
        return {
            "subject_id": subject_id,
            "selected_channels": enr.selected_channels,
            "n_selected": len(enr.selected_channels),
            "selection_method": selector.result_.method if selector.result_ else "n/a",
            "encoder": getattr(encoder, "name", type(encoder).__name__),
            "embed_dim": int(g_emb.shape[1]),
            "recognizer_backend": recognizer.backend,
            "threshold": round(float(enr.threshold), 3),
        }

    # -------------------------------------------------------------- verify
    def verify(self, claimed_id: str, raw_trial: EEGTrial, challenge: Challenge) -> AuthResult:
        """Verify a raw trial against the enrolled ``claimed_id`` under ``challenge``.

        Liveness runs first on the RAW signal (fail-fast). Only if it passes do we
        ATAR-clean, select channels, embed, and score the biometric. Final accept =
        liveness AND recognition.
        """
        if claimed_id not in self.enrollments:
            return AuthResult(claimed_id, False, "no_enrollment", None, None,
                              reason=f"{claimed_id} is not enrolled")
        enr = self.enrollments[claimed_id]

        # 1) Active liveness on the raw (pre-ATAR) signal.
        live = self.liveness.verify(raw_trial, challenge)
        if not live.passed:
            return AuthResult(claimed_id, False, "liveness_reject", live, None,
                              reason="; ".join(live.reasons) or "liveness_failed")

        # 2) Biometric path on the ATAR-cleaned signal.
        cleaned = self.atar.transform(raw_trial)
        idx = enr.selector.selected_indices_for(cleaned)
        emb = enr.encoder.embed(cleaned, channel_idx=idx)
        accept, scores = enr.recognizer.verify(emb, threshold=enr.threshold)

        decision = bool(live.passed and accept)
        stage = "accept" if decision else "recognition_reject"
        reason = "liveness+biometric_match" if decision else \
                 f"biometric_score {scores['fused']:.3f} < thr {enr.threshold:.3f}"
        return AuthResult(claimed_id, decision, stage, live, scores, reason=reason)

    # ------------------------------------------------------------ metrics
    def biometric_metrics(
        self,
        claimed_id: str,
        genuine_trials: Sequence[EEGTrial],
        impostor_trials: Sequence[EEGTrial],
    ) -> Dict[str, float]:
        """FAR/FRR/ACC of the biometric branch alone (ATAR-cleaned, liveness aside)."""
        enr = self.enrollments[claimed_id]
        g = self._embed_trials(enr.selector, enr.encoder, self.atar.transform_many(genuine_trials))
        i = self._embed_trials(enr.selector, enr.encoder, self.atar.transform_many(impostor_trials))
        return enr.recognizer.evaluate(list(g), list(i), threshold=enr.threshold)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _gather(source: EEGDataSource, ids, n_each, base_seed=0, with_blink=False, blink_times=None):
    trials: List[EEGTrial] = []
    for k, sid in enumerate(ids):
        trials += source.get_subject_trials(
            sid, n_each, with_blink=with_blink, blink_times=blink_times, base_seed=base_seed + 10 * k)
    return trials


def _print_backends(pipe: EEGBiometricPipeline, source: EEGDataSource) -> None:
    print("Active backends")
    print(f"  data source        : {source.active_source}")
    print(f"  ATAR               : {pipe.atar.backend}")
    print(f"  SciPy              : {'yes' if dsp.have_scipy() else 'no (numpy fallback)'}")
    print(f"  PyTorch (MAEEG)    : {'yes' if _features._HAVE_TORCH else 'no (handcrafted encoder)'}")
    print(f"  scikit-learn       : {'yes' if _rec._HAVE_SKLEARN else 'no (numpy fallback)'}")
    print(f"  LightGBM           : {'yes' if _rec._HAVE_LGBM else 'no (gradient-boosting/logreg fallback)'}")


def _showcase_deep_encoder(config: PipelineConfig, trial: EEGTrial) -> None:
    """Prove the MAEEG transformer runs a real forward pass (architecture check)."""
    if not _features._HAVE_TORCH:
        print("  [MAEEG] PyTorch not installed — deep encoder skipped (handcrafted in use).")
        return
    try:
        enc = make_encoder(prefer="deep", variant=config.encoder_variant,
                           n_channels=trial.n_channels, input_len=trial.n_times,
                           embed_dim=config.embed_dim, seed=config.random_state)
        emb = enc.embed(trial)
        n_params = sum(p.numel() for p in enc.module.parameters())
        print(f"  [MAEEG] {enc.name}: forward OK | params={n_params:,} | "
              f"embedding shape={emb.shape} (frozen; load_pretrained() to use for scoring)")
    except Exception as exc:  # pragma: no cover
        print(f"  [MAEEG] showcase failed: {exc}")


def main() -> None:
    """Run the full pipeline init → preprocess → selection → features → open-set →
    liveness → decision on demo data, exercising four attack/usage scenarios."""
    config = PipelineConfig()
    rng = np.random.default_rng(config.random_state + 1)
    pipe = EEGBiometricPipeline(config)

    print("=" * 74)
    print(" EEG Biometric Verification — defensive research prototype (1:1)")
    print("=" * 74)

    source = EEGDataSource(source=config.data_source, sfreq=config.sfreq,
                           trial_seconds=config.trial_seconds, seed=config.random_state)
    _print_backends(pipe, source)

    # ---- cohorts (background cohort is DISJOINT from evaluation impostors) ----
    enrollee = "S001"
    eval_impostors = ["S002", "S003", "S004"]
    background_ids = ["B01", "B02", "B03", "B04"]

    genuine_all = source.get_subject_trials(enrollee, n_trials=24, base_seed=1)
    genuine_enroll, genuine_calib = genuine_all[:18], genuine_all[18:]
    background = _gather(source, background_ids, n_each=8, base_seed=100)
    calib_impostor = source.get_subject_trials(eval_impostors[0], n_trials=8, base_seed=200)

    print("\nEnrolling", enrollee, "...")
    summary = pipe.enroll(enrollee, genuine_enroll, background,
                          calib_genuine=genuine_calib, calib_impostor=calib_impostor)
    print(f"  selected {summary['n_selected']} channels via {summary['selection_method']}: "
          f"{summary['selected_channels']}")
    print(f"  encoder={summary['encoder']} (dim={summary['embed_dim']}), "
          f"recognizer[{summary['recognizer_backend']}], threshold={summary['threshold']}")

    # Deep-encoder architectural showcase (separate from scoring encoder).
    _showcase_deep_encoder(config, pipe.atar.transform(genuine_enroll[0]))

    # ---- scenarios -----------------------------------------------------------
    challenge = pipe.liveness.make_challenge(trial_duration=config.trial_seconds, rng=rng)
    cue = challenge.blink_cue_time
    early = max(0.1, challenge.prompt_time - 0.7)
    print(f"\nLiveness challenge: blink within window "
          f"[{challenge.window[0]:.2f}s, {challenge.window[1]:.2f}s]  nonce={challenge.nonce[:8]}")

    # Defensive red-team (survey hypothesis 4): synthesise a spoof of the enrollee
    # with NO on-cue blink and confirm the pipeline rejects it.
    try:
        from .adversarial import PresentationAttackSimulator, make_generator
    except ImportError:
        from adversarial import PresentationAttackSimulator, make_generator
    attacker = PresentationAttackSimulator(
        make_generator(config.gan_backend, seed=config.random_state)).fit(genuine_enroll)
    spoof_trial = attacker.synthesize_spoofs(1, inject_blink=False)[0]
    print(f"Attack simulator: {attacker.generator.name} "
          f"(defensive red-team; spoof carries no on-cue blink)")

    scenarios = [
        ("S1 genuine + on-cue blink     (expect ACCEPT)",
         source.get_subject_trials(enrollee, 1, with_blink=True, blink_times=[cue], base_seed=900)[0]),
        ("S2 impostor + on-cue blink    (expect REJECT: identity)",
         source.get_subject_trials(eval_impostors[1], 1, with_blink=True, blink_times=[cue], base_seed=901)[0]),
        ("S3 genuine replay, NO blink   (expect REJECT: liveness)",
         source.get_subject_trials(enrollee, 1, with_blink=False, base_seed=902)[0]),
        ("S4 genuine, mistimed blink    (expect REJECT: liveness)",
         source.get_subject_trials(enrollee, 1, with_blink=True, blink_times=[early], base_seed=903)[0]),
        ("S5 GAN/surrogate spoof, no blink (expect REJECT: liveness)", spoof_trial),
    ]

    print("\nResults")
    print("-" * 74)
    for label, trial in scenarios:
        res = pipe.verify(enrollee, trial, challenge)
        verdict = "ACCEPT" if res.decision else "REJECT"
        live = res.liveness
        live_str = f"live={'pass' if (live and live.passed) else 'fail'}" + \
                   (f"(blinks_in_win={live.observed_in_window})" if live else "")
        bio = res.recognition
        bio_str = f"bio_fused={bio['fused']:.3f}" if bio else "bio=skipped"
        print(f"{verdict:6} | {label}")
        print(f"         {live_str}, {bio_str}  ->  {res.reason}")
    print("-" * 74)

    # ---- batch biometric metrics (liveness held aside) -----------------------
    eval_genuine = source.get_subject_trials(enrollee, n_trials=20, base_seed=300)
    eval_impostor = _gather(source, eval_impostors, n_each=8, base_seed=400)
    metrics = pipe.biometric_metrics(enrollee, eval_genuine, eval_impostor)
    print(f"\nBiometric branch metrics (threshold={metrics['threshold']:.3f}, "
          f"mode={metrics['mode']}):")
    print(f"  FAR={metrics['FAR']:.3f}  FRR={metrics['FRR']:.3f}  ACC={metrics['ACC']:.3f}  "
          f"(genuine n={metrics['n_genuine']}, impostor n={metrics['n_impostor']})")
    print("\nDone. This is a defensive research prototype on public/synthetic data only.")


if __name__ == "__main__":
    main()
