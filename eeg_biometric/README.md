# eeg_biometric — defensive EEG 1:1 verification (research prototype)

A modular, server-side inference pipeline that **verifies a claimed identity from EEG**
and **rejects impostors and presentation attacks**. It is a defensive, academic
prototype: it runs only on **MNE public sample data** or **NumPy-synthesised waveforms**,
and is **not** for real-person data collection or production use.

## Architecture

```
RAW trial ─┬─ frontal taps ─► LivenessDetector (pre-ATAR)  ──fail──► REJECT
           │                                   │pass
           └─ all channels ─► ATAR ─► Elastic-Net channels ─► MAEEG/Handcrafted
                                       embedding ─► OC-SVM ⊕ LightGBM ─┐
                                                                       │
                              ACCEPT  ◄────────────── AND ─────────────┘
```

Liveness reads the **raw** signal (so the blink/EOG evidence survives); the biometric
path reads the **ATAR-cleaned** signal (so identity features are not dominated by that
same blink). Final acceptance requires **both** stages to pass.

## Modules

| File | Class | Role |
|------|-------|------|
| `dsp.py` | — | Shared DSP (PSD, band-power, robust z, peak finding); SciPy→NumPy fallbacks |
| `data.py` | `EEGDataSource`, `EEGTrial` | MNE EEGBCI loader with synthetic fallback + per-subject signatures |
| `preprocess.py` | `ATARPreprocessor` | Wavelet artifact removal — tunable, single-channel, low-latency (WPD/DWT) |
| `channels.py` | `ElasticNetChannelSelector` | Stable L1/L2 channel/feature selection (stability selection) |
| `features.py` | `MAEEGEncoder`, `GMAEEGEncoder`, `HandcraftedSpectralEncoder` | Frozen embedding: 6-conv → 8-layer Transformer (192→64) + handcrafted fallback |
| `recognition.py` | `OpenSetRecognizer` | One-Class SVM (SVDD) ⊕ LightGBM, calibrated fusion |
| `liveness.py` | `LivenessDetector`, `Challenge` | ISO/IEC 30107 active challenge–response PAD |
| `adversarial.py` | `EEGGAN`, `SurrogateEEGGenerator`, `PresentationAttackSimulator` | GAN/surrogate augmentation + spoof red-teaming (defensive) |
| `pipeline.py` | `EEGBiometricPipeline`, `main()` | Integration + end-to-end demo |

## Key design decisions

**ATAR instead of ICA.** ICA needs the full multichannel block and a relatively costly
unmixing estimate. ATAR works one channel at a time in short overlapping windows, which
suits low-latency streaming (ESP32→server). Its core premise is the *opposite* of
classic wavelet denoising: artifacts are the **high-amplitude** wavelet coefficients, so
ATAR suppresses the *large* coefficients and keeps the smaller neural rhythm. A single
knob `beta` sets the operating point (gentle→aggressive); modes are `soft`/`linatten`/`elim`.
The decomposition defaults to **Wavelet Packet Decomposition (WPD)** — the variant the
survey specifies, splitting both approximation and detail branches for uniform frequency
resolution — with a lighter multi-level **DWT** as an option.

**Elastic Net for channel selection.** Volume conduction makes neighbouring electrodes
highly correlated. Pure L1 (Lasso) would keep an arbitrary one of a correlated cluster
and flip its choice across resamples; the L2 term adds Elastic Net's *grouping effect* so
correlated channels are kept/dropped together. We wrap it in **stability selection**
(bootstrap refits, keep frequently-selected features) for a reproducible channel set.

**MAEEG/GMAEEG with an honest fallback.** The intended extractor is a pretrained MAEEG
masked auto-encoder used *frozen*. Following Chien et al., `MAEEGEncoder` uses a 6-layer
convolutional frontend (GroupNorm + GELU + Dropout) → 64-d tokens → an 8-layer Transformer
at `model_dim=192` → a 64-d context embedding; its `MaskedReconstructionPretrainer`
illustrates the Gaussian-noise masking + cosine-similarity reconstruction objective.
`GMAEEGEncoder` (Fu et al.) adds a **learnable dynamic adjacency matrix** following the
paper's exact update `A = ReLU(tanh(W₂·ELU(W₁·Ã_init)))` (with `Ã_init = E·Eᵀ`,
row-normalised with self-loops) and a graph convolution (`ELU(Â·X)`) across electrodes,
so the *connectivity topology* itself becomes part of the signature. Because no pretrained weights ship here, a random-init transformer is **not**
identity-discriminative, so the factory defaults to a `HandcraftedSpectralEncoder`
(band-power + Hjorth + spectral-edge) that genuinely separates subjects on the demo data.
The deep modules are still real and runnable (the demo runs a forward pass and reports the
parameter count); `load_pretrained(path)` or `prefer="deep"` promotes them to the scoring
encoder. This keeps the architecture truthful without faking learned discriminability.

**OC-SVM ⊕ LightGBM for open-set.** 1:1 verification faces unbounded unknown impostors.
The One-Class SVM (SVDD) is trained on the enrollee's genuine embeddings *only* and bounds
the false-accept rate against *novel* attackers (open-set term). LightGBM, trained
genuine-vs-background, sharpens the boundary against the *known* impostor distribution
(closed-set term). Scores are Platt-calibrated and fused; an `and` gate (both must pass) is
available as a stricter, lower-FAR alternative. The demo keeps the background cohort
**disjoint** from the evaluation impostors so the open-set claim is honest.

**Liveness before ATAR.** ATAR removes exactly the blink/EOG the liveness check relies on,
so the detector taps the raw stream. The active challenge carries a nonce and a random
time window; it checks blink **presence/count**, **timing inside the window**, and
**no blink before the prompt** — together these reject static replays and spliced clips.

**GAN as a dual-use *defensive* tool.** `adversarial.py` provides a generator (a small
`EEGGAN`, or a NumPy phase-randomised `SurrogateEEGGenerator` fallback) for two purposes:
(1) **augmenting** the few enrollment trials, and (2) **red-teaming** — synthesising spoof
EEG and confirming the pipeline rejects it. The key defensive insight is that a generator
can imitate *resting* EEG statistics but cannot produce a *bona fide on-cue blink*
synchronised to a random challenge, so liveness rejects these spoofs regardless of spectral
realism (demo scenario S5). This module only generates and scores signals against our own
pipeline — it has no injection capability and targets no real person.

## Dependency / fallback matrix

Everything runs with just NumPy. Each optional library upgrades a stage:

- **SciPy** → Butterworth/Welch/`find_peaks`; else FFT/periodogram/local-maxima.
- **PyWavelets** → true ATAR (WPD/DWT); else robust-amplitude attenuation.
- **scikit-learn** → Elastic-Net logistic + One-Class SVM + scaling; else NumPy logreg + Mahalanobis one-class.
- **LightGBM** → boosted branch; else gradient boosting / logistic regression.
- **PyTorch** → MAEEG/GMAEEG encoder + EEG-GAN; else handcrafted encoder + phase-randomised surrogate.
- **MNE** → PhysioNet EEGBCI data; else synthetic waveforms.

## Run

```bash
pip install -r requirements.txt          # or just `pip install numpy` for the minimal path
python -m eeg_biometric.pipeline         # from the parent directory
# or:  cd eeg_biometric && python pipeline.py
```

The demo prints the active backends, enrolls subject `S001`, then runs five scenarios —
genuine+blink (accept), impostor+blink (reject on identity), genuine replay without a
blink (reject on liveness), a mistimed blink (reject on liveness), and a GAN/surrogate
spoof without a blink (reject on liveness) — followed by FAR/FRR/ACC for the biometric
branch. Enrollment-time GAN augmentation is available via
`PipelineConfig(use_gan_augmentation=True)`.

## Limitations & next steps

This is a skeleton. The MAEEG path needs real pretrained weights to become the scoring
encoder; thresholds and `nu`/`beta` need tuning on real recordings; the liveness
`require_clean_pre_prompt` rule may need relaxing for data with spontaneous blinks; and the
synthetic generator is a stand-in, not a model of real inter-subject EEG variability.
Natural extensions: template ageing / re-enrollment, score-level fusion calibration on a
held-out cohort, multi-session evaluation, and a streaming (block-wise) ATAR + liveness
front end matching the ESP32 capture path.
