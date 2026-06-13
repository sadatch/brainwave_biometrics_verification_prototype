"""eeg_biometric — defensive EEG-based 1:1 biometric verification (research prototype).

A modular, server-side inference pipeline for **verifying** a claimed identity from
EEG and **rejecting** impostors and presentation attacks. It composes well-established
signal-processing and machine-learning components:

    dsp          Shared DSP helpers (PSD, band-power, robust stats, peak finding).
    data         EEGTrial container + EEGDataSource (MNE public sample -> synthetic).
    preprocess   ATARPreprocessor — wavelet artifact removal, tunable, single-channel.
    channels     ElasticNetChannelSelector — stable L1/L2 channel/feature selection.
    features     MAEEGEncoder / GMAEEGEncoder (PyTorch) + HandcraftedSpectralEncoder.
    recognition  OpenSetRecognizer — One-Class SVM (SVDD) + LightGBM ensemble.
    liveness     LivenessDetector — ISO/IEC 30107 active challenge-response PAD.
    adversarial  EEG-GAN / surrogate generator + presentation-attack red-teaming (defensive).
    pipeline     EEGBiometricPipeline integration + main() end-to-end demo.

Scope & ethics
--------------
This is a **defensive**, academic prototype. It authenticates legitimate users and
detects spoofing. It uses only public MNE sample data or NumPy-synthesised waveforms;
real-person data collection and production deployment are out of scope.
"""

from .data import EEGTrial, EEGDataSource  # noqa: F401

__all__ = ["EEGTrial", "EEGDataSource"]
__version__ = "0.1.0"
