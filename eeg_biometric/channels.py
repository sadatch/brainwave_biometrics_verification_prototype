"""Elastic-Net channel / feature selection with stability selection.

Goal
----
Pick a small, *stable* set of channels (and per-channel features) that carry the
identity signal, while coping with the multicollinearity that volume conduction
imposes on scalp EEG (neighbouring electrodes see overlapping sources).

Why Elastic Net (L1 + L2)
-------------------------
* Pure **L1** (Lasso) would arbitrarily keep one of a cluster of correlated channels
  and drop the rest — and *which* one it keeps flips between data resamples.
* The **L2** term adds Elastic Net's *grouping effect*: correlated predictors receive
  similar coefficients, so a volume-conduction cluster tends to be kept or dropped
  *together*, which is far more reproducible.
* We further wrap the fit in **stability selection** (refit over bootstrap resamples,
  keep features chosen frequently), turning a single noisy fit into a robust ranking.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .data import EEGTrial
    from .dsp import EEG_BANDS, band_powers, hjorth_parameters, spectral_edge_frequency
except ImportError:
    from data import EEGTrial
    from dsp import EEG_BANDS, band_powers, hjorth_parameters, spectral_edge_frequency

try:
    from sklearn.linear_model import LogisticRegression
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


class PerChannelFeatureExtractor:
    """Compute interpretable per-channel features used for channel selection.

    Per channel: 5 relative band powers (δ θ α β γ), 3 Hjorth parameters
    (activity, mobility, complexity) and the 95% spectral-edge frequency → 9
    features/channel. Producing per-channel features (rather than a single global
    vector) is what lets the selector reason at *channel* granularity.
    """

    FEATURE_NAMES = list(EEG_BANDS.keys()) + ["hjorth_activity", "hjorth_mobility",
                                              "hjorth_complexity", "spectral_edge"]

    def extract(self, trial: EEGTrial) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
        """Return ``(vector, feature_names, channel_of_feature, channel_names)``."""
        feats: List[float] = []
        names: List[str] = []
        ch_of_feat: List[int] = []
        for ci, ch in enumerate(trial.channels):
            x = trial.data[ci]
            bp = band_powers(x, trial.sfreq, relative=True)
            act, mob, comp = hjorth_parameters(x)
            sef = spectral_edge_frequency(x, trial.sfreq)
            values = [bp[b] for b in EEG_BANDS] + [act, mob, comp, sef]
            for fname, val in zip(self.FEATURE_NAMES, values):
                feats.append(float(val))
                names.append(f"{ch}:{fname}")
                ch_of_feat.append(ci)
        return np.asarray(feats), names, np.asarray(ch_of_feat, dtype=int), list(trial.channels)

    def extract_matrix(self, trials: Sequence[EEGTrial]) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
        """Stack feature vectors for many trials into ``X`` and shared metadata."""
        rows = []
        names = ch_of_feat = channel_names = None
        for t in trials:
            vec, names, ch_of_feat, channel_names = self.extract(t)
            rows.append(vec)
        return np.vstack(rows), names, ch_of_feat, channel_names


def build_selection_dataset(
    positives: Sequence[EEGTrial],
    negatives: Sequence[EEGTrial],
    extractor: Optional[PerChannelFeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build ``(X, y, channel_of_feature, channel_names)`` for selector training.

    ``positives`` are the enrollee's (genuine) trials, ``negatives`` the background
    cohort. Labels are 1/0 respectively.
    """
    extractor = extractor or PerChannelFeatureExtractor()
    Xp, _, ch_of_feat, ch_names = extractor.extract_matrix(positives)
    Xn, _, _, _ = extractor.extract_matrix(negatives)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return X, y, ch_of_feat, ch_names


@dataclass
class ChannelSelectionResult:
    """Outcome of channel selection."""

    selected_channels: List[str]
    channel_scores: Dict[str, float]
    selected_feature_mask: np.ndarray
    method: str
    feature_selection_freq: np.ndarray = field(default_factory=lambda: np.array([]))


class ElasticNetChannelSelector:
    """Stable channel/feature selection via bootstrapped Elastic-Net logistic fits.

    Parameters
    ----------
    l1_ratio : Elastic-Net mix (1.0 = pure L1, 0.0 = pure L2).
    C : inverse regularisation strength.
    n_bootstrap : number of resampled fits for stability selection.
    selection_threshold : minimum per-channel selection frequency to keep a channel.
    min_channels, max_channels : guard rails on the selected count.
    standardize : z-score features before fitting (recommended).
    random_state : RNG seed.
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        C: float = 1.0,
        n_bootstrap: int = 25,
        selection_threshold: float = 0.6,
        min_channels: int = 3,
        max_channels: Optional[int] = None,
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.l1_ratio = float(l1_ratio)
        self.C = float(C)
        self.n_bootstrap = int(n_bootstrap)
        self.selection_threshold = float(selection_threshold)
        self.min_channels = int(min_channels)
        self.max_channels = max_channels
        self.standardize = standardize
        self.random_state = int(random_state)
        # learned state
        self.result_: Optional[ChannelSelectionResult] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    @property
    def backend(self) -> str:
        return "elasticnet-logreg" if _HAVE_SKLEARN else "anova-fscore-fallback"

    @property
    def selected_channels(self) -> List[str]:
        return list(self.result_.selected_channels) if self.result_ else []

    # --------------------------------------------------------------- fitting
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channel_of_feature: np.ndarray,
        channel_names: Sequence[str],
    ) -> "ElasticNetChannelSelector":
        """Fit the selector and record the chosen channels.

        ``channel_of_feature[j]`` is the channel index that feature column ``j``
        belongs to; ``channel_names`` maps channel indices to names.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        channel_of_feature = np.asarray(channel_of_feature, dtype=int)
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
            Xs = (X - self.mean_) / self.std_
        else:
            Xs = X
        if _HAVE_SKLEARN and len(np.unique(y)) == 2:
            feat_freq = self._stability_selection(Xs, y)
            method = "elasticnet-stability"
        else:
            feat_freq = self._fscore_selection(Xs, y)
            method = self.backend
        self.result_ = self._aggregate_to_channels(feat_freq, channel_of_feature, channel_names, method)
        return self

    def _stability_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fraction of bootstrap fits in which each feature has a non-zero coef."""
        rng = np.random.default_rng(self.random_state)
        n_feat = X.shape[1]
        counts = np.zeros(n_feat)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        successful = 0
        for _ in range(self.n_bootstrap):
            bs = np.concatenate([
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ])
            max_iter = 3000
            try:
                clf = LogisticRegression(
                    penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio,
                    C=self.C, max_iter=max_iter, tol=1e-3,
                )
                with warnings.catch_warnings():
                    # saga 未収束 / penalty 非推奨の大量警告は抑制（収束は n_iter_ で判定）。
                    warnings.simplefilter("ignore")
                    clf.fit(X[bs], y[bs])
            except Exception:
                continue
            # 未収束の当て嵌めは選択頻度推定を歪めるため計数しない。
            if int(np.max(np.atleast_1d(getattr(clf, "n_iter_", [max_iter])))) >= max_iter:
                continue
            counts += (np.abs(clf.coef_.ravel()) > 1e-6).astype(float)
            successful += 1
        if successful == 0:  # 収束した当て嵌めが皆無 → F-score にフォールバック
            warnings.warn(
                "[ElasticNetChannelSelector] no Elastic-Net fit converged; "
                "falling back to ANOVA F-score selection.", RuntimeWarning)
            return self._fscore_selection(X, y)
        return counts / successful

    def _fscore_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ANOVA F-score per feature, min-max normalised to ``[0, 1]`` as a proxy."""
        classes = np.unique(y)
        n, d = X.shape
        grand = X.mean(axis=0)
        ssb = np.zeros(d)
        ssw = np.zeros(d)
        for c in classes:
            Xc = X[y == c]
            nc = len(Xc)
            mc = Xc.mean(axis=0)
            ssb += nc * (mc - grand) ** 2
            ssw += ((Xc - mc) ** 2).sum(axis=0)
        msb = ssb / max(len(classes) - 1, 1)
        msw = ssw / max(n - len(classes), 1) + 1e-12
        f = msb / msw
        if np.ptp(f) < 1e-12:
            return np.zeros(d)
        return (f - f.min()) / (f.max() - f.min())

    def _aggregate_to_channels(
        self,
        feat_freq: np.ndarray,
        channel_of_feature: np.ndarray,
        channel_names: Sequence[str],
        method: str,
    ) -> ChannelSelectionResult:
        """Pool feature scores within each channel and apply selection rules."""
        channel_names = list(channel_names)
        n_ch = len(channel_names)
        ch_score = np.zeros(n_ch)
        for ci in range(n_ch):
            cols = np.where(channel_of_feature == ci)[0]
            ch_score[ci] = float(feat_freq[cols].mean()) if len(cols) else 0.0

        if float(np.max(ch_score)) <= 1e-9:  # 全特徴が無情報 → 機械的に先頭を採用
            warnings.warn(
                "[ElasticNetChannelSelector] selection is uninformative (all channel "
                f"scores ~0); arbitrarily taking the first {self.min_channels} channels.",
                RuntimeWarning)

        keep = np.where(ch_score >= self.selection_threshold)[0]
        order = np.argsort(ch_score)[::-1]
        if len(keep) < self.min_channels:  # ensure a usable minimum
            keep = order[: self.min_channels]
        if self.max_channels is not None and len(keep) > self.max_channels:
            keep = np.array(sorted(keep, key=lambda i: -ch_score[i])[: self.max_channels])

        keep = sorted(keep.tolist())
        selected_names = [channel_names[i] for i in keep]
        feature_mask = np.isin(channel_of_feature, keep)
        scores = {channel_names[i]: float(ch_score[i]) for i in range(n_ch)}
        return ChannelSelectionResult(
            selected_channels=selected_names,
            channel_scores=scores,
            selected_feature_mask=feature_mask,
            method=method,
            feature_selection_freq=feat_freq,
        )

    # ------------------------------------------------------------- transform
    def selected_indices_for(self, trial: EEGTrial) -> List[int]:
        """Row indices of the selected channels *within a given trial* (by name)."""
        if self.result_ is None:
            raise RuntimeError("selector is not fitted")
        idxs = []
        for name in self.result_.selected_channels:
            i = trial.channel_index(name)
            if i is not None:
                idxs.append(i)
        return idxs

    def transform_channels(self, trial: EEGTrial) -> EEGTrial:
        """Return a trial restricted to the selected channels (order preserved)."""
        idxs = self.selected_indices_for(trial)
        if not idxs:  # nothing matched (e.g. different montage) → pass through
            return trial
        return EEGTrial(
            data=trial.data[idxs], channels=[trial.channels[i] for i in idxs],
            sfreq=trial.sfreq, subject=trial.subject,
            has_blink=trial.has_blink, blink_times=trial.blink_times,
        )

    def report(self) -> Dict[str, object]:
        """Human-readable summary of the selection outcome."""
        if self.result_ is None:
            return {"fitted": False}
        top = sorted(self.result_.channel_scores.items(), key=lambda kv: -kv[1])
        return {
            "fitted": True,
            "method": self.result_.method,
            "selected_channels": self.result_.selected_channels,
            "n_selected": len(self.result_.selected_channels),
            "top_channel_scores": top[:8],
        }
