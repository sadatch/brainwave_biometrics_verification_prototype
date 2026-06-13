"""Open-set recognition: One-Class SVM (SVDD) + LightGBM ensemble.

The problem is open-set
-----------------------
1:1 verification must reject an *unbounded* set of impostors that were never seen at
enrollment. A purely discriminative classifier trained on "enrollee vs. some known
others" can overfit the *known* others and accept a novel attacker that lands in an
unguarded region of feature space.

Two complementary views, ensembled
----------------------------------
* **One-Class SVM / SVDD** — trained on the enrollee's genuine embeddings *only*. It
  draws a tight boundary around the enrollee and treats everything outside as
  impostor. This is what bounds the **false-accept rate (FAR)** against *unknown*
  attackers (the open-set term).
* **LightGBM** — a supervised genuine-vs-background classifier. Where we *do* have
  cohort (background) data, it sharpens the boundary against the *known* impostor
  distribution (the closed-set term).

Their calibrated scores are fused. Fusion balances open-set robustness (OC-SVM) with
discriminative sharpness (LightGBM); an ``"and"`` gate (both must accept) is offered
as a stricter, lower-FAR alternative.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:  # pragma: no cover
    _HAVE_LGBM = False


# --------------------------------------------------------------------------- #
# NumPy fallbacks (used only when scikit-learn is unavailable)
# --------------------------------------------------------------------------- #
class _NumpyLogReg:
    """Minimal multivariate logistic regression (gradient descent, L2)."""

    def __init__(self, l2: float = 1e-2, iters: int = 500, lr: float = 0.5) -> None:
        self.l2, self.iters, self.lr = l2, iters, lr
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLogReg":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
            g = p - y
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * float(g.mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(np.asarray(X, float) @ self.w + self.b)))
        return np.column_stack([1.0 - p, p])


class _MahalanobisOneClass:
    """One-class scorer via Mahalanobis distance to the genuine cluster."""

    def fit(self, genuine: np.ndarray, background: np.ndarray) -> "_MahalanobisOneClass":
        self.mu = genuine.mean(axis=0)
        cov = np.cov(genuine, rowvar=False)
        cov = np.atleast_2d(cov)
        self.inv = np.linalg.pinv(cov + 1e-3 * np.eye(cov.shape[0]))
        dg, db = self._dist(genuine), self._dist(background)
        self.mid = 0.5 * (np.median(dg) + (np.median(db) if len(db) else np.median(dg) * 2))
        self.scale = float(np.std(np.concatenate([dg, db]))) + 1e-8
        return self

    def _dist(self, X: np.ndarray) -> np.ndarray:
        diff = np.asarray(X, float) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv, diff) + 1e-12)

    def prob(self, X: np.ndarray) -> np.ndarray:
        d = self._dist(X)
        return 1.0 / (1.0 + np.exp((d - self.mid) / self.scale))  # small dist → high prob


# --------------------------------------------------------------------------- #
# Open-set recognizer
# --------------------------------------------------------------------------- #
class OpenSetRecognizer:
    """Calibrated OC-SVM (+SVDD) ⊕ LightGBM verifier for one enrolled identity.

    Parameters
    ----------
    nu : OC-SVM upper bound on the fraction of training outliers / margin errors.
    gamma : RBF kernel coefficient (``"scale"`` recommended).
    fusion_weight : weight ``w`` on the OC-SVM probability; LightGBM gets ``1−w``.
    threshold : accept threshold on the decision score.
    mode : ``"and"`` (default; both branches must pass their *own* calibrated
        threshold — the secure choice that closes the open-set hole where a high
        LightGBM probability could override a low OC-SVM novelty score) or
        ``"fusion"`` (threshold the single fused score).
    random_state : RNG seed for the discriminative model.
    """

    def __init__(
        self,
        nu: float = 0.1,
        gamma: str = "scale",
        fusion_weight: float = 0.5,
        threshold: float = 0.5,
        mode: str = "and",
        lgbm_params: Optional[dict] = None,
        random_state: int = 0,
    ) -> None:
        if mode not in ("fusion", "and"):
            raise ValueError("mode must be 'fusion' or 'and'")
        self.nu = float(nu)
        self.gamma = gamma
        self.fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))
        self.threshold = float(threshold)        # fused-score threshold ("fusion" mode)
        self.threshold_oc = float(threshold)     # per-branch thresholds ("and" mode)
        self.threshold_lgbm = float(threshold)
        self.mode = mode
        self.lgbm_params = lgbm_params
        self.random_state = int(random_state)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, genuine_embeds: np.ndarray, background_embeds: np.ndarray) -> "OpenSetRecognizer":
        """Train both branches on genuine (positive) and background (negative) embeddings."""
        g = np.atleast_2d(np.asarray(genuine_embeds, float))
        b = np.atleast_2d(np.asarray(background_embeds, float))
        if _HAVE_SKLEARN:
            self.scaler = StandardScaler().fit(np.vstack([g, b]))
        else:
            self.scaler = _StandardScalerNP().fit(np.vstack([g, b]))
        gs, bs = self.scaler.transform(g), self.scaler.transform(b)
        self._fit_oneclass(gs, bs)
        self._fit_discriminative(gs, bs)
        self._fitted = True
        return self

    def _fit_oneclass(self, gs: np.ndarray, bs: np.ndarray) -> None:
        if _HAVE_SKLEARN:
            self.ocsvm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma).fit(gs)
            raw = self.ocsvm.decision_function(np.vstack([gs, bs])).reshape(-1, 1)
            lab = np.r_[np.ones(len(gs)), np.zeros(len(bs))]
            if len(np.unique(lab)) == 2:
                self._platt = LogisticRegression(C=1e3, max_iter=1000).fit(raw, lab)
            else:  # degenerate (no background) → sigmoid on standardized score
                self._platt = None
                self._raw_mu, self._raw_sd = float(raw.mean()), float(raw.std()) + 1e-8
            self._oc_backend = "ocsvm-rbf"
        else:
            self._maha = _MahalanobisOneClass().fit(gs, bs)
            self._oc_backend = "mahalanobis-fallback"

    def _fit_discriminative(self, gs: np.ndarray, bs: np.ndarray) -> None:
        X = np.vstack([gs, bs])
        y = np.r_[np.ones(len(gs)), np.zeros(len(bs))].astype(int)
        if _HAVE_LGBM:
            params = dict(n_estimators=80, num_leaves=15, max_depth=4, learning_rate=0.1,
                          min_child_samples=5, subsample=0.9, colsample_bytree=0.9,
                          verbosity=-1, random_state=self.random_state)
            params.update(self.lgbm_params or {})
            self.clf = LGBMClassifier(**params).fit(X, y)
            self._disc_backend = "lightgbm"
        elif _HAVE_SKLEARN:
            self.clf = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, random_state=self.random_state).fit(X, y)
            self._disc_backend = "gradient-boosting-fallback"
        else:
            self.clf = _NumpyLogReg().fit(X, y)
            self._disc_backend = "numpy-logreg-fallback"

    # --------------------------------------------------------------- scoring
    def _oc_prob(self, Xs: np.ndarray) -> np.ndarray:
        if getattr(self, "_oc_backend", "").startswith("ocsvm"):
            raw = self.ocsvm.decision_function(Xs).reshape(-1, 1)
            if self._platt is not None:
                return self._platt.predict_proba(raw)[:, 1]
            return 1.0 / (1.0 + np.exp(-(raw.ravel() - self._raw_mu) / self._raw_sd))
        return self._maha.prob(Xs)

    def _disc_prob(self, Xs: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(Xs)[:, 1]

    def score(self, embed: np.ndarray) -> Dict[str, float]:
        """Return per-branch and fused probabilities for a single embedding."""
        if not self._fitted:
            raise RuntimeError("recognizer is not fitted")
        xs = self.scaler.transform(np.atleast_2d(np.asarray(embed, float)))
        p_oc = float(self._oc_prob(xs)[0])
        p_disc = float(self._disc_prob(xs)[0])
        fused = self.fusion_weight * p_oc + (1.0 - self.fusion_weight) * p_disc
        return {"ocsvm_p": p_oc, "lgbm_p": p_disc, "fused": float(fused)}

    def verify(
        self, embed: np.ndarray, threshold: Optional[float] = None, mode: Optional[str] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Decide accept/reject for one embedding; returns ``(accept, scores)``.

        In ``"and"`` mode each branch is compared against its *own* calibrated
        threshold (``threshold_oc`` / ``threshold_lgbm``); a single ``threshold``
        argument, if given, overrides both. In ``"fusion"`` mode the fused score is
        compared against ``threshold``.
        """
        m = self.mode if mode is None else mode
        s = self.score(embed)
        if m == "and":
            t_oc = self.threshold_oc if threshold is None else float(threshold)
            t_lg = self.threshold_lgbm if threshold is None else float(threshold)
            accept = (s["ocsvm_p"] >= t_oc) and (s["lgbm_p"] >= t_lg)
            s.update(decision=bool(accept), threshold_oc=t_oc, threshold_lgbm=t_lg, mode=m)
        else:
            t = self.threshold if threshold is None else float(threshold)
            accept = s["fused"] >= t
            s.update(decision=bool(accept), threshold=t, mode=m)
        return bool(accept), s

    # ------------------------------------------------------------- utilities
    def calibrate_threshold(
        self,
        genuine_embeds: Sequence[np.ndarray],
        impostor_embeds: Sequence[np.ndarray],
        target_far: float = 0.01,
        grid: Optional[Sequence[float]] = None,
    ):
        """Calibrate the decision threshold(s) **consistently with the active mode**.

        Uses *both* genuine and impostor scores. For each relevant score (the fused
        score in ``"fusion"`` mode; the OC-SVM and LightGBM branch scores separately
        in ``"and"`` mode) it picks the threshold meeting ``target_far`` with the
        lowest FRR. If no threshold can meet ``target_far`` it does **not** silently
        keep the default (the previous fail-open behaviour): it emits a
        ``RuntimeWarning`` and falls back to the equal-error-rate (EER) point.
        """
        grid = sorted(grid) if grid is not None else list(np.linspace(0.05, 0.95, 19))
        gs = [self.score(e) for e in genuine_embeds]
        isc = [self.score(e) for e in impostor_embeds]
        if self.mode == "and":
            self.threshold_oc = self._calibrate_branch(
                [g["ocsvm_p"] for g in gs], [i["ocsvm_p"] for i in isc], target_far, grid, "OC-SVM")
            self.threshold_lgbm = self._calibrate_branch(
                [g["lgbm_p"] for g in gs], [i["lgbm_p"] for i in isc], target_far, grid, "LightGBM")
            return (self.threshold_oc, self.threshold_lgbm)
        self.threshold = self._calibrate_branch(
            [g["fused"] for g in gs], [i["fused"] for i in isc], target_far, grid, "fused")
        return self.threshold

    @staticmethod
    def _calibrate_branch(g_scores, i_scores, target_far, grid, name) -> float:
        """Choose a per-branch threshold (target-FAR feasible → min FRR, else EER)."""
        def far(t: float) -> float:
            return float(np.mean([s >= t for s in i_scores])) if i_scores else 0.0

        def frr(t: float) -> float:
            return float(np.mean([s < t for s in g_scores])) if g_scores else 0.0

        feasible = [t for t in grid if far(t) <= target_far]
        if feasible:
            return float(min(feasible, key=lambda t: (frr(t), t)))
        eer = float(min(grid, key=lambda t: abs(far(t) - frr(t))))
        warnings.warn(
            f"[OpenSetRecognizer] {name}: no threshold meets target FAR={target_far}; "
            f"falling back to EER point t={eer:.2f} (FAR={far(eer):.2f}, FRR={frr(eer):.2f}).",
            RuntimeWarning,
        )
        return eer

    def evaluate(
        self,
        genuine_embeds: Sequence[np.ndarray],
        impostor_embeds: Sequence[np.ndarray],
        threshold: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute FAR / FRR / accuracy on labelled genuine and impostor embeddings."""
        tg = [self.verify(e, threshold, mode)[0] for e in genuine_embeds]
        ti = [self.verify(e, threshold, mode)[0] for e in impostor_embeds]
        ng, ni = max(len(tg), 1), max(len(ti), 1)
        frr = 1.0 - (sum(tg) / ng)
        far = sum(ti) / ni
        acc = (sum(tg) + (len(ti) - sum(ti))) / (len(tg) + len(ti) or 1)
        m = self.mode if mode is None else mode
        return {
            "FAR": float(far), "FRR": float(frr), "ACC": float(acc),
            "n_genuine": len(tg), "n_impostor": len(ti),
            "threshold": self.threshold if threshold is None else float(threshold),
            "threshold_oc": self.threshold_oc, "threshold_lgbm": self.threshold_lgbm,
            "mode": m,
        }

    @property
    def backend(self) -> str:
        if not self._fitted:
            return "unfitted"
        return f"oneclass={self._oc_backend}, discriminative={self._disc_backend}"


class _StandardScalerNP:
    """NumPy StandardScaler used only when scikit-learn is unavailable."""

    def fit(self, X: np.ndarray) -> "_StandardScalerNP":
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) / self.scale_
