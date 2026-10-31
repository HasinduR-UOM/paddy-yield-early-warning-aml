"""Probabilistic forecasting with NGBoost, plus the scoring rules used to evaluate it.

Why a probabilistic model
  A planner asks two questions: "how much paddy will this district harvest?" and
  "how likely is a shortfall?". A point forecast answers only the first. NGBoost
  (Natural Gradient Boosting, Duan et al., 2020) predicts a full distribution
  for every district-season. Here it is a Normal distribution N(mu, sigma), where
  BOTH mu and sigma depend on the inputs. The shortfall probability then follows
  directly: P(yield < 0.85 x 5-season mean) = Phi((threshold - mu) / sigma).

How NGBoost differs from the boosting taught in class
  * Gradient boosting / XGBoost fit one number per row (the mean) by following
    the gradient of squared error.
  * NGBoost fits the PARAMETERS of a distribution (here mu and log sigma) by
    following the NATURAL gradient of a proper scoring rule (the negative log
    likelihood). The natural gradient rescales the ordinary gradient by the
    Fisher information, so steps are taken in "distribution space" and mu and
    sigma learn at comparable speeds.
  * Each boosting stage fits one base learner per parameter; the base learner
    (a shallow tree or a ridge regression) is a hyperparameter tuned by CV.

Calibration
  Errors on future seasons are larger than errors on the training seasons, so
  raw NGBoost intervals are too narrow. A single variance-inflation factor k is
  estimated from the OUT-OF-FOLD predictions of the expanding-window CV
  (training seasons only) and applied to every sigma. The test seasons are not
  used for this.

Scoring rules
  * CRPS (continuous ranked probability score): the mean absolute error
    generalised to a distribution, in kg/ha. Lower is better. It rewards a
    distribution that is both close to the truth and appropriately narrow.
  * NLL: negative log likelihood per row.
  * Coverage: share of actual yields inside the 80 % / 95 % intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C

Z80 = norm.ppf(0.9)    # 1.2816: half-width of a central 80 % interval in sigmas
Z95 = norm.ppf(0.975)  # 1.9600


# --------------------------------------------------------------------------
# NGBoost estimator that scikit-learn recognises as fitted
# --------------------------------------------------------------------------
try:
    from ngboost import NGBRegressor as _NGBRegressor
except ImportError:  # pragma: no cover - ngboost is a core dependency
    _NGBRegressor = object


class NGBRegressor(_NGBRegressor):
    """ngboost.NGBRegressor plus the hook scikit-learn uses to check that a model is fitted.

    ngboost stores its fitted state in `base_models` (no trailing underscore), so
    without this hook a Pipeline refuses to call `predict` inside cross-validation."""

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "base_models", None))


# --------------------------------------------------------------------------
# Scoring rules for a Normal forecast
# --------------------------------------------------------------------------
def crps_normal(y, mu, sigma) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma) at observation y (Gneiting & Raftery, 2007)."""
    y, mu, sigma = map(np.asarray, (y, mu, sigma))
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def nll_normal(y, mu, sigma) -> np.ndarray:
    return -norm.logpdf(np.asarray(y), np.asarray(mu), np.asarray(sigma))


def probabilistic_metrics(y, mu, sigma) -> dict:
    y, mu, sigma = map(np.asarray, (y, mu, sigma))
    return {"CRPS": float(np.mean(crps_normal(y, mu, sigma))),
            "NLL": float(np.mean(nll_normal(y, mu, sigma))),
            "coverage_80": float(np.mean(np.abs(y - mu) <= Z80 * sigma)),
            "coverage_95": float(np.mean(np.abs(y - mu) <= Z95 * sigma)),
            "width_80": float(np.mean(2 * Z80 * sigma))}


# --------------------------------------------------------------------------
# Getting (mu, sigma) out of a fitted model
# --------------------------------------------------------------------------
def _is_ngb(est) -> bool:
    return hasattr(est, "pred_dist")


def raw_normal(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """(mu, sigma) from a Pipeline(pre, NGBRegressor); sigma is None for point models."""
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if _is_ngb(final):
        Xt = model[:-1].transform(X) if hasattr(model, "steps") else X
        d = final.pred_dist(Xt)
        return np.asarray(d.params["loc"]), np.asarray(d.params["scale"])
    return np.asarray(model.predict(X)), None


class NormalForecaster(RegressorMixin, BaseEstimator):
    """Wraps a fitted point or NGBoost pipeline and returns calibrated Normal forecasts.

    * NGBoost pipeline: sigma comes from the model (it varies by row) and is
      multiplied by `sigma_scale` (the CV calibration factor).
    * Point model (e.g. Ridge): sigma is the constant `const_sigma`, the RMSE of
      its out-of-fold CV predictions. This is the "Ridge + Normal errors"
      baseline that NGBoost has to beat.

    `fit` re-fits a clone of the inner pipeline on new data and keeps the
    calibration, so the same object can be refitted on all seasons for the app.
    """

    def __init__(self, pipeline=None, sigma_scale: float = 1.0, const_sigma: float | None = None):
        self.pipeline = pipeline
        self.sigma_scale = sigma_scale
        self.const_sigma = const_sigma

    def fit(self, X, y):
        self.pipeline_ = clone(self.pipeline).fit(X, y)
        return self

    def _inner(self):
        return getattr(self, "pipeline_", self.pipeline)

    def predict(self, X) -> np.ndarray:
        return self.predict_normal(X)[0]

    def predict_normal(self, X) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = raw_normal(self._inner(), X)
        if sigma is None:
            sigma = np.full(len(mu), float(self.const_sigma))
        return mu, sigma * float(self.sigma_scale)

    def predict_sigma(self, X) -> np.ndarray:
        return self.predict_normal(X)[1]

    def predict_interval(self, X, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.predict_normal(X)
        z = norm.ppf(0.5 + level / 2)
        return mu - z * sigma, mu + z * sigma

    def prob_below(self, X, threshold) -> np.ndarray:
        mu, sigma = self.predict_normal(X)
        return norm.cdf((np.asarray(threshold, dtype=float) - mu) / sigma)

    @property
    def is_ngboost(self) -> bool:
        final = self._inner().steps[-1][1] if hasattr(self._inner(), "steps") else self._inner()
        return _is_ngb(final)


class SigmaView:
    """Adapter whose `predict` returns the forecast spread sigma, so SHAP/LIME can explain uncertainty."""

    def __init__(self, model):
        self.model = model

    def predict(self, X) -> np.ndarray:
        return predict_normal(self.model, X)[1]


def predict_normal(model, X) -> tuple[np.ndarray, np.ndarray | None]:
    """(mu, sigma) for any saved model: NormalForecaster, NGBoost pipeline or point model."""
    if hasattr(model, "predict_normal"):
        return model.predict_normal(X)
    return raw_normal(model, X)


# --------------------------------------------------------------------------
# Out-of-fold predictions, calibration and the shortfall decision threshold
# --------------------------------------------------------------------------
def oof_normal(estimator, train: pd.DataFrame, features: list[str], target: str,
               folds: list) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window out-of-fold (mu, sigma). Rows never in a validation block are NaN.

    For a point model sigma is the in-sample residual SD of each training fold;
    it is replaced later by the OOF RMSE (see `fit_forecaster`)."""
    mu = np.full(len(train), np.nan)
    sd = np.full(len(train), np.nan)
    for tr, va in folds:
        m = clone(estimator).fit(train.iloc[tr][features], train.iloc[tr][target])
        m_mu, m_sd = raw_normal(m, train.iloc[va][features])
        if m_sd is None:
            res = train.iloc[tr][target].to_numpy() - m.predict(train.iloc[tr][features])
            m_sd = np.full(len(va), res.std())
        mu[va], sd[va] = m_mu, m_sd
    return mu, sd


def calibration_factor(y, mu, sigma) -> float:
    """k such that the standardised OOF errors (y - mu) / (k sigma) have unit variance."""
    z = (np.asarray(y) - mu) / sigma
    return float(np.sqrt(np.nanmean(z ** 2)))


def shortfall_threshold_values(df: pd.DataFrame) -> np.ndarray:
    return C.SHORTFALL_THRESHOLD * df["yield_mean5_same"].to_numpy(dtype=float)


def shortfall_truth(df: pd.DataFrame, target: str) -> np.ndarray:
    thr = shortfall_threshold_values(df)
    return np.where(np.isnan(thr), np.nan, (df[target].to_numpy() < thr).astype(float))


def best_f1_cut(labels, prob) -> float:
    """Probability cut-off that maximises F1 on out-of-fold predictions."""
    labels, prob = np.asarray(labels), np.asarray(prob)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(labels, prob >= c, zero_division=0) for c in grid]
    return float(grid[int(np.argmax(scores))])


def shortfall_prob_metrics(labels, prob, cut: float) -> dict:
    labels, prob = np.asarray(labels).astype(int), np.asarray(prob)
    flag = (prob >= cut).astype(int)
    both = len(np.unique(labels)) == 2
    return {"shortfall_cases": int(labels.sum()),
            "ROC_AUC": float(roc_auc_score(labels, prob)) if both else float("nan"),
            "PR_AUC": float(average_precision_score(labels, prob)) if both else float("nan"),
            "Brier": float(brier_score_loss(labels, prob)),
            "precision": float(precision_score(labels, flag, zero_division=0)),
            "recall": float(recall_score(labels, flag, zero_division=0)),
            "F1": float(f1_score(labels, flag, zero_division=0)),
            "confusion": confusion_matrix(labels, flag, labels=[0, 1]).tolist()}


def fit_forecaster(estimator, train: pd.DataFrame, features: list[str], target: str, folds: list) -> dict:
    """Calibrate on out-of-fold CV predictions, then fit on all training seasons.

    Returns the fitted NormalForecaster plus the OOF quantities used to calibrate it."""
    mu, sd = oof_normal(estimator, train, features, target, folds)
    y = train[target].to_numpy()
    ok = ~np.isnan(mu)
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if not _is_ngb(final):   # constant sigma = OOF RMSE (already calibrated by construction)
        const = float(np.sqrt(np.mean((y[ok] - mu[ok]) ** 2)))
        k, sd = 1.0, np.full(len(mu), const)
    else:
        const = None
        k = calibration_factor(y[ok], mu[ok], sd[ok])
    sig = sd * k
    truth = shortfall_truth(train, target)
    lab_ok = ok & ~np.isnan(truth)
    prob = norm.cdf((shortfall_threshold_values(train) - mu) / sig)
    cut = best_f1_cut(truth[lab_ok], prob[lab_ok])
    model = NormalForecaster(estimator, sigma_scale=k, const_sigma=const).fit(train[features], train[target])
    return {"model": model, "sigma_scale": k, "const_sigma": const, "cut": cut,
            "oof_mu": mu, "oof_sigma": sig, "oof_prob": prob, "oof_truth": truth,
            "oof_metrics": {**probabilistic_metrics(y[ok], mu[ok], sig[ok]),
                            **{f"shortfall_{k_}": v for k_, v in
                               shortfall_prob_metrics(truth[lab_ok], prob[lab_ok], cut).items()
                               if k_ != "confusion"}}}
