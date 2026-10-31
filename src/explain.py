"""SHAP and LIME helpers that work on any fitted Pipeline from modeling.py.

SHAP (SHapley Additive exPlanations)
  * Tree models -> shap.TreeExplainer on the model inside the pipeline (exact,
    fast). One-hot / missing-indicator columns are summed back to the original
    feature so plots show "district", not 25 dummy columns.
  * NGBoost -> explains each predicted PARAMETER separately: the mean (loc,
    in kg/ha) and the spread (log sigma). With ridge base learners the boosted
    ensemble is a sum of linear models, i.e. one linear model per parameter, so
    exact (interventional) SHAP values are coefficient x (x - background mean).
    With tree base learners shap.TreeExplainer(model_output=k) is used.
  * Other models (Ridge, SVR, KNN) -> model-agnostic permutation
    explainer on the raw features.
LIME (Local Interpretable Model-agnostic Explanations)
  * Fits a small weighted linear model around ONE prediction by perturbing
    the inputs. Categorical features are label-encoded for LIME and decoded
    before calling the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import CAT_FEATURES

TREE_TYPES = ("RandomForest", "GradientBoosting", "XGB", "LGBM","DecisionTree", "ExtraTrees")


class RawCodec:
    """Label-encode categorical columns so explainers can work on a numeric matrix."""

    def __init__(self, X: pd.DataFrame, features: list[str]):
        self.features = features
        self.cats = {c: sorted(X[c].dropna().astype(str).unique()) for c in features if c in CAT_FEATURES}

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.features].copy()
        for c, levels in self.cats.items():
            Z[c] = Z[c].astype(str).map({v: i for i, v in enumerate(levels)}).fillna(-1)
        return Z.astype(float).to_numpy()

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(A, columns=self.features)
        for c, levels in self.cats.items():
            idx = np.clip(np.round(df[c]).astype(int), 0, len(levels) - 1)
            df[c] = [levels[i] for i in idx]
        return df

    @property
    def cat_idx(self):
        return [self.features.index(c) for c in self.cats]


def _final(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def is_tree_pipeline(model) -> bool:
    return any(t in type(_final(model)).__name__ for t in TREE_TYPES)


def _group(values: np.ndarray, cols: list[str], features: list[str]) -> np.ndarray:
    """Sum transformed-column SHAP values back to original features."""
    out = np.zeros((values.shape[0], len(features)))
    index = {f: i for i, f in enumerate(features)}
    for j, col in enumerate(cols):
        src = col.replace("missingindicator_", "")
        if src in index:                      # numeric feature (or its missing indicator)
            out[:, index[src]] += values[:, j]
            continue
        # one-hot column "<categorical>_<level>"; match the longest categorical name so that
        # e.g. "season_year" is never mistaken for a level of "season"
        for f in sorted((f for f in features if f in CAT_FEATURES), key=len, reverse=True):
            if src.startswith(f + "_"):
                out[:, index[f]] += values[:, j]
                break
    return out


# --------------------------------------------------------------------------
# NGBoost: SHAP for the mean and for the spread
# --------------------------------------------------------------------------
def _ngb_parts(model):
    """(pipeline, NGBRegressor, sigma_scale) if `model` is an NGBoost model, else None."""
    from .probabilistic import NormalForecaster
    k = 1.0
    if isinstance(model, NormalForecaster):
        k = float(model.sigma_scale)
        model = model._inner()
    final = _final(model)
    if not hasattr(final, "pred_dist"):
        return None
    return model, final, k


def is_ngboost(model) -> bool:
    return _ngb_parts(model) is not None


def ngboost_linear_coefficients(ngb, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a ridge-based NGBoost ensemble into one linear model per parameter.

    params(x) = init - lr * sum_i s_i * (x[cols_i] . coef_ij + b_ij), so the effective
    weights are W[j] = -lr * sum_i s_i[j] * coef_ij (scattered to the sampled columns)."""
    n_par = len(ngb.init_params)
    W = np.zeros((n_par, n_cols))
    b = np.array(ngb.init_params, dtype=float)
    for models, s, cols in zip(ngb.base_models, ngb.scalings, ngb.col_idxs):
        s = np.broadcast_to(np.ravel(np.asarray(s, dtype=float)), (n_par,))  # one line-search scale per stage
        for j, m in enumerate(models):
            W[j, cols] -= ngb.learning_rate * s[j] * np.ravel(m.coef_)
            b[j] -= ngb.learning_rate * s[j] * float(m.intercept_)
    return W, b


def ngboost_shap(model, X: pd.DataFrame, features: list[str], background: pd.DataFrame,
                 param: str = "loc") -> shap.Explanation:
    """SHAP over the original features for NGBoost's `loc` (kg/ha) or `scale` (log sigma)."""
    pipe, ngb, k = _ngb_parts(model)
    j = {"loc": 0, "scale": 1}[param]
    pre = pipe[:-1] if hasattr(pipe, "steps") else None
    Xt = pre.transform(X[features]) if pre is not None else X[features]
    Bt = pre.transform(background[features]) if pre is not None else background[features]
    cols = list(Xt.columns)
    if all(hasattr(m, "coef_") for ms in ngb.base_models for m in ms):
        W, b = ngboost_linear_coefficients(ngb, Xt.shape[1])
        mean_bg = Bt.to_numpy(dtype=float).mean(0)
        vals = (Xt.to_numpy(dtype=float) - mean_bg) * W[j]
        base = float(b[j] + mean_bg @ W[j])
    else:
        explainer = shap.TreeExplainer(ngb, model_output=j)
        vals = np.asarray(explainer.shap_values(Xt))
        base = float(np.ravel(explainer.expected_value)[0])
    if param == "scale":
        base += np.log(k)          # calibration multiplies sigma by k: a constant shift in log sigma
    grouped = _group(vals, cols, features)
    return shap.Explanation(values=grouped, base_values=np.repeat(base, len(Xt)), data=X[features].to_numpy(),
                            feature_names=features)


def shap_explanation(model, X: pd.DataFrame, features: list[str],
                     background: pd.DataFrame | None = None, max_evals: int = 800) -> shap.Explanation:
    """Return a shap.Explanation over the ORIGINAL features for rows of X.

    For NGBoost this explains the expected yield (loc, kg/ha). Wrap the model in
    probabilistic.SigmaView to explain the spread instead (values in log sigma)."""
    from .probabilistic import SigmaView
    X = X[features]
    bg_ng = background if background is not None else X
    if isinstance(model, SigmaView) and is_ngboost(model.model):
        return ngboost_shap(model.model, X, features, bg_ng, "scale")
    if is_ngboost(model):
        return ngboost_shap(model, X, features, bg_ng, "loc")
    if is_tree_pipeline(model):
        pre, est = model.named_steps["pre"], _final(model)
        Xt = pre.transform(X)
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xt)
        base = np.repeat(np.ravel(explainer.expected_value)[0], len(Xt))
        grouped = _group(np.asarray(sv), list(Xt.columns), features)
        return shap.Explanation(values=grouped, base_values=np.asarray(base), data=X.to_numpy(),
                                feature_names=features)
    # model-agnostic path
    bg = background if background is not None else X
    codec = RawCodec(pd.concat([bg, X]), features)
    bg_enc = codec.encode(bg.sample(min(100, len(bg)), random_state=0))
    f = lambda A: model.predict(codec.decode(A))  # noqa: E731
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(bg_enc), max_evals=max_evals)
    ex = explainer(codec.encode(X))
    return shap.Explanation(values=ex.values, base_values=ex.base_values, data=X.to_numpy(),
                            feature_names=features)


def shap_importance(ex: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)


# --------------------------------------------------------------------------
def lime_explainer(model, X_train: pd.DataFrame, features: list[str], seed: int = 42):
    from lime.lime_tabular import LimeTabularExplainer
    codec = RawCodec(X_train, features)
    med = X_train[features].median(numeric_only=True)
    A = codec.encode(X_train[features].fillna(med))       # LIME cannot handle NaN
    # treat binary / constant numeric columns as categorical (avoids zero-variance bins)
    extra = [i for i, f in enumerate(features)
             if f not in codec.cats and set(np.unique(A[:, i])) <= {0.0, 1.0}]
    cat_idx = codec.cat_idx + extra
    names = {codec.features.index(c): lv for c, lv in codec.cats.items()}
    for i in extra:
        names[i] = ["0", "1"]
    explainer = LimeTabularExplainer(
        A, feature_names=features, mode="regression", categorical_features=cat_idx,
        categorical_names=names, discretize_continuous=True, random_state=seed)
    predict = lambda Z: model.predict(codec.decode(Z))  # noqa: E731
    return explainer, codec, predict, med


def lime_explain_row(model, X_train: pd.DataFrame, row: pd.DataFrame, features: list[str],
                     num_features: int = 10, num_samples: int = 3000, seed: int = 42, explainer_bundle=None):
    explainer, codec, predict, med = explainer_bundle or lime_explainer(model, X_train, features, seed)
    r = row[features].copy().fillna(med)
    return explainer.explain_instance(codec.encode(r)[0], predict,
                                      num_features=num_features, num_samples=num_samples)


def lime_table(exp) -> pd.DataFrame:
    return pd.DataFrame(exp.as_list(), columns=["condition", "weight_kg_ha"])


def compare_shap_lime(shap_row: shap.Explanation, exp, top: int = 8) -> pd.DataFrame:
    """Side-by-side ranking of the top features for one prediction."""
    s = pd.Series(np.abs(shap_row.values), index=shap_row.feature_names).sort_values(ascending=False)
    lime_feats = []
    for cond, w in exp.as_list():
        for f in sorted(shap_row.feature_names, key=len, reverse=True):
            if f in cond:
                lime_feats.append((f, w))
                break
    lime_s = pd.Series({f: abs(w) for f, w in lime_feats}).sort_values(ascending=False)
    n = max(top, 1)
    return pd.DataFrame({"SHAP rank": list(s.index[:n]) + [""] * (n - len(s.index[:n])),
                         "LIME rank": list(lime_s.index[:n]) + [""] * (n - len(lime_s.index[:n]))},
                        index=range(1, n + 1))
