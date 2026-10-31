"""End-to-end training script (the notebooks walk through the same steps).

    python -m src.train                    # 30 random-search draws per model
    python -m src.train --n-iter 10        # faster run

Outputs (artifacts/):
    models/<model>.joblib       tuned on train (<= 2015) - used for test evaluation
    models/final_model.joblib   selected model re-fitted on ALL seasons - used by the app,
                                the API and the Azure ML endpoint
    cv_results.csv, test_metrics.csv, prob_metrics.csv, test_predictions.csv,
    app_metadata.json, metrics.json

Model selection
    Every model is tuned with the same expanding-window CV. The early-warning
    system needs a probability of shortfall, so the deployed model is the one
    named in params.yaml (train.selected_model, default NGBoost). It is compared
    with the best point model (lowest CV RMSE) given Normal errors, on CV CRPS and
    on out-of-fold shortfall detection, before the test seasons are opened.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.base import clone

from . import config as C
from .features import FEATURE_DESCRIPTIONS, TARGET, WEATHER_FEATURES, feature_list
from .modeling import (
    evaluate,
    expanding_window_cv,
    naive_baseline,
    regression_metrics,
    temporal_split,
    tune_all,
)
from .probabilistic import (
    fit_forecaster,
    probabilistic_metrics,
    shortfall_prob_metrics,
    shortfall_threshold_values,
    shortfall_truth,
)

RIDGE = "Ridge (baseline)"
RIDGE_NORMAL = "Ridge + Normal errors"


def load_params() -> dict:
    p = C.ROOT / "params.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def crisis_mask(df: pd.DataFrame) -> np.ndarray:
    return df.apply(lambda r: (r.season, int(r.season_year)) in C.FERTILIZER_CRISIS, axis=1).to_numpy()


def probabilistic_comparison(candidates: dict, train: pd.DataFrame, test: pd.DataFrame,
                             features: list[str]) -> tuple[dict, pd.DataFrame]:
    """Calibrate each candidate on out-of-fold CV predictions, then score it once on the test seasons."""
    folds = expanding_window_cv(train)
    truth_te = shortfall_truth(test, TARGET)
    ok = ~np.isnan(truth_te)
    normal = ~crisis_mask(test)
    fcs, rows = {}, []
    for name, est in candidates.items():
        fc = fit_forecaster(est, train, features, TARGET, folds)
        mu, sigma = fc["model"].predict_normal(test[features])
        prob = norm.cdf((shortfall_threshold_values(test) - mu) / sigma)
        allm = probabilistic_metrics(test[TARGET], mu, sigma)
        norm_m = probabilistic_metrics(test.loc[normal, TARGET], mu[normal], sigma[normal])
        sm = shortfall_prob_metrics(truth_te[ok], prob[ok], fc["cut"])
        sm_n = shortfall_prob_metrics(truth_te[ok & normal], prob[ok & normal], fc["cut"])
        fc.update({"test_mu": mu, "test_sigma": sigma, "test_prob": prob, "test_shortfall": sm})
        fcs[name] = fc
        o = fc["oof_metrics"]
        rows.append({"model": name, "sigma_scale": round(fc["sigma_scale"], 3), "prob_cut": fc["cut"],
                     "cv_CRPS": o["CRPS"], "cv_coverage_80": o["coverage_80"],
                     "cv_shortfall_PR_AUC": o["shortfall_PR_AUC"], "cv_shortfall_F1": o["shortfall_F1"],
                     "test_CRPS": allm["CRPS"], "test_CRPS_normal": norm_m["CRPS"], "test_NLL": allm["NLL"],
                     "test_coverage_80": allm["coverage_80"], "test_coverage_95": allm["coverage_95"],
                     "test_width_80": allm["width_80"],
                     "test_shortfall_ROC_AUC": sm["ROC_AUC"], "test_shortfall_PR_AUC": sm["PR_AUC"],
                     "test_shortfall_Brier": sm["Brier"], "test_shortfall_precision": sm["precision"],
                     "test_shortfall_recall": sm["recall"], "test_shortfall_F1": sm["F1"],
                     "test_shortfall_recall_normal": sm_n["recall"], "test_shortfall_F1_normal": sm_n["F1"]})
    return fcs, pd.DataFrame(rows)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_table() -> pd.DataFrame:
    path = C.PROCESSED / "model_table.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.features`")
    return pd.read_csv(path)


def build_app_metadata(df: pd.DataFrame, features: list[str], best_name: str,
                       cv_table: pd.DataFrame, test_table: pd.DataFrame,
                       prob_table: pd.DataFrame | None = None, prob_info: dict | None = None,
                       selection: dict | None = None) -> dict:
    defaults = {}
    for (d, s), g in df.sort_values("t").groupby(["district", "season"]):
        g_obs = g[g[TARGET].notna()]
        last = g_obs.iloc[-1]
        base = g[g.season_year.between(*C.CLIMATE_BASE)]
        clim = base[WEATHER_FEATURES].median(numeric_only=True)
        clim_q = {f: [float(base[f].quantile(.05)), float(base[f].quantile(.95))] for f in WEATHER_FEATURES
                  if f in base and base[f].notna().any()}
        harvest_ratio = (g["harvested_ha"] / g["sown_ha"]).median() if "harvested_ha" in g else np.nan
        defaults[f"{d}|{s}"] = {
            "last_season_year": int(last.season_year),
            "prev_yield_same": float(last[TARGET]),
            "yield_mean3_same": float(g_obs[TARGET].tail(3).mean()),
            "yield_mean5_same": float(g_obs[TARGET].tail(5).mean()),
            "sown_ha": float(last.get("sown_ha", np.nan)),
            "harvest_ratio": float(harvest_ratio) if pd.notna(harvest_ratio) else 0.95,
            "climatology": {k: float(v) for k, v in clim.items() if pd.notna(v)},
            "climatology_range": clim_q,
            "history": g_obs[["season_year", TARGET]].tail(15).values.tolist(),
        }
        if "oni_mean" in features:
            defaults[f"{d}|{s}"]["climatology"]["oni_mean"] = 0.0
    last_any = (df[df[TARGET].notna()].sort_values("t").groupby("district").tail(1)
                .set_index("district")[TARGET].to_dict())
    return {
        "features": features,
        "feature_descriptions": {f: FEATURE_DESCRIPTIONS.get(f, f) for f in features},
        "districts": sorted(df.district.unique()),
        "best_model": best_name,
        "defaults": defaults,
        "last_yield_any": {k: float(v) for k, v in last_any.items()},
        "cv_table": cv_table.to_dict(orient="records"),
        "test_table": test_table.to_dict(orient="records"),
        "prob_table": prob_table.to_dict(orient="records") if prob_table is not None else [],
        "probabilistic": prob_info or {},
        "selection": selection or {},
        "train_end": C.TRAIN_END,
        "shortfall_threshold": C.SHORTFALL_THRESHOLD,
        "trained_on": f"{int(df.season_year.min())}-{int(df.season_year.max())}",
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }


def summary_metrics(best_name, cv_table, preds, best_slug, n_train, n_test,
                    prob_row: dict | None = None, best_by_cv: str | None = None) -> dict:
    """Headline numbers for the quality gate, the model tags and MLflow."""
    col = f"pred_{best_slug}"
    crisis = crisis_mask(preds)
    allm = regression_metrics(preds[TARGET], preds[col])
    norm = regression_metrics(preds.loc[~crisis, TARGET], preds.loc[~crisis, col])
    row = cv_table.set_index("model").loc[best_name]
    return {
        "best_model": best_name,
        "cv_rmse": round(float(row["cv_rmse"]), 2),
        "cv_rmse_std": round(float(row["cv_rmse_std"]), 2),
        "test_rmse": round(allm["RMSE"], 2), "test_mae": round(allm["MAE"], 2),
        "test_mape": round(allm["MAPE_%"], 3), "test_r2": round(allm["R2"], 4),
        "test_rmse_normal": round(norm["RMSE"], 2), "test_mape_normal": round(norm["MAPE_%"], 3),
        "test_r2_normal": round(norm["R2"], 4),
        "n_train": int(n_train), "n_test": int(n_test),
        "best_by_cv_rmse": best_by_cv or best_name,
        **({"test_crps": round(prob_row["test_CRPS"], 2), "test_crps_normal": round(prob_row["test_CRPS_normal"], 2),
            "test_coverage_80": round(prob_row["test_coverage_80"], 3),
            "shortfall_roc_auc": round(prob_row["test_shortfall_ROC_AUC"], 3),
            "shortfall_pr_auc": round(prob_row["test_shortfall_PR_AUC"], 3),
            "shortfall_precision": round(prob_row["test_shortfall_precision"], 3),
            "shortfall_recall": round(prob_row["test_shortfall_recall"], 3),
            "shortfall_f1": round(prob_row["test_shortfall_F1"], 3),
            "shortfall_prob_cut": prob_row["prob_cut"]} if prob_row else {}),
    }


def run(n_iter: int = 30, only: list[str] | None = None) -> dict:
    df = load_table()
    features = feature_list(df)
    train, test = temporal_split(df)
    print(f"Train {len(train)} rows (<= {C.TRAIN_END}), test {len(test)} rows (> {C.TRAIN_END}); "
          f"{len(features)} features")

    results = tune_all(train, features, n_iter=n_iter, only=only)
    fitted = {name: r["search"].best_estimator_ for name, r in results.items()}

    cv_table = pd.DataFrame([{"model": k, "cv_rmse": v["cv_rmse"], "cv_rmse_std": v["cv_rmse_std"],
                              "cv_crps": v.get("cv_crps"), "tuning_seconds": v["fit_s"],
                              "best_params": json.dumps({p.split("__")[-1]: (round(x, 5) if isinstance(x, float) else str(x) if hasattr(x, "fit") else x)
                                                         for p, x in v["search"].best_params_.items()}, default=str)}
                             for k, v in results.items()]).sort_values("cv_rmse").reset_index(drop=True)

    test_table = evaluate(fitted, test, features)
    naive = {"model": "Naive (last same-type season)", **regression_metrics(test[TARGET], naive_baseline(test))}
    test_table = pd.concat([test_table, pd.DataFrame([naive])], ignore_index=True)

    best_by_cv = cv_table.iloc[0]["model"]
    selected = load_params().get("train", {}).get("selected_model") or best_by_cv
    if selected not in fitted:
        print(f"WARNING: selected_model '{selected}' was not trained; using {best_by_cv}")
        selected = best_by_cv
    print(f"\nBest point model by CV RMSE: {best_by_cv} | selected for the early-warning system: {selected}")
    print(test_table.round(3).to_string(index=False))

    # Probabilistic comparison: NGBoost vs the best point model with Normal errors
    candidates = {}
    if "NGBoost" in fitted:
        candidates["NGBoost"] = results["NGBoost"]["search"].best_estimator_
    if RIDGE in fitted:
        candidates[RIDGE_NORMAL] = fitted[RIDGE]
    fcs, prob_table = probabilistic_comparison(candidates, train, test, features) if candidates else ({}, None)
    if prob_table is not None:
        print("\nProbabilistic forecasts (calibrated on CV, scored once on test):")
        print(prob_table.round(3).to_string(index=False))

    # Save
    for name, m in fitted.items():
        joblib.dump(m, C.MODELS / f"{slug(name)}.joblib")
    preds = test[["district", "season", "season_year", TARGET, "yield_mean5_same"]].copy()
    for name, m in fitted.items():
        preds[f"pred_{slug(name)}"] = m.predict(test[features])
    for name, fc in fcs.items():
        preds[f"sigma_{slug(name)}"] = fc["test_sigma"]
        preds[f"p_shortfall_{slug(name)}"] = fc["test_prob"]
    preds.to_csv(C.ARTIFACTS / "test_predictions.csv", index=False)
    cv_table.to_csv(C.ARTIFACTS / "cv_results.csv", index=False)
    test_table.to_csv(C.ARTIFACTS / "test_metrics.csv", index=False)
    if prob_table is not None:
        prob_table.to_csv(C.ARTIFACTS / "prob_metrics.csv", index=False)

    if selected in fcs:              # probabilistic model: keep its CV calibration
        sel_train = fcs[selected]["model"]
    else:
        sel_train = fitted[selected]
    final = clone(sel_train).fit(df[features], df[TARGET])
    joblib.dump(final, C.MODELS / "final_model.joblib")
    joblib.dump(sel_train, C.MODELS / "best_model_train_only.joblib")
    train[features + [TARGET]].to_csv(C.ARTIFACTS / "train_sample.csv", index=False)
    prob_row = (prob_table.set_index("model").loc[selected].to_dict() | {"model": selected}
                if prob_table is not None and selected in set(prob_table.model) else None)
    prob_info = ({"sigma_scale": fcs[selected]["sigma_scale"], "prob_cut": fcs[selected]["cut"],
                  "interval_level": 0.8} if selected in fcs else {})
    selection = {"selected": selected, "best_by_cv_rmse": best_by_cv,
                 "reason": ("NGBoost is the only candidate that outputs a probability of shortfall; "
                            "it is compared with the best point model given Normal errors on CV CRPS "
                            "and out-of-fold shortfall detection") if selected == "NGBoost" else "lowest CV RMSE"}
    meta = build_app_metadata(df, features, selected, cv_table, test_table, prob_table, prob_info, selection)
    (C.ARTIFACTS / "app_metadata.json").write_text(json.dumps(meta, indent=1, default=float))
    metrics = summary_metrics(selected, cv_table, preds, slug(selected), len(train), len(test),
                              prob_row, best_by_cv)
    (C.ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models and metadata to {C.ARTIFACTS.relative_to(C.ROOT)}/")
    try:  # MLflow tracking is optional: on inside Azure ML jobs or when MLFLOW_TRACKING_URI is set
        from .tracking import log_training_run
        log_training_run(metrics, cv_table, test_table, final, features, df)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001  tracking must never break training
        print(f"WARNING: MLflow logging failed: {e}")
    return {"df": df, "train": train, "test": test, "features": features, "results": results,
            "fitted": fitted, "cv_table": cv_table, "test_table": test_table, "best_name": selected,
            "best_by_cv": best_by_cv, "forecasters": fcs, "prob_table": prob_table}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    a = ap.parse_args()
    run(a.n_iter)
