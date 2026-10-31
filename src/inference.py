"""Build one model-ready row from a small request, using the saved metadata.

Shared by the FastAPI service. A caller only has to give district, season and
year; history and weather default to the district's last observed values and
its 1981-2010 climatology, and can be overridden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAIN_FEATURES = ["rain_total", "rain_establish", "rain_late", "rain_max5d"]


def scenario_weather(clim: dict, rain_pct_of_normal: float) -> dict:
    """Scale climatology by a rainfall scenario (100 = normal, 50 = half the normal rain)."""
    s = rain_pct_of_normal / 100.0
    v = dict(clim)
    for f in RAIN_FEATURES:
        if f in v:
            v[f] = clim[f] * s
    if "rain_days" in v:
        v["rain_days"] = clim["rain_days"] * (0.6 + 0.4 * s)
    if "longest_dry_spell" in v:
        v["longest_dry_spell"] = clim["longest_dry_spell"] / max(s, 0.3) ** 0.8
    if "et0_total" in v:
        v["et0_total"] = clim["et0_total"] * (1 + 0.08 * (1 - s))
    v["rain_anom_pct"] = 100 * (s - 1)
    if {"rain_total", "et0_total"} <= set(v):
        v["water_balance"] = v["rain_total"] - v["et0_total"]
    return v


def build_row(meta: dict, district: str, season: str, season_year: int | None = None,
              rain_pct_of_normal: float = 100.0, fertilizer_crisis: int = 0,
              overrides: dict | None = None) -> tuple[pd.DataFrame, dict]:
    key = f"{district}|{season}"
    if key not in meta["defaults"]:
        raise KeyError(f"No history for {district} / {season}")
    d = meta["defaults"][key]
    year = season_year or d["last_season_year"] + 1
    row = {"district": district, "season": season, "season_year": year,
           "fertilizer_crisis": int(fertilizer_crisis),
           "prev_yield_same": d["prev_yield_same"], "yield_mean3_same": d["yield_mean3_same"],
           "prev_yield_any": meta["last_yield_any"].get(district, d["prev_yield_same"]),
           "sown_ha": d["sown_ha"], "sown_change_pct": 0.0}
    row.update(scenario_weather(d["climatology"], rain_pct_of_normal))
    row.update(overrides or {})
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in meta["features"]}])
    return X, d


def forecast(model, meta: dict, X: pd.DataFrame, ref: float) -> dict:
    """Point forecast, 80 % interval and shortfall probability for one row.

    `ref` is the district's 5-season same-season mean; a shortfall means
    yield < shortfall_threshold x ref. For a probabilistic model (NGBoost) the
    flag is raised when P(shortfall) reaches the cut-off chosen on the CV folds;
    for a point model it falls back to comparing the point forecast with the
    threshold."""
    from scipy.stats import norm

    from .probabilistic import Z80, predict_normal

    mu, sigma = predict_normal(model, X)
    mu = float(mu[0])
    thr = meta["shortfall_threshold"] * ref
    out = {"yield_kg_ha": mu, "five_season_mean_kg_ha": float(ref), "change_pct": 100 * (mu / ref - 1),
           "shortfall_threshold_kg_ha": thr}
    if sigma is None:
        out.update({"sigma_kg_ha": None, "interval_80": None, "shortfall_probability": None,
                    "shortfall": bool(mu < thr)})
    else:
        s = float(sigma[0])
        p = float(norm.cdf((thr - mu) / s))
        cut = float(meta.get("probabilistic", {}).get("prob_cut", 0.5))
        out.update({"sigma_kg_ha": s, "interval_80": [float(mu - Z80 * s), float(mu + Z80 * s)],
                    "shortfall_probability": p, "shortfall": bool(p >= cut), "probability_cut": cut})
    return out
