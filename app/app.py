"""Paddy Yield Early-Warning - Streamlit app.

    streamlit run app/app.py

Loads artifacts/models/final_model.joblib and artifacts/app_metadata.json
(written by `python -m src.train`), takes a district / season and a weather
scenario, and shows:
  * the NGBoost forecast as a distribution: expected yield, 80 % interval and
    the probability of a shortfall (yield < 85 % of the 5-season mean);
  * SHAP and LIME explanations of the expected yield, and SHAP for the
    forecast uncertainty (what makes this season harder to predict).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import explain as E  # noqa: E402
from src.inference import forecast  # noqa: E402
from src.probabilistic import SigmaView  # noqa: E402

ART = ROOT / "artifacts"
st.set_page_config(page_title="Paddy Yield Early Warning - Sri Lanka", page_icon="🌾", layout="wide")


@st.cache_resource
def load():
    meta = json.loads((ART / "app_metadata.json").read_text())
    model = joblib.load(ART / "models" / "final_model.joblib")
    train = pd.read_csv(ART / "train_sample.csv")
    return meta, model, train


@st.cache_resource
def lime_bundle(_model, _train, features):
    return E.lime_explainer(_model, _train[features], features)


try:
    meta, model, train = load()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python -m src.train` first.")
    st.stop()

F = meta["features"]
DESC = meta["feature_descriptions"]
WEATHER_RAIN = ["rain_total", "rain_establish", "rain_late", "rain_max5d"]


st.title("🌾 Paddy Yield Early Warning — Sri Lanka")
st.caption(f"District-level pre-harvest yield forecast · model: **{meta['best_model']}** "
           f"(probabilistic: predicts a distribution, not one number) · trained on seasons {meta['trained_on']} · "
           f"DCS paddy statistics + ERA5 weather")

tab_fc, tab_models, tab_about = st.tabs(["Forecast", "Model comparison", "About"])

# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Season")
    district = st.selectbox("District", meta["districts"],
                            index=meta["districts"].index("Polonnaruwa") if "Polonnaruwa" in meta["districts"] else 0)
    season = st.radio("Season", ["Maha", "Yala"], horizontal=True,
                      help="Maha: Oct–Mar (north-east monsoon). Yala: Apr–Sep (south-west monsoon).")
    key = f"{district}|{season}"
    if key not in meta["defaults"]:
        st.error("No history for this district/season.")
        st.stop()
    dflt = meta["defaults"][key]
    year = st.number_input("Season start year", 1980, 2035, value=dflt["last_season_year"] + 1)

    st.header("Weather scenario")
    preset = st.radio("Rainfall vs the 1981–2010 normal",
                      ["Normal", "Dry (−30 %)", "Severe drought (−50 %)", "Wet (+30 %)", "Custom"], index=0)
    scale = {"Normal": 1.0, "Dry (−30 %)": 0.7, "Severe drought (−50 %)": 0.5, "Wet (+30 %)": 1.3}.get(preset, 1.0)
    fert = st.checkbox("Fertilizer supply crisis (as in 2021/22)", value=False)

clim = dflt["climatology"]
rng = dflt["climatology_range"]


def scenario_values(scale: float) -> dict:
    v = dict(clim)
    for f in WEATHER_RAIN:
        if f in v:
            v[f] = clim[f] * scale
    if "rain_days" in v:
        v["rain_days"] = clim["rain_days"] * (0.6 + 0.4 * scale)
    if "longest_dry_spell" in v:
        v["longest_dry_spell"] = clim["longest_dry_spell"] / max(scale, 0.3) ** 0.8
    v["rain_anom_pct"] = 100 * (scale - 1)
    if "et0_total" in v:
        v["et0_total"] = clim["et0_total"] * (1 + 0.08 * (1 - scale))
    if {"rain_total", "et0_total"} <= set(v):
        v["water_balance"] = v["rain_total"] - v["et0_total"]
    return v


with tab_fc:
    vals = scenario_values(scale)
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Recent yield history and planting (known before harvest)", expanded=True):
            prev_same = st.number_input("Previous same-season yield (kg/ha)", 500.0, 9000.0,
                                        float(dflt["prev_yield_same"]), 50.0)
            mean3 = st.number_input("Mean of last 3 same-season yields (kg/ha)", 500.0, 9000.0,
                                    float(dflt["yield_mean3_same"]), 50.0)
            prev_any = st.number_input("Immediately preceding season yield (kg/ha)", 500.0, 9000.0,
                                       float(meta["last_yield_any"].get(district, prev_same)), 50.0)
            sown = st.number_input("Extent sown this season (ha)", 0.0, 500000.0,
                                   float(np.nan_to_num(dflt["sown_ha"], nan=10000.0)), 500.0)
    with c2:
        with st.expander("Seasonal weather (pre-filled from the scenario)", expanded=preset == "Custom"):
            for f in ["rain_total", "rain_anom_pct", "longest_dry_spell", "antecedent_rain_6m", "tmax_mean", "rain_max5d"]:
                if f in F and f in vals:
                    lo, hi = rng.get(f, [vals[f] * 0.3, vals[f] * 1.7])
                    lo, hi = min(lo, vals[f]) , max(hi, vals[f])
                    if f == "rain_anom_pct":
                        lo, hi = -90.0, 150.0
                    span = hi - lo if hi > lo else abs(hi) + 1
                    vals[f] = st.slider(DESC.get(f, f), float(lo - 0.2 * span), float(hi + 0.2 * span),
                                        float(vals[f]), help=f, disabled=preset != "Custom")
            if "oni_mean" in F:
                vals["oni_mean"] = st.slider("ENSO (ONI): negative = La Niña, positive = El Niño", -2.5, 2.5, 0.0, 0.1)

    row = {"district": district, "season": season, "season_year": year,
           "fertilizer_crisis": int(fert), "prev_yield_same": prev_same, "prev_yield_any": prev_any,
           "yield_mean3_same": mean3, "sown_ha": sown,
           "sown_change_pct": 100 * (sown / dflt["sown_ha"] - 1) if dflt["sown_ha"] else 0.0}
    for f in F:
        if f not in row:
            row[f] = vals.get(f, clim.get(f, np.nan))
    X = pd.DataFrame([row])[F]

    ref = dflt["yield_mean5_same"]
    fc = forecast(model, meta, X, ref)
    pred, thr, p_short = fc["yield_kg_ha"], fc["shortfall_threshold_kg_ha"], fc["shortfall_probability"]
    ratio = pred / ref
    production = pred * sown * dflt["harvest_ratio"] / 1000
    shortfall = fc["shortfall"]
    probabilistic = p_short is not None

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected yield", f"{pred:,.0f} kg/ha", f"{100 * (ratio - 1):+.1f} % vs 5-season mean")
    if probabilistic:
        lo, hi = fc["interval_80"]
        m2.metric("80 % range", f"{lo:,.0f} – {hi:,.0f}",
                  help="The actual yield should fall inside this range 8 times out of 10")
        m3.metric("Chance of a shortfall", f"{100 * p_short:.0f} %",
                  help=f"P(yield < {thr:,.0f} kg/ha), i.e. below {int(meta['shortfall_threshold'] * 100)} % "
                       f"of the 5-season mean ({ref:,.0f} kg/ha)")
    else:
        m2.metric("5-season mean", f"{ref:,.0f} kg/ha")
        m3.metric("Estimated production", f"{production:,.0f} t")
    m4.metric("Status", "⚠️ SHORTFALL RISK" if shortfall else "✅ Normal",
              help=(f"Flag raised when the shortfall probability reaches {100 * fc['probability_cut']:.0f} % "
                    "(cut-off chosen on the cross-validation folds)") if probabilistic else
                   f"Shortfall = forecast below {int(meta['shortfall_threshold'] * 100)} % of the 5-season mean")
    st.caption(f"Estimated production ≈ {production:,.0f} t (yield × sown extent × typical harvested/sown ratio)")

    hist = pd.DataFrame(dflt["history"], columns=["season_year", "yield_kg_ha"])
    g1, g2 = st.columns([3, 2]) if probabilistic else (st.container(), None)
    with g1:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(hist.season_year, hist.yield_kg_ha, "o-", color="#2a7ab9", label="observed")
        colour = "#c0504d" if shortfall else "#3a9a5b"
        if probabilistic:
            ax.errorbar([year], [pred], yerr=[[pred - lo], [hi - pred]], fmt="D", ms=9, color=colour,
                        capsize=6, lw=2, label="forecast, 80 % range")
        else:
            ax.plot([year], [pred], "D", ms=10, color=colour, label="forecast")
        ax.axhline(thr, ls="--", c="grey", lw=1, label="shortfall threshold")
        ax.set_ylabel("kg/ha"); ax.set_title(f"{district} — {season}"); ax.legend(loc="lower left", fontsize=8)
        st.pyplot(fig, width="stretch")
    if probabilistic:
        with g2:
            from scipy.stats import norm
            s_ = fc["sigma_kg_ha"]
            xs = np.linspace(pred - 4 * s_, pred + 4 * s_, 300)
            fig3, ax3 = plt.subplots(figsize=(5, 3.4))
            ax3.plot(xs, norm.pdf(xs, pred, s_), color="#16302A")
            ax3.fill_between(xs, norm.pdf(xs, pred, s_), where=xs < thr, color="#c0504d", alpha=.45,
                             label=f"shortfall: {100 * p_short:.0f} %")
            ax3.axvline(thr, ls="--", c="grey", lw=1)
            ax3.set_yticks([]); ax3.set_xlabel("yield kg/ha"); ax3.set_title("Forecast distribution (NGBoost)")
            ax3.legend(fontsize=8, loc="upper right")
            st.pyplot(fig3, width="stretch")

    st.subheader("Why this forecast?")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**SHAP** — contribution of each feature (kg/ha) to the expected yield")
        with st.spinner("Computing SHAP ..."):
            ex = E.shap_explanation(model, X, F, background=train.sample(min(200, len(train)), random_state=0),
                                    max_evals=2 * len(F) + 1 if not E.is_tree_pipeline(model) else 800)
        plt.figure()
        shap.plots.waterfall(ex[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), width="stretch")
        plt.close("all")
    with e2:
        st.markdown("**LIME** — local linear surrogate around this input")
        with st.spinner("Computing LIME ..."):
            exp = E.lime_explain_row(model, train[F], X, F, num_features=10,
                                     explainer_bundle=lime_bundle(model, train, F))
        lt = E.lime_table(exp)
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        ax2.barh(lt.condition[::-1], lt.weight_kg_ha[::-1],
                 color=np.where(lt.weight_kg_ha[::-1] > 0, "#3a9a5b", "#c0504d"))
        ax2.set_xlabel("kg/ha"); ax2.axvline(0, c="k", lw=.7)
        st.pyplot(fig2, width="stretch")
        st.caption(f"LIME surrogate prediction {exp.local_pred[0]:,.0f} kg/ha (surrogate R² {exp.score:.2f})")

    if probabilistic and getattr(model, "is_ngboost", False):
        with st.expander("What makes this forecast more or less certain? (SHAP on the predicted spread σ)"):
            with st.spinner("Computing SHAP for σ ..."):
                ex_s = E.shap_explanation(SigmaView(model), X, F,
                                          background=train.sample(min(200, len(train)), random_state=0),
                                          max_evals=2 * len(F) + 1)
            plt.figure()
            shap.plots.waterfall(ex_s[0], max_display=8, show=False)
            st.pyplot(plt.gcf(), width="stretch")
            plt.close("all")
            st.caption("Positive values widen the forecast distribution (less certain); negative values narrow it. "
                       f"σ for this season: {fc['sigma_kg_ha']:,.0f} kg/ha.")

    top = pd.Series(ex.values[0], index=F).sort_values()
    neg, pos = top.head(2), top.tail(2)[::-1]
    risk = (f" There is a **{100 * p_short:.0f} %** chance that yield falls below the shortfall threshold "
            f"of {thr:,.0f} kg/ha." if probabilistic else "")
    st.info(
        f"**Plain-language summary.** The model expects **{pred:,.0f} kg/ha** for {district} {season} {year}.{risk} "
        f"Biggest upward drivers: {', '.join(f'{k} (+{v:,.0f})' for k, v in pos.items() if v > 0) or 'none'}. "
        f"Biggest downward drivers: {', '.join(f'{k} ({v:,.0f})' for k, v in neg.items() if v < 0) or 'none'}.")

    with st.expander("Input row sent to the model"):
        st.dataframe(pd.DataFrame({"feature": F, "value": [str(round(v, 2)) if isinstance(v, float) else str(v) for v in X.iloc[0]],
                                  "description": [DESC.get(f, "") for f in F]}), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_models:
    st.subheader("Cross-validation (model selection, seasons ≤ %d)" % meta["train_end"])
    st.dataframe(pd.DataFrame(meta["cv_table"]).round(1), width="stretch")
    st.subheader("Held-out test seasons (> %d): point forecasts" % meta["train_end"])
    st.dataframe(pd.DataFrame(meta["test_table"]).round(3), width="stretch")
    if meta.get("prob_table"):
        st.subheader("Probabilistic forecasts: NGBoost vs Ridge with Normal errors")
        st.caption("CRPS = error of the whole forecast distribution in kg/ha (lower is better). Coverage = share of "
                   "actual yields inside the 80 % range (ideal 0.80). Shortfall metrics use P(shortfall).")
        st.dataframe(pd.DataFrame(meta["prob_table"]).set_index("model").T.round(3), width="stretch")
    for f, cap in [("cv_rmse.png", "CV RMSE by model"), ("test_pred_vs_actual.png", "Test: predicted vs actual"),
                   ("ngb_calibration.png", "NGBoost: calibration of the forecast distribution"),
                   ("ngb_shortfall_pr.png", "Shortfall probability: precision-recall"),
                   ("shap_beeswarm.png", "Global SHAP summary")]:
        p = ART / "figures" / f
        if p.exists():
            st.image(str(p), caption=cap)

with tab_about:
    st.markdown("""
**Problem.** Paddy is Sri Lanka's staple crop. Planners need to know before harvest whether a district's
season will fall short, so stocks, imports and support can be arranged early.

**Data.** Department of Census and Statistics district paddy statistics (Maha and Yala, 1979–2025);
ERA5 daily weather via the Open-Meteo historical API (one paddy-area point per district);
NOAA Oceanic Niño Index.

**Model.** Nine model families were tuned with expanding-window cross-validation on seasons up to the
training cut-off and tested on later seasons. The app uses **NGBoost** (Natural Gradient Boosting), which predicts a
Normal distribution for each district-season instead of a single number: an expected yield, an uncertainty (σ) and,
from those, the probability of a shortfall. Its intervals were calibrated on the cross-validation folds. The model was
then re-fitted on all seasons for this app.

**Limits.** One weather point per district; shocks outside the training history (e.g. the 2021/22 fertilizer
ban) cannot be learned from weather; forecasts are for planning support, not official statistics.
""")
    st.dataframe(pd.DataFrame({"feature": F, "description": [DESC.get(f, "") for f in F]}), width="stretch")
