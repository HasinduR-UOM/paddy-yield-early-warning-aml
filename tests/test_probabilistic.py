"""NGBoost / probabilistic-forecast helpers: scoring rules, calibration and the forecaster wrapper."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import Pipeline

from src.modeling import make_preprocessor, ngboost_regressor
from src.probabilistic import (
    NormalForecaster,
    calibration_factor,
    crps_normal,
    predict_normal,
    probabilistic_metrics,
    shortfall_prob_metrics,
)


def test_crps_matches_numerical_integral():
    y, mu, s = 3.0, 2.0, 1.5
    grid = np.linspace(-20, 25, 200_001)
    num = np.trapezoid((norm.cdf(grid, mu, s) - (grid >= y)) ** 2, grid)
    assert abs(crps_normal(y, mu, s) - num) < 1e-4


def test_crps_rewards_sharp_correct_forecasts():
    assert crps_normal(0.0, 0.0, 0.5) < crps_normal(0.0, 0.0, 2.0) < crps_normal(0.0, 3.0, 2.0)


def test_calibration_factor_recovers_inflation():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2.0, 20_000)
    assert abs(calibration_factor(y, np.zeros_like(y), np.ones_like(y)) - 2.0) < 0.05


def test_coverage_of_calibrated_normal():
    rng = np.random.default_rng(1)
    y = rng.normal(10, 3, 50_000)
    m = probabilistic_metrics(y, np.full_like(y, 10), np.full_like(y, 3))
    assert abs(m["coverage_80"] - 0.8) < 0.01 and abs(m["coverage_95"] - 0.95) < 0.01


def test_shortfall_metrics_perfect_ranking():
    m = shortfall_prob_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], cut=0.5)
    assert m["ROC_AUC"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0


def _toy():
    rng = np.random.default_rng(2)
    n = 300
    X = pd.DataFrame({"district": rng.choice(["A", "B"], n), "season": rng.choice(["Maha", "Yala"], n),
                      "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    noise = np.where(X.district == "A", 50, 300)              # district B is much noisier
    y = 4000 + 400 * X.x1 + rng.normal(0, noise)
    return X, y


def test_ngboost_learns_heteroscedastic_sigma_and_wrapper_works():
    X, y = _toy()
    feats = ["district", "season", "x1", "x2"]
    pipe = Pipeline([("pre", make_preprocessor(feats)),
                     ("model", ngboost_regressor(n_estimators=300, learning_rate=0.05))])
    fc = NormalForecaster(pipe, sigma_scale=1.2).fit(X, y)
    mu, sigma = predict_normal(fc, X)
    assert sigma[(X.district == "B").to_numpy()].mean() > 2 * sigma[(X.district == "A").to_numpy()].mean()
    lo, hi = fc.predict_interval(X, 0.8)
    assert np.all(lo < mu) and np.all(mu < hi)
    p = fc.prob_below(X, np.full(len(X), 4000.0))
    assert np.all((p >= 0) & (p <= 1))


def test_shap_grouping_keeps_season_year_separate_from_season():
    from src.explain import _group
    cols = ["season_year", "missingindicator_rain_total", "rain_total", "season_Maha", "season_Yala", "district_A"]
    vals = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    feats = ["district", "season", "season_year", "rain_total"]
    out = _group(vals, cols, feats)
    assert out.tolist() == [[6.0, 9.0, 1.0, 5.0]]
