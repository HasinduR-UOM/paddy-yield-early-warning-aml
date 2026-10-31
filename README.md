# Paddy Yield Early Warning for Sri Lanka

Pre-harvest forecasting of district paddy yield (kg/ha) for the Maha and Yala seasons. The selected model, **NGBoost** (Natural Gradient Boosting, not covered in the lectures), predicts a full probability distribution for each district-season, which gives an 80 % range and a **probability of shortfall**. Includes SHAP and LIME explanations, a Streamlit app, a FastAPI service and an MLOps pipeline on Azure Machine Learning.

IT 5504 Applied Machine Learning, Assignment 2 · MSc in Artificial Intelligence, University of Moratuwa · Hasindu Ramanayake

## Problem

Paddy is Sri Lanka's staple crop, and district yields swing with the monsoon, droughts and policy shocks such as the 2021/22 fertilizer import ban. Planners need to know before harvest whether a district's season will fall short, so that stocks, imports and support can be arranged in time.

The task is regression: given a district and season, predict yield in kg/ha from information available before harvest. A season is a **shortfall** when yield is below 85% of the district's previous five same-season yields. The early-warning system reports P(shortfall) and raises a flag when that probability reaches a cut-off chosen on the cross-validation folds.

## Data

| Source | Content | Coverage |
|---|---|---|
| [Department of Census and Statistics: Paddy Statistics](https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics) | District average yield, production, sown and harvested extent | Maha and Yala, 1978-2025 |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 reanalysis) | Daily rainfall, max/min temperature and FAO-56 reference evapotranspiration at one paddy-area point per district | 1979-present |
| [NOAA CPC Oceanic Niño Index](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | ENSO state | 1950-present |

One row is one district in one season. The 23 model inputs fall into three groups:

- **Seasonal weather:** totals and anomalies against the 1981-2010 normal, dry spells, 5-day maxima, temperature, evapotranspiration, and rainfall in the six months before the season.
- **Yield history and planting:** lagged yields and sown extent.
- **Context:** district, season, year, ENSO and a fertilizer-crisis flag.

Harvested extent and production are only known after harvest, so they are excluded from the inputs to avoid target leakage.

## Method

1. **Split by time.** Train on seasons up to 2015 (1,784 rows), test on 2016-2025 (465 rows). Inside the training period, hyperparameters are tuned with expanding-window cross-validation (five 3-year validation blocks).
2. **Preprocess inside a pipeline.** Median imputation with missing indicators, scaling for the distance-, margin- and ridge-based models, and one-hot encoding.
3. **Compare nine models.** Ridge, KNN, SVR (RBF), decision tree, random forest, gradient boosting, XGBoost and LightGBM are tuned on CV RMSE with `RandomizedSearchCV` (30 draws each). A naive "same as last season" forecast is the baseline.
4. **NGBoost, the new algorithm.** NGBoost (Duan et al., 2020) boosts the *parameters* of a Normal distribution (mean μ and log σ) using the natural gradient of the log-likelihood, so every forecast comes with its own uncertainty. It is tuned on CV **CRPS** (a proper scoring rule for distributions). The base learner (shallow trees or ridge regression) is itself a hyperparameter; ridge learners won. σ is then calibrated on the out-of-fold CV predictions (factor k = 1.29), and the P(shortfall) cut-off that maximises F1 on those predictions is chosen (0.16). The test seasons are not used for any of this.
5. **Select and evaluate.** NGBoost is selected for the early-warning system because it is the only model that outputs a calibrated probability of shortfall, and on CV it matches the best point model (Ridge) on RMSE while beating Ridge + Normal errors on CRPS and on shortfall detection. All models are then evaluated once on the test seasons: RMSE, MAE, MAPE and R² for the point forecast; CRPS, interval coverage, and ROC-AUC, PR-AUC, precision, recall and F1 for the shortfall probability.
6. **Explain.** SHAP explains NGBoost's expected yield (kg/ha) and its uncertainty (log σ). Because the tuned NGBoost uses ridge base learners, the ensemble collapses to one linear model per parameter and its SHAP values are exact. LIME gives local explanations for three case studies and is compared with SHAP; `TreeExplainer` on the best tree model checks that another model family learns the same drivers.
7. **Deploy.** A Streamlit app and a FastAPI service run the NGBoost model refitted on all seasons (keeping its CV calibration).

## Repository layout

```
src/
  config.py         districts and coordinates, seasons, split years, data URLs
  dcs.py            download and parse DCS tables (PDF or Excel)
  weather.py        Open-Meteo daily weather per district, NOAA ONI
  features.py       seasonal weather features and lagged history -> model table
  contract.py       pandera data contract for the model table
  modeling.py       pipelines, model zoo (incl. NGBoost), search spaces, time-aware CV, metrics
  probabilistic.py  NGBoost wrapper, CRPS / coverage, CV calibration, shortfall probability
  train.py          tuning, probabilistic comparison, test evaluation, artifacts, MLflow logging
  explain.py        SHAP (exact for NGBoost mean and spread) and LIME helpers
  inference.py      builds a model input row from a short request; forecast + P(shortfall)
  tracking.py       MLflow tracking (automatic inside Azure ML jobs)
  gate.py           quality gate against the registered champion
notebooks/          01 EDA, 02 modelling (section 9: NGBoost), 03 explainability
app/app.py          Streamlit app: inputs -> forecast distribution -> P(shortfall) -> SHAP + LIME
serve/api.py        FastAPI service: /health, /version, /predict, /feedback
aml/                Azure ML pipeline, environment, endpoint, scoring script, drift schedule
monitoring/         drift check and demo traffic for the endpoint
tests/              unit, data-contract, probabilistic, gate, API and scoring tests
params.yaml         training, selection, gate and monitoring settings
.github/workflows   CI, Train, Deploy
```

## Quick start

```bash
# Python 3.11 or 3.12
uv sync --all-extras          # or: pip install -r requirements.txt

python run_pipeline.py        # download -> parse -> weather -> features -> validate -> train
# data/processed/model_table.csv is included, so `python -m src.train` also works without downloading
jupyter notebook notebooks    # 01 -> 02 -> 03
streamlit run app/app.py      # interactive forecast with explanations
uvicorn serve.api:app --port 8000   # REST API, docs at /docs
pytest -q
```

On Windows, activate the environment with `.venv\Scripts\activate` before running these commands.

## Results

Test seasons 2016-2025 (465 district-seasons), from `artifacts/metrics.json`, `test_metrics.csv` and `prob_metrics.csv`. "Normal seasons" excludes the fertilizer-crisis seasons Maha 2021/22 and Yala 2022 (49 rows), which have no precedent in the training period.

**Point forecasts**

| Model | CV RMSE | Test RMSE | Test RMSE, normal seasons | MAPE, normal seasons | R², normal seasons |
|---|---|---|---|---|---|
| NGBoost (selected) | 409.8 | 680.3 | 472.4 | 9.7 % | 0.72 |
| Ridge (best CV RMSE) | 409.6 | 658.6 | 467.8 | 9.5 % | 0.72 |
| Naive (same as last season) | n/a | 758.8 | n/a | n/a | n/a |

**Probability of shortfall** (80 real shortfalls in the test seasons, 45 of them in the crisis seasons)

| | NGBoost | Ridge + Normal errors | NGBoost point forecast < threshold |
|---|---|---|---|
| CV CRPS (kg/ha, used for selection) | **218.2** | 220.8 | n/a |
| Test ROC-AUC | **0.61** | 0.57 | n/a |
| Test recall / precision / F1 | **0.16 / 0.25 / 0.20** | 0.06 / 0.21 / 0.10 | 0 / 0 / 0 |
| Test recall, normal seasons | **0.29** | 0.11 | n/a |
| Test CRPS (kg/ha), all / normal seasons | 361 / 260 | **351 / 257** | n/a |
| Test 80 % interval coverage, normal seasons | 0.73 | 0.77 | n/a |

NGBoost does not make the point forecast more accurate. What it adds is a usable early warning: a probability, an interval, and an uncertainty that differs by district. σ is about twice as large in Kilinochchi, Vavuniya and Mannar as in Galle or Kurunegala, because those districts have the most volatile and patchy yield records.

## MLOps on Azure Machine Learning

| Stage | Implementation |
|---|---|
| Environment | `uv.lock` locally; Azure ML environment `paddy-env` pinned to the same versions; Codespaces devcontainer |
| Data versioning | Azure ML data assets: `paddy-raw` (downloaded DCS + weather files) and `paddy-model-table` (pipeline output), new version per run |
| Training | Azure ML pipeline job (`aml/pipeline.yml`) on a CPU cluster that scales to zero. After a dependency change (e.g. adding `ngboost`), register the environment again: `az ml environment create -f aml/environment.yml` |
| Data quality | pandera data contract inside the prep step |
| Experiment tracking | MLflow, logged automatically to the pipeline job in Azure ML studio |
| Model registry | Azure ML model `paddy-yield`; each version tagged with its metrics, source job and `champion=true/false` |
| Quality gate | a new model must not have a higher normal-season test CRPS than the champion (`src/gate.py`); a champion without that metric (the earlier Ridge model) counts as none |
| Serving | Azure ML managed online endpoint `paddy-yield-ep`, key auth, scoring script `aml/score.py` |
| Release | GitHub Actions with OIDC login: manual approval, in-place or blue-green (0% -> smoke test -> 10% -> 100%) |
| Monitoring | endpoint data collector + weekly Azure ML schedule running a Kolmogorov-Smirnov drift test, metrics in MLflow |
| Retraining | Train workflow monthly with fresh data (and on demand); only gated models are registered and deployed |

## Limitations

- Weather comes from one ERA5 grid point per district rather than an average over the paddy area.
- Shocks outside the training history, such as the 2021/22 fertilizer ban, cannot be learned from weather and history alone. NGBoost gives those seasons an average P(shortfall) of only 5 %.
- Even with probabilities, most shortfalls are still missed (recall 0.16 overall, 0.29 in normal seasons); the warning is a planning aid, not a guarantee.
- Intervals are calibrated on the training-period CV; on the test seasons the 80 % interval covers 73 % of normal-season outcomes, so it is still slightly too narrow.
- Tree models cannot extrapolate the long-run yield trend; the lagged-yield features carry it instead.
- The Major, Minor and Rainfed split of sown extent is only published from 2004, so it is not part of the main model.
