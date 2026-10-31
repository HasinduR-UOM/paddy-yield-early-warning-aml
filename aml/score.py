"""Scoring script for the Azure ML managed online endpoint.

Request (one object or a list):
    {"district": "Kurunegala", "season": "Yala", "season_year": 2026,
     "rain_pct_of_normal": 60, "fertilizer_crisis": 0, "overrides": {"sown_ha": 70000}}
Only district and season are required.

Response: one object per request with the forecast (kg/ha), its 80 % interval,
the district's 5-season mean, the % change, the probability of a shortfall and
the shortfall flag. Inputs and outputs are
sent to the deployment's data collector, which the weekly drift job reads.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import build_row, forecast  # noqa: E402

MODEL = META = None
COLLECT_IN = COLLECT_OUT = None


def _model_root() -> Path:
    base = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
    return next(base.rglob("app_metadata.json")).parent  # registered folder may be nested one level


def init() -> None:
    global MODEL, META, COLLECT_IN, COLLECT_OUT
    root = _model_root()
    META = json.loads((root / "app_metadata.json").read_text())
    MODEL = joblib.load(root / "models" / "final_model.joblib")
    try:
        from azureml.ai.monitoring import Collector
        COLLECT_IN, COLLECT_OUT = Collector(name="model_inputs"), Collector(name="model_outputs")
    except Exception as e:  # noqa: BLE001  collector only works inside a deployment with data_collector on
        logging.info(f"Data collection off: {e}")
    logging.info(f"Loaded {META['best_model']} trained on {META['trained_on']}")


def predict_one(req: dict) -> tuple[pd.DataFrame, dict]:
    X, d = build_row(META, req["district"], req["season"], req.get("season_year"),
                     float(req.get("rain_pct_of_normal", 100.0)), int(req.get("fertilizer_crisis", 0)),
                     req.get("overrides"))
    f = forecast(MODEL, META, X, float(d["yield_mean5_same"]))
    lo, hi = f["interval_80"] or (None, None)
    return X, {"request_id": str(uuid.uuid4()), "district": req["district"], "season": req["season"],
               "yield_kg_ha": round(f["yield_kg_ha"], 1),
               "interval_80_low": None if lo is None else round(lo, 1),
               "interval_80_high": None if hi is None else round(hi, 1),
               "five_season_mean_kg_ha": round(f["five_season_mean_kg_ha"], 1),
               "change_pct": round(f["change_pct"], 2),
               "shortfall_probability": None if f["shortfall_probability"] is None
               else round(f["shortfall_probability"], 4),
               "shortfall": f["shortfall"]}


def run(raw_data: str):
    body = json.loads(raw_data)
    reqs = body if isinstance(body, list) else body.get("input", [body]) if isinstance(body, dict) else []
    rows, out = [], []
    for r in reqs:
        try:
            X, res = predict_one(r)
            rows.append(X)
            out.append(res)
        except KeyError as e:
            out.append({"error": f"unknown or missing field: {e}", "request": r})
    if rows and COLLECT_IN is not None:
        try:
            ctx = COLLECT_IN.collect(pd.concat(rows, ignore_index=True))
            COLLECT_OUT.collect(pd.DataFrame([o for o in out if "error" not in o]), ctx)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"collect failed: {e}")
    return out[0] if isinstance(body, dict) and "input" not in body else out
