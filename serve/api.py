"""Prediction API (FastAPI).

    uvicorn serve.api:app --port 8000          # local
    docker run -p 8000:8000 paddy-yield:local   # container

Endpoints: GET /health, GET /version, POST /predict, POST /feedback, GET /docs.
Every prediction and feedback is printed as one JSON line; on Azure Container
Apps these lines go to Log Analytics and feed the drift monitor.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import build_row, forecast  # noqa: E402

ART = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
META = json.loads((ART / "app_metadata.json").read_text())
MODEL = joblib.load(ART / "models" / "final_model.joblib")

app = FastAPI(title="Sri Lanka paddy yield forecast", version=MODEL_VERSION,
              description="Pre-harvest district paddy yield (kg/ha) with an 80 % interval, "
                          "a shortfall probability and a shortfall flag.")


class PredictRequest(BaseModel):
    district: str = Field(examples=["Polonnaruwa"])
    season: Literal["Maha", "Yala"] = Field(examples=["Maha"])
    season_year: int | None = Field(None, description="Year the season starts; default = next season")
    rain_pct_of_normal: float = Field(100.0, ge=0, le=300, description="100 = normal rainfall, 50 = half")
    fertilizer_crisis: int = Field(0, ge=0, le=1)
    overrides: dict[str, float] | None = Field(None, description="Optional raw feature values, e.g. sown_ha")


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    yield_kg_ha: float
    interval_80_low: float | None = None
    interval_80_high: float | None = None
    five_season_mean_kg_ha: float
    change_pct: float
    shortfall_probability: float | None = None
    shortfall: bool


class Feedback(BaseModel):
    request_id: str
    actual_yield_kg_ha: float


def log(event: str, **kw) -> None:
    print(json.dumps({"event": event, "ts": time.time(), "model_version": MODEL_VERSION, **kw}), flush=True)


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION, "best_model": META["best_model"]}


@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION, "best_model": META["best_model"], "trained_on": META["trained_on"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X, d = build_row(META, req.district, req.season, req.season_year, req.rain_pct_of_normal,
                         req.fertilizer_crisis, req.overrides)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    f = forecast(MODEL, META, X, float(d["yield_mean5_same"]))
    lo, hi = f["interval_80"] or (None, None)
    out = PredictResponse(request_id=str(uuid.uuid4()), model_version=MODEL_VERSION,
                          yield_kg_ha=round(f["yield_kg_ha"], 1),
                          interval_80_low=None if lo is None else round(lo, 1),
                          interval_80_high=None if hi is None else round(hi, 1),
                          five_season_mean_kg_ha=round(f["five_season_mean_kg_ha"], 1),
                          change_pct=round(f["change_pct"], 2),
                          shortfall_probability=None if f["shortfall_probability"] is None
                          else round(f["shortfall_probability"], 4),
                          shortfall=f["shortfall"])
    log("prediction", request_id=out.request_id, features=X.iloc[0].to_dict(), prediction=out.yield_kg_ha)
    return out


@app.post("/feedback")
def feedback(fb: Feedback):
    log("feedback", request_id=fb.request_id, actual=fb.actual_yield_kg_ha)
    return {"status": "ok"}
