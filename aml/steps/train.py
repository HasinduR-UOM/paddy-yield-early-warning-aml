"""Pipeline step 2 (Azure ML): model table -> tuned models, metrics and the deployable model folder.

    python -m aml.steps.train --table <folder with model_table.csv> --n-iter 30 --out <output folder>

The output folder is what gets registered as the Azure ML model:
    models/final_model.joblib (calibrated NGBoost), app_metadata.json, metrics.json,
    cv_results.csv, test_metrics.csv, prob_metrics.csv, test_predictions.csv
MLflow logging goes to the job automatically (see src/tracking.py).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config as C
from src import train as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.table)
    shutil.copy(src / "model_table.csv" if src.is_dir() else src, C.PROCESSED / "model_table.csv")
    T.run(n_iter=a.n_iter)

    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(C.MODELS / "final_model.joblib", out / "models" / "final_model.joblib")
    for f in ("app_metadata.json", "metrics.json", "cv_results.csv", "test_metrics.csv", "prob_metrics.csv",
              "test_predictions.csv"):
        if (C.ARTIFACTS / f).exists():
            shutil.copy(C.ARTIFACTS / f, out / f)
    print(f"Model folder written to {out}")


if __name__ == "__main__":
    main()
