"""Quality gate: is the newly trained model (challenger) at least as good as the champion?

    python -m src.gate --challenger artifacts/metrics.json --champion champion.json
    # exit 0 = pass, 1 = fail; writes gate_report.md

The champion's metrics come from the tags of the Azure ML model version that
carries champion=true (the train workflow writes them to champion.json). If
there is no champion yet, the first model passes. Metric and tolerance are
set in params.yaml (default: CRPS of the NGBoost forecast distribution on normal
seasons, lower is better). A champion that has no value for the metric (for
example an older point model) is treated as "no champion yet".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import config as C

ROWS = ["test_crps_normal", "test_crps", "test_rmse_normal", "test_mape_normal", "test_r2_normal", "test_rmse",
        "cv_rmse", "shortfall_recall", "shortfall_pr_auc"]


def load(path: str | None) -> dict | None:
    if not path or not Path(path).exists() or Path(path).stat().st_size == 0:
        return None
    data = json.loads(Path(path).read_text() or "null")
    return data or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", default=str(C.ARTIFACTS / "metrics.json"))
    ap.add_argument("--champion", default=None, help="JSON with the champion's metrics (empty/missing = none)")
    ap.add_argument("--report", default=str(C.ROOT / "gate_report.md"))
    a = ap.parse_args(argv)
    params = yaml.safe_load((C.ROOT / "params.yaml").read_text())["gate"]
    metric, tol = params["metric"], float(params["tolerance"])
    challenger = json.loads(Path(a.challenger).read_text())
    champion = load(a.champion)

    lines = ["## Quality gate", "", f"Rule: challenger `{metric}` <= champion + {tol} (lower is better)", "",
             "| Metric | Champion | Challenger |", "|---|---|---|"]
    for m in ROWS:
        c = champion.get(m) if champion else None
        lines.append(f"| {m} | {'none yet' if c is None else round(float(c), 3)} | {challenger.get(m)} |")
    if champion is None or champion.get(metric) is None:
        passed, why = True, "No champion yet, so the first model passes."
    else:
        passed = challenger.get(metric) is not None and float(challenger[metric]) <= float(champion[metric]) + tol
        why = (f"{metric}: challenger {challenger.get(metric)} vs champion {float(champion[metric]):.2f}. "
               + ("PASS" if passed else "FAIL: the new model is worse, the champion stays."))
    lines += ["", f"**Result: {'PASS' if passed else 'FAIL'}** - {why}", "",
              f"Challenger: {challenger['best_model']}"
              + (f" | Champion: version {champion.get('version', '?')}" if champion else "")]
    report = "\n".join(lines)
    Path(a.report).write_text(report + "\n")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
