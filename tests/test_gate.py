import json
from pathlib import Path

import yaml

from src.gate import main

METRIC = yaml.safe_load((Path(__file__).resolve().parents[1] / "params.yaml").read_text())["gate"]["metric"]


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_first_model_passes(tmp_path):
    ch = write(tmp_path, "c.json", {"best_model": "NGBoost", METRIC: 250.0})
    assert main(["--challenger", ch, "--champion", write(tmp_path, "n.json", None),
                 "--report", str(tmp_path / "r.md")]) == 0


def test_worse_model_fails(tmp_path):
    ch = write(tmp_path, "c.json", {"best_model": "NGBoost", METRIC: 270.0})
    champ = write(tmp_path, "champ.json", {METRIC: "250.0", "version": "2"})
    assert main(["--challenger", ch, "--champion", champ, "--report", str(tmp_path / "r.md")]) == 1


def test_champion_without_metric_counts_as_none(tmp_path):
    ch = write(tmp_path, "c.json", {"best_model": "NGBoost", METRIC: 270.0})
    champ = write(tmp_path, "champ.json", {"test_rmse_normal": "460.0", "version": "4"})
    assert main(["--challenger", ch, "--champion", champ, "--report", str(tmp_path / "r.md")]) == 0
