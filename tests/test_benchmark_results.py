from pathlib import Path
import json
import pytest

from visionguard.benchmark.results import load_experiment, summarize_experiment


def test_load_and_summarize(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "records": [
            {"baseline_f1": 0.8, "metric_delta": -0.1, "drift_score": 0.4, "severity": 10},
            {"baseline_f1": 0.8, "metric_delta": -0.3, "drift_score": 0.9, "severity": 20},
        ]
    }), encoding="utf-8")
    payload = load_experiment(path)
    summary = summarize_experiment(payload)
    assert summary["records"] == 2
    assert summary["baseline_f1"] == 0.8
    assert summary["largest_f1_drop"] == -0.3
    assert summary["largest_drift_score"] == 0.9


def test_missing_records_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(path)
