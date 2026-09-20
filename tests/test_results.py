from pathlib import Path

from visionguard.benchmark.results import write_results
from visionguard.benchmark.runner import ExperimentRecord


def test_results_are_written_as_csv(tmp_path: Path):
    target = tmp_path / "results.csv"
    write_results(
        [ExperimentRecord("brightness", 20, 0.3, 0.8, 1.0, -0.2)],
        target,
    )
    text = target.read_text(encoding="utf-8")
    assert "condition,severity,drift_score" in text
    assert "brightness,20,0.3" in text
