import numpy as np

from visionguard.benchmark.runner import run_appearance_experiment


def test_runner_records_each_severity():
    frames = [np.full((8, 8, 3), 100, dtype=np.uint8) for _ in range(4)]

    def features(frame):
        return frame.mean(axis=(0, 1))

    def task(frame):
        return 1.0 - abs(float(frame.mean()) - 100.0) / 100.0

    records = run_appearance_experiment(
        frames,
        condition="brightness",
        severities=[0, 20],
        feature_fn=features,
        task_fn=task,
        baseline_metric=1.0,
    )

    assert len(records) == 2
    assert records[0].severity == 0
    assert records[1].severity == 20
    assert records[0].drift_score < records[1].drift_score
    assert records[1].task_metric < records[0].task_metric


def test_runner_rejects_unknown_condition():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

    try:
        run_appearance_experiment(
            frames,
            "unknown",
            [1],
            lambda x: x.mean(),
            lambda x: 1.0,
            1.0,
        )
    except ValueError as exc:
        assert "unknown drift condition" in str(exc)
    else:
        raise AssertionError("expected ValueError")
