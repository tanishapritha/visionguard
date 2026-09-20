from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable

import numpy as np

from .drift_generator import DriftConfig, apply_drift
from ..drift.metrics import rbf_mmd_squared


@dataclass(frozen=True)
class ExperimentRecord:
    condition: str
    severity: float
    drift_score: float
    task_metric: float
    baseline_metric: float
    metric_delta: float


def run_appearance_experiment(
    frames: Iterable[np.ndarray],
    condition: str,
    severities: Iterable[float],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    task_fn: Callable[[np.ndarray], float],
    baseline_metric: float,
    seed: int = 42,
) -> list[ExperimentRecord]:
    """Run a controlled appearance-drift sweep.

    The supplied feature_fn and task_fn are intentionally model-agnostic so the
    benchmark can later wrap a real detector/classifier without changing the
    experiment protocol.
    """
    source = [np.asarray(frame) for frame in frames]
    if not source:
        raise ValueError("frames must contain at least one frame")

    rng = np.random.default_rng(seed)
    reference_features = np.vstack([np.asarray(feature_fn(frame)).reshape(1, -1) for frame in source])
    records: list[ExperimentRecord] = []

    for severity in severities:
        config = _config_for(condition, severity)
        drifted = [apply_drift(frame, config, rng) for frame in source]
        current_features = np.vstack(
            [np.asarray(feature_fn(frame)).reshape(1, -1) for frame in drifted]
        )
        drift_score = rbf_mmd_squared(reference_features, current_features)
        task_values = [float(task_fn(frame)) for frame in drifted]
        task_metric = float(np.mean(task_values))

        records.append(
            ExperimentRecord(
                condition=condition,
                severity=float(severity),
                drift_score=float(drift_score),
                task_metric=task_metric,
                baseline_metric=float(baseline_metric),
                metric_delta=task_metric - float(baseline_metric),
            )
        )

    return records


def _config_for(condition: str, severity: float) -> DriftConfig:
    if condition == "brightness":
        return DriftConfig(brightness_delta=severity)
    if condition == "gaussian_noise":
        return DriftConfig(gaussian_noise_std=severity)
    if condition == "blur":
        return DriftConfig(blur_radius=int(severity))
    if condition == "contrast":
        return DriftConfig(contrast_scale=severity)
    raise ValueError(f"unknown drift condition: {condition}")
