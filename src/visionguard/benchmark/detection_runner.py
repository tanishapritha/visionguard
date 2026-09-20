from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import numpy as np
from ..baseline.protocol import Detector, evaluate_detections
from ..data.manifest import VideoManifest
from ..data.video import read_video
from ..drift.metrics import rbf_mmd_squared
from .drift_generator import DriftConfig, apply_drift

@dataclass(frozen=True)
class DetectionDriftRecord:
    video_id: str
    condition: str
    severity: float
    frames_evaluated: int
    drift_score: float
    baseline_f1: float
    drifted_f1: float
    metric_delta: float
    baseline_mean_confidence: float
    drifted_mean_confidence: float

def run_detection_drift_experiment(
    manifest: VideoManifest, detector: Detector,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    condition: str, severities: Iterable[float],
    root: str | None = None, stride: int = 1,
    max_frames: int | None = None, iou_threshold: float = 0.5,
    seed: int = 42,
) -> list[DetectionDriftRecord]:
    """Measure appearance drift and detector degradation on real video."""
    frames = list(read_video(manifest.resolve_path(root), manifest.video_id, stride, max_frames))
    if not frames:
        raise ValueError("no frames were evaluated")
    annotations = manifest.load_annotations(root)
    source_images = [frame.image_bgr for frame in frames]
    reference_features = _feature_matrix(source_images, feature_fn)
    baseline_f1s, baseline_confidences = [], []
    for frame in frames:
        predictions = list(detector.predict(frame.image_bgr))
        ground_truth = [(x.class_id, x.bbox_xyxy) for x in annotations.get(frame.frame_index, [])]
        metrics = evaluate_detections(predictions, ground_truth, iou_threshold)
        baseline_f1s.append(metrics["f1"])
        baseline_confidences.extend(x.confidence for x in predictions)
    baseline_f1 = float(np.mean(baseline_f1s))
    baseline_mean_confidence = float(np.mean(baseline_confidences)) if baseline_confidences else 0.0
    rng = np.random.default_rng(seed)
    records = []
    for severity in severities:
        config = _config_for(condition, float(severity))
        drifted_images = [apply_drift(image, config, rng) for image in source_images]
        drift_score = rbf_mmd_squared(reference_features, _feature_matrix(drifted_images, feature_fn))
        drifted_f1s, drifted_confidences = [], []
        for frame, image in zip(frames, drifted_images):
            predictions = list(detector.predict(image))
            ground_truth = [(x.class_id, x.bbox_xyxy) for x in annotations.get(frame.frame_index, [])]
            metrics = evaluate_detections(predictions, ground_truth, iou_threshold)
            drifted_f1s.append(metrics["f1"])
            drifted_confidences.extend(x.confidence for x in predictions)
        drifted_f1 = float(np.mean(drifted_f1s))
        records.append(DetectionDriftRecord(
            manifest.video_id, condition, float(severity), len(frames),
            float(drift_score), baseline_f1, drifted_f1, drifted_f1 - baseline_f1,
            baseline_mean_confidence,
            float(np.mean(drifted_confidences)) if drifted_confidences else 0.0,
        ))
    return records

def _feature_matrix(images: Sequence[np.ndarray], feature_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(feature_fn(image), dtype=float).reshape(1, -1) for image in images])

def image_statistics(frame_bgr: np.ndarray) -> np.ndarray:
    """Small deterministic feature vector for baseline experiments."""
    image = np.asarray(frame_bgr, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    return np.concatenate([image.mean(axis=(0, 1)), image.std(axis=(0, 1))])

def _config_for(condition: str, severity: float) -> DriftConfig:
    if condition == "brightness": return DriftConfig(brightness_delta=severity)
    if condition == "gaussian_noise": return DriftConfig(gaussian_noise_std=severity)
    if condition == "blur": return DriftConfig(blur_radius=int(severity))
    if condition == "contrast": return DriftConfig(contrast_scale=severity)
    raise ValueError(f"unknown drift condition: {condition}")
