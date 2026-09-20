from __future__ import annotations
from dataclasses import dataclass
from ..data.manifest import VideoManifest
from ..data.video import read_video
from .protocol import Detector, evaluate_detections

@dataclass(frozen=True)
class BaselineSummary:
    frames_evaluated: int
    precision: float
    recall: float
    f1: float
    mean_confidence: float

def evaluate_video(manifest: VideoManifest, detector: Detector, root: str | None = None, stride: int = 1, max_frames: int | None = None) -> BaselineSummary:
    annotations = manifest.load_annotations(root)
    metrics, confidences = [], []
    for frame in read_video(manifest.resolve_path(root), manifest.video_id, stride, max_frames):
        predictions = list(detector.predict(frame.image_bgr))
        ground_truth = [(x.class_id, x.bbox_xyxy) for x in annotations.get(frame.frame_index, [])]
        metrics.append(evaluate_detections(predictions, ground_truth))
        confidences.extend(x.confidence for x in predictions)
    if not metrics:
        raise ValueError("no frames were evaluated")
    return BaselineSummary(
        len(metrics),
        sum(x["precision"] for x in metrics) / len(metrics),
        sum(x["recall"] for x in metrics) / len(metrics),
        sum(x["f1"] for x in metrics) / len(metrics),
        sum(confidences) / len(confidences) if confidences else 0.0,
    )
