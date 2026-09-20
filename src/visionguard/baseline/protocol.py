from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

class Detector(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> Sequence[Detection]:
        ...

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def evaluate_detections(predictions: Sequence[Detection], ground_truth: Sequence[tuple[int, tuple[float, float, float, float]]], iou_threshold: float = 0.5) -> dict[str, float]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    remaining = list(enumerate(ground_truth))
    tp = 0
    for pred in sorted(predictions, key=lambda x: x.confidence, reverse=True):
        best, best_iou = None, 0.0
        for index, (class_id, bbox) in remaining:
            if class_id == pred.class_id:
                overlap = _iou(pred.bbox_xyxy, bbox)
                if overlap >= iou_threshold and overlap > best_iou:
                    best, best_iou = index, overlap
        if best is not None:
            tp += 1
            remaining = [(i, gt) for i, gt in remaining if i != best]
    fp, fn = len(predictions) - tp, len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "true_positive": float(tp), "false_positive": float(fp), "false_negative": float(fn)}
