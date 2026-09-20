from __future__ import annotations
from typing import Sequence
import numpy as np
from .protocol import Detection

class UltralyticsDetector:
    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Ultralytics is required for this adapter. Install visionguard[cv].") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def predict(self, frame_bgr: np.ndarray) -> Sequence[Detection]:
        kwargs = {"conf": self.confidence, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(source=frame_bgr, **kwargs)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes
        return [Detection(int(c), float(conf), tuple(float(v) for v in xyxy))
                for xyxy, conf, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]
