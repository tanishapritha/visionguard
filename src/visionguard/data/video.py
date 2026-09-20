from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np

@dataclass(frozen=True)
class VideoFrame:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    image_bgr: np.ndarray

def read_video(path: str | Path, video_id: str | None = None, stride: int = 1, max_frames: int | None = None) -> Iterator[VideoFrame]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for video ingestion. Install visionguard[cv].") from exc
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    emitted, frame_index = 0, 0
    resolved_id = video_id or path.stem
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                yield VideoFrame(resolved_id, frame_index, frame_index / fps, np.asarray(frame))
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
