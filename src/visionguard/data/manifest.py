from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class FrameAnnotation:
    frame_index: int
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]

@dataclass(frozen=True)
class VideoManifest:
    video_id: str
    path: str
    split: str = "unspecified"
    annotations_path: str | None = None

    def resolve_path(self, root: str | Path | None = None) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() or root is None else Path(root) / path

    def load_annotations(self, root: str | Path | None = None) -> dict[int, list[FrameAnnotation]]:
        if not self.annotations_path:
            return {}
        path = Path(self.annotations_path)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if not path.exists():
            raise FileNotFoundError(path)
        annotations: dict[int, list[FrameAnnotation]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                frame = int(item["frame_index"])
                annotations.setdefault(frame, []).append(FrameAnnotation(
                    frame_index=frame,
                    class_id=int(item["class_id"]),
                    bbox_xyxy=tuple(float(v) for v in item["bbox_xyxy"]),
                ))
        return annotations
