from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from .manifest import VideoManifest

@dataclass(frozen=True)
class AICityScene:
    scene_root: Path
    ground_truth_path: Path

    def manifests(self) -> list[VideoManifest]:
        payload = json.loads(self.ground_truth_path.read_text(encoding="utf-8"))
        camera_ids = _camera_ids(payload)
        manifests = []
        for video in sorted((self.scene_root / "videos").glob("*.mp4")):
            camera_id = _camera_id_from_name(video.stem)
            if camera_id is None or camera_id not in camera_ids:
                continue
            annotation_path = self._materialize_camera_annotations(camera_id, payload)
            manifests.append(VideoManifest(
                video_id=f"{self.scene_root.name}:{camera_id}",
                path=str(video), split="train", annotations_path=str(annotation_path)
            ))
        return manifests

    def _materialize_camera_annotations(self, camera_id: str, payload: dict) -> Path:
        target = self.scene_root / ".visionguard" / f"camera_{camera_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        with target.open("w", encoding="utf-8") as handle:
            for frame_id, objects in payload.items():
                for obj in objects:
                    box = obj.get("2d_bounding_box_visible", {}).get(camera_id)
                    if box is None:
                        continue
                    handle.write(json.dumps({
                        "frame_index": int(frame_id),
                        "class_id": object_type_id(obj.get("object_type", "unknown")),
                        "bbox_xyxy": box,
                        "object_id": obj.get("object_id"),
                    }) + "\n")
        return target

def object_type_id(name: str) -> int:
    return {
        "Person": 0, "Forklift": 1, "NovaCarter": 2,
        "Transporter": 3, "FourierGR1T2": 4, "AgilityDigit": 5,
    }.get(name, 99)

def _camera_ids(payload: dict) -> set[str]:
    ids = set()
    for objects in payload.values():
        for obj in objects:
            ids.update(obj.get("2d_bounding_box_visible", {}).keys())
    return ids

def _camera_id_from_name(stem: str) -> str | None:
    parts = stem.lower().split("_")
    return parts[-1] if len(parts) >= 2 and parts[-1].isdigit() else None
