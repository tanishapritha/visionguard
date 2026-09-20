from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import cv2
import yaml

from visionguard.data.aicity import AICityScene


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AI City 2025 scenes to YOLO detection format.")
    parser.add_argument("--config", default="configs/aicity_yolo.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    classes = config["classes"]
    class_to_id = {name: i for i, name in enumerate(classes)}
    stride = int(config.get("frame_stride", 30))
    max_frames = config.get("max_frames_per_camera")
    seed = int(config.get("seed", 42))

    scenes = []
    for split in ("train", "val"):
        for scene_root in config.get(split, []):
            scenes.append((split, Path(scene_root)))

    for split, scene_root in scenes:
        scene = AICityScene(scene_root, scene_root / "ground_truth.json")
        manifests = scene.manifests()
        for manifest in manifests:
            _export_camera(
                manifest, scene_root, output, split, class_to_id,
                stride=stride, max_frames=max_frames, seed=seed,
            )


def _export_camera(manifest, scene_root, output, split, class_to_id, stride, max_frames, seed):
    annotations = manifest.load_annotations()
    video_path = manifest.resolve_path()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    camera_id = manifest.video_id.rsplit(":", 1)[-1]
    scene_id = manifest.video_id.split(":", 1)[0]
    rng = random.Random(seed)
    frame_index = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue
        if max_frames is not None and written >= int(max_frames):
            break

        frame_annotations = annotations.get(frame_index, [])
        labels = []
        height, width = frame.shape[:2]
        for ann in frame_annotations:
            if ann.class_id < 0:
                continue
            # The adapter uses a stable class-id mapping; only requested classes
            # are exported to keep the detector label space explicit.
            class_name = _class_name_for_id(ann.class_id)
            if class_name not in class_to_id:
                continue
            x1, y1, x2, y2 = ann.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            labels.append(f"{class_to_id[class_name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if labels:
            stem = f"{scene_id}__{camera_id}__{frame_index:06d}"
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_dir / f"{stem}.jpg"), frame)
            (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            written += 1
        frame_index += 1

    cap.release()


def _class_name_for_id(class_id: int) -> str:
    return {
        0: "Person",
        1: "Forklift",
        2: "NovaCarter",
        3: "Transporter",
        4: "FourierGR1T2",
        5: "AgilityDigit",
    }.get(class_id, "unknown")


if __name__ == "__main__":
    main()
