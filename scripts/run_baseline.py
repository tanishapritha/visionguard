from __future__ import annotations
import argparse
import json
import yaml
from visionguard.baseline.evaluator import evaluate_video
from visionguard.baseline.yolo import UltralyticsDetector
from visionguard.data.manifest import VideoManifest

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a detector baseline on a manifest video.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--video-id", default="example")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config.get("root")
    item = next(v for v in config["videos"] if v["video_id"] == args.video_id)
    manifest = VideoManifest(item["video_id"], item["video_path"], item.get("split", "unspecified"), item.get("annotation_path"))
    summary = evaluate_video(manifest, UltralyticsDetector(args.model, device=args.device), root=root, max_frames=args.max_frames)
    print(json.dumps(summary.__dict__, indent=2))

if __name__ == "__main__":
    main()
