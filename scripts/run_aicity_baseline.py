from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from visionguard.baseline.evaluator import evaluate_video
from visionguard.baseline.yolo import UltralyticsDetector
from visionguard.data.aicity import AICityScene

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a detector baseline on an AI City warehouse scene.")
    parser.add_argument("--config", default="configs/aicity_warehouse.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["scene_root"])
    scene = AICityScene(root, root / "ground_truth.json")
    manifests = scene.manifests()
    if args.camera is not None:
        manifests = [m for m in manifests if m.video_id.endswith(f":{args.camera}")]
    if not manifests:
        raise SystemExit("No matching camera videos found.")
    detector = UltralyticsDetector(args.model, device=args.device)
    results = {}
    for manifest in manifests:
        summary = evaluate_video(manifest, detector, max_frames=args.max_frames)
        results[manifest.video_id] = summary.__dict__
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
