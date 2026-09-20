from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from visionguard.baseline.yolo import UltralyticsDetector
from visionguard.benchmark.detection_runner import image_statistics, run_detection_drift_experiment
from visionguard.data.aicity import AICityScene


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first end-to-end VisionGuard experiment.")
    parser.add_argument("--config", default="configs/aicity_warehouse.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--condition", default="brightness",
                        choices=["brightness", "gaussian_noise", "blur", "contrast"])
    parser.add_argument("--severities", nargs="+", type=float, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/first_experiment.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["scene_root"])
    scene = AICityScene(root, root / "ground_truth.json")
    manifests = [m for m in scene.manifests() if m.video_id.endswith(f":{args.camera}")]
    if not manifests:
        raise SystemExit(f"No matching camera found: {args.camera}")

    detector = UltralyticsDetector(args.model, device=args.device)
    records = run_detection_drift_experiment(
        manifests[0],
        detector,
        feature_fn=image_statistics,
        condition=args.condition,
        severities=args.severities,
        max_frames=args.max_frames,
        stride=args.stride,
        iou_threshold=float(config.get("evaluation", {}).get("iou_threshold", 0.5)),
    )

    payload = {
        "experiment": "controlled_appearance_drift",
        "video_id": manifests[0].video_id,
        "condition": args.condition,
        "model": args.model,
        "records": [record.__dict__ for record in records],
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
