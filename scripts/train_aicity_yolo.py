from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on prepared AI City frames.")
    parser.add_argument("--data", default="configs/aicity_yolo_dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/visionguard")
    parser.add_argument("--name", default="aicity-baseline")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(results)


if __name__ == "__main__":
    main()
