# AI City YOLO baseline

This pipeline converts the NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2025 JSON annotations into standard YOLO detection labels, trains a detector, and keeps raw media outside Git.

The dataset's 2025 schema provides per-frame objects with camera-specific visible 2D boxes. The released 2025 data is synthetic, so detector results here are a controlled benchmark rather than evidence of real-world warehouse robustness.

## Workflow

1. Download only the selected scenes/cameras you need.
2. Update `configs/aicity_yolo.yaml` with local scene paths.
3. Run `python scripts/prepare_aicity_yolo.py`.
4. Run `python scripts/train_aicity_yolo.py --epochs 20`.
5. Use the resulting checkpoint with the VisionGuard baseline and controlled-drift runners.

The default configuration deliberately uses a small subset and frame stride so the first experiment is tractable. Increase coverage after the pipeline is validated.
