# Dataset: PhysicalAI-SmartSpaces MTMC_Tracking_2025

VisionGuard now has a concrete dataset adapter for the AI City / NVIDIA PhysicalAI-SmartSpaces warehouse benchmark.

The 2025 format provides MP4 camera videos plus frame-level 2D and 3D annotations. The ground truth JSON stores per-frame objects and a camera-specific visible 2D bounding box. NVIDIA documents the dataset as 1080p video at 30 FPS.

The adapter converts camera-specific 2D boxes into VisionGuard's detector-agnostic annotation contract. Media is never committed to the repository.

The released warehouse data is synthetic. It is therefore appropriate for controlled experiments and multi-camera structure, but results must not be presented as evidence of robustness on real warehouse deployments.

Local setup:
1. Download one scene.
2. Place it under data/raw/MTMC_Tracking_2025/train/Warehouse_000.
3. Install the CV extra.
4. Run scripts/run_aicity_baseline.py with a detector checkpoint.

Expected scene files: videos/, ground_truth.json, calibration.json.
