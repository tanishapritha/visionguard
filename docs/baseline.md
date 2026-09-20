# Real video baseline

VisionGuard now has a real-video ingestion and detector-evaluation boundary.

## Data contract

VideoManifest stores a video path, split, and optional frame-level annotation JSONL. Each annotation line contains frame_index, class_id, and bbox_xyxy. Raw media stays outside git.

## Sampling

read_video streams frames through OpenCV and emits frame index plus timestamp. A frame stride provides a deterministic temporal sampling contract for later drift analysis.

## Detector

UltralyticsDetector is an optional adapter. The core package does not require Ultralytics. The adapter converts detector output into VisionGuard Detection records.

## Evaluation

The baseline evaluator computes frame-level precision, recall, and F1 with class-aware IoU matching. These are a common minimal metric, not a replacement for a dataset's official mAP/tracking evaluation.

Run:
python scripts/run_baseline.py --model <checkpoint> --video-id <id>
