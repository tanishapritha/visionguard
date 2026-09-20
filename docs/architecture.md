# Architecture

VisionGuard is an evaluation pipeline rather than a dashboard-first application.

1. Video ingestion supplies frames.
2. A baseline CV model produces task outputs and confidence values.
3. Feature extraction produces representations for drift comparison.
4. The drift engine compares a reference window with a current window.
5. Routing decides whether ordinary CV inference is sufficient.
6. A VLM provider can inspect selected frames or short clips.
7. Evidence schemas bind claims to frame IDs and timestamps.
8. Evaluation records task quality, drift metrics, routing behavior, latency, and evidence quality.

The benchmark is the source of truth. Thresholds are experimental parameters to be calibrated, not universal constants.
