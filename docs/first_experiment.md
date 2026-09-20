# First VisionGuard experiment

After preparing the AI City YOLO dataset and training a warehouse detector, run:

    python scripts/run_first_experiment.py --model path/to/best.pt --camera 0001 --condition brightness --severities -30 -15 0 15 30 --max-frames 100

The resulting JSON records the detector baseline, controlled drift score, F1 delta, and confidence shift for each severity.

The experiment is intentionally limited to appearance transforms whose ground-truth boxes remain valid. It is the first measurement linking a controlled visual shift to downstream detector behavior; it is not yet a semantic drift experiment.
