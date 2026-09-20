import numpy as np
from visionguard.benchmark.drift_generator import DriftConfig, apply_drift

def test_brightness_drift_changes_frame():
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    shifted = apply_drift(frame, DriftConfig(brightness_delta=30), np.random.default_rng(42))
    assert shifted.mean() > frame.mean()

def test_output_is_valid_image_range():
    frame = np.full((8, 8, 3), 250, dtype=np.uint8)
    shifted = apply_drift(frame, DriftConfig(brightness_delta=100, gaussian_noise_std=20), np.random.default_rng(42))
    assert shifted.dtype == np.uint8
    assert shifted.min() >= 0 and shifted.max() <= 255
