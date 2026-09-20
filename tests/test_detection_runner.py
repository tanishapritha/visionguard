from __future__ import annotations
import numpy as np
from visionguard.benchmark.detection_runner import image_statistics

def test_image_statistics_is_deterministic() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = 10, 20, 30
    first, second = image_statistics(frame), image_statistics(frame)
    assert first.shape == (6,)
    assert np.array_equal(first, second)
    assert np.allclose(first[:3], [10, 20, 30])
