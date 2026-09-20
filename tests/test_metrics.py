import numpy as np
from visionguard.drift.metrics import class_frequency_shift, confidence_drop, rbf_mmd_squared

def test_mmd_identical_samples_is_near_zero():
    x = np.array([[0.0], [1.0], [2.0]])
    assert rbf_mmd_squared(x, x) < 1e-8

def test_confidence_drop():
    assert confidence_drop(np.array([0.9, 0.8]), np.array([0.6, 0.7])) == 0.2

def test_class_frequency_shift():
    assert class_frequency_shift({"a": 10}, {"b": 10}) == 1.0
