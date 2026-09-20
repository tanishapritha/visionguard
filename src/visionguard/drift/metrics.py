from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def rbf_mmd_squared(reference: np.ndarray, current: np.ndarray, gamma: float | None = None) -> float:
    """Compute biased squared MMD with an RBF kernel."""
    x, y = np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("reference and current must be 2D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("feature dimensions must match")
    if not len(x) or not len(y):
        raise ValueError("both samples must be non-empty")
    if gamma is None:
        sample = np.vstack([x, y])
        d = cdist(sample, sample, metric="sqeuclidean")
        nz = d[d > 0]
        gamma = 1.0 / (float(np.median(nz)) + 1e-12) if len(nz) else 1.0
    k_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def confidence_drop(reference: np.ndarray, current: np.ndarray) -> float:
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("confidence arrays must be non-empty")
    return float(ref.mean() - cur.mean())


def class_frequency_shift(reference_counts: dict[str, int], current_counts: dict[str, int]) -> float:
    """Total-variation distance between class-frequency distributions."""
    classes = sorted(set(reference_counts) | set(current_counts))
    if not classes:
        raise ValueError("at least one class is required")
    ref_total, cur_total = max(sum(reference_counts.values()), 1), max(sum(current_counts.values()), 1)
    return float(0.5 * sum(abs(reference_counts.get(c, 0)/ref_total - current_counts.get(c, 0)/cur_total) for c in classes))
