from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DriftConfig:
    brightness_delta: float = 0.0
    gaussian_noise_std: float = 0.0
    blur_radius: int = 0
    contrast_scale: float = 1.0

def _box_blur(frame: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return frame
    k = 2 * radius + 1
    padded = np.pad(frame, ((radius,radius),(radius,radius),(0,0)), mode="edge")
    out = np.zeros_like(frame, dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy+frame.shape[0], dx:dx+frame.shape[1]]
    return out / (k*k)

def apply_drift(frame: np.ndarray, config: DriftConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply controlled appearance drift to an HxWxC image."""
    image = np.asarray(frame, dtype=np.float32)
    if image.ndim != 3: raise ValueError("frame must have shape HxWxC")
    image = (image - 127.5) * config.contrast_scale + 127.5 + config.brightness_delta
    if config.gaussian_noise_std > 0:
        image += rng.normal(0.0, config.gaussian_noise_std, size=image.shape)
    if config.blur_radius > 0:
        image = _box_blur(image, config.blur_radius)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)
