from math import ceil

def stride_for_fps(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    return max(1, int(ceil(source_fps / target_fps)))
