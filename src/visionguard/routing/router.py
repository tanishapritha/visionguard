from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str

def choose_route(drift_score: float, uncertainty: float, drift_threshold: float = 0.70,
                 uncertainty_threshold: float = 0.55) -> RouteDecision:
    if drift_score >= drift_threshold:
        return RouteDecision("vlm", "drift_threshold_exceeded")
    if uncertainty >= uncertainty_threshold:
        return RouteDecision("vlm", "uncertainty_threshold_exceeded")
    return RouteDecision("cv", "within_normal_operating_range")
