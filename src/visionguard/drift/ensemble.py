from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DriftSignals:
    embedding: float
    confidence: float
    output_shift: float

@dataclass(frozen=True)
class DriftAssessment:
    score: float
    triggered: bool
    reason: str

def combine_signals(signals: DriftSignals, embedding_threshold: float = 0.70,
                    confidence_threshold: float = 0.15, output_shift_threshold: float = 0.20) -> DriftAssessment:
    exceeded = []
    if signals.embedding >= embedding_threshold: exceeded.append("embedding")
    if signals.confidence >= confidence_threshold: exceeded.append("confidence")
    if signals.output_shift >= output_shift_threshold: exceeded.append("output_shift")
    score = max(signals.embedding/max(embedding_threshold,1e-12),
                signals.confidence/max(confidence_threshold,1e-12),
                signals.output_shift/max(output_shift_threshold,1e-12))
    return DriftAssessment(float(score), bool(exceeded), ",".join(exceeded) if exceeded else "within_thresholds")
