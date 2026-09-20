from typing import Literal
from pydantic import BaseModel, Field

class EvidenceFrame(BaseModel):
    frame_id: str
    timestamp_seconds: float
    relevance: float = Field(ge=0.0, le=1.0)

class DriftClaim(BaseModel):
    claim: str
    evidence: list[EvidenceFrame] = Field(default_factory=list)
    status: Literal["supported", "contradicted", "insufficient_evidence"]
    confidence: float = Field(ge=0.0, le=1.0)
