from __future__ import annotations

import json
from pathlib import Path


def load_experiment(path: str | Path) -> dict:
    """Load a VisionGuard experiment result artifact."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "records" not in payload:
        raise ValueError("experiment artifact must contain records")
    return payload


def summarize_experiment(payload: dict) -> dict:
    records = payload.get("records", [])
    if not records:
        return {"records": 0}
    return {
        "records": len(records),
        "baseline_f1": records[0]["baseline_f1"],
        "largest_f1_drop": min(r["metric_delta"] for r in records),
        "largest_drift_score": max(r["drift_score"] for r in records),
        "severities": [r["severity"] for r in records],
    }
