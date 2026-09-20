from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .runner import ExperimentRecord


def write_results(records: Iterable[ExperimentRecord], path: str | Path) -> Path:
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("records must not be empty")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return destination
