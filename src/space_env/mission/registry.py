"""Имена учебных сценариев. Отложенный набор помечен отдельно и не смешивается с Кируной."""
from __future__ import annotations

from pathlib import Path

SCENARIOS = (
    {"id": "family_nominal", "file": "family_nominal.yaml", "holdout": False},
    {"id": "family_outage", "file": "family_outage.yaml", "holdout": False},
    {"id": "family_infeasible", "file": "family_infeasible.yaml", "holdout": False},
    {"id": "mission_short", "file": "mission_short.yaml", "holdout": False},
    {"id": "mission_holdout", "file": "mission_holdout.yaml", "holdout": True},
)


def scenario_registry() -> list[dict]:
    root = Path(__file__).resolve().parents[1] / "configs"
    rows = []
    for item in SCENARIOS:
        path = root / item["file"]
        rows.append({**item, "path": str(path), "present": path.is_file()})
    return rows
