"""Один отчёт миссии для CLI, среды и студии."""
from __future__ import annotations

from pathlib import Path

from space_env.mission.tasking import request_lifecycle, requests_from_scenario, run_scenario
from space_env.mission.scenario_v2 import load_scenario_v2


def mission_report(
    scenario: str | Path,
    *,
    policy: str = "priority",
    step_sec: float = 30.0,
    refresh_passes: bool = False,
) -> dict:
    spec = load_scenario_v2(scenario)
    report = run_scenario(
        scenario,
        step_sec=step_sec,
        policy=policy,
        refresh_passes=refresh_passes,
    )
    requests = requests_from_scenario(spec)
    statuses = [
        {
            "request_id": item.request_id,
            "status": request_lifecycle(item, report["receipts"], spec["horizon_sec"]),
            "destination": item.destination,
            "quality": item.quality,
        }
        for item in requests
    ]
    return {
        "scenario": Path(scenario).name,
        "on_time": list(report["score"]["on_time"]),
        "unmet": report["unmet"],
        "statuses": statuses,
        "weighted_on_time": report["score"]["weighted_on_time"],
        "safety_violations": report["kpis"]["safety_violations"],
    }
