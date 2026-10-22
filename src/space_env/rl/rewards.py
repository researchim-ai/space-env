"""Веса наград. Источник — configs/rewards.yaml, при отсутствии файла берутся значения по умолчанию."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REWARDS: dict[str, dict[str, float]] = {
    "mission_planner": {
        "imaging_success": 0.2,
        "comms_success": 0.2,
        "maneuver_cost": 0.05,
        "goal_complete": 1.0,
        "invalid_action": 0.05,
        "delta_v_overrun": 0.2,
    },
    "resource_manager": {
        "soc_failure": 1.0,
        "thermal_failure": 0.5,
        "survival_step": 0.01,
        "episode_complete": 0.1,
    },
    "anomaly": {
        "step_cost": 0.05,
        "recovered": 1.0,
        "soc_failure": 1.0,
        "thermal_failure": 0.5,
        "premature_reboot": 0.3,
    },
    "conjunction": {
        "risk_reduction": 0.3,
        "delta_v_cost": 0.05,
        "avoided": 1.0,
        "collision_risk": 1.0,
        "unnecessary_burn": 0.2,
    },
    "workplan": {
        "task_done": 1.0,
        "soc_failure": 1.0,
    },
    "conjunction_queue": {
        "avoided": 1.0,
        "missed": 1.0,
    },
    "decay_watch": {
        "correct": 1.0,
        "wrong": 0.5,
    },
    "phase_burn": {
        "stopped": 1.0,
        "drifting": 1.0,
    },
    "reroute": {
        "delivered": 1.0,
        "stuck": 1.0,
    },
    "fleet_shift": {
        "delivered": 1.0,
        "cold_burn": 1.0,
    },
    "rendezvous_hold": {"held": 1.0, "drift": 1.0},
    "priority_coverage": {"priority": 1.0, "missed": 1.0},
    "pass_fault": {"recovered": 1.0, "lost_pass": 1.0},
    "debris_avoid": {"avoided": 1.0, "missed": 1.0},
    "disposal": {"lowered": 1.0, "left": 1.0},
    "station_contention": {"scheduled": 1.0, "conflict": 1.0},
    "order_book": {"filled": 1.0, "wasted": 1.0},
    "day_mission": {"accepted": 1.0, "rejected": 1.0},
    "day_shift": {"task_done": 1.0, "missed": 1.0},
    "dish_award": {"awarded": 1.0, "missed": 1.0},
    "operator_shift": {"task_done": 1.0, "step_done": 0.1, "missed": 1.0},
    "radar_pass": {"scene": 1.0, "missed": 1.0},
    "longitude_keep": {"held": 1.0, "drift": 1.0},
}


def _config_path() -> Path:
    packaged = Path(__file__).resolve().parents[1] / "configs" / "rewards.yaml"
    cwd = Path.cwd() / "configs" / "rewards.yaml"
    if cwd.is_file():
        return cwd
    return packaged


@lru_cache(maxsize=1)
def load_rewards() -> dict[str, dict[str, float]]:
    """Сливает YAML поверх значений по умолчанию. Ключи, которых нет в файле, остаются."""
    merged: dict[str, dict[str, float]] = {name: dict(weights) for name, weights in DEFAULT_REWARDS.items()}
    path = _config_path()
    if not path.exists():
        return merged
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle) or {}
    for name, weights in loaded.items():
        if isinstance(weights, dict) and name in merged:
            for key, value in weights.items():
                merged[name][key] = float(value)
    return merged
