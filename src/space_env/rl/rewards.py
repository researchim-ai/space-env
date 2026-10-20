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
}


def _config_path() -> Path:
    name = Path("configs") / "rewards.yaml"
    candidates = [Path.cwd() / name, Path(__file__).resolve().parents[3] / name]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


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
