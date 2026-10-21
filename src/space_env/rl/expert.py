"""Эвристики, которые задачи иногда решают. Нужны, чтобы в траекториях был не только шум."""
from __future__ import annotations

import numpy as np


def expert_action(env_id: str, observation: np.ndarray) -> int:
    obs = np.asarray(observation, dtype=float).ravel()
    if "ResourceManager" in env_id:
        in_eclipse = obs[1] > 0.5
        soc = obs[2]
        return 2 if in_eclipse or soc < 0.45 else 0
    if "AnomalyResponse" in env_id:
        temp = obs[1]
        radio = obs[3]
        rate = obs[4]
        sun = obs[5] if obs.size > 5 else 1.0
        if radio < 0.5:
            return 3
        if temp > 0.65:
            return 1
        if rate < 0.35 and sun > 0.5:
            return 2
        return 0
    if "ConjunctionAssessment" in env_id:
        radial = abs(obs[0] - 0.5)
        cross = abs(obs[1] - 0.5)
        probability = obs[2]
        if probability < 0.1:
            return 0
        return 1 if radial >= cross else 2
    if "MissionPlanner" in env_id:
        in_eclipse = obs[1] > 0.5
        imaging = obs[4]
        comms = obs[5]
        contact = obs[6] > 0.5
        hold = obs[7] if obs.size > 7 else 1.0
        if hold < 0.55:
            return 3
        if not in_eclipse and imaging < 1.0:
            return 1
        if contact and comms < 1.0:
            return 2
        return 0
    if "WorkPlan" in env_id:
        kind = obs[5] if obs.size > 6 else 0.0
        in_window = obs[6] > 0.5 if obs.size > 6 else False
        if in_window and kind > 0.75:
            return 2
        if in_window and kind > 0.25:
            return 1
        return 0
    if "ConstellationOps" in env_id:
        count = (obs.size - 2) // 4
        if count < 2:
            return 0
        buffers = [obs[4 * index + 1] for index in range(count)]
        elevations = [obs[4 * index + 2] for index in range(count)]
        holder = int(np.argmax(buffers))
        gateway = int(np.argmax(elevations))
        if buffers[holder] < 0.5 or holder == gateway:
            return np.array([holder, 0], dtype=int)
        return np.array([holder, 2], dtype=int)
    return 0
