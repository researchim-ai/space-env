"""Эвристики, которые задачи иногда решают. Нужны, чтобы в траекториях был не только шум."""
from __future__ import annotations

import numpy as np

from space_env.envs.cards import ENV_CARDS, observation_index


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
    if "ConjunctionQueue" in env_id:
        hot = observation_index(env_id, "Pc горячей пары")
        return 1 if obs[hot] > 0.01 else 0
    if "RendezvousHold" in env_id or "DebrisAvoid" in env_id or "Disposal" in env_id:
        return 1
    if "PriorityCoverage" in env_id:
        return 1 if obs[0] > 0.5 else 0
    if "SentinelPassFault" in env_id:
        if obs[0] > 0.5 and obs[1] > 0.5:
            return 1
        if obs[2] < 0.5:
            return 2
        return 0
    if "StationContention" in env_id:
        count = obs.size // 2
        buffers = obs[:count]
        visible = obs[count:]
        candidates = [index for index in range(count) if visible[index] > 0.5]
        if not candidates:
            return 0
        return max(candidates, key=lambda index: float(buffers[index]))
    if "OrderBook" in env_id:
        return 1 if obs[0] > 0.5 else 0
    if "PassFault" in env_id:
        if obs[0] > 0.5:
            return 1
        if obs[2] < 0.5:
            return 2
        return 0
    if "PhaseBurn" in env_id or "Reroute" in env_id:
        return 1
    if "DecayWatch" in env_id:
        return 1 if obs.size > 1 and obs[1] > 0.5 else 0
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
    if "FleetShift" in env_id or "ConstellationOps" in env_id:
        if "FleetShift" in env_id:
            count = (obs.size - 4) // 4
            packet, elevation, stride = 1, 2, 4
            task_index = count * 4 + 2
        else:
            names = ENV_CARDS[env_id].observation
            count = sum(1 for name in names if name.startswith("доля пакета"))
            packet = observation_index(env_id, "доля пакета 1")
            elevation = observation_index(env_id, "угол места 1")
            stride = observation_index(env_id, "доля пакета 2") - packet
            task_index = None
        if count < 2:
            return 0
        buffers = [obs[packet + stride * index] for index in range(count)]
        elevations = [obs[elevation + stride * index] for index in range(count)]
        holder = int(np.argmax(buffers))
        gateway = int(np.argmax(elevations))
        relay = buffers[holder] >= 0.5 and holder != gateway
        if task_index is not None:
            relay = relay and obs[task_index] > 0.5
        if not relay:
            return np.array([holder, 0], dtype=int)
        return np.array([holder, 2], dtype=int)
    return 0
