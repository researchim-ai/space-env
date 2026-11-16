"""Bounded central-gravity constellation phasing recipes, not long-horizon deployment."""

from __future__ import annotations

import copy
import math

import numpy as np

from space_env.runtime import dynamic_reference_scenario
from space_env.tasks.modeling_suite import _inputs, _orbit_target
from space_env.tasks.schema import TaskSpec


def deployment_recipe(actors=2):
    if type(actors) is not int or not 2 <= actors <= 8:
        raise ValueError("phasing recipe supports 2..8 actors")
    original = _inputs()["chemical"]["definition"]
    definition = {
        k: copy.deepcopy(original[k])
        for k in ("schema", "eop", "horizon_sec", "operations", "orbits")
    }
    definition["horizon_sec"] = 60.0
    operations = definition["operations"]
    operations["products"] = []
    operations["vehicles"] = {}
    template = original["orbits"]["a"]
    definition["orbits"] = {}
    roles, actions, conditions, limits = {}, [], [], {}
    for index in range(actors):
        actor = "sat-" + str(index).zfill(2)
        orbit = copy.deepcopy(template)
        orbit["forces"] = {}
        angle = 0.002 * index
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        for key in ("r_m", "v_m_s"):
            orbit["initial"][key] = (
                rotation @ np.array(orbit["initial"][key])
            ).tolist()
        direction = np.array(orbit["initial"]["v_m_s"])
        direction /= np.linalg.norm(direction)
        direction *= 1 if index % 2 == 0 else -1
        target = _orbit_target(orbit, direction, thrust=2.0, firing=20.0, end=60.0)
        definition["orbits"][actor] = orbit
        vehicle = copy.deepcopy(original["operations"]["vehicles"]["a"])
        vehicle["solar_mode"] = "ideal_sun_tracking"
        operations["vehicles"][actor] = vehicle
        roles[actor] = {"actors": [actor], "action_slots": 1, "stations": []}
        actions.append(
            {
                "id": "burn-" + actor,
                "role": actor,
                "kind": "submit",
                "commands": [
                    {
                        "command_id": "burn-" + actor,
                        "actor_id": actor,
                        "kind": "finite_burn",
                        "duration_sec": 20.0,
                        "burn": {
                            "frame": "GCRS",
                            "direction_gcrs": direction.tolist(),
                            "thrust_n": 2.0,
                        },
                    }
                ],
            }
        )
        conditions.append(
            {
                "id": "slot-" + actor,
                "kind": "orbit_state",
                "actor": actor,
                "at_sec": 60.0,
                "position_m": target[:3],
                "velocity_m_s": target[3:],
                "position_tolerance_m": 0.01,
                "velocity_tolerance_m_s": 0.00001,
            }
        )
        limits[actor] = {"min_propellant_kg": 9.98}
    scenario = dynamic_reference_scenario(
        definition, id="native-local-phasing", lineage="central-local-phasing/v1"
    )
    task = TaskSpec.model_validate(
        {
            "id": "native-local-phasing",
            "roles": roles,
            "actions": actions,
            "decision_sec": 5.0,
            "max_decision_sec": 5.0,
            "max_decisions": 16,
            "max_joint_actions": actors,
            "goal": {"deadline_sec": 60.0, "conditions": conditions, "limits": limits},
        }
    ).model_dump(mode="json", by_alias=True)
    steps = [{"action_ids": [a["id"] for a in actions], "advance_sec": 5.0}] + [
        {"action_ids": ["wait"], "advance_sec": 5.0} for _ in range(11)
    ]
    return {
        "scenario": scenario,
        "task": task,
        "reference_steps": steps,
        "oracle": "Separate fixed-step RK4 h=0.005 s, common constants/inputs; constant inertial finite force, analytic mass depletion",
        "scope": "60-second central-gravity local phasing; explicit resource surrogate, no continuous conjunction guarantee, no physical deployment of antennas/panels, no end-to-end service coupling",
    }
