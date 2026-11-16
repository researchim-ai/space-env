"""Constructive delayed-burn targets and explicit synthetic dispatch windows."""

from __future__ import annotations

import copy
import hashlib

from space_env.runtime import dynamic_reference_scenario
from space_env.simulation.snapshot import digest
from space_env.tasks.modeling_suite import _orbit_target
from space_env.tasks.schema import TaskSpec


def windowed_recipe(actors, generation):
    from space_env.tasks.phasing_generation import generated_phasing_recipe

    if type(actors) is not int or not 2 <= actors <= 4:
        raise ValueError("windowed phasing supports 2..4 actors")
    base = generated_phasing_recipe(
        actors, generation.model_copy(update={"windows": False})
    )
    definition = base["scenario"]["definition"]
    goal = base["task"]["goal"]
    horizon = goal["deadline_sec"]
    correct = set(base["reference_steps"][0]["action_ids"])
    # Unique constructive start epochs respect one shared dispatch slot per step.
    epochs = sorted(
        (5.0, 10.0, 15.0, 20.0),
        key=lambda t: hashlib.sha256(
            f"windows/v1:{generation.split}:{generation.seed}:{t}".encode()
        ).digest(),
    )[:actors]
    windows = {}
    schedule = {}
    parameters = []
    for index, (actor, orbit) in enumerate(definition["orbits"].items()):
        start = epochs[index]
        action = next(
            a
            for a in base["task"]["actions"]
            if a["id"] in correct and a["role"] == actor
        )
        command = action["commands"][0]
        burn = command["burn"]
        coast = _orbit_target(
            orbit, [0.0, 0.0, 0.0], thrust=0.0, firing=start, end=start
        )
        delayed = copy.deepcopy(orbit)
        delayed["initial"]["r_m"] = coast[:3]
        delayed["initial"]["v_m_s"] = coast[3:]
        target = _orbit_target(
            delayed,
            burn["direction_gcrs"],
            thrust=burn["thrust_n"],
            firing=command["duration_sec"],
            end=horizon - start,
        )
        condition = next(c for c in goal["conditions"] if c["actor"] == actor)
        condition["position_m"] = target[:3]
        condition["velocity_m_s"] = target[3:]
        schedule.setdefault(start, []).append(action["id"])
        parameters.append({"actor": actor, "reference_start_sec": start})
        alternatives = [t for t in (5.0, 10.0, 15.0, 20.0) if t != start]
        decoy = min(
            alternatives,
            key=lambda t: hashlib.sha256(
                f"windows-choice/v1:{generation.split}:{generation.seed}:{actor}:{t}".encode()
            ).digest(),
        )
        starts = sorted((0.0, start, decoy))
        for candidate in base["task"]["actions"]:
            if candidate["role"] == actor:
                windows[candidate["id"]] = [[float(t), float(t + 5)] for t in starts]
    base["scenario"] = dynamic_reference_scenario(
        definition, id="windowed-local-phasing", lineage="windowed-local-phasing/v1"
    )
    base["task"]["id"] = "windowed-local-phasing"
    base["task"] = TaskSpec.model_validate(base["task"]).model_dump(
        mode="json", by_alias=True
    )
    base["reference_steps"] = [
        {"action_ids": schedule.get(float(t), ["wait"]), "advance_sec": 5.0}
        for t in range(0, int(horizon), 5)
    ]
    base["launch_windows"] = windows
    base["dispatch_capacity"] = 1
    base["generation"]["schema"] = "space-env.windowed-phasing-generation/1"
    base["generation"]["windows"] = True
    base["generation"]["private_timing"] = parameters
    base["generation"]["physics_sha256"] = digest(
        {
            "definition": definition,
            "goal": goal,
            "controls": sorted(
                [
                    {k: v for k, v in a["commands"][0].items() if k != "command_id"}
                    for a in base["task"]["actions"]
                ],
                key=digest,
            ),
            "launch_windows_by_actor": {
                actor: windows[
                    next(a["id"] for a in base["task"]["actions"] if a["role"] == actor)
                ]
                for actor in definition["orbits"]
            },
            "dispatch_capacity": 1,
            "one_burn_per_actor": True,
        }
    )
    base["scope"] = (
        "40-90 second central-gravity delayed-burn phasing, 2..4 actors. Published half-open dispatch windows and one shared command-start slot per step; synthetic scheduling constraint, not a simulated RF uplink. One burn per actor, no continuous conjunction guarantee."
    )
    return base
