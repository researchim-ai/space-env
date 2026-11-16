"""Individually reachable goals competing for one common dispatch epoch."""

from __future__ import annotations

import copy

from space_env.runtime import dynamic_reference_scenario
from space_env.simulation.snapshot import digest
from space_env.tasks.modeling_suite import _orbit_target
from space_env.tasks.schema import TaskSpec


def competitive_recipe(actors, generation):
    from space_env.tasks.phasing_generation import generated_phasing_recipe

    base = generated_phasing_recipe(
        actors, generation.model_copy(update={"competition": False})
    )
    definition = base["scenario"]["definition"]
    horizon = definition["horizon_sec"]
    correct = {
        name
        for step in base["reference_steps"]
        for name in step["action_ids"]
        if name != "wait"
    }
    for actor, orbit in definition["orbits"].items():
        action = next(
            a
            for a in base["task"]["actions"]
            if a["role"] == actor and a["id"] in correct
        )
        command = action["commands"][0]
        burn = command["burn"]
        coast = _orbit_target(orbit, [0.0, 0.0, 0.0], thrust=0.0, firing=5.0, end=5.0)
        delayed = copy.deepcopy(orbit)
        delayed["initial"]["r_m"] = coast[:3]
        delayed["initial"]["v_m_s"] = coast[3:]
        target = _orbit_target(
            delayed,
            burn["direction_gcrs"],
            thrust=burn["thrust_n"],
            firing=command["duration_sec"],
            end=horizon - 5,
        )
        condition = next(
            c for c in base["task"]["goal"]["conditions"] if c["actor"] == actor
        )
        condition.update(position_m=target[:3], velocity_m_s=target[3:])
    base["launch_windows"] = {a["id"]: [[5.0, 10.0]] for a in base["task"]["actions"]}
    base["scenario"] = dynamic_reference_scenario(
        definition, id="competing-local-phasing", lineage="competing-local-phasing/v1"
    )
    base["task"]["id"] = "competing-local-phasing"
    base["task"] = TaskSpec.model_validate(base["task"]).model_dump(
        mode="json", by_alias=True
    )
    base["reference_steps"] = []
    base["generation"].update(
        schema="space-env.competing-phasing-generation/1", competition=True
    )
    base["generation"].pop("private_timing", None)
    base["dispatch_bound"] = {
        "maximum_completed_conditions": 1,
        "dispatch_epoch_sec": 5.0,
        "window_end_sec": 10.0,
        "reason": "One common five-second decision epoch, one shared dispatch start and required own-actor commands for distinct goals.",
    }
    base["generation"]["physics_sha256"] = digest(
        {
            "definition": definition,
            "goal": base["task"]["goal"],
            "commands": sorted(
                [
                    {k: v for k, v in a["commands"][0].items() if k != "command_id"}
                    for a in base["task"]["actions"]
                ],
                key=digest,
            ),
            "launch_window": [5.0, 10.0],
            "dispatch_capacity": 1,
            "one_burn_per_actor": True,
        }
    )
    base["scope"] = (
        "Individually reachable strict orbital goals compete for one common dispatch epoch; at most one condition can be met under declared template and dispatch rules. No all-goals reference or global physical reachability claim."
    )
    return base
