"""Paired recovery/critical closures sharing the same public initial mission."""

from __future__ import annotations

import copy

from space_env.runtime import dynamic_reference_scenario
from space_env.simulation.snapshot import digest
from space_env.tasks.modeling_suite import _orbit_target
from space_env.tasks.schema import TaskSpec


def disrupted_recipe(actors, generation):
    from space_env.tasks.phasing_generation import generated_phasing_recipe

    base = generated_phasing_recipe(
        actors, generation.model_copy(update={"disruptions": None})
    )
    definition = base["scenario"]["definition"]
    horizon = definition["horizon_sec"] + 10.0
    definition["horizon_sec"] = horizon
    goal = base["task"]["goal"]
    goal["deadline_sec"] = horizon
    correct = {
        name
        for step in base["reference_steps"]
        for name in step["action_ids"]
        if name != "wait"
    }
    timing = {
        r["actor"]: r["reference_start_sec"]
        for r in base["generation"]["private_timing"]
    }
    for actor, orbit in definition["orbits"].items():
        start = timing[actor]
        action = next(
            a
            for a in base["task"]["actions"]
            if a["id"] in correct and a["role"] == actor
        )
        command = action["commands"][0]
        burn = command["burn"]
        duration = command["duration_sec"]
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
            firing=duration,
            end=horizon - start,
        )
        mass = orbit["propulsion"]["dry_mass_kg"] + orbit["propulsion"]["propellant_kg"]
        condition = next(c for c in goal["conditions"] if c["actor"] == actor)
        condition.update(
            at_sec=horizon,
            position_m=target[:3],
            velocity_m_s=target[3:],
            position_tolerance_m=1.2 * 5 * burn["thrust_n"] * duration / mass,
            velocity_tolerance_m_s=0.001,
        )
        for a in base["task"]["actions"]:
            if a["role"] == actor:
                intervals = {tuple(w) for w in base["launch_windows"][a["id"]]}
                intervals.add((start + 5, start + 10))
                base["launch_windows"][a["id"]] = [
                    list(pair) for pair in sorted(intervals)
                ]
    # Both variants have identical public goals/windows, including a reserve slot.
    actor = max(timing, key=lambda a: (timing[a], a))
    at = timing[actor]
    end = at + 5 if generation.disruptions == "recoverable" else horizon
    base["dispatch_events"] = [
        {
            "at_sec": at,
            "kind": "close_windows",
            "actor": actor,
            "start_sec": at,
            "end_sec": end,
        }
    ]
    reference = []
    for index, step in enumerate(base["reference_steps"]):
        ids = [name for name in step["action_ids"] if name != "wait"]
        ids = [
            name
            for name in ids
            if not any(
                a["id"] == name and a["role"] == actor for a in base["task"]["actions"]
            )
        ]
        if generation.disruptions == "recoverable" and index * 5 == at + 5:
            ids.extend(
                a["id"]
                for a in base["task"]["actions"]
                if a["role"] == actor and a["id"] in correct
            )
        reference.append({"action_ids": ids or ["wait"], "advance_sec": 5.0})
    reference.extend([{"action_ids": ["wait"], "advance_sec": 5.0}] * 2)
    base["reference_steps"] = reference
    base["scenario"] = dynamic_reference_scenario(
        definition, id="dynamic-local-phasing", lineage="dynamic-local-phasing/v1"
    )
    base["task"]["id"] = "dynamic-local-phasing"
    base["task"] = TaskSpec.model_validate(base["task"]).model_dump(
        mode="json", by_alias=True
    )
    base["generation"].update(
        schema="space-env.disrupted-phasing-generation/1",
        horizon_sec=horizon,
        disruptions=generation.disruptions,
    )
    base["generation"]["physics_sha256"] = digest(
        {
            "definition": definition,
            "goal": goal,
            "commands": sorted(
                [
                    {k: v for k, v in a["commands"][0].items() if k != "command_id"}
                    for a in base["task"]["actions"]
                ],
                key=digest,
            ),
            "windows_by_actor": {
                a: base["launch_windows"][
                    next(
                        row["id"] for row in base["task"]["actions"] if row["role"] == a
                    )
                ]
                for a in definition["orbits"]
            },
            "dispatch_events": base["dispatch_events"],
            "dispatch_capacity": 1,
            "one_burn_per_actor": True,
        }
    )
    base["scope"] = (
        "50-100 second central-gravity flexible-goal phasing with a released dispatch closure. Both variants publish identical broadened goal tolerances from the start; critical closure retains partial completion and does not retarget the mission."
    )
    return base
