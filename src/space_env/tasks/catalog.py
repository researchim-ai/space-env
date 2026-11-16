"""Discover, export and execute native mission recipes without changing their semantics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def mission_catalog():
    """Return detached JSON metadata; recipe construction is deferred until requested."""
    recipes = _recipes()
    rows = []
    for key, case in recipes.items():
        task = case["task"]
        steps = case.get("reference_steps", case.get("steps", []))
        actions = {a["id"]: a for a in task.get("actions", [])}
        phases = []
        elapsed = 0.0
        for index, step in enumerate(steps):
            duration = float(step["advance_sec"])
            phases.append(
                {
                    "index": index,
                    "start_sec": elapsed,
                    "end_sec": elapsed + duration,
                    "action_ids": list(step["action_ids"]),
                    "operations": sorted(
                        {
                            command["kind"]
                            for name in step["action_ids"]
                            if name != "wait"
                            for command in actions[name].get("commands", [])
                        }
                    ),
                }
            )
            elapsed += duration
        rows.append(
            {
                "id": key,
                "family": case.get("family", "estimated_navigation_delivery"),
                "actors": list(case["scenario"]["actor_order"]),
                "roles": copy.deepcopy(task["roles"]),
                "goal": copy.deepcopy(task.get("goal")),
                "navigation_sensors": sorted(task.get("navigation_sensors", {})),
                "requirements": copy.deepcopy(task.get("model_requirements", [])),
                "schedule": phases,
                "recipe_sha256": digest(case),
                "observability": "estimated_navigation"
                if task.get("navigation_sensors")
                else "native_task_view",
            }
        )
    return {
        "schema": "space-env.mission-catalog/1",
        "missions": rows,
        "scope": "Public synthetic regression recipes, including negative cases. Schedules are reference-policy operations, not autonomous phase transitions or feasibility certificates.",
        "unsupported_compositions": [
            "sensor-aware physical goal residuals",
            "continuous collision avoidance certification",
            "contact docking and refuelling",
            "full reentry and atmospheric flight",
            "arbitrary cross-recipe concatenation",
        ],
    }


def _recipes():
    # Imports remain local so discovery never creates runtimes or executes policies.
    from space_env.navigation.finite_suite import compile_suite as finite
    from space_env.navigation.task_suite import compile_suite as sensors
    from space_env.tasks.modeling_suite import compile_suite as physical

    groups = {
        "physical": physical()["cases"],
        "navigation": sensors(),
        "finite": finite(),
    }
    result = {}
    for group, cases in groups.items():
        for case in cases:
            key = group + "/" + case["id"]
            if key in result:
                raise ValueError("Duplicate mission id: " + key)
            result[key] = case
    return result


def mission_recipe(mission_id):
    """Export exact original task/scenario plus labelled privileged reference schedule."""
    recipes = _recipes()
    if mission_id not in recipes:
        raise ValueError("Unknown mission id: " + str(mission_id))
    return copy.deepcopy(recipes[mission_id])


def run_mission(mission_id, *, policy="wait"):
    """Use the existing actor-only LLM tool contract; no direct runtime shortcuts."""
    if policy not in ("wait", "reference"):
        raise ValueError("policy must be wait or reference")
    from space_env.llm.benchmark import public_policy, reference_policy, run_agent
    from space_env.llm.episode import LLMEpisode, verify_record
    from space_env.tasks import TaskSession

    case = mission_recipe(mission_id)
    session = TaskSession(case["scenario"], case["task"])
    episode = LLMEpisode(
        session,
        task_text=case.get("task_text", "Execute the declared native mission goal."),
        group=mission_id,
        split="test",
        max_calls=128,
        policy_origin="scripted_privileged"
        if policy == "reference"
        else "scripted_public",
    )
    chooser = public_policy("wait", session._task.decision_sec)
    if policy == "reference":
        chooser = reference_policy(
            case.get("reference_steps", case.get("steps", [])),
            {a.id: a.role for a in session._task.actions},
        )
    record = run_agent(episode, chooser)
    verify_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission")
    parser.add_argument("--run", choices=["wait", "reference"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run and not args.mission:
        parser.error("--run requires --mission")
    value = (
        run_mission(args.mission, policy=args.run)
        if args.run
        else (mission_recipe(args.mission) if args.mission else mission_catalog())
    )
    if args.output.exists():
        parser.error("output already exists; choose a new artifact path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, canonical_json(value))
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
