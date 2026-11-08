"""Frozen physical control tasks over one native goal/reward contract.

Public synthetic regression inputs. No feasibility filtering, real model calls,
or training. Orbital targets use a separately implemented fixed-step RK4 oracle.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from importlib.resources import files
from pathlib import Path

import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, seal, sources, verify_record
from space_env.llm.operations_suite import compile_suite as operations_suite
from space_env.reference.spacecraft_bus import rotate_body
from space_env.runtime import dynamic_reference_scenario
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.schema import TaskSpec
from space_env.tasks.session import TaskSession

PHYSICS_SHA = "5927c4615867646f00fa90ae237847ee6fcf755b94114b9962e03ec82321cb84"


def _inputs():
    value = json.loads(
        files("space_env").joinpath("bundled_data/modeling_physics_v1.json").read_text()
    )
    if (
        value["sha256"] != PHYSICS_SHA
        or digest({k: v for k, v in value.items() if k != "sha256"}) != PHYSICS_SHA
    ):
        raise ValueError("Frozen modeling physics inputs mismatch")
    return {c["id"]: c["scenario"] for c in value["cases"]}


def _rotation(q, angle):
    w, x, y, z = q
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return [w * c - x * s, x * c + w * s, y * c + z * s, z * c - y * s]


def _orbit_target(orbit, direction, *, thrust=0.5, firing=2.0, end=5.0):
    """Central gravity + constant inertial force / analytic changing mass.

    Separate RK4 implementation, h=0.005s, ending exactly on firing cutoff.
    Shares physical constants/initial values, not native derivative or solver.
    """
    y = np.array([*orbit["initial"]["r_m"], *orbit["initial"]["v_m_s"]], dtype=float)
    p = orbit["propulsion"]
    m0 = p["dry_mass_kg"] + p["propellant_kg"]
    flow = thrust / (p["isp_s"] * 9.80665)
    t = 0.0
    for finish, force in ((firing, thrust), (end, 0.0)):
        if finish <= t:
            continue

        def derivative(clock, state, force=force):
            r = state[:3]
            a = -MU_EARTH * r / math.hypot(*r) ** 3 + force * np.array(direction) / (
                m0 - flow * min(clock, firing)
            )
            return np.concatenate((state[3:], a))

        n = round((finish - t) / 0.005)
        h = (finish - t) / n
        for _ in range(n):
            k1 = derivative(t, y)
            k2 = derivative(t + h / 2, y + h * k1 / 2)
            k3 = derivative(t + h / 2, y + h * k2 / 2)
            k4 = derivative(t + h, y + h * k3)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
        t = finish
    return y.tolist()


def compile_suite():
    inputs = _inputs()
    operations = {c["id"]: c for c in operations_suite()["cases"]}
    cases = []

    def add(name, scenario, condition, command, deadline, *, limits=None, oracle=None):
        task = {
            "id": name,
            "roles": {a: {"actors": [a]} for a in scenario["actor_order"]},
            "actions": [
                {"id": "control", "role": "a", "kind": "submit", "commands": [command]}
            ],
            "decision_sec": 1.0,
            "max_decision_sec": 5.0,
            "max_joint_actions": len(scenario["actor_order"]),
            "max_decisions": 64,
            "goal": {
                "conditions": [condition],
                "deadline_sec": deadline,
                "limits": limits or {},
            },
        }
        task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
        cases.append(
            {
                "id": name,
                "family": condition["kind"],
                "scenario": scenario,
                "task": task,
                "task_text": "Execute the declared physical mission goal within its time, energy and fuel limits. Only native state validates success.",
                "oracle": oracle,
                "reference_steps": [
                    {
                        "action_ids": ["control"] if i == 0 else ["wait"],
                        "advance_sec": 1.0,
                    }
                    for i in range(round(deadline))
                ],
            }
        )

    for mode in ("nominal", "wrong-direction", "rigid", "rigid-loose"):
        scenario = copy.deepcopy(operations["lanes-wide"]["scenario"])
        q = scenario["definition"]["spacecraft_bus"]["a"]["attitude"][
            "initial_quaternion_wxyz"
        ]
        target = _rotation(q, math.radians(8.0))
        if mode in {"rigid", "rigid-loose"}:
            rigid = copy.deepcopy(
                operations["unload-nominal"]["scenario"]["definition"][
                    "spacecraft_bus"
                ]["a"]["attitude"]["rigid"]
            )
            rigid["initial_wheel_momentum_nms"] = [0.0, 0.0, 0.0]
            if mode == "rigid":
                rigid["rate_tolerance_rad_s"] = math.radians(0.005)
                scenario["definition"]["spacecraft_bus"]["a"]["attitude"][
                    "rigid_pointing_tolerance_deg"
                ] = 0.05
            scenario["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"] = rigid
        command = {
            "command_id": "point",
            "actor_id": "a",
            "kind": "slew",
            "duration_sec": 18.0
            if mode == "rigid"
            else 15.0
            if mode == "rigid-loose"
            else 5.0,
            "slew": {
                "frame": "GCRS",
                "quaternion_wxyz": target
                if mode != "wrong-direction"
                else _rotation(q, math.radians(-8.0)),
            },
        }
        add(
            "attitude-" + mode,
            scenario,
            {
                "id": "point",
                "kind": "attitude",
                "actor": "a",
                "quaternion_wxyz": target,
                "tolerance_deg": 0.1,
                "max_rate_deg_s": 0.01,
            },
            command,
            20.0 if mode == "rigid" else 18.0 if mode == "rigid-loose" else 10.0,
            limits={"a": {"max_energy_j": 1000.0, "require_healthy": True}},
        )
    for mode in ("nominal", "limited"):
        base = operations["unload-" + mode]
        command = copy.deepcopy(base["task"]["actions"][0]["commands"][0])
        add(
            "momentum-" + mode,
            copy.deepcopy(base["scenario"]),
            {"id": "unload", "kind": "momentum", "actor": "a", "tolerance_nms": 1e-5},
            command,
            3.0,
            limits={"a": {"max_energy_j": 200.0, "require_healthy": True}},
        )
    for kind in ("chemical", "electric", "coupled"):
        scenario = copy.deepcopy(inputs["chemical" if kind == "coupled" else kind])
        thrust = 1.0 if kind == "chemical" else 0.01
        if kind == "coupled":
            rigid = copy.deepcopy(
                operations["unload-nominal"]["scenario"]["definition"][
                    "spacecraft_bus"
                ]["a"]["attitude"]["rigid"]
            )
            rigid["initial_wheel_momentum_nms"] = [0.0, 0.0, 0.0]
            scenario["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"] = rigid
            scenario["definition"]["actuation"]["engines"]["a"]["rigid_coupling"] = {
                "lever_arm_body_m": [0.2, 0.0, 0.0],
                "max_mass_loss_fraction": 0.001,
            }
        direction = [math.sin(math.radians(10.0)), 0.0, math.cos(math.radians(10.0))]
        q = scenario["definition"]["spacecraft_bus"]["a"]["attitude"][
            "initial_quaternion_wxyz"
        ]
        target = (rotate_body(q, direction) * thrust).tolist()
        command = {
            "command_id": "engine",
            "actor_id": "a",
            "kind": "body_burn",
            "duration_sec": 5.0,
            "body_burn": {
                "frame": "BODY",
                "direction_body": direction,
                "thrust_n": thrust,
                "firing_sec": 1.0,
            },
        }
        add(
            "impulse-" + kind,
            scenario,
            {
                "id": "impulse",
                "kind": "body_impulse",
                "actor": "a",
                "target_ns": target,
                "tolerance_ns": 1e-4 if kind == "coupled" else 1e-8,
            },
            command,
            8.0,
            limits={"a": {"max_energy_j": 10000.0, "require_healthy": True}},
        )
    for mode in ("nominal", "wrong-direction"):
        definition = copy.deepcopy(inputs["chemical"]["definition"])
        for key in (
            "spacecraft_bus",
            "execution",
            "actuation",
            "unloading",
            "lifecycle",
            "acquisition_targets",
            "acquisition_sensors",
        ):
            definition.pop(key, None)
        definition["operations"]["products"] = []
        for vehicle in definition["operations"]["vehicles"].values():
            vehicle["solar_mode"] = "ideal_sun_tracking"
        for orbit in definition["orbits"].values():
            orbit["forces"] = {}
        scenario = dynamic_reference_scenario(
            definition, id="orbit-goal-" + mode, lineage="modeling-orbit-regression"
        )
        orbit = definition["orbits"]["a"]
        speed = math.hypot(*orbit["initial"]["v_m_s"])
        direction = [v / speed for v in orbit["initial"]["v_m_s"]]
        target = _orbit_target(orbit, direction)
        command = {
            "command_id": "burn",
            "actor_id": "a",
            "kind": "finite_burn",
            "duration_sec": 2.0,
            "burn": {
                "frame": "GCRS",
                "direction_gcrs": direction
                if mode == "nominal"
                else [-v for v in direction],
                "thrust_n": 0.5,
            },
        }
        add(
            "orbit-" + mode,
            scenario,
            {
                "id": "orbit",
                "kind": "orbit_state",
                "actor": "a",
                "at_sec": 5.0,
                "position_m": target[:3],
                "velocity_m_s": target[3:],
                "position_tolerance_m": 1e-3,
                "velocity_tolerance_m_s": 1e-6,
            },
            command,
            5.0,
            limits={
                "a": {"min_propellant_kg": orbit["propulsion"]["propellant_kg"] - 0.001}
            },
            oracle="Independent fixed-step RK4, h=0.005s, central gravity + constant inertial F/m(t); shared SI constants, no other perturbations",
        )
        if mode == "nominal":
            staged = copy.deepcopy(cases[-1])
            staged["id"] = "orbit-waypoints"
            staged["task"]["id"] = "orbit-waypoints"
            early = _orbit_target(orbit, direction, end=2.0)
            first = copy.deepcopy(staged["task"]["goal"]["conditions"][0])
            first.update(
                id="burn-cutoff",
                at_sec=2.0,
                position_m=early[:3],
                velocity_m_s=early[3:],
            )
            staged["task"]["goal"]["conditions"].insert(0, first)
            cases.append(staged)
    base = copy.deepcopy(operations["lanes-wide"])
    task = base["task"]
    ids = [r["id"] for r in base["scenario"]["definition"]["lifecycle"]["requests"]]
    task["goal"] = {
        "conditions": [{"id": "delivery", "kind": "delivery", "request_ids": ids}],
        "deadline_sec": base["scenario"]["definition"]["horizon_sec"],
    }
    cases.append(
        {
            "id": "delivery",
            "family": "delivery",
            "scenario": base["scenario"],
            "task": TaskSpec.model_validate(task).model_dump(
                mode="json", by_alias=True
            ),
            "task_text": "Deliver the declared native requests before the goal deadline.",
            "reference_steps": base["steps"],
            "oracle": None,
        }
    )
    return seal(
        {
            "schema": "space-env.modeling-tasks/1",
            "cases": cases,
            "scope": "thirteen public synthetic regression cases, five goal types, rigid pointing with matched/unmatched controller precision, rotating off-center thrust and staged orbit waypoints; scripted control, no model inference, blind/OOD or hardware qualification",
        }
    )


def run_suite(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    suite = compile_suite()
    before = sources()
    atomic_write(root / "suite.json", canonical_json(suite))
    rows = []
    for case in suite["cases"]:
        for policy in ("wait", "reference"):
            row = {"case": case["id"], "family": case["family"], "policy": policy}
            try:
                session = TaskSession(case["scenario"], case["task"])
                ep = LLMEpisode(
                    session,
                    task_text=case["task_text"],
                    group=case["id"],
                    split="test",
                    max_calls=128,
                    policy_origin="scripted_public"
                    if policy == "wait"
                    else "scripted_privileged",
                )
                chooser = (
                    public_policy("wait", session._task.decision_sec)
                    if policy == "wait"
                    else reference_policy(
                        case["reference_steps"],
                        {a.id: a.role for a in session._task.actions},
                    )
                )
                record = run_agent(ep, chooser)
                verify_record(record)
                name = case["id"] + "--" + policy + ".json"
                atomic_write(root / name, canonical_json(record))
                row.update(
                    status="completed",
                    artifact=name,
                    episode_sha256=record["sha256"],
                    metrics=record["metrics"],
                )
            except Exception as exc:  # noqa: BLE001 -- retain every attempted benchmark cell
                row.update(
                    status="error",
                    metrics=None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            rows.append(row)
            atomic_write(root / "progress.json", canonical_json(seal({"rows": rows})))
    if sources() != before:
        raise ValueError("Model sources changed during study")
    report = seal(
        {
            "schema": "space-env.modeling-tasks-report/1",
            "suite_sha256": suite["sha256"],
            "model_sources": before,
            "rows": rows,
            "external_llm_measured": False,
        }
    )
    atomic_write(root / "report.json", canonical_json(report))
    return report


def audit(directory):
    root = Path(directory)
    suite = json.loads((root / "suite.json").read_text())
    report = json.loads((root / "report.json").read_text())
    if (
        suite != compile_suite()
        or report != seal({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
        or report["suite_sha256"] != suite["sha256"]
    ):
        raise ValueError("Modeling suite/source/seal mismatch")
    expected = [(c["id"], p) for c in suite["cases"] for p in ("wait", "reference")]
    if [(r["case"], r["policy"]) for r in report["rows"]] != expected:
        raise ValueError("Incomplete modeling matrix")
    for case, row in zip([c for c in suite["cases"] for _ in range(2)], report["rows"]):
        if row["family"] != case["family"] or row["status"] not in {
            "completed",
            "error",
        }:
            raise ValueError("Case identity mismatch")
        if row["status"] == "error":
            if row["metrics"] is not None:
                raise ValueError("Error cannot claim metrics")
            continue
        name = case["id"] + "--" + row["policy"] + ".json"
        if row["artifact"] != name:
            raise ValueError("Unexpected artifact path")
        record = json.loads((root / name).read_text())
        verify_record(record)
        initial = TaskSession(case["scenario"], case["task"])
        expected_cfg = {
            "task_text": case["task_text"],
            "group": case["id"],
            "split": "test",
            "max_calls": 128,
            "max_dialogue_bytes": 16000000,
            "policy_origin": "scripted_public"
            if row["policy"] == "wait"
            else "scripted_privileged",
        }
        if (
            record["initial_checkpoint"] != initial.checkpoint(compact=True)
            or record["configuration"] != expected_cfg
            or record["metrics"] != row["metrics"]
            or record["sha256"] != row["episode_sha256"]
        ):
            raise ValueError("Native initial task/metrics mismatch")
        chooser = (
            public_policy("wait", initial._task.decision_sec)
            if row["policy"] == "wait"
            else reference_policy(
                case["reference_steps"], {a.id: a.role for a in initial._task.actions}
            )
        )
        if run_agent(LLMEpisode(initial, **expected_cfg), chooser) != record:
            raise ValueError("Declared policy trace mismatch")
    return {
        "valid": True,
        "rows": len(report["rows"]),
        "native_replayed": sum(r["status"] == "completed" for r in report["rows"]),
        "runtime_errors": sum(r["status"] == "error" for r in report["rows"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--audit", type=Path)
    args = p.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
    elif args.output:
        r = run_suite(args.output)
        print(
            canonical_json(
                {
                    "sha256": r["sha256"],
                    "rows": [
                        {k: v for k, v in row.items() if k not in {"metrics"}}
                        | {"success": (row["metrics"] or {}).get("success")}
                        for row in r["rows"]
                    ],
                }
            )
        )
    else:
        p.error("Use --output or --audit")


if __name__ == "__main__":
    main()
