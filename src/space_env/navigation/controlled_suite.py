"""Public synthetic controlled-sensor missions, with retained resource/control failures."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import (
    LLMEpisode,
    _training_row,
    export_training,
    seal,
    sources,
    verify_record,
)
from space_env.navigation.controls import own_impulses
from space_env.navigation.task_suite import compile_suite as sensor_cases
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks import TaskSession, TaskSpec


def compile_suite():
    original = sensor_cases()[0]
    cases = []
    for name in [
        "powered-impulse",
        "rejected-control",
        "no-contact",
        "power-exhaustion",
        "uncertain-control",
        "boundary-delay",
    ]:
        c = copy.deepcopy(original)
        c["id"] = name
        c["task"]["id"] = "controlled-sensor-" + name
        for spec in c["task"]["navigation_sensors"].values():
            spec.update(continuous_power_w=3.0, latency_sec=0.5)
        command = {
            "command_id": "kick",
            "actor_id": "a",
            "kind": "impulse",
            "duration_sec": 0.0,
            "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 0.0, 0.5]},
        }
        if name == "rejected-control":
            command["impulse"]["delta_v_m_s"] = [0.0, 0.0, 100000.0]
        c["task"]["actions"].insert(
            0, {"id": "kick", "role": "a", "kind": "submit", "commands": [command]}
        )
        c["steps"] = [
            {"action_ids": ["kick"], "advance_sec": 1.0},
            {"action_ids": ["deliver"], "advance_sec": 2.0},
        ] + [{"action_ids": ["wait"], "advance_sec": 1.0} for _ in range(5)]
        if name == "no-contact":
            c["scenario"]["definition"]["operations"]["stations"][0]["lat_deg"] = 70.0
        if name == "power-exhaustion":
            c["task"]["navigation_sensors"]["a"]["continuous_power_w"] = 36.0
            c["scenario"]["definition"]["operations"]["vehicles"]["a"].update(
                base_power_w=0.0, battery_wh=0.01, initial_soc=1.0
            )
        if name == "uncertain-control":
            c["task"]["navigation_sensors"]["a"]["impulse_velocity_cov_m2_s2"] = (
                np.eye(3) * 0.01
            ).tolist()
        if name == "boundary-delay":
            for spec in c["task"]["navigation_sensors"].values():
                spec.update(latency_sec=1.5, dropout_probability=0.3)
            c["steps"].insert(1, {"action_ids": ["kick"], "advance_sec": 1.0})
        c["task"] = TaskSpec.model_validate(c["task"]).model_dump(
            mode="json", by_alias=True
        )
        c["task_text"] = (
            "Deliver the ready native product to Ground. Known inertial impulse actions update the estimated orbit; sensor electronics draw native continuous power. Sensor coordinates are noisy estimates; masks are declarative."
        )
        cases.append(c)
    return cases


def execute(case, policy):
    session = TaskSession(case["scenario"], case["task"])
    ep = LLMEpisode(
        session,
        task_text=case["task_text"],
        group="controlled-sensor-" + case["id"],
        split="train",
        policy_origin="scripted_privileged"
        if policy == "reference"
        else "scripted_public",
    )
    chooser = (
        reference_policy(case["steps"], {a.id: a.role for a in session._task.actions})
        if policy == "reference"
        else public_policy("wait", 1.0)
    )
    record = run_agent(ep, chooser)
    verify_record(record)
    final = ep._session
    backend = final._runtime._backend
    report = backend.report()
    witness = {
        "scope": "trainer-only native ledgers and estimator state; never agent inputs",
        "time_sec": final._runtime.time_sec,
        "energy_ledger": report["energy_ledger"],
        "battery_wh": report["energy_wh"],
        "accepted_impulses": sum(
            r["command"]["kind"] == "impulse" and r["status"] == "completed"
            for r in final._runtime.command_status()
        ),
        "sensors": {},
    }
    for actor, f in final._navigation._filters.items():
        impulses = own_impulses(final._runtime, actor)
        nominal_mass = final._navigation._environments[actor].nominal_mass(
            final._runtime.time_sec,
            impulses,
            {r["command"]["command_id"] for r in impulses},
            f["burns"],
        )
        actual_mass = backend.mission.ephemerides[actor].mass(final._runtime.time_sec)
        witness["sensors"][actor] = {
            "nominal_mass_kg": nominal_mass,
            "actual_mass_kg": actual_mass,
            "mass_error_kg": nominal_mass - actual_mass,
            "finite_burn_history": copy.deepcopy(f["burns"]),
            "acquired": f["index"],
            "received": len(f["history"]),
            "pending": len(f["pending"]),
            "unavailable_power": sum(
                r.get("unavailable") == "power" for r in f["history"]
            ),
            "filter_controls": len(f["controls"]),
            "diagnostics": final._navigation.diagnostics(actor),
        }
    return record, witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("choose output or audit")
    cases = compile_suite()
    if args.audit:
        report = json.loads((args.audit / "report.json").read_text())
        if report.get("model_sources") != sources() or report != seal(
            {k: v for k, v in report.items() if k != "sha256"}
        ):
            raise ValueError("controlled-sensor source/seal mismatch")
        expected = [(c["id"], p) for c in cases for p in ["wait", "reference"]]
        if [(r["case"], r["policy"]) for r in report["rows"]] != expected:
            raise ValueError("controlled-sensor matrix mismatch")
        for c in cases:
            for p in ["wait", "reference"]:
                record, witness = execute(c, p)
                stem = c["id"] + "--" + p
                if json.loads((args.audit / (stem + ".json")).read_text()) != record:
                    raise ValueError("controlled-sensor trace mismatch")
                for fmt in ("rl", "sft"):
                    export_path = args.audit / (stem + "-" + fmt + ".jsonl")
                    if fmt == "sft" and not record["metrics"]["sft_eligible"]:
                        if export_path.exists():
                            raise ValueError("ineligible controlled-sensor SFT export")
                        continue
                    if json.loads(export_path.read_text()) != _training_row(
                        record, format=fmt
                    ):
                        raise ValueError("controlled-sensor training export mismatch")
                row = next(
                    r
                    for r in report["rows"]
                    if r["case"] == c["id"] and r["policy"] == p
                )
                if (
                    row["episode_sha256"] != record["sha256"]
                    or row["metrics"] != record["metrics"]
                    or row["witness"] != witness
                ):
                    raise ValueError("controlled-sensor metrics/witness mismatch")
        print(canonical_json({"valid": True, "native_replayed": 12}))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for c in cases:
            for p in ["wait", "reference"]:
                record, witness = execute(c, p)
                stem = c["id"] + "--" + p
                atomic_write(args.output / (stem + ".json"), canonical_json(record))
                export_training(record, args.output / (stem + "-rl.jsonl"))
                if record["metrics"]["sft_eligible"]:
                    export_training(
                        record, args.output / (stem + "-sft.jsonl"), format="sft"
                    )
                rows.append(
                    {
                        "case": c["id"],
                        "policy": p,
                        "episode_sha256": record["sha256"],
                        "metrics": record["metrics"],
                        "witness": witness,
                    }
                )
        report = seal(
            {
                "schema": "space-env.controlled-sensor-study/1",
                "model_sources": sources(),
                "rows": rows,
                "scope": "12 public scripted train rollouts; fixed orbital geometry; no learned model, external comparator, blind holdout or hardware calibration",
            }
        )
        atomic_write(args.output / "report.json", canonical_json(report))
        print(
            canonical_json(
                {
                    "sha256": report["sha256"],
                    "successes": sum(r["metrics"]["success"] for r in rows),
                    "sft_eligible": sum(r["metrics"]["sft_eligible"] for r in rows),
                }
            )
        )


if __name__ == "__main__":
    main()
