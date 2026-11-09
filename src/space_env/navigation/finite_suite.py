"""Fixed finite-thrust/force-profile native missions and explicit preflight rejections."""

import argparse
import copy
import json
from pathlib import Path

from space_env.llm.episode import _training_row, export_training, seal, sources
from space_env.navigation.controlled_suite import execute
from space_env.navigation.task_suite import compile_suite as delivery_cases
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks import TaskSession, TaskSpec


def compile_suite():
    original = delivery_cases()[0]
    cases = []
    for name in [
        "finite-central",
        "finite-j2",
        "finite-all-forces",
        "cancelled-burn",
        "rejected-thrust",
        "insufficient-fuel",
    ]:
        c = copy.deepcopy(original)
        c["id"] = name
        c["task"]["id"] = name
        for spec in c["task"]["navigation_sensors"].values():
            spec.update(continuous_power_w=3.0, latency_sec=0.5)
        c["task"]["navigation_sensors"]["a"].update(
            force_profile="matched_native", control_profile="inertial_finite_burn"
        )
        if name in ["finite-j2", "finite-all-forces"]:
            c["scenario"]["definition"]["orbits"]["a"]["forces"] = {"j2": True}
        if name == "finite-all-forces":
            c["scenario"]["definition"]["orbits"]["a"]["forces"] = {
                "j2": True,
                "drag": True,
                "srp": True,
                "third_body": True,
            }
            c["scenario"]["definition"]["weather"] = {
                "source": "synthetic constant weather parameters, not measured",
                "status": "synthetic",
                "valid_from_utc": "2026-07-18T11:00:00Z",
                "valid_until_utc": "2026-07-18T13:00:00Z",
            }
        command = {
            "command_id": "burn",
            "actor_id": "a",
            "kind": "finite_burn",
            "duration_sec": 1.0,
            "burn": {
                "frame": "GCRS",
                "direction_gcrs": [0.0, 0.0, 1.0],
                "thrust_n": 0.5,
            },
        }
        if name == "rejected-thrust":
            command["burn"]["thrust_n"] = 1e6
        if name == "insufficient-fuel":
            c["scenario"]["definition"]["orbits"]["a"]["propulsion"][
                "propellant_kg"
            ] = 1e-6
        if name == "cancelled-burn":
            command["duration_sec"] = 4.0
        c["task"]["actions"].insert(
            0, {"id": "burn", "role": "a", "kind": "submit", "commands": [command]}
        )
        c["task"]["actions"].append(
            {"id": "stop", "role": "a", "kind": "cancel_actor", "actor": "a"}
        )
        c["steps"] = [
            {"action_ids": ["burn"], "advance_sec": 1.0},
            {"action_ids": ["deliver"], "advance_sec": 2.0},
        ] + [{"action_ids": ["wait"], "advance_sec": 1.0} for _ in range(5)]
        if name == "cancelled-burn":
            c["steps"] = [
                {"action_ids": ["burn"], "advance_sec": 0.5},
                {"action_ids": ["stop"], "advance_sec": 0.1},
                {"action_ids": ["deliver"], "advance_sec": 2.0},
            ] + [{"action_ids": ["wait"], "advance_sec": 1.0} for _ in range(5)]
        c["task"] = TaskSpec.model_validate(c["task"]).model_dump(
            mode="json", by_alias=True
        )
        c["task_text"] = (
            "Deliver the native ready product using declared inertial finite thrust and sensor-estimated orbit. Matched force inputs and nominal mass are known; no true orbit is available in observations. Sensor electronics draw continuous native power."
        )
        cases.append(c)
    return cases


def rejections():
    c = next(c for c in compile_suite() if c["id"] == "finite-j2")
    rows = []
    for name in ["force-optin", "control-optin", "body-thrust", "mixed-epochs"]:
        x = copy.deepcopy(c)
        x["id"] = name
        if name == "force-optin":
            x["task"]["navigation_sensors"]["a"]["force_profile"] = "central"
        if name == "control-optin":
            x["task"]["navigation_sensors"]["a"]["control_profile"] = "impulse"
        if name == "body-thrust":
            x["task"]["actions"][0]["commands"][0] = {
                "command_id": "body",
                "actor_id": "a",
                "kind": "body_burn",
                "duration_sec": 1.0,
                "body_burn": {
                    "frame": "BODY",
                    "firing_sec": 1.0,
                    "thrust_n": 0.5,
                    "direction_body": [0.0, 0.0, 1.0],
                },
            }
        if name == "mixed-epochs":
            x["scenario"]["definition"]["orbits"]["b"]["initial"]["epoch_utc"] = (
                "2026-07-18T12:00:01Z"
            )
        try:
            TaskSession(x["scenario"], x["task"])
        except ValueError as exc:
            rows.append(
                {
                    "id": name,
                    "rejected": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            raise ValueError("unsupported profile accepted: " + name)
    return rows


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
            raise ValueError("finite navigation source/seal mismatch")
        if report["rejections"] != rejections():
            raise ValueError("finite navigation preflight mismatch")
        if [(r["case"], r["policy"]) for r in report["rows"]] != [
            (c["id"], p) for c in cases for p in ["wait", "reference"]
        ]:
            raise ValueError("finite navigation matrix mismatch")
        for c in cases:
            for p in ["wait", "reference"]:
                record, witness = execute(c, p)
                stem = c["id"] + "--" + p
                if json.loads((args.audit / (stem + ".json")).read_text()) != record:
                    raise ValueError("finite navigation replay mismatch")
                row = next(
                    r
                    for r in report["rows"]
                    if r["case"] == c["id"] and r["policy"] == p
                )
                if (
                    row["metrics"] != record["metrics"]
                    or row["episode_sha256"] != record["sha256"]
                    or row["witness"] != witness
                ):
                    raise ValueError("finite navigation witness/metrics mismatch")
                for fmt in ["rl", "sft"]:
                    path = args.audit / (stem + "-" + fmt + ".jsonl")
                    if fmt == "sft" and not record["metrics"]["sft_eligible"]:
                        if path.exists():
                            raise ValueError("ineligible finite navigation SFT")
                        continue
                    if json.loads(path.read_text()) != _training_row(
                        record, format=fmt
                    ):
                        raise ValueError("finite navigation training export mismatch")
        print(
            canonical_json(
                {"valid": True, "native_replayed": 12, "preflight_rejections": 4}
            )
        )
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
                "schema": "space-env.finite-navigation-study/1",
                "model_sources": sources(),
                "rows": rows,
                "rejections": rejections(),
                "scope": "fixed public synthetic cases, same orbit; matched shared force functions/inputs, not independent physics calibration; scripted train rollouts, no learned model or blind evaluation",
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
