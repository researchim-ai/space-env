"""Continuous two-wave spacecraft endurance; synthetic constructive geometry."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from examples.modeling.adaptive_campaign_study import accounting
from examples.modeling.audit_reactive_campaign import verify_policy_trace
from examples.modeling.reactive_campaign_study import run_reactive
from space_env.industrial.agent import export_training
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.long_campaign import long_campaign


def endpoint(record):
    view = record["events"][-1]["response"]["observation"]["native"]
    return {
        r["id"]: [
            r["values"][k]
            for k in ["r_x_m", "r_y_m", "r_z_m", "v_x_m_s", "v_y_m_s", "v_z_m_s"]
        ]
        for r in view["entities"]
        if r["kind"] == "actor"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    records = {}
    settings = [
        ("two-hours", {}),
        ("orbit-fine-two-hours", {"sample_sec": 1.0, "max_step_sec": 5.0}),
        ("six-hours", {"horizon_sec": 21600.0}),
        ("low-battery", {"horizon_sec": 600.0, "release_sec": 60.0, "battery_wh": 1.0}),
    ]
    for name, params in settings:
        start = time.perf_counter()
        case = long_campaign(params)
        case["public_brief"] = (
            f"Operate both spacecraft until t={case['setup']['horizon_sec']} seconds. "
            "Deliver the three initial requests by t=20 and newly announced demand by its declared deadline. "
            "Preserve successful delivery records through the end of the mission, without resets. "
            "No spacecraft faults or safe modes are allowed in the late/endurance phases. "
            f"Consumed energy per actor must remain below {case['setup']['horizon_sec'] * 16 + 2000} J from mission origin. "
            "Future arrival and station-event times are not provided."
        )
        build_sec = time.perf_counter() - start
        directory = args.output / name
        directory.mkdir()
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        start = time.perf_counter()
        record = run_reactive(
            case["configuration"],
            public_contract=case["public_contract"],
            max_calls=512,
            task_text=case["public_brief"],
        )
        execution_sec = time.perf_counter() - start
        trace = verify_policy_trace(record, case["public_contract"])
        atomic_write(directory / "episode.json", canonical_json(record))
        replay_export = export_training(record, directory / "rl.jsonl")
        if (
            name == "two-hours"
            and record["metrics"]["success"]
            and all(
                e["response"].get("feedback", {}).get("accepted")
                for e in record["events"]
            )
        ):
            export_training(record, directory / "sft.jsonl", format="sft")
        row = {
            "id": name,
            "setup": case["setup"],
            "metrics": record["metrics"],
            "calls": len(record["events"]),
            "build_wall_sec": build_sec,
            "execution_wall_sec": execution_sec,
            "replay_verification": {
                "performed_by": "export_training",
                "rl_export": replay_export,
            },
            "policy_trace": trace,
            "accounting": accounting(record),
            "sha256": record["sha256"],
        }
        rows.append(row)
        records[name] = record
        print(name, row["calls"], row["metrics"]["success"], execution_sec, flush=True)
    left, right = (
        endpoint(records["two-hours"]),
        endpoint(records["orbit-fine-two-hours"]),
    )
    differences = {
        a: {
            "position_m": math.dist(left[a][:3], right[a][:3]),
            "velocity_m_s": math.dist(left[a][3:], right[a][3:]),
        }
        for a in left
    }
    coarse = records["two-hours"]["initial"]["configuration"]
    fine = records["orbit-fine-two-hours"]["initial"]["configuration"]
    import copy

    normalized = copy.deepcopy(coarse)
    for orbit in normalized["scenario"]["definition"]["orbits"].values():
        orbit["integration"].update(sample_sec=1.0, max_step_sec=5.0)
    assert normalized == fine, (
        "convergence pair changed more than declared orbit settings"
    )
    report = {
        "scope": "Constructive synthetic two-wave EO/CPU/ground mission; J2 and RC thermal, battery-only, two active actors; no ISL, multi-agent autonomy, hardware calibration or independent orbital reference.",
        "rows": rows,
        "orbit_endpoint_convergence": differences,
        "timing_scope": "Local wall clock includes construction and record generation as separately listed; uncontrolled host load; replay/export time excluded from execution_wall_sec.",
    }
    atomic_write(args.output / "report.json", canonical_json(report))
    print(canonical_json({"orbit_endpoint_convergence": differences}))


if __name__ == "__main__":
    main()
