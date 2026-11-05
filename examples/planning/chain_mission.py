"""Independent fixed geography, paid kinematic pointing and no preloaded data.

Write spec before running. Synthetic hardware and metadata only, no calibration.
"""

import argparse
from pathlib import Path

from space_env.planning.chain import (
    propose_chain,
    replan_chain,
    verify_chain_proposal,
    verify_chain_replan,
)
from space_env.planning.mission_bundle import publish_plan_bundle
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.scaling.coupled import coupled_workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def configuration(mode="nominal"):
    cfg = {
        "planes": [
            {"id": "a", "inclination_deg": 55.0, "raan_deg": 0.0, "phases_deg": [0.0]},
            {
                "id": "b",
                "inclination_deg": 75.0,
                "raan_deg": 120.0,
                "phases_deg": [0.0],
            },
        ],
        "targets": [
            {"id": f"t{i:02d}", "lat_deg": 0.0, "lon_deg": float(lon)}
            for i, lon in enumerate(range(-180, 180, 30))
        ],
        "stations": [
            {"name": f"g{i:02d}", "lat_deg": 0.0, "lon_deg": float(lon)}
            for i, lon in enumerate(range(-180, 180, 30))
        ],
        "id": "forecast-chain-" + mode,
        "horizon_sec": 600.0,
        "decision_sec": 5.0,
    }
    if mode == "shifted":
        cfg["planes"][0].update(inclination_deg=48.0, raan_deg=-73.0, phases_deg=[47.0])
        cfg["planes"][1].update(inclination_deg=82.0, raan_deg=11.0, phases_deg=[199.0])
    elif mode == "outage":
        cfg["mode"] = "all_outage"
    elif mode != "nominal":
        raise ValueError("unknown mode")
    w = coupled_workload(cfg)
    for b in w["scenario"]["definition"]["spacecraft_bus"].values():
        b["attitude"].pop("rigid", None)
    w["coupled_gates"]["native_spec_sha256"] = digest(w["scenario"]["definition"])
    return {
        "scenario": w["scenario"],
        "task": w["task"],
        "goals": {
            "min_fulfilled": 1,
            "require_no_failed_operations": True,
            "coupled": w["coupled_gates"],
        },
    }


def run(value, output, *, replan=False):
    output = Path(output)
    if output.exists():
        raise ValueError("output must not exist")
    output.mkdir(parents=True)
    atomic_write(output / "input.json", canonical_json(value))
    runtime = MissionRuntime(value["scenario"], access=AccessPolicy(forecast=True))
    proposal = propose_chain(
        runtime, value["scenario"], value["task"], goals=value["goals"]
    )
    atomic_write(output / "proposal.json", canonical_json(proposal))
    atomic_write(
        output / "forecast_audit.json",
        canonical_json(verify_chain_proposal(proposal, native=True)),
    )
    execution = PlanExecution(proposal["plan"])
    if replan:
        ends = [a["finish_sec"] for a in proposal["schedule"]["allocations"]]
        if not ends:
            raise ValueError("replan example requires at least one selected chain")
        # First globally idle boundary, chosen from proposal, never outcome search.
        boundary = max(ends)
        count = round(boundary / value["task"]["decision_sec"])
        if count >= len(proposal["plan"]["steps"]):
            raise ValueError("no future idle suffix")
        execution.run(decisions=count)
        record = execution.save(output / "prefix.json")
        updated = replan_chain(proposal, record)
        atomic_write(output / "replan.json", canonical_json(updated))
        atomic_write(
            output / "replan_audit.json", canonical_json(verify_chain_replan(updated))
        )
        execution = PlanExecution(updated["plan"])
        execution.run(decisions=count)
        proposal = updated
    execution.run()
    record = execution.save(output / "execution.json")
    atomic_write(output / "plan.json", canonical_json(proposal["plan"]))
    bundle = publish_plan_bundle(proposal["plan"], record, output / "dataset")
    result = {
        "predicted_fulfilled": proposal["schedule"]["predicted_fulfilled"],
        "total": proposal["schedule"]["total"],
        "mission_goal_passed": bundle["mission_goal_passed"],
        "sft_exported": bundle["sft_exported"],
        "expert_teacher_qualified": False,
    }
    atomic_write(output / "result.json", canonical_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode", choices=["nominal", "shifted", "outage"], default="nominal"
    )
    parser.add_argument("--write-spec", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--replan", action="store_true")
    args = parser.parse_args()
    if args.spec:
        from space_env.planning.mission_bundle import read_json

        value = read_json(args.spec)
    else:
        value = configuration(args.mode)
    if args.output.exists():
        raise ValueError("output must not exist")
    if args.write_spec:
        atomic_write(args.output, canonical_json(value))
    else:
        print(canonical_json(run(value, args.output, replan=args.replan)))
