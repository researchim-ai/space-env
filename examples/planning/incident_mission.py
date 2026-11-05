"""Frozen synthetic incident cases. Run from repository using python -m examples.planning.incident_mission."""

import argparse
from pathlib import Path

from examples.planning.chain_mission import configuration as chain_configuration
from space_env.planning.chain import propose_chain
from space_env.planning.incident import recover_incident, verify_incident
from space_env.planning.mission_bundle import publish_plan_bundle
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def configuration(mode="partial_tx"):
    if mode not in ("partial_tx", "paused_cpu", "lost_capture", "blocked", "slew"):
        raise ValueError("unknown incident mode")
    value = chain_configuration()
    d = value["scenario"]["definition"]
    d["horizon_sec"] = 180.0
    for r in d["lifecycle"]["requests"]:
        r["deadline_sec"] = 180.0
    value["task"].update(decision_sec=2.0, max_decision_sec=2.0, max_decisions=91)
    if mode in ("partial_tx", "blocked"):
        for v in d["operations"]["vehicles"].values():
            v["rate_bps"] = 50000.0
    if mode == "paused_cpu":
        for j in d["lifecycle"]["jobs"]:
            j["work_sec"] = 8.0
    if mode == "lost_capture":
        for r in d["lifecycle"]["requests"]:
            r["acquisition_sec"] = 6.0
    value["goals"].update(min_fulfilled=2, require_no_failed_operations=False)
    value["goals"]["coupled"]["native_spec_sha256"] = digest(d)
    value.update(
        mode=mode,
        outage_end_sec=180.0
        if mode == "blocked"
        else 120.0
        if mode != "slew"
        else 20.0,
    )
    return value


def run(value, output):
    root = Path(output)
    if root.exists():
        raise ValueError("output must not exist")
    root.mkdir(parents=True)
    atomic_write(root / "input.json", canonical_json(value))
    proposal = propose_chain(
        MissionRuntime(value["scenario"], access=AccessPolicy(forecast=True)),
        value["scenario"],
        value["task"],
        goals=value["goals"],
    )
    atomic_write(root / "source_proposal.json", canonical_json(proposal))
    first = proposal["schedule"]["allocations"][0]
    q = value["task"]["decision_sec"]
    field = {
        "partial_tx": "delivery_sec",
        "blocked": "delivery_sec",
        "paused_cpu": "cpu_sec",
        "lost_capture": "capture_sec",
        "slew": "start_sec",
    }[value["mode"]]
    boundary = first[field] + q
    x = PlanExecution(proposal["plan"])
    x.run(decisions=round(boundary / q))
    record = x.save(root / "source_prefix.json")
    stations = sorted({a["station"] for a in proposal["schedule"]["allocations"]})
    incident = {
        "disruptions": [
            {"station": s, "end_sec": value["outage_end_sec"]} for s in stations
        ]
    }
    atomic_write(root / "incident.json", canonical_json(incident))
    updated = recover_incident(proposal, record, incident)
    atomic_write(root / "recovery.json", canonical_json(updated))
    atomic_write(root / "prefix_audit.json", canonical_json(verify_incident(updated)))
    execution = PlanExecution(updated["plan"])
    execution.run()
    record = execution.save(root / "execution.json")
    atomic_write(root / "plan.json", canonical_json(updated["plan"]))
    bundle = publish_plan_bundle(updated["plan"], record, root / "dataset")
    fulfilled = record["episode"]["runtime"]["backend_report"]["lifecycle"]["metrics"][
        "fulfilled_requests"
    ]
    result = {
        "mode": value["mode"],
        "fulfilled": fulfilled,
        "total": updated["schedule"]["total"],
        "predicted_fulfilled": updated["schedule"]["predicted_fulfilled"],
        "mission_goal_passed": bundle["mission_goal_passed"],
        "sft_exported": bundle["sft_exported"],
        "phases": {
            phase: sum(s["phase"] == phase for s in updated["states"].values())
            for phase in ("fresh", "raw", "final", "completed", "terminal")
        },
        "expert_teacher_qualified": False,
    }
    atomic_write(root / "result.json", canonical_json(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument(
        "--mode",
        choices=["partial_tx", "paused_cpu", "lost_capture", "blocked", "slew"],
        default="partial_tx",
    )
    p.add_argument("--write-spec", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must not exist")
    value = configuration(args.mode)
    if args.write_spec:
        atomic_write(args.output, canonical_json(value))
    else:
        print(canonical_json(run(value, args.output)))
