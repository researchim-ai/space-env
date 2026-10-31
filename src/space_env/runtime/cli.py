"""One scenario/plan/checkpoint workflow for both runtime adapters."""

from __future__ import annotations

import json
from pathlib import Path

from space_env.runtime.mission import MissionRuntime
from space_env.runtime.schema import AccessPolicy, load_scenario, scenario_json_schema
from space_env.simulation.snapshot import atomic_write, canonical_json


def apply_plan(runtime: MissionRuntime, plan: dict):
    if (
        not isinstance(plan, dict)
        or set(plan) != {"operations"}
        or not isinstance(plan["operations"], list)
    ):
        raise ValueError("runtime plan requires exactly an operations list")
    if (
        len(plan["operations"])
        > runtime.describe_capabilities()["limits"]["max_operations"]
    ):
        raise ValueError("plan exceeds operation budget")
    for row in plan["operations"]:
        if not isinstance(row, dict) or "op" not in row:
            raise ValueError("each operation requires op")
        op = row["op"]
        fields = {
            "submit": {"commands"},
            "advance": {"dt_sec"},
            "cancel": {"command_id"},
            "inject": {"events"},
        }
        if (
            op not in fields
            or set(row) != {"op", "expected_revision", "idempotency_key"} | fields[op]
        ):
            raise ValueError("invalid runtime operation fields")
        getattr(runtime, op)(**{k: v for k, v in row.items() if k != "op"})
    return runtime.report()


def run_command(args):
    if args.command == "mission-schema":
        atomic_write(args.output, canonical_json(scenario_json_schema()))
        return {"schema_version": 3, "written": str(args.output)}
    if args.command == "mission-replay":
        runtime = MissionRuntime.load(args.record)
        return {"ok": True, "reason": "verified", "report": runtime.report()}
    if args.command == "mission-validate":
        runtime = MissionRuntime(load_scenario(args.scenario))
        return {
            "ok": True,
            "capabilities": runtime.describe_capabilities(),
            "report": runtime.report(),
        }
    source = args.record if args.command == "mission-resume" else args.scenario
    paths = [Path(p).resolve() for p in (source, args.plan, args.output)]
    if len(set(paths)) != 3:
        raise ValueError("source, plan and output paths must differ")
    access = AccessPolicy(truth=True)
    runtime = (
        MissionRuntime.load(source, access=access)
        if args.command == "mission-resume"
        else MissionRuntime(load_scenario(source), access=access)
    )
    apply_plan(runtime, json.loads(paths[1].read_text()))
    payload = runtime.save(paths[2])
    return {"ok": True, "sha256": payload["sha256"], "report": runtime.report()}
