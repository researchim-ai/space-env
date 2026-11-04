"""Explicit operator forecast scheduling and retained evaluation workflows."""

import os
import shutil
import tempfile
from pathlib import Path

from space_env.planning.forecast_benchmark import (
    run_forecast_benchmark,
    verify_forecast_benchmark,
)
from space_env.planning.forecast_delivery import (
    collect_delivery_forecast,
    forecast_mission_plan,
)
from space_env.planning.mission_bundle import read_json
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest


def run_command(args):
    if args.command == "forecast-benchmark":
        result = run_forecast_benchmark(read_json(args.spec), args.output)
        return {
            "execution_integrity_passed": result["execution_integrity_passed"],
            "teacher_status": result["teacher"]["status"],
            "sha256": result["sha256"],
        }
    if args.command == "verify-forecast-benchmark":
        return verify_forecast_benchmark(args.directory, native=args.native)
    scenario, task = read_json(args.scenario), read_json(args.task)
    goals = read_json(args.goals, limit=2_000_000) if args.goals else None
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("forecast output must not exist")
    # This command is an explicit operator authorization; ordinary policy/agent
    # views do not acquire the Runtime forecast permission automatically.
    runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
    view = collect_delivery_forecast(runtime, scenario, task)
    result = forecast_mission_plan(scenario, task, view, goals=goals)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        for name, value in result.items():
            atomic_write(stage / (name + ".json"), canonical_json(value))
        body = {
            "schema": "space-env.forecast-proposal-bundle/1",
            "plan_sha256": result["plan"]["sha256"],
            "forecast_sha256": view.sha256,
            "schedule_sha256": result["schedule"]["sha256"],
            "files_sha256": {p.name: file_digest(p) for p in stage.iterdir()},
            "native_feasibility_validated": False,
        }
        receipt = body | {"sha256": digest(body)}
        atomic_write(stage / "proposal.json", canonical_json(receipt))
        if output.exists() or output.is_symlink():
            raise ValueError("forecast output appeared during compilation")
        os.rename(stage, output)
        return {
            "plan_sha256": result["plan"]["sha256"],
            "predicted_fulfilled": result["schedule"]["predicted_fulfilled"],
            "total": result["schedule"]["total"],
            "native_feasibility_validated": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
