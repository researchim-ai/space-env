"""CLI integration for the same baseline runner and portable records."""

from __future__ import annotations

import json
from pathlib import Path

from space_env.planning.benchmark import run_benchmark, verify_evaluation
from space_env.planning.runner import PlanningRun
from space_env.simulation.snapshot import atomic_write, canonical_json


def run_command(args):
    if args.command == "planning-verify-evaluation":
        return verify_evaluation(json.loads(Path(args.record).read_text()))
    if args.command == "planning-replay":
        run = PlanningRun.load(args.record)
        return {"ok": True, "decisions": len(run.trace), "metrics": run.metrics()}
    source = (
        args.record
        if args.command == "planning-resume"
        else args.manifest
        if args.command == "planning-evaluate"
        else args.scenario
    )
    inputs = [source]
    if args.command == "planning-run":
        inputs.extend([args.task, args.policy])
    if Path(args.output).resolve() in {Path(p).resolve() for p in inputs}:
        raise ValueError("planning output cannot overwrite an input")
    if args.command == "planning-evaluate":
        if args.artifacts is not None and Path(args.output).resolve().is_relative_to(
            Path(args.artifacts).resolve()
        ):
            raise ValueError(
                "evaluation output must be outside the artifacts directory"
            )
        result = run_benchmark(
            json.loads(Path(source).read_text()), artifacts=args.artifacts
        )
        atomic_write(args.output, canonical_json(result))
        return {
            "ok": result["passed_execution"]
            and result["teacher"]["status"] != "failed_declared_gates",
            "sha256": result["sha256"],
            "teacher": result["teacher"],
        }
    if args.command == "planning-resume":
        run = PlanningRun.load(source)
    else:
        from space_env.runtime.schema import load_scenario

        run = PlanningRun(
            load_scenario(source),
            json.loads(Path(args.task).read_text()),
            json.loads(Path(args.policy).read_text()),
            episode_seed=args.seed,
        )
    metrics = run.run(args.decisions)
    record = run.save(args.output)
    return {"ok": True, "sha256": record["sha256"], "metrics": metrics}
