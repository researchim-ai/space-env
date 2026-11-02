"""Task plan/checkpoint/export workflow over the same TaskSession."""

from __future__ import annotations

import json
from pathlib import Path

from space_env.runtime.schema import load_scenario
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.dataset import export_dataset
from space_env.tasks.schema import TaskSpec
from space_env.tasks.session import TaskSession


def apply_decisions(session, plan):
    if (
        not isinstance(plan, dict)
        or set(plan) != {"decisions"}
        or not isinstance(plan["decisions"], list)
    ):
        raise ValueError("task plan requires exactly a decisions list")
    if len(plan["decisions"]) > session.describe()["capacities"]["max_decisions"]:
        raise ValueError("task plan decision budget exceeded")
    for request in plan["decisions"]:
        session.step(**request)
    return session.observe()


def run_command(args):
    if args.command == "task-schema":
        atomic_write(args.output, canonical_json(TaskSpec.model_json_schema()))
        return {"ok": True}
    if args.command == "task-replay":
        session = TaskSession.load(args.record)
        return {
            "ok": True,
            "decisions": session.revision,
            "time_sec": session.observe()["time_sec"],
        }
    if args.command == "task-export":
        return export_dataset(
            json.loads(Path(args.record).read_text()),
            args.output,
            format=args.format,
            role=args.role,
        )
    source = args.record if args.command == "task-resume" else args.scenario
    paths = [Path(p).resolve() for p in [source, args.plan, args.output]]
    if len(set(paths)) != 3:
        raise ValueError("source, plan and output must differ")
    if args.command == "task-run" and Path(args.task).resolve() in paths[1:]:
        raise ValueError("task specification cannot be overwritten")
    session = (
        TaskSession.load(source)
        if args.command == "task-resume"
        else TaskSession(load_scenario(source), json.loads(Path(args.task).read_text()))
    )
    apply_decisions(session, json.loads(Path(args.plan).read_text()))
    record = session.save(args.output)
    return {
        "ok": True,
        "sha256": record["sha256"],
        "decisions": session.revision,
        "time_sec": session.observe()["time_sec"],
    }
