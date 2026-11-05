"""Explicit operator compilation/replanning; proposals are not certificates."""

from space_env.planning.chain import (
    propose_chain,
    replan_chain,
    verify_chain_proposal,
    verify_chain_replan,
)
from space_env.planning.mission_bundle import read_json
from space_env.planning.mission_cli import write_new
from space_env.runtime import AccessPolicy, MissionRuntime


def run_command(args):
    if args.command == "verify-chain-plan":
        value = read_json(args.proposal)
        if value.get("schema") == "space-env.chain-replan/1":
            return verify_chain_replan(value)
        return verify_chain_proposal(value, native=args.native)
    if args.command == "chain-replan":
        value = replan_chain(read_json(args.proposal), read_json(args.record))
    else:
        scenario, task = read_json(args.scenario), read_json(args.task)
        goals = read_json(args.goals, limit=2000000) if args.goals else None
        runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
        value = propose_chain(runtime, scenario, task, goals=goals)
    write_new(args.output, value)
    return {
        "sha256": value["sha256"],
        "plan_sha256": value["plan"]["sha256"],
        "predicted_fulfilled": value["schedule"]["predicted_fulfilled"],
        "total": value["schedule"]["total"],
        "native_feasibility_validated": False,
        "expert_teacher_qualified": False,
    }
