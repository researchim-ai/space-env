"""Bounded mission plan compilation, execution, retention and audit commands."""
from pathlib import Path

from space_env.planning.mission_bundle import read_json, publish_plan_bundle, verify_plan_bundle
from space_env.planning.mission_plan import MissionPlan, PlanExecution, recorded_mission_plan
from space_env.simulation.snapshot import atomic_write, canonical_json


def write_new(path,value):
    path=Path(path)
    if path.exists() or path.is_symlink():raise ValueError('mission artifact output must not exist')
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(path,canonical_json(value))


def run_command(args):
    if args.command=='mission-plan-schema':
        write_new(args.output,MissionPlan.model_json_schema());return {'passed':True}
    if args.command=='verify-mission-plan-bundle':
        return verify_plan_bundle(args.directory,native=args.native)
    if args.command=='mission-plan-from-record':
        goals=read_json(args.goals,limit=2_000_000) if args.goals else None
        result=recorded_mission_plan(read_json(args.record),goals=goals)
        write_new(args.output,result)
        return {'passed':True,'plan_sha256':result['sha256'],'steps':len(result['steps']),
                'origin':result['origin'],'source_policy_replayed':False}
    plan=read_json(args.plan)
    if args.command=='mission-plan-export':
        receipt=publish_plan_bundle(plan,read_json(args.record),args.output,max_bytes=args.max_bytes)
        return {'mission_goal_passed':receipt['mission_goal_passed'],'sha256':receipt['sha256'],
                'sft_exported':receipt['sft_exported']}
    output=Path(args.output)
    if output.exists() or output.is_symlink():raise ValueError('mission artifact output must not exist')
    execution=(PlanExecution.from_checkpoint(plan,read_json(args.record))
               if args.command=='mission-plan-resume' else PlanExecution(plan))
    status=execution.run(decisions=args.decisions)
    record=execution.checkpoint()
    write_new(output,record)
    return status|{'sha256':record['sha256']}
