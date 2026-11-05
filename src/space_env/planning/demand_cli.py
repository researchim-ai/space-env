"""New demand and explicit actor catalogue rebinding file workflows."""

from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.demand_journal import (
    append_demand,
    create_demand_journal,
    verify_demand_journal,
)
from space_env.planning.mission_bundle import read_json
from space_env.planning.mission_cli import write_new


def run_command(args):
    if args.command == "verify-demand-journal":
        return verify_demand_journal(read_json(args.journal), native=args.native)
    if args.command == "actor-controller-rebind":
        actor = ActorCoordinator.from_checkpoint(read_json(args.checkpoint))
        rebound = actor.rebind(read_json(args.journal))
        record = rebound.checkpoint()
        write_new(args.output, record)
        return {
            "sha256": record["sha256"],
            "catalogue_version": rebound.catalogue_version,
            "cursor": rebound.cursor,
        }
    if args.command == "demand-journal-create":
        value = create_demand_journal(read_json(args.reference))
    else:
        record = read_json(args.record)
        if set(record) == {"plan", "execution"}:
            if record["plan"]["sha256"] != record["execution"]["plan_sha256"]:
                raise ValueError("demand execution container mismatch")
            record = record["execution"]
        value = append_demand(
            read_json(args.journal), record, read_json(args.batch, limit=2000000)
        )
    write_new(args.output, value)
    return {
        "sha256": value["sha256"],
        "epochs": len(value["epochs"]),
        "catalogue_version": value["catalogue_version"],
    }
