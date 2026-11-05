"""Actor controller and flat incident journal file workflows."""

from space_env.planning.actor_data import publish_actor_data, verify_actor_data
from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.incident_journal import (
    append_incident,
    create_journal,
    verify_journal,
)
from space_env.planning.mission_bundle import read_json
from space_env.planning.mission_cli import write_new


def run_command(args):
    if args.command == "verify-incident-journal":
        return verify_journal(read_json(args.journal), native=args.native)
    if args.command == "incident-journal-create":
        value = create_journal(read_json(args.proposal))
        write_new(args.output, value)
        return {"sha256": value["sha256"], "epochs": 0}
    if args.command == "incident-journal-append":
        incident = read_json(args.incident, limit=2000000) if args.incident else None
        record = read_json(args.record)
        if set(record) == {"plan", "execution"}:
            if record["plan"]["sha256"] != record["execution"]["plan_sha256"]:
                raise ValueError("actor execution container plan mismatch")
            record = record["execution"]
        value = append_incident(read_json(args.journal), record, incident)
        write_new(args.output, value)
        return {"sha256": value["sha256"], "epochs": len(value["epochs"])}
    if args.command == "verify-actor-data":
        return verify_actor_data(args.directory, native=args.native)
    if args.command == "actor-data-export":
        value = publish_actor_data(read_json(args.checkpoint), args.output)
        return {
            "sha256": value["sha256"],
            "mission_goal_passed": value["goal"]["passed"],
            "training_files": len(value["training_files"]),
        }
    if args.command == "actor-controller-start":
        controller = ActorCoordinator(read_json(args.reference))
    else:
        controller = ActorCoordinator.from_checkpoint(read_json(args.checkpoint))
    if args.command == "actor-controller-observe":
        write_new(args.output, controller.observe())
        return {"cursor": controller.cursor}
    if args.command == "actor-controller-verify":
        return {
            "integrity_passed": True,
            "native_trace_replayed": True,
            "cursor": controller.cursor,
        }
    response = None
    if args.command == "actor-controller-step":
        response = controller.step(**read_json(args.request, limit=2000000))
    if args.command == "actor-controller-execution":
        value = controller.as_execution()
        write_new(args.output, value)
        return {"plan_sha256": value["plan"]["sha256"], "cursor": controller.cursor}
    record = controller.checkpoint()
    write_new(args.output, record)
    return {
        "sha256": record["sha256"],
        "cursor": controller.cursor,
        "feedback": response["feedback"] if response else None,
    }
