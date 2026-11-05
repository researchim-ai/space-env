"""Operator incident branch commands. Compilation does not certify future suffix."""

from space_env.planning.incident import recover_incident, verify_incident
from space_env.planning.mission_bundle import read_json
from space_env.planning.mission_cli import write_new


def run_command(args):
    if args.command == "incident-data-export":
        from space_env.planning.incident_data import publish_incident_data

        r = publish_incident_data(
            read_json(args.proposal), read_json(args.record), args.output
        )
        return {
            "mission_goal_passed": r["mission_goal_passed"],
            "training_files": len(r["training_files"]),
            "sha256": r["sha256"],
        }
    if args.command == "verify-incident-data":
        from space_env.planning.incident_data import verify_incident_data

        return verify_incident_data(args.directory, native=args.native)
    if args.command == "verify-chain-incident":
        return verify_incident(read_json(args.proposal))
    incident = read_json(args.incident, limit=2000000) if args.incident else None
    value = recover_incident(read_json(args.proposal), read_json(args.record), incident)
    write_new(args.output, value)
    return {
        "sha256": value["sha256"],
        "plan_sha256": value["plan"]["sha256"],
        "predicted_fulfilled": value["schedule"]["predicted_fulfilled"],
        "total": value["schedule"]["total"],
        "committed_prefix_steps": value["committed_prefix_steps"],
        "native_feasibility_validated": False,
        "expert_teacher_qualified": False,
    }
