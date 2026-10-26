"""Команды миссии. Тот же исполнитель, что у политики и среды."""
from __future__ import annotations

import argparse
import json

from space_env.mission.session import mission_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="space_env")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("scenario")
    run.add_argument("--policy", default="priority", choices=("fifo", "deadline", "priority"))
    run.add_argument("--step", type=float, default=60.0)
    validate = sub.add_parser("validate")
    validate.add_argument("scenario")
    args = parser.parse_args(argv)
    if args.command == "validate":
        from space_env.mission.scenario_v2 import load_scenario_v2

        spec = load_scenario_v2(args.scenario)
        print(json.dumps({"instances": [item["id"] for item in spec["instances"]], "requests": [item["id"] for item in spec["requests"]]}, ensure_ascii=False))
        return
    report = mission_report(args.scenario, step_sec=args.step, policy=args.policy)
    print(
        json.dumps(
            {
                "on_time": report["on_time"],
                "unmet": report["unmet"],
                "weighted_on_time": report["weighted_on_time"],
                "safety_violations": report["safety_violations"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
