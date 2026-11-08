"""Joint EO owner + ISL route + ground ACK, without preloaded products."""

import argparse
import json
from pathlib import Path

from space_env.api.network_workspace import default_network_configuration
from space_env.workspace import MissionWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output directory must be new")
    config = (
        json.loads(args.configuration.read_text())
        if args.configuration
        else default_network_configuration()
    )
    mission = MissionWorkspace.network(config)
    try:
        result = mission.run()["network"]
        if args.output:
            args.output.mkdir(parents=True)
            (args.output / "configuration.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n"
            )
            (args.output / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            mission.export(args.output / "training.zip")
        print(
            f"Planned {result['planned']}/{result['total']}; actual ACK {result['fulfilled']}/{result['total']}"
        )
        for request in result["requests"]:
            print(
                request["id"],
                request["status"],
                " -> ".join([*request["plan"]["path"], request["plan"]["station"]])
                if request["plan"]
                else request["plan_reason"],
            )
        return 0 if result["fulfilled"] == result["total"] else 2
    finally:
        mission.close()


if __name__ == "__main__":
    raise SystemExit(main())
