"""One local mission: optional JSON configuration, native run and optional export."""

import argparse
import json
from pathlib import Path

from space_env.workspace import MissionWorkspace, default_configuration


def main():
    parser = argparse.ArgumentParser(
        description="Настройка → план → миссия → результат и данные"
    )
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Каталог --output должен быть новым")
    config = (
        json.loads(args.configuration.read_text())
        if args.configuration
        else default_configuration()
    )
    mission = MissionWorkspace(config)
    state = mission.run(decisions=args.steps)
    result = state["workspace"]
    if args.output:
        args.output.mkdir(parents=True)
        (args.output / "configuration.json").write_text(
            json.dumps(result["configuration"], ensure_ascii=False, indent=2) + "\n"
        )
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        mission.export(args.output / "training.zip")
    print(
        f"{result['name']}: запланировано {result['planned']}/{result['total']}, доставлено {result['fulfilled']}/{result['total']}, время {state['time_sec']} с"
    )
    for r in result["requests"]:
        print(
            f"  {r['id']}: {r['status']}"
            + (f" — {r['plan_reason']}" if r["plan_reason"] else "")
        )


if __name__ == "__main__":
    main()
