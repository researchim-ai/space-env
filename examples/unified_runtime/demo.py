"""Generate a self-contained scenario and demonstrate active-command resume."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, reference_scenario


def main(directory: Path):
    # Reuse the repository's published physical fixture, never the test suite.
    path = Path(__file__).resolve().parents[1] / "reference_operations" / "demo.py"
    loader = importlib.util.spec_from_file_location("reference_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario = reference_scenario(module.build_spec(), id="unified-demo", lineage="B01")
    plan = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "orders",
                "commands": [
                    {
                        "command_id": "a-radio",
                        "actor_id": "a",
                        "kind": "downlink",
                        "duration_sec": 10,
                        "destination_id": "Ground",
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 1,
                "idempotency_key": "prefix",
                "dt_sec": 1,
            },
        ]
    }
    continuation = {
        "operations": [
            {
                "op": "advance",
                "expected_revision": 2,
                "idempotency_key": "continue",
                "dt_sec": 9,
            }
        ]
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scenario", scenario),
        ("plan", plan),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    from space_env.runtime.cli import apply_plan

    runtime = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(runtime, plan)
    runtime.save(directory / "prefix.json")
    restored = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    apply_plan(restored, continuation)
    restored.save(directory / "final.json")
    replayed = MissionRuntime.load(directory / "final.json")
    assert replayed.report() == restored.report()
    assert replayed.report()["metrics"]["on_time_bits"] == 2e6
    print(json.dumps({"ok": True, "report": replayed.report()}, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
