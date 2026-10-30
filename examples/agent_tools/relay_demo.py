"""A real serialized data handover, followed by replay and dialogue export.

Run from a checkout after package installation:
python examples/agent_tools/relay_demo.py /tmp/space-agent-demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from space_env.data_generators.agent_run import (
    AgentRunRecorder,
    export_tool_dialogue,
    replay_agent_run,
)
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.session import MissionSession
from space_env.mission.tools import MissionTools
from space_env.simulation.fleet import Fleet
from space_env.simulation.world import MissionWorld


def main(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fleet = Fleet.string_of_pearls(2, separation_deg=.02, epoch="2026-10-20T10:00:00Z",
                                  scenario=ScenarioConfig(ground_stations=[]))
    world = MissionWorld(fleet.vehicles, epoch="2026-10-20T10:00:00Z", horizon_sec=30)
    world.engines["SC-1"].accept_bits(5e6, origin="existing-scene", request_id="scene")
    tools = MissionTools.from_session(MissionSession.from_world(world), seed=7, call_budget=8)
    recorder = AgentRunRecorder(tools, task="Transfer the stored scene from SC-1 to SC-2 in five seconds.",
                                lineage="serial-handover-demo", provenance={"teacher": "manual-script", "expert": False})
    recorder.invoke("observe", {})
    result = recorder.invoke("relay", {"source": "SC-1", "destination": "SC-2", "dt_sec": 5, "expected_revision": 0})
    recorder.invoke("observe", {})
    run = directory / "agent-run.json"
    recorder.save(run)
    verified = replay_agent_run(run)
    if not verified["ok"] or not result["ok"] or result["result"]["bits"] != 5e6:
        raise RuntimeError(f"handover/replay failed: {result}, {verified}")
    dataset = export_tool_dialogue(run, directory / "dialogue.jsonl")
    print(json.dumps({"relay": result["result"], "replay": verified, "dataset": dataset}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
