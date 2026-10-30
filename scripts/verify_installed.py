"""Offline smoke checks against a wheel installed outside the source checkout.

Run with the target environment's `python -I` and a working directory outside repo.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np

import space_env
from space_env.api.server import handle_get
from space_env.core.frames import ecef_to_geodetic
from space_env.core.ground import GroundStation
from space_env.data_generators.agent_run import (
    AgentRunRecorder,
    export_tool_dialogue,
    replay_agent_run,
)
from space_env.data_generators.portable_run import (
    RunRecorder,
    export_transitions,
    replay_run,
)
from space_env.data_generators.reference_operations import (
    ReferenceOperationsRecorder,
    replay_reference_operations,
)
from space_env.eval.mission_benchmark import run_mission_suite
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.session import MissionSession
from space_env.mission.tools import MissionTools
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.rl.trajectories import rollout_episode
from space_env.simulation.fleet import Fleet
from space_env.simulation.snapshot import digest, load_world, save_world
from space_env.simulation.world import MissionWorld
from space_env.validation.gcrs_report import reference_physics_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prefix", type=Path, required=True)
    args = parser.parse_args()
    root = Path(space_env.__file__).resolve().parent
    if not root.is_relative_to(args.expected_prefix.resolve()):
        raise RuntimeError(f"импорт идёт не из проверяемого окружения: {root}")
    env = gym.make("space_env/MissionDict-v0", max_episode_steps=2)
    try:
        rows = rollout_episode(env, seed=7)
    finally:
        env.close()
    assert isinstance(rows[0]["observation"], dict)
    assert rows[-1]["truncated"]
    json.dumps(rows, allow_nan=False)
    suite = run_mission_suite("mission_v1_smoke.yaml")
    assert suite["passed"]
    for path in ("/studio", "/viz/app.js"):
        status, body, _kind = handle_get(path, {})
        assert status == 200 and body
    scenario = root / "configs" / "family_nominal.yaml"
    session = MissionSession(scenario)
    session.advance(1, expected_revision=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        save_world(path, session.world, scenario)
        restored = load_world(path)
        restored_time = restored.now_sec()
        assert restored_time == 1
        assert restored.engines["sentinel-a"].stations
        recorder = RunRecorder(restored, lineage="installed-wheel-smoke", provenance={"teacher": "wait"})
        recorder.step(1, [])
        run_path = Path(tmp) / "run.json"
        recorder.save(run_path)
        portable = replay_run(run_path)
        assert portable["ok"], portable
        dataset = export_transitions(run_path, Path(tmp) / "transitions.jsonl")
        assert dataset["rows"] == 1
        fleet = Fleet.string_of_pearls(2, separation_deg=.02, epoch="2026-10-20T10:00:00Z",
                                      scenario=ScenarioConfig(ground_stations=[]))
        relay_world = MissionWorld(fleet.vehicles, epoch="2026-10-20T10:00:00Z", horizon_sec=5)
        relay_world.engines["SC-1"].accept_bits(1e6, origin="imaging", request_id="scene")
        agent = AgentRunRecorder(MissionTools.from_session(MissionSession.from_world(relay_world)),
                                 task="Relay the scene to SC-2", lineage="installed-tool-smoke")
        assert agent.invoke("observe", {})["ok"]
        relayed = agent.invoke("relay", {"source": "SC-1", "destination": "SC-2", "dt_sec": 1,
                                         "expected_revision": 0})
        assert relayed["ok"] and relayed["result"]["bits"] == 1e6
        agent_path = Path(tmp) / "agent.json"
        agent.save(agent_path)
        agent_replay = replay_agent_run(agent_path)
        assert agent_replay["ok"], agent_replay
        dialogue = export_tool_dialogue(agent_path, Path(tmp) / "dialogue.jsonl")
        assert dialogue["messages"] == 5
    physics = reference_physics_report(long_arcs=True, crosscheck=True)
    reference_result = propagate(ReferenceState(Epoch.utc("2026-07-18T12:00:00Z"),
                                 np.array([7e6, 0, 0]), np.array([0, 0, 7546.053290107542])),
                                 1200, FrameContext(EOPSeries.packaged()), sample_sec=30)
    ephemeris = ReferenceEphemeris.from_result(reference_result)
    fixed = ephemeris.context.matrix(ephemeris.epoch.shifted(600)) @ ephemeris.state(600)[:3]
    lat, lon, _alt = ecef_to_geodetic(fixed)
    access = ephemeris.access(GroundStation(name="ReferenceGround", lat_deg=lat, lon_deg=lon))
    assert any(window["aos_sec"] <= 600 <= window["los_sec"] for window in access["windows"])
    operations_spec = {"schema": "space-env.reference-downlink-spec/1",
                       "ephemerides": {"a": {**reference_result, "sha256": digest(reference_result)}},
                       "vehicles": {"a": {"memory_bits": 2e6, "battery_wh": 1, "base_power_w": 0, "tx_power_w": 0}},
                       "stations": [access["station"]],
                       "products": [{"id": "scene", "actor": "a", "size_bits": 2e6, "deadline_sec": 1200}]}
    operations_env = ReferenceDownlinkEnv(operations_spec, dt_sec=1200)
    operations_env.reset(seed=7)
    _, reference_reward, terminated, _, _ = operations_env.step(np.array([1]))
    assert terminated and reference_reward == 2
    with tempfile.TemporaryDirectory() as tmp:
        operation_run = Path(tmp) / "reference-run.json"
        recorder = ReferenceOperationsRecorder(operations_spec, lineage="installed-reference-operations")
        recorder.step(1200, {"a": "ReferenceGround"})
        recorder.save(operation_run)
        reference_operations_replay = replay_reference_operations(operation_run)
        assert reference_operations_replay["ok"], reference_operations_replay
    assert physics["passed"], physics
    cli = subprocess.check_output(
        [sys.executable, "-I", "-m", "space_env.cli", "validate", str(scenario)], text=True,
    )
    print(json.dumps({
        "package": str(root), "dict_steps": len(rows), "suite_passed": suite["passed"],
        "cases": len(suite["cases"]), "reference_access_windows": len(access["windows"]),
        "reference_reward_mbit": reference_reward, "reference_operations_replay": reference_operations_replay,
        "external_force_crosscheck": physics["force_crosscheck"]["passed"], "gcrs_reference": {"passed": physics["passed"], "checks": len(physics["checks"]), "scope": "upstream rotation and circular two-body 600/24h/72h"}, "studio_status": 200,
        "restored_time_sec": restored_time, "portable_replay": portable, "agent_replay": agent_replay, "dialogue_messages": dialogue["messages"], "transition_rows": dataset["rows"], "cli_validate": json.loads(cli),
        "dependencies": {name: version(name) for name in ("numpy", "boinor", "gymnasium", "astropy", "pydantic")},
    }))


if __name__ == "__main__":
    main()
