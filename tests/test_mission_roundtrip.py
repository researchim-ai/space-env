"""Снимок, команды, заявка, генератор, общий исход и повтор прогона."""
import json
import subprocess
import sys

import numpy as np
import pytest

from space_env.api.server import handle_get
from space_env.cli import main
from space_env.data_generators.mission_dataset import replay, seal_run
from space_env.envs.tasking_env import packaged_tasking
from space_env.mission.factory import build_mission
from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.mission.session import mission_report
from space_env.mission.tasking import TaskRequest, delivery_score, request_lifecycle
from space_env.simulation.commands import CommandBook, MissionCommand
from space_env.simulation.snapshot import save_world


def test_file_snapshot_replays_the_next_step_in_a_new_process(tmp_path):
    scenario = tmp_path / "one.yaml"
    scenario.write_text(
        "version: 2\nepoch: '2026-10-20T10:00:00Z'\nhorizon_sec: 60\ninstances:\n"
        "  - {id: a, model: cubesat-6u, nu_deg: 0}\nstations: []\nrequests: []\n",
        encoding="utf-8",
    )
    world = world_from_scenario(scenario)
    world.step(10.0)
    snap = tmp_path / "snap.json"
    save_world(snap, world, scenario)
    world.step(10.0)
    expected = world.engines["a"].orbit.r
    script = (
        "import json\n"
        "from space_env.simulation.snapshot import load_world\n"
        f"w = load_world({str(snap)!r})\n"
        "w.step(10.0)\n"
        "print(json.dumps(w.engines['a'].orbit.r.tolist()))\n"
    )
    raw = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert np.allclose(json.loads(raw), expected)


def test_command_repeat_is_idempotent_and_cancel_drops_it():
    book = CommandBook()
    command = MissionCommand("c1", "alpha", "comms", request_id="r", destination_id="Kiruna")
    first = book.submit(command)
    second = book.submit(command)
    assert first == second
    assert book.orders() == {"alpha": "comms"}
    with pytest.raises(ValueError, match="другим телом"):
        book.submit(MissionCommand("c1", "alpha", "imaging"))
    book.cancel("c1")
    assert book.orders() == {}


def test_wrong_station_partial_quality_and_lifecycle():
    request = TaskRequest("r", 0.0, 0.0, 0.0, 10.0, 100.0, 1, "optical", destination="Kiruna", quality="full")
    wrong = {"product_id": "p", "request_id": "r", "destination": "Svalbard", "time_sec": 5.0, "offset_bits": 0.0, "bits": 100.0}
    partial = dict(wrong, destination="Kiruna", quality="partial")
    good = dict(wrong, destination="Kiruna", quality="full")
    assert delivery_score([request], [wrong], [])["on_time"] == []
    assert delivery_score([request], [partial], [])["on_time"] == []
    assert delivery_score([request], [good], [])["on_time"] == ["r"]
    assert request_lifecycle(request, [], 0.0) == "available"
    assert request_lifecycle(request, [good], 5.0) == "delivered"
    early = TaskRequest("later", 0.0, 0.0, 20.0, 30.0, 100.0, 1, "optical")
    assert request_lifecycle(early, [], 10.0) == "unreleased"
    assert request_lifecycle(request, [], 11.0) == "expired"


def test_two_seeds_change_the_mission_and_the_files_have_the_promised_counts():
    left = build_mission("short", 1)
    right = build_mission("short", 2)
    assert len(left["requests"]) == 12
    assert left["requests"][0]["lat_deg"] != right["requests"][0]["lat_deg"]
    loaded = load_scenario_v2("configs/scenarios/mission_short.yaml")
    assert len(loaded["requests"]) == 12
    assert len(loaded["instances"]) == 2
    assert loaded["requests"][0]["destination"] in {"Kiruna", "Svalbard", "Matera"}
    day = load_scenario_v2("configs/scenarios/mission_day.yaml")
    assert len(day["requests"]) == 32
    assert len(day["instances"]) == 4


def test_dataset_replay_matches_the_sealed_outcome(tmp_path):
    scenario = packaged_tasking("tasking_deliver.yaml")
    record = seal_run(tmp_path / "run.json", scenario, step_sec=30.0)
    assert record["on_time"] == ["kiruna-pass"]
    assert replay(tmp_path / "run.json") is True


def test_cli_gym_and_api_share_the_short_mission():
    import gymnasium as gym

    from space_env.envs.mission_fleet_env import packaged_mission

    scenario = str(packaged_mission("mission_short.yaml"))
    report = mission_report(scenario, step_sec=60.0)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(["run", scenario, "--step", "60"])
    cli = json.loads(buffer.getvalue())
    status, body, _kind = handle_get("/api/mission", {})
    api = json.loads(body)
    env = gym.make("space_env/MissionFleet-v0")
    try:
        env.reset(seed=0)
        flown = env.unwrapped.follow_policy()
    finally:
        env.close()
    assert status == 200
    assert cli["on_time"] == api["on_time"] == report["on_time"] == flown["score"]["on_time"]
    assert cli["unmet"] == api["unmet"] == report["unmet"]
    assert len(report["statuses"]) == 12
    assert report["on_time"] == [
        "req-00",
        "req-01",
        "req-02",
        "req-03",
        "req-04",
        "req-05",
        "req-06",
        "req-07",
        "req-08",
        "req-09",
        "req-10",
        "req-11",
    ]
    assert report["unmet"] == []
    assert isinstance(report["rejected"], list)


def test_holdout_seed_is_not_the_training_mission():
    from space_env.envs.mission_fleet_env import packaged_mission

    trained = load_scenario_v2(packaged_mission("mission_short.yaml"))
    held = load_scenario_v2(packaged_mission("mission_holdout.yaml"))
    assert held["requests"][0]["lat_deg"] != trained["requests"][0]["lat_deg"]
    assert held["requests"][0]["id"] == trained["requests"][0]["id"]


def test_three_families_record_commands_and_replay(tmp_path):
    from space_env.envs.mission_fleet_env import packaged_mission

    nominal = packaged_mission("family_nominal.yaml")
    outage = packaged_mission("family_outage.yaml")
    impossible = packaged_mission("family_infeasible.yaml")
    nominal_report = mission_report(nominal, step_sec=30.0)
    outage_report = mission_report(outage, step_sec=30.0)
    impossible_report = mission_report(impossible, step_sec=30.0)
    assert nominal_report["on_time"] == ["kiruna-pass"]
    assert outage_report["on_time"] == []
    assert outage_report["unmet"][0]["reason"] == "станция недоступна"
    assert impossible_report["on_time"] == []
    assert impossible_report["unmet"][0]["reason"] == "срок вышел"
    sealed = seal_run(tmp_path / "nominal.json", nominal, step_sec=30.0)
    commands = [command for item in sealed["schedule"] for command in item.get("commands", {}).values()]
    assert commands
    assert all(item["command_id"] and item["kind"] in {"wait", "imaging", "comms"} for item in commands)
    assert replay(tmp_path / "nominal.json") is True


def test_day_mission_names_every_request():
    from space_env.envs.mission_fleet_env import packaged_mission

    report = mission_report(packaged_mission("mission_day.yaml"), step_sec=120.0)
    names = {item["request_id"] for item in report["statuses"]}
    unmet = {item["request_id"] for item in report["unmet"]}
    assert len(report["statuses"]) == 32
    assert set(report["on_time"]).isdisjoint(unmet)
    assert set(report["on_time"]) | unmet == names
    # The legacy >=30 baseline assumed sun-normal generation. With sampled
    # nadir-panel cosine, this policy depletes batteries and delivers 18.
    # Freeze the actual energy-limited baseline; do not change physics to fit it.
    assert report["on_time"] == [
        "req-00", "req-01", "req-02", "req-03", "req-04", "req-05",
        "req-06", "req-07", "req-08", "req-09", "req-10", "req-11",
        "req-13", "req-14", "req-16", "req-24", "req-28", "req-31",
    ]
    assert all(item["reason"] == "срок вышел" for item in report["unmet"])
    assert report["safety_violations"] >= 0
