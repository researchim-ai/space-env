"""Дефекты M0: пачка команд, неизвестная заявка, ранняя съёмка и станция в снимке."""
import tempfile
from pathlib import Path

import yaml

from space_env.mission.scenario_v2 import world_from_scenario
from space_env.mission.tasking import TaskRequest, delivery_score
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import load_world, save_world
from tests.test_mission_contract import _world


def test_a_conflicting_batch_does_not_keep_the_first_command():
    world = _world()
    before = len(world.commands.export())
    try:
        world.accept([MissionCommand("same", "a", "wait"), MissionCommand("same", "a", "imaging")])
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert len(world.commands.export()) == before
    assert world.now_sec() == 0.0


def test_an_unknown_request_is_rejected_rather_than_completed():
    world = _world()
    orders = world.accept([MissionCommand("ghost-request", "a", "imaging", request_id="not-a-request")])
    world.step(1.0, orders)
    world.settle()
    record = world.commands.get("ghost-request")
    assert record["status"] == "rejected"
    assert record["reason"] == "неизвестная заявка"


def test_the_earliest_full_acquisition_closes_the_request():
    request = TaskRequest("r", 0.0, 0.0, 0.0, 10.0, 100.0, 1, "optical")
    slips = [
        {"request_id": "r", "product_id": "early", "offset_bits": 0.0, "bits": 100.0, "time_sec": 5.0},
        {"request_id": "r", "product_id": "late", "offset_bits": 0.0, "bits": 120.0, "time_sec": 20.0},
    ]
    assert delivery_score([request], slips, [])["on_time"] == ["r"]


def test_json_snapshot_keeps_the_kiruna_station():
    with tempfile.TemporaryDirectory() as tmp:
        scenario = Path(tmp) / "source.yaml"
        scenario.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "horizon_sec": 60,
                    "instances": [{"id": "a", "model": "cubesat-6u"}],
                    "stations": ["Kiruna"],
                    "requests": [],
                }
            ),
            encoding="utf-8",
        )
        world = world_from_scenario(scenario, attach_stations=True, refresh_passes=False)
        path = Path(tmp) / "state.json"
        save_world(path, world, scenario)
        loaded = load_world(path)
        assert [station.name for station in loaded.engines["a"].stations] == ["Kiruna"]


def test_a_bad_step_does_not_keep_the_accepted_command():
    from space_env.mission.session import MissionSession

    session = MissionSession("family_nominal.yaml")
    try:
        session.advance(-1.0, [MissionCommand("bad", "sentinel-a", "wait")], expected_revision=0)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert session.revision == 0
    assert session.world.commands.export() == {}
    assert session.world.now_sec() == 0.0
