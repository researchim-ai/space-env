"""Инструменты агента и базовые политики на одном исполнителе."""
from pathlib import Path

import pytest

from space_env.mission.session import RevisionError
from space_env.mission.tools import MissionTools, choose_request, run_baseline

_NOMINAL = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "family_nominal.yaml"


def test_forecast_does_not_move_the_clock_and_cancel_drops_a_pending_order():
    tools = MissionTools(_NOMINAL)
    before = tools.observe()
    ahead = tools.forecast("sentinel-a", 30.0)
    assert ahead["forecast_sec"] == pytest.approx(30.0)
    assert tools.session.world.now_sec() == pytest.approx(before["now_sec"])
    assert "health" not in ahead
    tools.submit(
        [{"command_id": "img-1", "actor_id": "sentinel-a", "kind": "imaging", "request_id": "kiruna-pass"}]
    )
    assert tools.observe()["pending"] == ["img-1"]
    dropped = tools.cancel("img-1")
    assert dropped["status"] == "dropped"
    assert tools.observe()["pending"] == []


def test_unknown_request_is_rejected_and_a_stale_revision_does_not_step():
    tools = MissionTools(_NOMINAL)
    tools.submit(
        [{"command_id": "bad", "actor_id": "sentinel-a", "kind": "imaging", "request_id": "no-such"}]
    )
    report = tools.advance(30.0, expected_revision=0)
    record = tools.session.world.commands.get("bad")
    assert record["status"] == "rejected"
    assert report["revision"] == 1
    with pytest.raises(RevisionError):
        tools.advance(30.0, expected_revision=0)
    assert tools.revision == 1


def test_fifo_and_priority_pick_different_open_requests():
    rows = [
        {"id": "early", "release_sec": 0.0, "deadline_sec": 100.0, "priority": 1, "released": True},
        {"id": "urgent", "release_sec": 10.0, "deadline_sec": 50.0, "priority": 5, "released": True},
        {"id": "closed", "release_sec": 0.0, "deadline_sec": 10.0, "priority": 9, "released": False},
    ]
    assert choose_request(rows, "fifo")["id"] == "early"
    assert choose_request(rows, "deadline")["id"] == "urgent"
    assert choose_request(rows, "priority")["id"] == "urgent"


def test_baseline_runs_the_nominal_pass_through_the_same_tools():
    report = run_baseline(_NOMINAL, "priority", step_sec=30.0, steps=4, actor="sentinel-a")
    assert report["revision"] == 4
    assert report["now_sec"] == pytest.approx(120.0)
    assert {item["request_id"] for item in report["statuses"]} == {"kiruna-pass"}
