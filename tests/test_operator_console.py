"""Пульт человека на живой смене OperatorShift."""
import json

import pytest

from space_env.api.server import handle_post
from space_env.rl.rewards import load_rewards


def _post(path: str, payload: dict | None = None):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    status, raw, kind = handle_post(path, body)
    return status, json.loads(raw.decode()), kind


def test_console_rejects_a_command_the_window_does_not_allow_and_keeps_a_worse_award():
    status, opened, kind = _post("/api/shift/reset", {"seed": 0})
    assert status == 200 and "json" in kind
    assert opened["phase"] == "award"
    assert opened["window"]["station"] == "Svalbard"
    assert opened["focus_sec"] == min(row["start_sec"] for row in opened["window"]["rows"])
    assert opened["queue"]
    assert opened["queue"][0]["current"] is True
    assert opened["queue"][0]["start_sec"] == opened["focus_sec"]
    assert len(opened["queue"]) == opened["remaining_windows"]
    assert any(item["kind"] == "maneuver" and item["vehicle"] == "sentinel-2a" for item in opened["plan"])
    assert sum(row["count"] for row in opened["imaging_planned"]) > 0
    assert {row["vehicle"] for row in opened["ledger"]} == set(opened["vehicles"])
    assert all(row["images"] == 0 and row["downlink_mb"] == 0.0 for row in opened["ledger"])
    assert len(opened["window"]["rows"]) == 2
    wait = next(item for item in opened["actions"] if item["id"] == 0)
    assert wait["blocked"]
    status, refused, _kind = _post("/api/shift/step", {"action": 0})
    weights = load_rewards()["operator_shift"]
    assert status == 200
    assert refused["terminated"] is True
    assert refused["unsafe"] is True
    assert refused["violations"]
    assert refused["reward"] == 0.0
    assert refused["success"] is False
    status, blocked, _kind = _post("/api/shift/step", {"action": 0})
    assert status == 409 and "открыта" in blocked["error"]

    status, again, _kind = _post("/api/shift/reset")
    assert status == 200 and again["terminated"] is False
    rows = again["window"]["rows"]
    keeper = min(range(len(rows)), key=lambda index: (-rows[index]["ready_bits"], rows[index]["start_sec"], index))
    loser = rows[1 - keeper]["vehicle"]
    action = next(item for item in again["actions"] if item["label"] == f"Зеркало {loser}")
    assert action["blocked"] is None
    status, stepped, _kind = _post("/api/shift/step", {"action": action["id"]})
    assert status == 200
    assert stepped["terminated"] is False
    assert stepped["unsafe"] is False
    assert stepped["wrong_awards"] == 1
    assert stepped["reward"] == -weights["missed"]
    assert stepped["phase"] == "award"
    assert any("занято" in note and loser in note for note in stepped["notes"])
    assert stepped["focus_sec"] == min(row["start_sec"] for row in stepped["window"]["rows"])
    assert stepped["queue"][0]["current"] is True
    assert len(stepped["queue"]) == stepped["remaining_windows"]
    assert len(stepped["queue"]) < len(again["queue"])

    def comms(state):
        return [
            (item["vehicle"], item["name"], item["start_sec"], item["required"])
            for item in state["plan"]
            if item["kind"] == "comms"
        ]

    assert comms(again) != comms(stepped)
    assert any("Зеркало занято" in item["reason"] and loser in item["reason"] for item in stepped["plan"])
    opened_burn = next(item for item in again["journal"] if item["kind"] == "maneuver")
    assert opened_burn["vehicle"] == "sentinel-2a" and opened_burn["status"] == "pending"

    def journal_comms(state):
        return [
            (item["vehicle"], item["name"], item["start_sec"], item["status"])
            for item in state["journal"]
            if item["kind"] == "comms"
        ]

    assert journal_comms(again) != journal_comms(stepped)
    assert any(
        item["status"] == "denied" and "Зеркало занято" in item["note"] and loser in item["note"]
        for item in stepped["journal"]
    )


_JOB_ACTION = {"imaging": 1, "comms": 2, "maneuver": 3}


def _next_action(state: dict) -> int:
    if state["phase"] == "award":
        return _keeper_action(state)
    return _JOB_ACTION[state["job"]["kind"]]


def _hot(state: dict) -> dict:
    return next(row for row in state["screening"]["rows"] if row["name"] == "HOT")


def _keeper_action(state: dict) -> int:
    rows = state["window"]["rows"]
    index = min(range(len(rows)), key=lambda item: (-rows[item]["ready_bits"], rows[item]["start_sec"], item))
    label = f"Зеркало {rows[index]['vehicle']}"
    return next(item["id"] for item in state["actions"] if item["label"] == label)


def test_console_burn_reduces_the_delta_v_left_on_the_ledger():
    status, state, _kind = _post("/api/shift/reset", {"seed": 0})
    assert status == 200
    assert state["screening"]["decision"] == "pending"
    assert state["screening"]["arc"] == "coast"
    opened = _hot(state)
    assert opened["coast_pc"] > 1.0e-4 > opened["flown_pc"]
    assert opened["active_pc"] == pytest.approx(opened["coast_pc"])
    for _ in range(16):
        if state["phase"] != "award" or state["terminated"]:
            break
        status, state, _kind = _post("/api/shift/step", {"action": _keeper_action(state)})
        assert status == 200 and state["unsafe"] is False
    assert state["phase"] == "execute"
    assert state["job"]["kind"] == "maneuver"
    assert state["screening"]["decision"] == "pending"
    before = next(row for row in state["ledger"] if row["vehicle"] == "sentinel-2a")
    status, state, _kind = _post("/api/shift/step", {"action": 3})
    assert status == 200
    assert state["burned"] is True
    assert state["terminated"] is False
    after = next(row for row in state["ledger"] if row["vehicle"] == "sentinel-2a")
    assert before["dv_left_m_s"] - after["dv_left_m_s"] == pytest.approx(1.0, abs=0.05)
    assert state["screening"]["decision"] == "done"
    assert state["screening"]["arc"] == "flown"
    burned = _hot(state)
    assert burned["active_pc"] == pytest.approx(burned["flown_pc"])
    assert burned["active_pc"] < 1.0e-4
    assert "уведена" in state["screening"]["note"]
    burned_row = next(item for item in state["journal"] if item["kind"] == "maneuver")
    assert burned_row["status"] == "done"
    assert set(state["flight"]["vehicles"]) == set(state["vehicles"])
    assert state["flight"]["clocks"]["sentinel-2a"] >= 0.5
    clocks = list(state["flight"]["clocks"].values())
    assert max(clocks) - min(clocks) < 2.0
    assert len(state["flight"]["vehicles"]["sentinel-2a"][-1]["r"]) == 3


def test_console_skips_a_job_and_keeps_the_shift_open():
    status, state, _kind = _post("/api/shift/reset", {"seed": 0})
    assert status == 200
    for _ in range(16):
        if state["phase"] != "award" or state["terminated"]:
            break
        status, state, _kind = _post("/api/shift/step", {"action": _keeper_action(state)})
        assert status == 200 and state["unsafe"] is False
    assert state["job"]["kind"] == "maneuver"
    status, state, _kind = _post("/api/shift/step", {"action": 0})
    weights = load_rewards()["operator_shift"]
    assert status == 200
    assert state["terminated"] is False
    assert state["unsafe"] is False
    assert state["burned"] is False
    assert state["done"] == 0
    assert state["phase"] == "execute"
    assert state["reward"] == pytest.approx(-weights["missed"])
    assert state["job"]["kind"] != "maneuver"
    assert "пропущено" in state["log"][-1]["summary"]
    assert sum("раздача" in row["summary"] for row in state["log"][:-1]) == state["awards"]
    assert state["screening"]["decision"] == "skipped"
    assert state["screening"]["arc"] == "coast"
    skipped = _hot(state)
    assert skipped["active_pc"] == pytest.approx(skipped["coast_pc"])
    assert skipped["active_pc"] > 1.0e-4
    assert "пропущен" in state["screening"]["note"]
    skipped_row = next(item for item in state["journal"] if item["kind"] == "maneuver")
    assert skipped_row["status"] == "skipped"
    assert skipped_row["vehicle"] == "sentinel-2a"


def test_console_closes_the_shift_when_dishes_follow_the_buffer():
    status, state, _kind = _post("/api/shift/reset", {"seed": 0})
    assert status == 200
    for _ in range(2000):
        if state["terminated"]:
            break
        status, state, _kind = _post("/api/shift/step", {"action": _next_action(state)})
        assert status == 200 and state["unsafe"] is False
    assert state["terminated"] is True
    assert state["success"] is True
    assert state["wrong_awards"] == 0
    assert state["done"] == state["jobs"]
    assert state["jobs"] > 1
    for row in state["ledger"]:
        assert row["images"] > 0
        assert row["downlink_mb"] > 0.0
    assert len(state["log"]) == state["awards"] + state["done"]
    assert state["log"][-1]["success"] is True
    assert "успех" in state["log"][-1]["summary"]
    assert state["screening"]["decision"] == "done"
    assert state["screening"]["arc"] == "flown"
    assert sum(item["status"] == "done" for item in state["journal"]) == state["jobs"]
    assert all(item["status"] != "skipped" for item in state["journal"])
    assert any(item["status"] == "denied" for item in state["journal"])
    assert state["flight"]["time_sec"] > 10_000
    assert set(state["flight"]["clocks"]) == set(state["vehicles"])
