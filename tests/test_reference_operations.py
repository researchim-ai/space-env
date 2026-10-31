"""Actual GCRS-window downlink resource outcomes and RL episode contracts."""
import copy
import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from space_env.core.frames import ecef_to_geodetic
from space_env.data_generators.reference_operations import (
    ReferenceOperationsRecorder,
    replay_reference_operations,
)
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.simulation.snapshot import digest


@pytest.fixture(scope="module")
def spec():
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    result = propagate(ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 0, 7546.])),
                       30, context, sample_sec=5)
    orbit = {**result, "sha256": digest(result)}
    lat, lon, _height = ecef_to_geodetic(context.matrix(epoch) @ np.array([7e6, 0, 0]))
    return {"schema": "space-env.reference-downlink-spec/1", "ephemerides": {"a": orbit, "b": orbit},
            "vehicles": {name: {"memory_bits": 10e6, "battery_wh": 1, "base_power_w": 0,
                                 "tx_power_w": 0, "rate_bps": 1e6} for name in ("a", "b")},
            "stations": [{"name": "Ground", "lat_deg": lat, "lon_deg": lon}],
            "products": [{"id": name, "actor": name, "size_bits": 2e6, "deadline_sec": 20} for name in ("a", "b")]}


@pytest.mark.parametrize("steps", [[10], [1]*10, [3, 7]])
def test_terminal_handover_at_product_completion_not_external_step_end(spec, steps):
    mission = ReferenceDownlinkMission(spec)
    for dt in steps:
        mission.step(dt, {"a": "Ground", "b": "Ground"})
    result = mission.report()
    assert result["on_time_bits"] == 4e6
    assert {p["id"]: p["completed_sec"] for p in result["products"]} == {"a": 2, "b": 4}
    assert sum(r["bits"] for r in result["receipts"]) == 4e6
    for interval in mission.intervals:
        assert len(interval["transmitters"]) <= 1


def test_battery_floor_cuts_transmission_at_physical_time(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(initial_soc=.5, min_tx_soc=.25, tx_power_w=900)
    mission = ReferenceDownlinkMission(data)
    report = mission.step(10, {"a": "Ground"})
    assert sum(r["bits"] for r in report["receipts"]) == pytest.approx(1e6)
    assert max(r["end_sec"] for r in report["receipts"]) == pytest.approx(1)
    assert report["energy_wh"]["a"] == pytest.approx(.25)
    assert report["on_time_bits"] == 0


def test_readiness_outage_deadline_and_memory_are_real_constraints(spec):
    data = copy.deepcopy(spec)
    data["products"][0].update(ready_sec=3, deadline_sec=7)
    data["outages"] = [{"station": "Ground", "start_sec": 2, "end_sec": 6}]
    mission = ReferenceDownlinkMission(data)
    report = mission.step(10, {"a": "Ground"})
    a = next(p for p in report["products"] if p["id"] == "a")
    assert a["status"] == "delivered_late" and a["completed_sec"] == 8
    assert report["on_time_bits"] == 0
    assert all(r["start_sec"] >= 6 for r in report["receipts"])
    assert report["observation"]["actors"]["a"]["stored_bits"] == 0
    data["vehicles"]["a"]["memory_bits"] = 1e6
    with pytest.raises(ValueError, match="memory"):
        ReferenceDownlinkMission(data)


def test_bad_action_and_internal_failure_leave_no_partial_state(spec, monkeypatch):
    mission = ReferenceDownlinkMission(spec)
    before = mission.report()
    with pytest.raises(ValueError):
        mission.step(5, {"a": "unknown"})
    assert mission.report() == before
    original = mission._generation

    def injected(actor, midpoint):
        if mission.clock > 0:
            raise RuntimeError("injected")
        return original(actor, midpoint)

    monkeypatch.setattr(mission, "_generation", injected)
    with pytest.raises(RuntimeError):
        mission.step(10, {"a": "Ground"})
    assert mission.report() == before
    assert mission.intervals == []


def test_replay_and_tampering_are_checked(spec, tmp_path):
    recorder = ReferenceOperationsRecorder(spec, lineage="downlink-family")
    recorder.step(10, {"a": "Ground", "b": "Ground"})
    path = tmp_path / "run.json"
    recorder.save(path)
    assert replay_reference_operations(path)["ok"]
    record = json.loads(path.read_text())
    record["steps"][0]["result"]["on_time_bits"] = 9e6
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    path.write_text(json.dumps(record))
    assert not replay_reference_operations(path)["ok"]


def test_gym_seed_reset_actions_and_success_reward(spec):
    env = ReferenceDownlinkEnv(spec, dt_sec=10)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    _, reward, terminated, truncated, report = env.step(np.array([1, 1]))
    assert reward == 4 and terminated and not truncated
    assert report["on_time_bits"] == 4e6
    with pytest.raises(RuntimeError):
        env.step(np.array([1, 1]))
    obs, _ = env.reset(seed=7)
    assert obs["stored_bits"].tolist() == [2e6, 2e6]
    before = env.mission.clock
    with pytest.raises(ValueError):
        env.step(np.array([9, 1]))
    assert env.mission.clock == before


def test_destination_and_two_terminals(spec):
    data = copy.deepcopy(spec)
    data["stations"].append({**data["stations"][0], "name": "Other"})
    data["products"][0]["destination"] = "Other"
    mission = ReferenceDownlinkMission(data)
    report = mission.step(5, {"a": "Ground"})
    assert not report["receipts"]
    data["stations"][0]["terminals"] = 2
    data["products"][0]["destination"] = ""
    report = ReferenceDownlinkMission(data).step(5, {"a": "Ground", "b": "Ground"})
    assert all(p["completed_sec"] == 2 for p in report["products"])


def test_sun_tracking_energy_is_reference_geometry_coupled_and_balanced(spec):
    from space_env.reference.orbit import _ephemerides

    data = copy.deepcopy(spec)
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    sun, _ = _ephemerides(epoch)
    r = sun/np.linalg.norm(sun)*7e6
    v = np.cross(r, [0, 0, 1.])
    v = v/np.linalg.norm(v)*7546
    result = propagate(ReferenceState(epoch, r, v), 30, FrameContext(EOPSeries.packaged()), sample_sec=5)
    data["ephemerides"] = {name: {**result, "sha256": digest(result)} for name in ("a", "b")}
    for name in ("a", "b"):
        data["vehicles"][name].update(initial_soc=.5, solar_area_m2=1)
    mission = ReferenceDownlinkMission(data)
    report = mission.step(10, {})
    assert report["energy_wh"]["a"] > .5
    assert report["energy_ledger"]["a"]["generated_wh"] > 0
    assert abs(report["energy_ledger"]["a"]["balance_error_wh"]) < 1e-12


def test_cli_portable_operations_and_existing_rollout_writer(spec, tmp_path, capsys):
    from space_env.cli import main
    from space_env.rl.trajectories import rollout_episode

    source, plan, target = (tmp_path / name for name in ("spec.json", "plan.json", "run.json"))
    source.write_text(json.dumps(spec))
    plan.write_text(json.dumps({"steps": [{"dt_sec": 10, "actions": {"a": "Ground", "b": "Ground"}}]}))
    main(["reference-run", str(source), str(plan), str(target), "--lineage", "cli-reference"])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    source.unlink()
    plan.unlink()
    main(["reference-replay", str(target)])
    assert json.loads(capsys.readouterr().out)["ok"]
    rows = rollout_episode(ReferenceDownlinkEnv(spec, dt_sec=10), seed=7,
                           policy=lambda observation: np.array([1, 1]))
    assert rows[-1]["success"] and rows[-1]["reward"] == 4
    assert isinstance(rows[0]["observation"], dict)
    json.dumps(rows, allow_nan=False)


def test_solar_sampling_does_not_change_with_external_step_partition(spec):
    data = copy.deepcopy(spec)
    for actor in ("a", "b"):
        data["vehicles"][actor].update(initial_soc=.5, solar_area_m2=1)
    left, right = ReferenceDownlinkMission(data), ReferenceDownlinkMission(data)
    left.step(10, {})
    for _ in range(10):
        right.step(1, {})
    assert left.report()["energy_wh"] == pytest.approx(right.report()["energy_wh"])


def test_baseline_power_shortfall_is_accounted_without_creating_charge(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(initial_soc=.5, base_power_w=900)
    report = ReferenceDownlinkMission(data).step(10, {})
    assert report["energy_wh"]["a"] == 0
    assert report["unserved_base_wh"]["a"] == pytest.approx(2)
    assert report["energy_ledger"]["a"]["consumed_wh"] == pytest.approx(.5)
    assert abs(report["energy_ledger"]["a"]["balance_error_wh"]) < 1e-12


def test_agent_observes_product_deadlines_readiness_and_actor_mapping(spec):
    env = ReferenceDownlinkEnv(spec)
    observation, info = env.reset(seed=1)
    assert observation["products"].shape == (2, 6)
    assert observation["products"][:, 2].tolist() == [20, 20]
    assert info["product_ids"] == ["a", "b"]
    assert info["product_fields"][1] == "ready_sec"
    assert env.observation_space.contains(observation)


def test_tiny_step_cannot_silently_produce_an_infinite_episode(spec):
    mission = ReferenceDownlinkMission(spec)
    with pytest.raises(ValueError):
        mission.step(1e-12, {})
    assert mission.clock == 0
    with pytest.raises(ValueError):
        ReferenceDownlinkEnv(spec, dt_sec=1e-12)


def test_scheduler_urgent_actor_wins_and_plan_reexecutes(spec):
    from space_env.reference.scheduling import plan_downlinks

    data = copy.deepcopy(spec)
    data["products"][1]["deadline_sec"] = 2
    result = plan_downlinks(data)
    assert result["verified"] and not result["expert"]
    assert not result["optimality_proven"]
    assert result["plan"]["steps"][0]["actions"] == {"b": "Ground"}
    assert result["report"]["on_time_bits"] == 4e6
    assert result["unfulfilled"] == []
    mission = ReferenceDownlinkMission(data)
    for row in result["plan"]["steps"]:
        mission.step(row["dt_sec"], row["actions"])
    assert mission.report() == result["report"]
    assert plan_downlinks(data) == result
    assert data["products"][0]["size_bits"] == 2e6


def test_scheduler_outage_readiness_and_incomplete_is_not_infeasibility_proof(spec):
    from space_env.reference.scheduling import plan_downlinks

    data = copy.deepcopy(spec)
    data["products"][0].update(ready_sec=4, deadline_sec=5)
    data["outages"] = [{"station": "Ground", "start_sec": 0, "end_sec": 3}]
    result = plan_downlinks(data)
    assert result["report"]["on_time_bits"] == 2e6
    assert result["unfulfilled"] == [{"id": "a", "status": "incomplete",
                                       "reason": "not_completed_on_time_by_baseline"}]
    assert all(r["start_sec"] >= 3 for r in result["report"]["receipts"])
    assert result["report"]["time_sec"] == 30


def test_scheduler_energy_floor_and_terminal_capacity(spec):
    from space_env.reference.scheduling import plan_downlinks

    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(initial_soc=.1, min_tx_soc=.1, tx_power_w=20)
    result = plan_downlinks(data)
    assert result["report"]["on_time_bits"] == 2e6
    assert all(r["actor"] == "b" for r in result["report"]["receipts"])
    data = copy.deepcopy(spec)
    data["stations"][0]["terminals"] = 2
    result = plan_downlinks(data)
    assert result["plan"]["steps"][0]["actions"] == {"a": "Ground", "b": "Ground"}
    assert result["report"]["time_sec"] == 2


@pytest.mark.parametrize("kwargs", [{"decision_sec": 0}, {"decision_sec": float("nan")},
                                    {"max_steps": True}, {"max_steps": 1}])
def test_scheduler_rejects_invalid_or_exhausted_budget(spec, kwargs):
    from space_env.reference.scheduling import plan_downlinks

    with pytest.raises(ValueError):
        plan_downlinks(spec, **kwargs)


def test_cli_generated_schedule_runs_as_portable_record(spec, tmp_path, capsys):
    from space_env.cli import main

    source, plan, report, record = [tmp_path / n for n in ("spec.json", "plan.json", "report.json", "run.json")]
    source.write_text(json.dumps(spec))
    main(["reference-plan", str(source), str(plan), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["verified"]
    main(["reference-run", str(source), str(plan), str(record), "--lineage", "scheduled-baseline"])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    assert replay_reference_operations(record)["ok"]
    previous = plan.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        main(["reference-plan", str(source), str(plan), "--report", str(report), "--max-steps", "1"])
    assert plan.read_bytes() == previous


def test_scheduler_respects_priority_destination_and_onboard_queue(spec):
    from space_env.reference.scheduling import plan_downlinks

    data = copy.deepcopy(spec)
    data["stations"].append({**data["stations"][0], "name": "Other"})
    data["products"][1].update(priority=10, destination="Other")
    result = plan_downlinks(data)
    assert result["plan"]["steps"][0]["actions"]["b"] == "Other"
    assert result["report"]["on_time_bits"] == 4e6
    # An impossible high-priority head blocks the lower-priority product: the
    # planner must not pretend it can command arbitrary product selection.
    data = copy.deepcopy(spec)
    data["products"][0].update(priority=10, deadline_sec=1)
    data["products"].append({"id": "small", "actor": "a", "size_bits": 1e6, "deadline_sec": 20})
    result = plan_downlinks(data)
    assert result["report"]["on_time_bits"] == 2e6
    assert {p["id"] for p in result["unfulfilled"]} == {"a", "small"}


def test_explicit_product_bypasses_queue_without_transmitting_next_product(spec):
    data = copy.deepcopy(spec)
    data["products"].append({"id": "small", "actor": "a", "size_bits": 1e6,
                             "deadline_sec": 10})
    mission = ReferenceDownlinkMission(data)
    report = mission.step(5, {"a": "Ground"}, products={"a": "small"})
    assert mission.remaining["a"] == 2e6 and mission.remaining["small"] == 0
    assert report["on_time_bits"] == 1e6
    assert {r["product_id"] for r in report["receipts"]} == {"small"}


@pytest.mark.parametrize("actions,products", [({"a": "Ground"}, {"a": "b"}),
                                            ({"a": None}, {"a": "a"}),
                                            ({"a": "Ground"}, {"ghost": "a"}),
                                            ({"a": "Ground"}, {"a": "missing"})])
def test_invalid_product_selection_is_atomic(spec, actions, products):
    mission = ReferenceDownlinkMission(spec)
    before = mission.report()
    with pytest.raises(ValueError, match="selected product"):
        mission.step(5, actions, products=products)
    assert mission.report() == before


def test_selected_product_readiness_and_destination_are_binding(spec):
    data = copy.deepcopy(spec)
    data["products"][0]["ready_sec"] = 3
    mission = ReferenceDownlinkMission(data)
    result = mission.step(10, {"a": "Ground"}, products={"a": "a"})
    assert min(r["start_sec"] for r in result["receipts"]) == 3
    assert result["products"][0]["completed_sec"] == 5
    data["stations"].append({**data["stations"][0], "name": "Other"})
    data["products"][0]["destination"] = "Other"
    mission = ReferenceDownlinkMission(data)
    with pytest.raises(ValueError, match="compatible"):
        mission.step(5, {"a": "Ground"}, products={"a": "a"})
    assert mission.clock == 0


def test_explicit_planner_avoids_impossible_queue_head(spec):
    from space_env.reference.scheduling import plan_downlinks

    data = copy.deepcopy(spec)
    data["products"][0].update(priority=10, deadline_sec=1)
    data["products"].append({"id": "small", "actor": "a", "size_bits": 1e6,
                             "deadline_sec": 10})
    implicit = plan_downlinks(data)
    explicit = plan_downlinks(data, select_products=True)
    assert implicit["report"]["on_time_bits"] == 2e6
    assert explicit["report"]["on_time_bits"] == 3e6
    assert explicit["algorithm"] == "priority-slack-explicit/1"


def test_replanning_preserves_partial_inventory_energy_and_source(spec):
    from space_env.reference.scheduling import (
        mission_fingerprint,
        plan_remaining_downlinks,
    )

    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["tx_power_w"] = 20
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {"a": "Ground"}, products={"a": "a"})
    before = mission_fingerprint(mission)
    result = plan_remaining_downlinks(mission)
    assert mission_fingerprint(mission) == before
    assert mission.remaining["a"] == 1e6
    assert result["plan"]["start_sec"] == 1
    assert result["plan"]["initial_state_sha256"] == before
    for row in result["plan"]["steps"]:
        mission.step(row["dt_sec"], row["actions"], products=row.get("products"))
    assert mission.report() == result["report"]
    assert sum(r["bits"] for r in mission.receipts) == 4e6
    assert mission.energy_wh["a"] == pytest.approx(1-40/3600)


def test_recorder_restore_append_and_tamper_rejection(spec, tmp_path):
    path = tmp_path / "run.json"
    recorder = ReferenceOperationsRecorder(spec, lineage="rolling-family")
    recorder.step(1, {"a": "Ground"}, products={"a": "a"})
    recorder.save(path)
    resumed = ReferenceOperationsRecorder.from_record(path)
    assert resumed.mission.report() == recorder.mission.report()
    resumed.step(5, {"a": "Ground", "b": "Ground"})
    resumed.save(tmp_path / "continued.json")
    assert resumed.lineage == "rolling-family" and len(resumed.steps) == 2
    assert replay_reference_operations(tmp_path / "continued.json")["ok"]
    record = json.loads(path.read_text())
    record["steps"][0]["products"] = {"a": "b"}
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        ReferenceOperationsRecorder.from_record(path)


def test_product_selection_gym_action_space_checker_and_reward(spec):
    env = ReferenceDownlinkEnv(spec, dt_sec=5, product_selection=True)
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=7)
    assert info["action_choices"]["a"][2] == {"station": "Ground", "product": "a"}
    _, reward, terminated, truncated, _ = env.step(np.array([2, 2]))
    assert reward == 4 and terminated and not truncated


def test_cli_rolling_plan_resume_and_wrong_state_guard(spec, tmp_path, capsys):
    from space_env.cli import main

    recorder = ReferenceOperationsRecorder(spec, lineage="rolling-cli")
    recorder.step(1, {"a": "Ground"}, products={"a": "a"})
    record, plan, report, output = [tmp_path / p for p in ("prefix.json", "plan.json", "report.json", "output.json")]
    recorder.save(record)
    main(["reference-replan", str(record), str(plan), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["verified"]
    main(["reference-resume", str(record), str(plan), str(output)])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    assert replay_reference_operations(output)["ok"]
    saved = output.read_bytes()
    rejected = tmp_path / "rejected.json"
    rejected.write_text("preserve-existing-output")
    with pytest.raises(SystemExit):
        main(["reference-resume", str(output), str(plan), str(rejected)])
    assert rejected.read_text() == "preserve-existing-output"
    assert output.read_bytes() == saved
    alternate = ReferenceOperationsRecorder(spec, lineage="rolling-cli")
    alternate.step(1, {})
    alternate_path = tmp_path / "same-time-different-inventory.json"
    alternate.save(alternate_path)
    with pytest.raises(SystemExit):
        main(["reference-resume", str(alternate_path), str(plan), str(rejected)])
    assert rejected.read_text() == "preserve-existing-output"
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(spec))
    with pytest.raises(SystemExit):
        main(["reference-run", str(source), str(plan), str(output), "--lineage", "wrong-start"])
    assert output.read_bytes() == saved


@pytest.mark.parametrize("field", ["runtime", "final"])
def test_resume_rejects_incompatible_or_forged_state_even_with_new_checksum(spec, tmp_path, field):
    recorder = ReferenceOperationsRecorder(spec, lineage="resume-validation")
    recorder.step(1, {})
    path = tmp_path / "run.json"
    record = recorder.save(path)
    if field == "runtime":
        record["runtime"]["frame_contract"] = "wrong-frame"
    else:
        record["final"]["energy_wh"]["a"] = 0
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cannot resume"):
        ReferenceOperationsRecorder.from_record(path)


def test_replan_after_delivery_has_no_duplicate_data_or_clock_reset(spec):
    from space_env.reference.scheduling import plan_remaining_downlinks

    mission = ReferenceDownlinkMission(spec)
    mission.step(5, {"a": "Ground", "b": "Ground"})
    result = plan_remaining_downlinks(mission)
    assert result["plan"]["steps"] == []
    assert result["start_sec"] == 5
    assert result["report"] == mission.report()


def _event(kind, time=1., payload=None, id="evt"):
    return {"id": id, "type": kind, "time_sec": time, "source": "operator-test",
            "reason": "operational update", "payload": payload or {}}


def test_event_arrival_uses_free_memory_and_has_no_pre_event_effect(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {"a": "Ground"})
    before = mission.energy_wh.copy()
    event = _event("onboard_product_added", payload={"id": "urgent", "actor": "a", "size_bits": 1e6,
                                                   "ready_sec": 1, "deadline_sec": 4, "priority": 9})
    report = mission.apply_events([event])
    assert mission.clock == 1 and mission.energy_wh == before
    assert mission.remaining["urgent"] == 1e6
    assert report["observation"]["actors"]["a"]["stored_bits"] == 2e6
    assert report["events"] == [event]
    mission.step(1, {"a": "Ground"}, products={"a": "urgent"})
    assert mission.remaining["urgent"] == 0 and mission.remaining["a"] == 1e6


@pytest.mark.parametrize("mutation", ["past", "future", "unknown", "memory", "duplicate", "source"])
def test_event_batch_validation_rolls_back_everything(spec, mutation):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    first = _event("product_priority_changed", payload={"product_id": "a", "priority": 8}, id="first")
    bad = _event("onboard_product_added", payload={"id": "new", "actor": "a", "size_bits": 1e6,
                                                 "ready_sec": 1, "deadline_sec": 10}, id="second")
    if mutation == "past":
        bad["time_sec"] = 0
    elif mutation == "future":
        bad["time_sec"] = 2
    elif mutation == "unknown":
        bad["type"] = "imaginary-event"
    elif mutation == "memory":
        bad["payload"]["size_bits"] = 20e6
    elif mutation == "duplicate":
        bad["id"] = "first"
    else:
        bad["source"] = ""
    before = mission.report()
    with pytest.raises(ValueError):
        mission.apply_events([first, bad])
    assert mission.report() == before
    assert mission.products["a"].priority == 0


def test_outage_event_and_early_recovery_are_half_open_and_preserve_future_outage(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([
        _event("station_outage_added", payload={"station": "Ground", "start_sec": 1, "end_sec": 10}),
        _event("station_outage_added", payload={"station": "Ground", "start_sec": 12, "end_sec": 15}, id="future")])
    mission.step(2, {"a": "Ground"})
    assert not mission.receipts
    mission.apply_events([_event("station_recovered", time=3, payload={"station": "Ground"}, id="recovered")])
    mission.step(10, {"a": "Ground", "b": "Ground"})
    assert {r["start_sec"] for r in mission.receipts} >= {3}
    assert all(not (12 <= r["start_sec"] < 15) for r in mission.receipts)
    assert mission._visible("a", "Ground") is False
    assert mission.report()["on_time_bits"] == 4e6


def test_event_ids_cannot_be_reused_after_restore_and_tampering_is_detected(spec, tmp_path):
    recorder = ReferenceOperationsRecorder(spec, lineage="event-family")
    recorder.step(1, {})
    event = _event("product_priority_changed", payload={"product_id": "a", "priority": 7})
    recorder.inject([event])
    path = tmp_path / "event-run.json"
    record = recorder.save(path)
    assert record["schema"] == "space-env.reference-operations-run/2"
    assert replay_reference_operations(path)["ok"]
    restored = ReferenceOperationsRecorder.from_record(path)
    with pytest.raises(ValueError, match="duplicate"):
        restored.inject([event])
    record["steps"][1]["events"][0]["payload"]["priority"] = 99
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    path.write_text(json.dumps(record))
    assert not replay_reference_operations(path)["ok"]


def test_adaptation_replans_new_data_and_fault_with_preserved_prefix(spec, tmp_path):
    from space_env.reference.adaptation import adapt_reference_run

    recorder = ReferenceOperationsRecorder(spec, lineage="adaptive-family")
    recorder.step(1, {"a": "Ground"})
    path = tmp_path / "prefix.json"
    recorder.save(path)
    original = path.read_bytes()
    events = [_event("onboard_product_added", payload={"id": "urgent", "actor": "a", "size_bits": 1e6,
                                                      "ready_sec": 1, "deadline_sec": 5, "priority": 10}),
              _event("station_outage_added", payload={"station": "Ground", "start_sec": 1, "end_sec": 2}, id="fault")]
    adapted, schedule = adapt_reference_run(path, events)
    assert path.read_bytes() == original
    assert schedule["verified"] and adapted.mission.report()["on_time_bits"] == 5e6
    assert adapted.mission.receipts[0] == recorder.mission.receipts[0]
    assert all(r["end_sec"] <= 1 or r["start_sec"] >= 2 for r in adapted.mission.receipts)
    adapted.save(tmp_path / "adapted.json")
    assert replay_reference_operations(tmp_path / "adapted.json")["ok"]


def test_cli_adapt_and_event_checkpoint(spec, tmp_path, capsys):
    from space_env.cli import main

    recorder = ReferenceOperationsRecorder(spec, lineage="cli-adapt")
    recorder.step(1, {})
    prefix, events, checkpoint, complete, report = [tmp_path / p for p in
                                                  ("prefix.json", "events.json", "checkpoint.json", "complete.json", "report.json")]
    recorder.save(prefix)
    events.write_text(json.dumps({"events": [_event("station_outage_added", payload={
        "station": "Ground", "start_sec": 1, "end_sec": 3})]}))
    main(["reference-events", str(prefix), str(events), str(checkpoint)])
    assert json.loads(capsys.readouterr().out)["report"]["time_sec"] == 1
    main(["reference-adapt", str(prefix), str(events), str(complete), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    assert replay_reference_operations(complete)["ok"]
    previous = complete.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        main(["reference-adapt", str(prefix), str(events), str(complete), "--report", str(report), "--max-steps", "1"])
    assert complete.read_bytes() == previous


def test_event_gym_fixed_slots_hidden_future_and_reaction_boundary(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    events = [_event("onboard_product_added", time=3, payload={"id": "urgent", "actor": "a",
                      "size_bits": 1e6, "ready_sec": 3, "deadline_sec": 10})]
    env = ReferenceEventDownlinkEnv(spec, events, product_slots=3, dt_sec=10)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert obs["slot_active"].sum() == 2 and info["report"]["events"] == []
    assert "urgent" not in json.dumps(info)
    obs, reward, terminated, truncated, info = env.step(np.array([1, 1]))
    assert info["elapsed_sec"] == 3 and not terminated and not truncated
    assert obs["slot_active"].sum() == 3
    assert info["product_ids"]["a"] == ["a", "urgent"]
    # station block: 1 automatic, 2 slot1, 3 slot2, 4 slot3.
    obs, reward, terminated, truncated, info = env.step(np.array([3, 1]))
    assert reward == 3 and terminated and not truncated and info["success"]
    assert env.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        env.step(np.array([1, 1]))


def test_event_gym_invalid_slot_waits_and_failed_event_rolls_back_step(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    bad = _event("onboard_product_added", time=3, payload={"id": "bad", "actor": "a",
                 "size_bits": 20e6, "ready_sec": 3, "deadline_sec": 10})
    env = ReferenceEventDownlinkEnv(spec, [bad], product_slots=3, dt_sec=10)
    env.reset(seed=1)
    before = env.mission.report()
    with pytest.raises(ValueError, match="memory"):
        env.step(np.array([1, 1]))
    assert env.mission.report() == before and env._cursor == 0
    env = ReferenceEventDownlinkEnv(spec, [], product_slots=3, dt_sec=1)
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.array([4, 0]))
    assert reward == 0 and info["invalid_actors"] == ["a"]
    assert env.mission.remaining["a"] == 2e6


def test_event_replanning_changes_state_binding(spec):
    from space_env.reference.scheduling import mission_fingerprint

    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    before = mission_fingerprint(mission)
    mission.apply_events([_event("product_priority_changed", payload={"product_id": "a", "priority": 9})])
    assert mission_fingerprint(mission) != before


def test_dynamic_inventory_ledger_uses_validated_numeric_values(spec):
    data = copy.deepcopy(spec)
    data["products"][0]["size_bits"] = "2000000"
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {"a": "Ground"})
    mission.apply_events([_event("onboard_product_added", payload={"id": "urgent", "actor": "a",
                                 "size_bits": "1000000", "ready_sec": 1, "deadline_sec": 10})])
    mission.step(5, {"a": "Ground"})
    ledger = mission.report()["inventory_ledger"]["a"]
    assert ledger == {"initial_bits": 2e6, "injected_bits": 1e6, "delivered_bits": 3e6,
                      "remaining_bits": 0, "discarded_bits": 0, "balance_error_bits": 0}


def test_failed_event_report_does_not_commit_staged_changes(spec, monkeypatch):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    before = mission.report()
    original = mission.report

    def fail():
        raise RuntimeError("injected report failure")

    monkeypatch.setattr(mission, "report", fail)
    with pytest.raises(RuntimeError, match="report failure"):
        mission.apply_events([_event("product_priority_changed", payload={"product_id": "a", "priority": 9})])
    monkeypatch.setattr(mission, "report", original)
    assert mission.report() == before


@pytest.mark.parametrize("change", ["duplicate", "unordered", "past", "capacity"])
def test_event_gym_rejects_invalid_tape_before_episode(spec, change):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    event = _event("onboard_product_added", time=3, payload={"id": "new", "actor": "a", "size_bits": 1e6,
                                                           "ready_sec": 3, "deadline_sec": 10})
    events, slots = [event], 3
    if change == "duplicate":
        events.append(copy.deepcopy(event))
    elif change == "unordered":
        second = copy.deepcopy(event)
        second.update(id="second", time_sec=2)
        events.append(second)
    elif change == "past":
        event["time_sec"] = -1
    else:
        slots = 1
    with pytest.raises(ValueError):
        ReferenceEventDownlinkEnv(spec, events, product_slots=slots)


def test_cancel_partial_product_preserves_delivered_bits_and_balances_discard(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {"a": "Ground"})
    report = mission.apply_events([_event("product_cancelled", payload={"product_id": "a"})])
    a = next(p for p in report["products"] if p["id"] == "a")
    assert a["status"] == "cancelled" and a["completed_sec"] is None and a["cancelled_sec"] == 1
    assert a["delivered_bits"] == 1e6 and a["discarded_bits"] == 1e6 and a["remaining_bits"] == 0
    assert report["on_time_bits"] == 0
    assert report["inventory_ledger"]["a"]["balance_error_bits"] == 0
    assert report["observation"]["actors"]["a"]["stored_bits"] == 0
    mission.step(5, {"a": "Ground"}, products={"a": "a"})
    assert sum(r["bits"] for r in mission.receipts) == 1e6


def test_cancel_then_arrive_reuses_memory_without_reusing_identity(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["memory_bits"] = 2e6
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    mission.apply_events([_event("product_cancelled", payload={"product_id": "a"}),
                          _event("onboard_product_added", id="replacement", payload={"id": "replacement", "actor": "a",
                                 "size_bits": 2e6, "ready_sec": 1, "deadline_sec": 10})])
    assert mission.remaining["replacement"] == 2e6 and mission.discarded["a"] == 2e6
    assert mission.report()["inventory_ledger"]["a"]["balance_error_bits"] == 0
    with pytest.raises(ValueError):
        mission.apply_events([_event("onboard_product_added", id="reuse-id", payload={"id": "a", "actor": "a",
                             "size_bits": 1e6, "ready_sec": 1, "deadline_sec": 10})])


def test_deadline_update_preserves_history_and_cannot_rewrite_resolved_product(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {"a": "Ground"})
    receipts = copy.deepcopy(mission.receipts)
    mission.apply_events([_event("product_deadline_changed", payload={"product_id": "a", "deadline_sec": 2})])
    mission.step(1, {"a": "Ground"})
    assert mission.report()["products"][0]["status"] == "delivered_on_time"
    assert mission.receipts[:len(receipts)] == receipts
    with pytest.raises(ValueError, match="resolved"):
        mission.apply_events([_event("product_deadline_changed", time=2, id="rewrite", payload={"product_id": "a", "deadline_sec": 3})])


@pytest.mark.parametrize("deadline", [0, 1, 31, float("inf")])
def test_bad_deadline_event_rolls_back(spec, deadline):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    before = mission.report()
    with pytest.raises(ValueError):
        mission.apply_events([_event("product_deadline_changed", payload={"product_id": "a", "deadline_sec": deadline})])
    assert mission.report() == before


def test_per_terminal_station_limit_and_transmitter_degradation_use_minimum(spec):
    data = copy.deepcopy(spec)
    data["stations"][0]["terminals"] = 2
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    mission.apply_events([_event("station_rate_changed", payload={"station": "Ground", "rate_bps": .5e6}),
                          _event("transmitter_rate_changed", id="tx", payload={"actor": "a", "rate_bps": .25e6})])
    report = mission.step(2, {"a": "Ground", "b": "Ground"})
    assert mission.remaining == {"a": 1.5e6, "b": 1e6}
    assert sum(r["bits"] for r in report["receipts"]) == 1.5e6
    mission.apply_events([_event("transmitter_rate_changed", time=3, id="repair", payload={"actor": "a", "rate_bps": 1e6})])
    assert mission.effective_rate("a", "Ground") == .5e6


def test_zero_rate_neither_delivers_nor_consumes_transmitter_energy(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["tx_power_w"] = 100
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    mission.apply_events([_event("station_rate_changed", payload={"station": "Ground", "rate_bps": 0})])
    mission.step(5, {"a": "Ground"})
    assert mission.energy_wh["a"] == 1 and not mission.receipts
    mission.apply_events([_event("station_rate_changed", time=6, id="repair", payload={"station": "Ground", "rate_bps": 1e6})])
    mission.step(2, {"a": "Ground"})
    assert mission.energy_wh["a"] == pytest.approx(1-200/3600)


@pytest.mark.parametrize("kind,payload", [
    ("transmitter_rate_changed", {"actor": "a", "rate_bps": 2e6}),
    ("transmitter_rate_changed", {"actor": "a", "rate_bps": -1}),
    ("station_rate_changed", {"station": "Ground", "rate_bps": True}),
    ("station_rate_changed", {"station": "missing", "rate_bps": 1}),
    ("product_cancelled", {"product_id": "missing"}),
])
def test_invalid_lifecycle_batch_does_not_commit_first_event(spec, kind, payload):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    before = mission.report()
    with pytest.raises(ValueError):
        mission.apply_events([_event("product_priority_changed", id="valid", payload={"product_id": "a", "priority": 10}),
                              _event(kind, payload=payload)])
    assert mission.report() == before


def test_planner_uses_effective_rate_and_skips_cancelled_and_blocked_data(spec):
    from space_env.reference.scheduling import plan_remaining_downlinks

    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_event("product_cancelled", payload={"product_id": "b"}),
                          _event("transmitter_rate_changed", id="degrade", payload={"actor": "a", "rate_bps": .5e6})])
    plan = plan_remaining_downlinks(mission)
    assert plan["report"]["products"][0]["completed_sec"] == 5
    assert plan["report"]["on_time_bits"] == 2e6
    assert plan["unfulfilled"] == [{"id": "b", "status": "cancelled", "reason": "cancelled_by_operator"}]
    mission.apply_events([_event("station_rate_changed", id="block", payload={"station": "Ground", "rate_bps": 0})])
    blocked = plan_remaining_downlinks(mission)
    assert blocked["report"]["on_time_bits"] == 0


def test_lifecycle_journal_cli_adapt_and_replay(spec, tmp_path, capsys):
    from space_env.cli import main

    recorder = ReferenceOperationsRecorder(spec, lineage="lifecycle-family")
    recorder.step(1, {"a": "Ground"})
    source, events, target, report = [tmp_path / p for p in ("prefix.json", "events.json", "run.json", "report.json")]
    recorder.save(source)
    events.write_text(json.dumps({"events": [
        _event("product_cancelled", payload={"product_id": "b"}),
        _event("product_deadline_changed", id="deadline", payload={"product_id": "a", "deadline_sec": 10}),
        _event("transmitter_rate_changed", id="tx", payload={"actor": "a", "rate_bps": .5e6}),
        _event("station_rate_changed", id="station", payload={"station": "Ground", "rate_bps": .75e6})]}))
    main(["reference-adapt", str(source), str(events), str(target), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 2e6
    assert replay_reference_operations(target)["ok"]
    resumed = ReferenceOperationsRecorder.from_record(target)
    assert resumed.mission.cancelled == {"b": 1}
    assert resumed.mission.discarded["b"] == 2e6
    assert resumed.mission.effective_rate("a", "Ground") == .5e6


def test_event_gym_exposes_cancelled_slots_and_live_rate_limits(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    tape = [_event("product_cancelled", payload={"product_id": "b"}),
            _event("transmitter_rate_changed", id="tx", payload={"actor": "a", "rate_bps": .5e6}),
            _event("station_rate_changed", id="station", payload={"station": "Ground", "rate_bps": .75e6})]
    env = ReferenceEventDownlinkEnv(spec, tape, product_slots=2, dt_sec=10)
    check_env(env, skip_render_check=True)
    env.reset(seed=1)
    obs, reward, terminated, _, _ = env.step(np.array([1, 1]))
    assert not terminated and reward == 0
    assert obs["slot_cancelled"].tolist() == [[0, 0], [1, 0]]
    assert obs["transmitter_rates_bps"].tolist() == [.5e6, 1e6]
    assert obs["station_rates_bps"].tolist() == [.75e6]
    obs, reward, terminated, truncated, info = env.step(np.array([1, 1]))
    assert terminated and not truncated and reward == 2 and not info["success"]
    assert env.observation_space.contains(obs)


def test_fractional_rate_completion_has_no_phantom_remainder_or_false_success(spec):
    from space_env.reference.scheduling import plan_remaining_downlinks

    data = copy.deepcopy(spec)
    data["products"][0]["size_bits"] = 1e6
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    mission.apply_events([_event("station_rate_changed", payload={"station": "Ground", "rate_bps": .75e6})])
    mission.step(1, {"a": "Ground"})
    assert mission.remaining["a"] == 250000
    mission.step(1/3, {"a": "Ground"})
    assert mission.remaining["a"] == 0
    result = plan_remaining_downlinks(mission)
    assert result["report"]["on_time_bits"] == 3e6
    assert all(abs(v["balance_error_bits"]) < 1e-8 for v in result["report"]["inventory_ledger"].values())
    # A physically meaningful partial transfer must not be snapped to success.
    other = ReferenceDownlinkMission(data)
    other.step(.999, {"a": "Ground"})
    assert other.remaining["a"] == pytest.approx(1000)
    assert other.report()["on_time_bits"] == 0


def _booking(id="booking-b", actor="b", station="Ground", start=1, end=3, event_id="reserve"):
    return _event("terminal_booking_added", id=event_id, payload={"id": id, "actor": actor,
                  "station": station, "start_sec": start, "end_sec": end})


def test_reserved_owner_beats_alphabetical_arbitration_and_releases_at_end(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking()])
    report = mission.step(5, {"a": "Ground", "b": "Ground"})
    assert {p["id"]: p["completed_sec"] for p in report["products"]} == {"a": 5, "b": 3}
    assert all(r["actor"] == "b" for r in report["receipts"] if r["start_sec"] < 3)
    assert mission.station_permitted("a", "Ground")


def test_idle_reserved_terminal_is_not_stolen(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(end=10)])
    report = mission.step(5, {"a": "Ground"})
    assert not report["receipts"] and not mission.station_permitted("a", "Ground")
    assert mission.booking_capacity("b", "Ground") == (True, 0)


def test_two_terminal_station_keeps_one_open_and_one_reserved_even_owner_idle(spec):
    data = copy.deepcopy(spec)
    data["ephemerides"]["c"] = data["ephemerides"]["a"]
    data["vehicles"]["c"] = data["vehicles"]["a"]
    data["products"].append({"id": "c", "actor": "c", "size_bits": 2e6, "deadline_sec": 20})
    data["stations"][0]["terminals"] = 2
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    mission.apply_events([_booking(end=10)])
    report = mission.step(5, {"a": "Ground", "c": "Ground"})
    assert report["on_time_bits"] == 4e6
    assert all(len(i["transmitters"]) <= 1 for i in mission.intervals)


@pytest.mark.parametrize("conflict", ["station", "actor"])
def test_booking_conflicts_are_atomic(spec, conflict):
    data = copy.deepcopy(spec)
    data["stations"].append({**data["stations"][0], "name": "Other"})
    mission = ReferenceDownlinkMission(data)
    mission.step(1, {})
    before = mission.report()
    second = _booking(id="another", actor="a" if conflict == "station" else "b",
                      station="Ground" if conflict == "station" else "Other", event_id="reserve-2")
    with pytest.raises(ValueError, match="conflict|overlapping"):
        mission.apply_events([_booking(), second])
    assert mission.report() == before and not mission.bookings


def test_adjacent_bookings_can_share_station_and_single_radio(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(), _booking(id="next", start=3, end=5, event_id="reserve-next")])
    mission.step(2, {})
    assert mission.booking_capacity("b", "Ground") == (True, 0)
    assert len(mission.bookings) == 2


def test_cancel_active_booking_preserves_past_and_allows_reassignment(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(end=10)])
    mission.step(1, {})
    mission.apply_events([_event("terminal_booking_cancelled", time=2, id="release", payload={"booking_id": "booking-b"}),
                          dict(_booking(id="new", actor="a", start=2, end=5, event_id="new-slot"), time_sec=2)])
    assert mission.booking_capacity("b", "Ground") == (False, 0)
    assert mission.booking_capacity("a", "Ground") == (True, 0)
    assert mission.bookings["booking-b"].end_sec == 10 and mission.booking_cancelled["booking-b"] == 2
    mission.step(3, {"a": "Ground", "b": "Ground"})
    assert mission.remaining["a"] == 0 and mission.remaining["b"] == 2e6


def test_cancel_future_booking_releases_it_without_reusing_id(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(start=5, end=10),
                          _event("terminal_booking_cancelled", id="cancel", payload={"booking_id": "booking-b"})])
    assert mission.booking_capacity("a", "Ground") == (False, 1)
    with pytest.raises(ValueError):
        mission.apply_events([_booking(event_id="reuse")])
    with pytest.raises(ValueError):
        mission.apply_events([_event("terminal_booking_cancelled", id="recancel", payload={"booking_id": "booking-b"})])


def test_reservation_policy_required_and_reopened_without_changing_geometry(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_event("station_reservation_policy_changed", payload={"station": "Ground", "required": True})])
    assert mission._visible("a", "Ground") and not mission.station_permitted("a", "Ground")
    mission.step(2, {"a": "Ground"})
    assert not mission.receipts
    mission.apply_events([_event("station_reservation_policy_changed", time=3, id="reopen", payload={"station": "Ground", "required": False})])
    mission.step(2, {"a": "Ground"})
    assert mission.remaining["a"] == 0


@pytest.mark.parametrize("change", ["past", "horizon", "unknown", "empty", "policy"])
def test_booking_validation_rejects_bad_intervals_and_policy(spec, change):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    event = _booking()
    if change == "past":
        event["payload"]["start_sec"] = 0
    elif change == "horizon":
        event["payload"]["end_sec"] = 31
    elif change == "unknown":
        event["payload"]["actor"] = "unknown"
    elif change == "empty":
        event["payload"]["end_sec"] = 1
    else:
        event = _event("station_reservation_policy_changed", payload={"station": "Ground", "required": 1})
    before = mission.report()
    with pytest.raises(ValueError):
        mission.apply_events([event])
    assert mission.report() == before


def test_planner_obeys_reservation_owner_and_future_release(spec):
    from space_env.reference.scheduling import plan_remaining_downlinks

    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking()])
    result = plan_remaining_downlinks(mission)
    assert result["plan"]["steps"][0]["actions"] == {"b": "Ground"}
    assert result["report"]["on_time_bits"] == 4e6
    assert {p["id"]: p["completed_sec"] for p in result["report"]["products"]} == {"a": 5, "b": 3}


def test_booking_does_not_override_outage_or_geometry(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(end=5), _event("station_outage_added", id="outage", payload={
        "station": "Ground", "start_sec": 1, "end_sec": 4})])
    mission.step(4, {"b": "Ground"})
    assert sum(r["bits"] for r in mission.receipts) == 1e6
    assert min(r["start_sec"] for r in mission.receipts) == 4


def test_booking_cli_adapt_and_replay_restores_capacity(spec, tmp_path, capsys):
    from space_env.cli import main

    recorder = ReferenceOperationsRecorder(spec, lineage="bookings")
    recorder.step(1, {})
    source, events, target, report = [tmp_path / p for p in ("prefix.json", "events.json", "run.json", "report.json")]
    recorder.save(source)
    events.write_text(json.dumps({"events": [_booking()]}))
    main(["reference-adapt", str(source), str(events), str(target), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    assert replay_reference_operations(target)["ok"]
    restored = ReferenceOperationsRecorder.from_record(target)
    assert len(restored.mission.bookings) == 1 and restored.mission.booking_capacity("a", "Ground") == (False, 1)


def test_event_gym_reservation_masks_and_hidden_future_booking(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    env = ReferenceEventDownlinkEnv(spec, [_booking()], product_slots=2, dt_sec=2)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    assert obs["reserved"].sum() == 0 and not info["report"]["bookings"]
    obs, _, _, _, _ = env.step(np.array([0, 0]))
    assert obs["reserved"].tolist() == [[0], [1]]
    assert obs["station_permitted"].tolist() == [[0], [1]]
    assert obs["open_terminals"].tolist() == [0]
    obs, reward, _, _, _ = env.step(np.array([1, 1]))
    assert reward == 2 and obs["reserved"].sum() == 0
    assert obs["station_permitted"].tolist() == [[1], [1]]
    assert env.observation_space.contains(obs)


def test_service_windows_intersect_contacts_outages_and_reservation_policy(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking(), _event("station_reservation_policy_changed", id="exclusive", payload={
        "station": "Ground", "required": True}), _event("station_outage_added", id="fault", payload={
        "station": "Ground", "start_sec": 2, "end_sec": 2.5})])
    assert mission.service_windows("a", "Ground") == []
    assert mission.service_windows("b", "Ground") == [
        {"start_sec": 1, "end_sec": 2, "mode": "reserved"},
        {"start_sec": 2.5, "end_sec": 3, "mode": "reserved"}]
    assert mission.clock == 1
    mission.apply_events([_event("station_rate_changed", id="rate-off", payload={"station": "Ground", "rate_bps": 0})])
    assert mission.service_windows("b", "Ground") == []


@pytest.mark.parametrize("transmit", [True, False])
def test_booking_utilization_distinguishes_reserved_idle_from_used_time(spec, transmit):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    mission.apply_events([_booking()])
    mission.step(3, {"b": "Ground"} if transmit else {})
    utilization = mission.report()["booking_utilization"][0]
    assert utilization == {"booking_id": "booking-b", "elapsed_reserved_sec": 2,
                           "transmitting_sec": 2 if transmit else 0,
                           "idle_reserved_sec": 0 if transmit else 2,
                           "utilization": 1 if transmit else 0}


def _thermal_spec(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(tx_power_w=100, thermal={"ambient_k": 280, "initial_k": 280,
        "capacity_j_per_k": 10, "conductance_w_per_k": 1, "heat_tx_w": 100, "trip_k": 300, "resume_k": 290})
    data["products"][0].update(size_bits=10e6, deadline_sec=30)
    return data


def test_thermal_trip_and_cooling_resume_use_analytic_boundary(spec):
    import math

    mission = ReferenceDownlinkMission(_thermal_spec(spec))
    trip = -10*math.log(.8)
    mission.step(trip, {"a": "Ground"})
    assert mission.tx_seconds["a"] == pytest.approx(trip)
    assert mission.temperature["a"] == pytest.approx(300)
    assert mission.thermal_locked["a"]
    delivered = sum(r["bits"] for r in mission.receipts)
    mission.step(2, {"a": "Ground"})
    assert sum(r["bits"] for r in mission.receipts) == delivered
    assert mission.temperature["a"] == pytest.approx(280+20*math.exp(-.2))
    cooldown_remaining = 10*math.log(2)-2
    mission.step(cooldown_remaining, {})
    assert not mission.thermal_locked["a"] and mission.temperature["a"] == pytest.approx(290)
    mission.step(.1, {"a": "Ground"})
    assert mission.tx_seconds["a"] == pytest.approx(trip+.1)


def test_thermal_whole_step_matches_partitions_and_energy_balance(spec):
    data = _thermal_spec(spec)
    left, right = ReferenceDownlinkMission(data), ReferenceDownlinkMission(data)
    left.step(20, {"a": "Ground"})
    for _ in range(20):
        right.step(1, {"a": "Ground"})
    assert left.temperature == pytest.approx(right.temperature)
    assert left.tx_seconds == pytest.approx(right.tx_seconds)
    assert left.remaining == pytest.approx(right.remaining)
    health = left.report()["transmitter_health"]["a"]
    assert abs(health["thermal_balance_error_j"]) < 1e-9
    assert health["heat_input_j"] == pytest.approx(100*left.tx_seconds["a"])
    assert left.energy_wh["a"] == pytest.approx(1-left.tx_seconds["a"]*100/3600)
    assert left.temperature["a"] <= 300+1e-9


def test_transmission_quota_cuts_data_and_power_exactly(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(max_tx_sec=.5, tx_power_w=100)
    mission = ReferenceDownlinkMission(data)
    report = mission.step(10, {"a": "Ground", "b": "Ground"})
    assert mission.remaining["a"] == 1.5e6 and mission.remaining["b"] == 0
    assert mission.tx_seconds["a"] == .5 and mission.tx_seconds["b"] == 2
    assert mission.energy_wh["a"] == pytest.approx(1-50/3600)
    assert not report["transmitter_health"]["a"]["ready"]
    assert mission.transmitter_resume_sec("a") is None


def test_zero_quota_never_uses_station_or_power(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(max_tx_sec=0, tx_power_w=100)
    mission = ReferenceDownlinkMission(data)
    report = mission.step(3, {"a": "Ground", "b": "Ground"})
    assert report["on_time_bits"] == 2e6 and mission.energy_wh["a"] == 1
    assert all(r["actor"] == "b" for r in mission.receipts)


def test_initially_hot_transmitter_cools_while_waiting(spec):
    data = _thermal_spec(spec)
    data["vehicles"]["a"]["thermal"]["initial_k"] = 310
    mission = ReferenceDownlinkMission(data)
    assert not mission.transmitter_ready("a")
    mission.step(1, {"a": "Ground"})
    assert not mission.receipts and mission.temperature["a"] < 310


@pytest.mark.parametrize("change", ["threshold", "ambient", "heat", "capacity", "quota", "derived"])
def test_bad_thermal_config_is_rejected(spec, change):
    data = _thermal_spec(spec)
    t = data["vehicles"]["a"]["thermal"]
    if change == "threshold":
        t["resume_k"] = 300
    elif change == "ambient":
        t["ambient_k"] = 290
    elif change == "heat":
        t["heat_tx_w"] = 101
    elif change == "capacity":
        t["capacity_j_per_k"] = 0
    elif change == "derived":
        t["conductance_w_per_k"] = 1e-320
    else:
        data["vehicles"]["a"]["max_tx_sec"] = -1
    with pytest.raises(ValueError):
        ReferenceDownlinkMission(data)


def test_internal_failure_rolls_back_thermal_and_lifetime_state(spec, monkeypatch):
    mission = ReferenceDownlinkMission(_thermal_spec(spec))
    before = mission.report()
    original = mission._generation

    def fail(actor, midpoint):
        if mission.clock > 0:
            raise RuntimeError("thermal rollback")
        return original(actor, midpoint)

    monkeypatch.setattr(mission, "_generation", fail)
    with pytest.raises(RuntimeError):
        mission.step(10, {"a": "Ground"})
    assert mission.report() == before


def test_thermal_planner_replay_and_gym_telemetry(spec, tmp_path):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv
    from space_env.reference.scheduling import plan_downlinks

    data = _thermal_spec(spec)
    result = plan_downlinks(data, decision_sec=10, select_products=True)
    recorder = ReferenceOperationsRecorder(data, lineage="thermal-planning")
    for row in result["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    path = tmp_path / "thermal.json"
    recorder.save(path)
    assert replay_reference_operations(path)["ok"]
    restored = ReferenceOperationsRecorder.from_record(path)
    assert restored.mission.temperature == recorder.mission.temperature
    for env in (ReferenceDownlinkEnv(data, dt_sec=5), ReferenceEventDownlinkEnv(data, [], dt_sec=5)):
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=1)
        assert obs["thermal_enabled"].tolist() == [1, 0]
        obs, _, _, _, _ = env.step(np.array([1, 0]))
        assert obs["thermal_locked"].tolist() == [1, 0]
        assert env.observation_space.contains(obs)


def test_planner_hands_station_over_at_thermal_trip_and_skips_insufficient_quota(spec):
    import math

    from space_env.reference.scheduling import plan_downlinks

    data = _thermal_spec(spec)
    data["products"][0]["priority"] = 10
    result = plan_downlinks(data, decision_sec=10, select_products=True)
    first_b = min(r["start_sec"] for r in result["report"]["receipts"] if r["actor"] == "b")
    assert first_b == pytest.approx(-10*math.log(.8))
    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["max_tx_sec"] = .5
    data["products"].append({"id": "small", "actor": "a", "size_bits": .25e6, "deadline_sec": 10})
    result = plan_downlinks(data, select_products=True)
    assert result["report"]["on_time_bits"] == 2.25e6
    assert not any(r["product_id"] == "a" for r in result["report"]["receipts"])


def test_initial_temperature_at_trip_tolerance_does_not_create_zero_step(spec):
    data = _thermal_spec(spec)
    data["vehicles"]["a"]["thermal"]["initial_k"] = 300-5e-10
    mission = ReferenceDownlinkMission(data)
    assert mission.thermal_locked["a"]
    mission.step(1, {"a": "Ground"})
    assert not mission.receipts and mission.temperature["a"] < 300


def _processing_spec(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"].update(processing_power_w=18, tx_power_w=20)
    data["products"][0].update(processing_sec=3, workspace_bits=1e6)
    return data


def test_processing_completion_unlocks_data_and_charges_cpu_before_downlink(spec):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    report = mission.step(6, {"a": "Ground", "b": "Ground"})
    a = next(p for p in report["products"] if p["id"] == "a")
    assert a["processing_completed_sec"] == 3 and a["completed_sec"] == 5
    assert min(r["start_sec"] for r in mission.receipts if r["actor"] == "a") == 3
    assert mission.processing_seconds == {"a": 3, "b": 0}
    assert mission.energy_wh["a"] == pytest.approx(1-(18*3+20*2)/3600)
    assert report["on_time_bits"] == 4e6
    assert report["inventory_ledger"]["a"]["balance_error_bits"] == 0


def test_processing_pause_preserves_work_and_raw_data_cannot_transmit(spec):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    mission.step(2, {"a": "Ground"}, processing={"a": False})
    assert not mission.receipts and mission.processing_remaining["a"] == 3
    assert mission.processing_seconds["a"] == 0 and mission.energy_wh["a"] == 1
    mission.step(5, {"a": "Ground"})
    assert mission.report()["products"][0]["completed_sec"] == 7


def test_processing_ready_time_is_start_not_completion(spec):
    data = _processing_spec(spec)
    data["products"][0]["ready_sec"] = 2
    mission = ReferenceDownlinkMission(data)
    mission.step(7, {"a": "Ground"})
    assert min(i["start_sec"] for i in mission.processing_intervals) == 2
    assert mission.processing_completed["a"] == 5
    assert mission.report()["products"][0]["completed_sec"] == 7


def test_downlink_can_free_workspace_for_blocked_processor(spec):
    data = _processing_spec(spec)
    data["vehicles"]["a"]["memory_bits"] = 2e6
    data["products"][0].update(size_bits=1e6, workspace_bits=.75e6, processing_sec=2)
    data["products"].append({"id": "ready", "actor": "a", "size_bits": 1e6, "deadline_sec": 10})
    mission = ReferenceDownlinkMission(data)
    mission.step(5, {"a": "Ground"})
    assert mission.processing_completed["a"] == 3
    assert min(r["start_sec"] for r in mission.receipts if r["product_id"] == "a") == 3
    assert mission.remaining["ready"] == mission.remaining["a"] == 0
    assert all(i["workspace_bits"] == .75e6 for i in mission.processing_intervals)


def test_processing_floor_stops_work_and_power_without_fabricated_progress(spec):
    data = _processing_spec(spec)
    data["vehicles"]["a"].update(initial_soc=.5, min_processing_soc=.25, processing_power_w=900)
    mission = ReferenceDownlinkMission(data)
    mission.step(5, {"a": "Ground"})
    assert mission.processing_seconds["a"] == 1
    assert mission.processing_remaining["a"] == 2
    assert mission.energy_wh["a"] == .25
    assert not mission.receipts


def test_cancel_processing_job_keeps_spent_cpu_energy_and_drops_remaining_work(spec):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    mission.step(1, {})
    energy = mission.energy_wh["a"]
    mission.apply_events([_event("product_cancelled", payload={"product_id": "a"})])
    mission.step(5, {"a": "Ground"})
    assert mission.processing_remaining["a"] == 0 and mission.processing_seconds["a"] == 1
    assert mission.processing_completed["a"] is None and mission.energy_wh["a"] == energy
    assert mission.report()["products"][0]["status"] == "cancelled"


def test_processing_step_failure_rolls_back_work_completion_logs_and_energy(spec, monkeypatch):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    before = mission.report()
    original = mission._generation

    def fail(actor, midpoint):
        if mission.clock > 0:
            raise RuntimeError("CPU rollback")
        return original(actor, midpoint)

    monkeypatch.setattr(mission, "_generation", fail)
    with pytest.raises(RuntimeError):
        mission.step(10, {"a": "Ground"})
    assert mission.report() == before and not mission.processing_intervals


def test_processing_step_partition_invariance(spec):
    left, right = [ReferenceDownlinkMission(_processing_spec(spec)) for _ in range(2)]
    left.step(10, {"a": "Ground", "b": "Ground"})
    for _ in range(10):
        right.step(1, {"a": "Ground", "b": "Ground"})
    assert left.processing_seconds == pytest.approx(right.processing_seconds)
    assert left.energy_wh == pytest.approx(right.energy_wh)
    assert left.remaining == right.remaining
    assert left.processing_completed == right.processing_completed


@pytest.mark.parametrize("bad", ["workspace", "duration", "control"])
def test_processing_invalid_inputs_fail_without_mutation(spec, bad):
    data = _processing_spec(spec)
    if bad == "workspace":
        data["products"][0]["workspace_bits"] = 11e6
    elif bad == "duration":
        data["products"][0]["processing_sec"] = -1
    else:
        mission = ReferenceDownlinkMission(data)
        before = mission.report()
        with pytest.raises(ValueError):
            mission.step(1, {}, processing={"a": 0})
        assert mission.report() == before
        return
    with pytest.raises(ValueError):
        ReferenceDownlinkMission(data)


def test_processing_record_resume_dynamic_arrival_and_planner(spec, tmp_path):
    from space_env.reference.scheduling import plan_remaining_downlinks

    data = _processing_spec(spec)
    recorder = ReferenceOperationsRecorder(data, lineage="processing-family")
    recorder.step(1, {}, processing={"a": False})
    recorder.inject([_event("onboard_product_added", payload={"id": "new", "actor": "b", "size_bits": 1e6,
                     "ready_sec": 1, "deadline_sec": 15, "processing_sec": 2, "workspace_bits": 1e6})])
    path = tmp_path / "prefix.json"
    recorder.save(path)
    resumed = ReferenceOperationsRecorder.from_record(path)
    assert resumed.mission.processing_remaining == {"a": 3, "b": 0, "new": 2}
    result = plan_remaining_downlinks(resumed.mission)
    for row in result["plan"]["steps"]:
        resumed.step(row["dt_sec"], row["actions"], products=row.get("products"))
    assert resumed.mission.report()["on_time_bits"] == 5e6
    resumed.save(tmp_path / "completed.json")
    assert replay_reference_operations(tmp_path / "completed.json")["ok"]


def test_both_gym_facades_expose_processing_and_allow_pause(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    data = _processing_spec(spec)
    for env, pause_action in ((ReferenceDownlinkEnv(data, dt_sec=1, cpu_control=True), 2),
                              (ReferenceEventDownlinkEnv(data, [], dt_sec=1, product_slots=2, cpu_control=True), 4)):
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        env.step(np.array([pause_action, 0]))
        assert env.mission.processing_seconds["a"] == 0
        obs, _, _, _, _ = env.step(np.array([0, 0]))
        assert env.mission.processing_seconds["a"] == 1
        assert env.observation_space.contains(obs)


def test_processing_priority_preemption_keeps_partial_work(spec):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    mission.step(1, {})
    mission.apply_events([_event("onboard_product_added", payload={"id": "urgent", "actor": "a", "size_bits": 1e6,
                         "ready_sec": 1, "deadline_sec": 10, "priority": 9, "processing_sec": 1})])
    mission.step(1, {})
    assert mission.processing_remaining["a"] == 2 and mission.processing_remaining["urgent"] == 0
    assert mission.processing_completed["urgent"] == 2
    mission.step(2, {})
    assert mission.processing_completed["a"] == 4 and mission.processing_seconds["a"] == 4
    assert all(i["total_memory_bits"] <= mission.vehicles[i["actor"]].memory_bits for i in mission.processing_intervals)


def test_cli_processing_controls_are_recorded_and_replayed(spec, tmp_path, capsys):
    from space_env.cli import main

    source, plan, target = [tmp_path / p for p in ("spec.json", "plan.json", "run.json")]
    source.write_text(json.dumps(_processing_spec(spec)))
    plan.write_text(json.dumps({"steps": [
        {"dt_sec": 2, "actions": {"a": "Ground"}, "processing": {"a": False}},
        {"dt_sec": 5, "actions": {"a": "Ground"}, "processing": {"a": True}}]}))
    main(["reference-run", str(source), str(plan), str(target), "--lineage", "processing-cli"])
    report = json.loads(capsys.readouterr().out)["report"]
    assert report["products"][0]["completed_sec"] == 7 and report["processing"]["seconds"]["a"] == 3
    assert replay_reference_operations(target)["ok"]


def _dag_spec(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["processing_power_w"] = 18
    data["products"][0]["processing_sec"] = 1
    data["products"].append({"id": "derived", "actor": "a", "size_bits": 1e6,
        "deadline_sec": 20, "processing_sec": 2, "priority": 9, "depends_on": ["a"]})
    return data


def test_dependency_order_overrides_priority_and_retains_input_until_consumed(spec):
    mission = ReferenceDownlinkMission(_dag_spec(spec))
    mission.step(1, {"a": "Ground"})
    assert mission.processing_completed["a"] == 1
    assert mission.processing_remaining["derived"] == 2
    assert mission.retained_for("a") == ["derived"] and not mission.receipts
    mission.step(2, {"a": "Ground"}, products={"a": "a"})
    assert mission.processing_completed["derived"] == 3 and not mission.receipts
    assert mission.retained_for("a") == []
    mission.step(5, {"a": "Ground"})
    assert mission.remaining["a"] == mission.remaining["derived"] == 0
    assert min(i["start_sec"] for i in mission.processing_intervals if i["product_id"] == "derived") == 1


@pytest.mark.parametrize("bad", ["cycle", "self", "duplicate", "unknown", "cross_actor", "zero_work"])
def test_dependency_validation_rejects_invalid_graphs(spec, bad):
    data = _dag_spec(spec)
    child = data["products"][-1]
    if bad == "cycle":
        data["products"][0]["depends_on"] = ["derived"]
    elif bad == "self":
        child["depends_on"] = ["derived"]
    elif bad == "duplicate":
        child["depends_on"] = ["a", "a"]
    elif bad == "unknown":
        child["depends_on"] = ["missing"]
    elif bad == "cross_actor":
        child["depends_on"] = ["b"]
    else:
        child["processing_sec"] = 0
    with pytest.raises(ValueError):
        ReferenceDownlinkMission(data)


def test_graph_validation_is_nonrecursive_for_deep_pipeline():
    from types import SimpleNamespace

    from space_env.reference.dependencies import dependency_order

    nodes = {str(i): SimpleNamespace(actor="a", depends_on=(str(i-1),) if i else ()) for i in range(2000)}
    order = dependency_order(nodes)
    assert len(order) == 2000 and order[0] == "0" and order[-1] == "1999"


def test_cancelled_input_blocks_waiting_child_and_releases_pin_after_child_cancel(spec):
    mission = ReferenceDownlinkMission(_dag_spec(spec))
    mission.step(.5, {})
    mission.apply_events([_event("product_cancelled", time=.5, payload={"product_id": "a"})])
    mission.step(3, {"a": "Ground"})
    assert mission.processing_remaining["derived"] == 2
    child = next(p for p in mission.report()["products"] if p["id"] == "derived")
    assert child["dependency_blockers"] == ["a"] and not child["downlink_ready"]
    mission.apply_events([_event("product_cancelled", time=3.5, id="cancel-child", payload={"product_id": "derived"})])
    assert mission.retained_for("a") == []


def test_completed_child_is_not_invalidated_by_later_input_cancellation(spec):
    mission = ReferenceDownlinkMission(_dag_spec(spec))
    mission.step(3, {})
    assert mission.processing_completed["derived"] == 3
    mission.apply_events([_event("product_cancelled", time=3, payload={"product_id": "a"})])
    mission.apply_events([_event("onboard_product_added", time=3, id="grandchild", payload={
        "id": "final", "actor": "a", "size_bits": 1e6, "ready_sec": 3, "deadline_sec": 20,
        "processing_sec": 1, "depends_on": ["derived"]})])
    mission.step(1, {})
    assert mission.processing_completed["final"] == 4
    assert next(p for p in mission.report()["products"] if p["id"] == "derived")["dependency_blockers"] == []


def test_dynamic_graph_batch_allows_forward_refs_and_rolls_back_cycles(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {})
    child = _event("onboard_product_added", id="child", payload={"id": "child", "actor": "a", "size_bits": 1e6,
        "ready_sec": 1, "deadline_sec": 20, "processing_sec": 1, "depends_on": ["parent"]})
    parent = _event("onboard_product_added", id="parent", payload={"id": "parent", "actor": "a", "size_bits": 1e6,
        "ready_sec": 1, "deadline_sec": 20, "processing_sec": 1})
    mission.apply_events([child, parent])
    assert mission._dependency_order.index("parent") < mission._dependency_order.index("child")
    before = mission.report()
    bad = _event("onboard_product_added", id="cycle", payload={"id": "bad", "actor": "a", "size_bits": 1e6,
        "ready_sec": 1, "deadline_sec": 20, "processing_sec": 1, "depends_on": ["bad"]})
    with pytest.raises(ValueError):
        mission.apply_events([bad])
    assert mission.report() == before


def test_new_dependency_cannot_reference_partially_downlinked_input(spec):
    mission = ReferenceDownlinkMission(spec)
    mission.step(1, {"a": "Ground"})
    before = mission.report()
    with pytest.raises(ValueError, match="intact"):
        mission.apply_events([_event("onboard_product_added", payload={"id": "new", "actor": "a", "size_bits": 1e6,
            "ready_sec": 1, "deadline_sec": 20, "processing_sec": 1, "depends_on": ["a"]})])
    assert mission.report() == before


def test_dependency_pins_still_apply_when_cpu_is_paused(spec):
    mission = ReferenceDownlinkMission(_dag_spec(spec))
    mission.step(1, {})
    mission.step(2, {"a": "Ground"}, products={"a": "a"}, processing={"a": False})
    assert mission.remaining["a"] == 2e6 and mission.processing_remaining["derived"] == 2
    assert not mission.receipts


def test_dag_planning_replay_and_both_gym_graph_observations(spec, tmp_path):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv
    from space_env.reference.scheduling import plan_downlinks

    data = _dag_spec(spec)
    result = plan_downlinks(data, select_products=True)
    assert result["report"]["on_time_bits"] == 5e6
    recorder = ReferenceOperationsRecorder(data, lineage="dag-family")
    for row in result["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    path = tmp_path / "dag.json"
    recorder.save(path)
    assert replay_reference_operations(path)["ok"]
    for env in (ReferenceDownlinkEnv(data, dt_sec=5), ReferenceEventDownlinkEnv(data, [], product_slots=3, dt_sec=5)):
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=1)
        assert obs["dependency_edges"].sum() == 1 and obs["retained"].sum() == 1
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("shape", ["fan_in", "fan_out"])
def test_dependency_join_and_shared_input_retention(spec, shape):
    data = _dag_spec(spec)
    child = data["products"][-1]
    if shape == "fan_in":
        data["products"].append({"id": "second-input", "actor": "a", "size_bits": 1e6,
                                 "deadline_sec": 20, "processing_sec": 1})
        child.update(processing_sec=1, depends_on=["a", "second-input"])
        mission = ReferenceDownlinkMission(data)
        mission.step(2, {})
        assert mission.processing_completed["derived"] is None
        assert mission.processing_remaining["derived"] == 1
        mission.step(1, {})
        assert mission.processing_completed["derived"] == 3
    else:
        child["processing_sec"] = 1
        data["products"].append({"id": "other-child", "actor": "a", "size_bits": 1e6,
                                 "deadline_sec": 20, "processing_sec": 2, "depends_on": ["a"]})
        mission = ReferenceDownlinkMission(data)
        mission.step(2, {})
        assert mission.retained_for("a") == ["other-child"]
        mission.step(2, {})
        assert mission.retained_for("a") == []


def _oracle_counterexample(spec):
    data = copy.deepcopy(spec)
    data["products"] = [
        {"id": "urgent", "actor": "a", "size_bits": 1e6, "deadline_sec": 4, "priority": 10},
        {"id": "early", "actor": "b", "size_bits": 2e6, "deadline_sec": 2},
        {"id": "later", "actor": "b", "size_bits": 2e6, "deadline_sec": 4}]
    return data


def test_calendar_oracle_proves_nonzero_greedy_regret_and_executes(spec):
    from space_env.reference.optimal import solve_small_downlinks

    data = _oracle_counterexample(spec)
    result = solve_small_downlinks(data)
    assert result["optimality_proven_in_scope"] and not result["general_resource_optimality"]
    assert result["report"]["on_time_bits"] == 4e6
    assert result["baseline"]["report"]["on_time_bits"] == 3e6
    assert result["regret_bits"] == 1e6 and result["relative_regret"] == .25
    assert result["selected_products"] == ["early", "later"]
    mission = ReferenceDownlinkMission(data)
    for row in result["plan"]["steps"]:
        mission.step(row["dt_sec"], row["actions"], products=row.get("products"))
    assert mission.report() == result["report"]
    assert solve_small_downlinks(data) == result


@pytest.mark.parametrize("seed", range(8))
def test_oracle_matches_independent_permutation_enumeration(spec, seed):
    import itertools
    import random

    from space_env.reference.optimal import solve_small_downlinks

    rng = random.Random(seed)
    data = copy.deepcopy(spec)
    for vehicle in data["vehicles"].values():
        vehicle["memory_bits"] = 20e6
    data["products"] = [{"id": f"p{i}", "actor": rng.choice(["a", "b"]), "size_bits": rng.randint(1, 3)*1e6,
                         "ready_sec": rng.randint(0, 1), "deadline_sec": rng.randint(3, 12),
                         "priority": rng.randint(0, 2)} for i in range(5)]
    # Whole fixture contact includes all deadlines. This brute force uses no DP,
    # service-window helper, scheduler or simulator transitions to determine value.
    best = 0
    for count in range(1, 6):
        for order in itertools.permutations(data["products"], count):
            clock = 0
            for p in order:
                clock = max(clock, p["ready_sec"])+p["size_bits"]/1e6
                if clock > p["deadline_sec"]:
                    break
            else:
                best = max(best, sum(p["size_bits"] for p in order))
    assert solve_small_downlinks(data)["report"]["on_time_bits"] == best


@pytest.mark.parametrize("constraint", ["power", "thermal", "quota", "processing", "terminals", "stations", "fractional_bits"])
def test_oracle_rejects_out_of_scope_models(spec, constraint):
    from space_env.reference.optimal import solve_small_downlinks

    data = copy.deepcopy(spec)
    if constraint == "power":
        data["vehicles"]["a"]["base_power_w"] = 1
    elif constraint == "thermal":
        data = _thermal_spec(spec)
    elif constraint == "quota":
        data["vehicles"]["a"]["max_tx_sec"] = 3
    elif constraint == "processing":
        data = _processing_spec(spec)
    elif constraint == "terminals":
        data["stations"][0]["terminals"] = 2
    elif constraint == "stations":
        data["stations"].append({**data["stations"][0], "name": "Other"})
    else:
        data["products"][0]["size_bits"] = 2e6+.5
    with pytest.raises(ValueError):
        solve_small_downlinks(data)


def test_oracle_release_outage_and_missed_deadlines(spec):
    from space_env.reference.optimal import solve_small_downlinks

    data = copy.deepcopy(spec)
    data["products"][0].update(ready_sec=2, deadline_sec=7)
    data["outages"] = [{"station": "Ground", "start_sec": 1, "end_sec": 4}]
    result = solve_small_downlinks(data)
    assert result["report"]["on_time_bits"] == 4e6
    assert all(r["start_sec"] >= 4 for r in result["report"]["receipts"])
    data["products"][0]["deadline_sec"] = 5
    assert solve_small_downlinks(data)["report"]["on_time_bits"] == 2e6


@pytest.mark.parametrize("budget", [0, True, 1, 65537])
def test_oracle_budget_never_returns_partial_optimality_certificate(spec, budget):
    from space_env.reference.optimal import solve_small_downlinks

    with pytest.raises(ValueError):
        solve_small_downlinks(spec, max_states=budget)


def test_cli_oracle_plan_executes_and_budget_failure_preserves_output(spec, tmp_path, capsys):
    from space_env.cli import main

    source, plan, report, record = [tmp_path / p for p in ("spec.json", "plan.json", "report.json", "run.json")]
    source.write_text(json.dumps(_oracle_counterexample(spec)))
    main(["reference-optimal", str(source), str(plan), "--report", str(report)])
    assert json.loads(capsys.readouterr().out)["regret_bits"] == 1e6
    main(["reference-run", str(source), str(plan), str(record), "--lineage", "oracle-test"])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 4e6
    assert replay_reference_operations(record)["ok"]
    previous = plan.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        main(["reference-optimal", str(source), str(plan), "--report", str(report), "--max-states", "1"])
    assert plan.read_bytes() == previous


def _calendar_cases(spec):
    train = _oracle_counterexample(spec)
    test = copy.deepcopy(train)
    test["products"][0]["deadline_sec"] = 5
    return [{"id": "train-case", "lineage": "declared-family-a", "split": "train", "spec": train},
            {"id": "test-case", "lineage": "declared-family-b", "split": "test", "spec": test}]


def test_calendar_suite_is_self_contained_and_recomputes_results(spec, tmp_path):
    from space_env.reference.benchmark import (
        evaluate_calendar_suite,
        save_calendar_benchmark,
        verify_calendar_benchmark,
    )

    report = evaluate_calendar_suite(_calendar_cases(spec))
    assert not report["expert_demonstrations"]
    assert report["summary"]["train"]["optimal_bits"] == 4e6
    assert isinstance(report["summary"]["train"]["optimal_bits"], int)
    assert report["summary"]["test"]["optimal_bits"] == 5e6
    path = tmp_path / "benchmark.json"
    save_calendar_benchmark(path, report)
    assert verify_calendar_benchmark(path)["ok"]
    body = json.loads(path.read_text())
    body["summary"]["test"]["optimal_bits"] = 99e6
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    path.write_text(json.dumps(body))
    assert verify_calendar_benchmark(path) == {"ok": False, "reason": "result_mismatch"}


@pytest.mark.parametrize("bad", ["lineage", "hash", "id", "budget"])
def test_calendar_suite_rejects_declared_split_leaks_and_budget(spec, bad):
    from space_env.reference.benchmark import evaluate_calendar_suite

    cases = _calendar_cases(spec)
    if bad == "lineage":
        cases[1]["lineage"] = cases[0]["lineage"]
    elif bad == "hash":
        cases[1]["spec"] = copy.deepcopy(cases[0]["spec"])
    elif bad == "id":
        cases[1]["id"] = cases[0]["id"]
    with pytest.raises(ValueError):
        evaluate_calendar_suite(cases, total_states=1 if bad == "budget" else 262144)


def test_cli_calendar_benchmark_survives_deleted_suite(spec, tmp_path, capsys):
    from space_env.cli import main

    source, target = tmp_path / "suite.json", tmp_path / "benchmark.json"
    source.write_text(json.dumps({"cases": _calendar_cases(spec)}))
    main(["reference-benchmark", str(source), str(target)])
    assert json.loads(capsys.readouterr().out)["summary"]["train"]["regret_bits"] == 1e6
    source.unlink()
    main(["reference-verify-benchmark", str(target)])
    assert json.loads(capsys.readouterr().out)["ok"]


def test_preemptive_project_policy_is_not_falsely_bounded_by_nonpreemptive_optimum(spec):
    from space_env.reference.optimal import solve_small_downlinks

    data = copy.deepcopy(spec)
    data["products"][0].update(deadline_sec=5, priority=10)
    data["outages"] = [{"station": "Ground", "start_sec": 1, "end_sec": 4}]
    result = solve_small_downlinks(data)
    assert result["report"]["on_time_bits"] == 2e6
    assert result["project_baseline"]["report"]["on_time_bits"] == 4e6
    assert not result["project_baseline"]["comparable"]
    assert result["project_baseline"]["regret_bits"] is None


def _cpu_choice_spec(spec):
    data = copy.deepcopy(spec)
    data["vehicles"]["a"]["processing_power_w"] = 18
    data["products"] = [
        {"id": "long", "actor": "a", "size_bits": 1e6, "ready_sec": 0, "deadline_sec": 25,
         "processing_sec": 10, "priority": 10},
        {"id": "short", "actor": "a", "size_bits": 1e6, "ready_sec": 0, "deadline_sec": 3,
         "processing_sec": 1}]
    return data


def test_explicit_cpu_selection_recovers_urgent_job_without_priority_relabeling(spec):
    data = _cpu_choice_spec(spec)
    auto, controlled = ReferenceDownlinkMission(data), ReferenceDownlinkMission(data)
    auto.step(13, {"a": "Ground"})
    controlled.step(2, {"a": "Ground"}, processing_jobs={"a": "short"})
    assert controlled.processing_remaining["long"] == 10
    assert controlled.processing_completed["short"] == 1
    controlled.step(11, {"a": "Ground"}, processing_jobs={"a": "long"})
    assert auto.report()["on_time_bits"] == 1e6
    assert controlled.report()["on_time_bits"] == 2e6
    assert controlled.products["long"].priority == 10


@pytest.mark.parametrize("jobs,processing", [({"a": "b"}, None), ({"a": "missing"}, None),
                                            ({"unknown": "a"}, None), ({"a": "a"}, {"a": False}),
                                            ({"a": None}, None)])
def test_bad_cpu_job_selection_rejected_before_any_state_change(spec, jobs, processing):
    mission = ReferenceDownlinkMission(_processing_spec(spec))
    before = mission.report()
    with pytest.raises(ValueError, match="selected CPU"):
        mission.step(1, {}, processing_jobs=jobs, processing=processing)
    assert mission.report() == before


def test_selected_dependency_blocked_job_waits_instead_of_running_parent(spec):
    mission = ReferenceDownlinkMission(_dag_spec(spec))
    mission.step(2, {}, processing_jobs={"a": "derived"})
    assert mission.processing_remaining["a"] == 1 and mission.processing_remaining["derived"] == 2
    assert mission.processing_status("derived")["reason"] == "dependency"
    mission.step(1, {}, processing_jobs={"a": "a"})
    assert mission.processing_status("derived")["eligible"]


def test_selected_job_waits_until_release_and_stops_after_completion(spec):
    data = _cpu_choice_spec(spec)
    data["products"][1].update(ready_sec=2, deadline_sec=10)
    mission = ReferenceDownlinkMission(data)
    assert mission.processing_status("short")["reason"] == "not_ready"
    mission.step(5, {}, processing_jobs={"a": "short"})
    assert mission.processing_completed["short"] == 3
    assert mission.processing_seconds["a"] == 1 and mission.processing_remaining["long"] == 10
    assert mission.processing_status("short")["reason"] == "completed"


def test_cpu_admission_reports_memory_power_and_cancelled(spec):
    data = _processing_spec(spec)
    data["vehicles"]["a"]["memory_bits"] = 2e6
    mission = ReferenceDownlinkMission(data)
    assert mission.processing_status("a")["reason"] == "memory"
    data["vehicles"]["a"].update(memory_bits=10e6, initial_soc=0, processing_power_w=100)
    mission = ReferenceDownlinkMission(data)
    assert mission.processing_status("a")["reason"] == "power"
    assert mission.processing_status("a", generation_w=100)["reason"] == "power"  # below SOC floor
    mission.step(1, {})
    mission.apply_events([_event("product_cancelled", payload={"product_id": "a"})])
    assert mission.processing_status("a")["reason"] == "cancelled"
    with pytest.raises(ValueError):
        mission.processing_status("a", generation_w=float("inf"))


def test_cpu_job_selection_cli_record_restores_actual_selected_work(spec, tmp_path, capsys):
    from space_env.cli import main

    source, plan, target = [tmp_path / p for p in ("spec.json", "plan.json", "run.json")]
    source.write_text(json.dumps(_cpu_choice_spec(spec)))
    plan.write_text(json.dumps({"steps": [
        {"dt_sec": 2, "actions": {"a": "Ground"}, "processing_jobs": {"a": "short"}},
        {"dt_sec": 11, "actions": {"a": "Ground"}, "processing_jobs": {"a": "long"}}]}))
    main(["reference-run", str(source), str(plan), str(target), "--lineage", "cpu-choice"])
    assert json.loads(capsys.readouterr().out)["report"]["on_time_bits"] == 2e6
    assert replay_reference_operations(target)["ok"]
    restored = ReferenceOperationsRecorder.from_record(target)
    assert restored.mission.processing_completed == {"long": 12, "short": 1}


def test_gym_cpu_job_dict_action_and_existing_rollout_writer(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv
    from space_env.rl.trajectories import rollout_episode

    data = _cpu_choice_spec(spec)
    for env in (ReferenceDownlinkEnv(data, dt_sec=1, cpu_job_selection=True),
                ReferenceEventDownlinkEnv(data, [], product_slots=3, dt_sec=1, cpu_job_selection=True)):
        check_env(env, skip_render_check=True)
        env.reset(seed=7)
        env.step({"radio": np.array([0, 0]), "cpu": np.array([3, 0])})
        assert env.mission.processing_completed["short"] == 1
        assert env.mission.processing_remaining["long"] == 10
        # Completed pinned short does not fall through to long; automatic CPU does.
        env.step({"radio": np.array([1, 0]), "cpu": np.array([3, 0])})
        assert env.mission.processing_remaining["long"] == 10
    env = ReferenceDownlinkEnv(data, dt_sec=1, cpu_job_selection=True)
    rows = rollout_episode(env, seed=7, policy=lambda obs: {"radio": np.array([1, 0]),
        "cpu": np.array([3 if obs["clock"][0] < 2 else 2, 0])})
    assert rows[-1]["success"] and sum(r["reward"] for r in rows) == 2
    json.dumps(rows, allow_nan=False)


def test_event_env_empty_cpu_slot_waits_and_slot_mapping_updates_on_arrival(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    data = _processing_spec(spec)
    event = _event("onboard_product_added", payload={"id": "new", "actor": "a", "size_bits": 1e6,
                  "ready_sec": 1, "deadline_sec": 15, "processing_sec": 1})
    env = ReferenceEventDownlinkEnv(data, [event], product_slots=3, dt_sec=1, cpu_job_selection=True)
    env.reset(seed=1)
    _, _, _, _, info = env.step({"radio": np.array([0, 0]), "cpu": np.array([3, 0])})
    assert info["invalid_cpu_actors"] == ["a"] and env.mission.processing_seconds["a"] == 0
    env.step({"radio": np.array([0, 0]), "cpu": np.array([3, 0])})
    assert env.mission.processing_completed["new"] == 2
    assert env.mission.processing_remaining["a"] == 3


def test_mutually_exclusive_cpu_control_modes_are_explicit(spec):
    from space_env.reference.event_env import ReferenceEventDownlinkEnv

    with pytest.raises(ValueError, match="choose"):
        ReferenceDownlinkEnv(spec, cpu_control=True, cpu_job_selection=True)
    with pytest.raises(ValueError, match="choose"):
        ReferenceEventDownlinkEnv(spec, [], cpu_control=True, cpu_job_selection=True)
