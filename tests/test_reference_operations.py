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
