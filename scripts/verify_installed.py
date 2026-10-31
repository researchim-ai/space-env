"""Offline smoke checks against a wheel installed outside the source checkout.

Run with the target environment's `python -I` and a working directory outside repo.
"""
from __future__ import annotations

import argparse
import json
import os
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
from space_env.reference.adaptation import adapt_reference_run
from space_env.reference.benchmark import (
    evaluate_calendar_suite,
    save_calendar_benchmark,
    verify_calendar_benchmark,
)
from space_env.reference.event_env import ReferenceEventDownlinkEnv
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.reference.optimal import solve_small_downlinks
from space_env.reference.scheduling import plan_remaining_downlinks
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
        recorder = ReferenceOperationsRecorder(operations_spec, lineage="installed-dynamic-reference")
        recorder.step(600, {})
        recorder.step(1, {"a": "ReferenceGround"})
        recorder.save(operation_run)
        events = [{"id": "new-scene", "type": "onboard_product_added", "time_sec": 601,
                   "source": "installed-smoke", "reason": "new onboard data",
                   "payload": {"id": "urgent", "actor": "a", "size_bits": 1e6,
                               "ready_sec": 601, "deadline_sec": 1200, "priority": 9}},
                  {"id": "station-fault", "type": "station_outage_added", "time_sec": 601,
                   "source": "installed-smoke", "reason": "temporary station fault",
                   "payload": {"station": "ReferenceGround", "start_sec": 601, "end_sec": 603}}]
        dynamic, schedule = adapt_reference_run(operation_run, events)
        dynamic.save(operation_run)
        dynamic_replay = replay_reference_operations(operation_run)
        assert dynamic_replay["ok"] and schedule["report"]["on_time_bits"] == 3e6
        dynamic_env = ReferenceEventDownlinkEnv(operations_spec, events, product_slots=2, dt_sec=1200)
        dynamic_rows = rollout_episode(dynamic_env, seed=7, policy=lambda observation: np.array([1]))
        assert dynamic_rows[-1]["success"] and sum(r["reward"] for r in dynamic_rows) == 3
        recorder = ReferenceOperationsRecorder(operations_spec, lineage="installed-lifecycle-reference")
        recorder.step(600, {})
        recorder.step(1, {"a": "ReferenceGround"})
        recorder.save(operation_run)
        lifecycle_events = [events[0],
            {"id": "cancel", "type": "product_cancelled", "time_sec": 601,
             "source": "installed-smoke", "reason": "replace old scene", "payload": {"product_id": "scene"}},
            {"id": "degrade", "type": "transmitter_rate_changed", "time_sec": 601,
             "source": "installed-smoke", "reason": "degraded hardware", "payload": {"actor": "a", "rate_bps": .5e6}},
            {"id": "station-cap", "type": "station_rate_changed", "time_sec": 601,
             "source": "installed-smoke", "reason": "ground cap", "payload": {"station": "ReferenceGround", "rate_bps": .75e6}}]
        lifecycle, lifecycle_schedule = adapt_reference_run(operation_run, lifecycle_events)
        lifecycle.save(operation_run)
        lifecycle_replay = replay_reference_operations(operation_run)
        assert lifecycle_replay["ok"] and lifecycle_schedule["report"]["on_time_bits"] == 1e6
        assert lifecycle.mission.report()["inventory_ledger"]["a"]["discarded_bits"] == 1e6
        recorder = ReferenceOperationsRecorder(operations_spec, lineage="installed-booking-reference")
        recorder.step(600, {})
        recorder.save(operation_run)
        booking_events = [
            {"id": "policy", "type": "station_reservation_policy_changed", "time_sec": 600,
             "source": "installed-smoke", "reason": "reserved service", "payload": {"station": "ReferenceGround", "required": True}},
            {"id": "reservation", "type": "terminal_booking_added", "time_sec": 600,
             "source": "installed-smoke", "reason": "mission pass", "payload": {"id": "pass-a", "actor": "a",
             "station": "ReferenceGround", "start_sec": 600, "end_sec": 605}}]
        booked, booking_schedule = adapt_reference_run(operation_run, booking_events)
        booked.save(operation_run)
        booking_replay = replay_reference_operations(operation_run)
        assert booking_replay["ok"] and booking_schedule["report"]["on_time_bits"] == 2e6
        assert booked.mission.report()["booking_utilization"][0]["utilization"] == 1
        thermal_spec = json.loads(json.dumps(operations_spec))
        thermal_spec["vehicles"]["a"].update(rate_bps=.5e6, max_tx_sec=5, tx_power_w=100,
            thermal={"ambient_k": 280, "initial_k": 280, "capacity_j_per_k": 10,
                     "conductance_w_per_k": 1, "heat_tx_w": 100, "trip_k": 300, "resume_k": 290})
        recorder = ReferenceOperationsRecorder(thermal_spec, lineage="installed-thermal-reference")
        recorder.step(600, {})
        schedule = plan_remaining_downlinks(recorder.mission, decision_sec=10)
        for row in schedule["plan"]["steps"]:
            recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
        recorder.save(operation_run)
        thermal_replay = replay_reference_operations(operation_run)
        assert thermal_replay["ok"] and recorder.mission.report()["on_time_bits"] == 2e6
        assert abs(recorder.mission.tx_seconds["a"]-4) < 1e-9
        assert abs(recorder.mission.report()["transmitter_health"]["a"]["thermal_balance_error_j"]) < 1e-9
        processing_spec = json.loads(json.dumps(operations_spec))
        processing_spec["vehicles"]["a"].update(memory_bits=3e6, processing_power_w=18)
        processing_spec["products"][0].update(processing_sec=2, workspace_bits=.5e6)
        recorder = ReferenceOperationsRecorder(processing_spec, lineage="installed-processing-reference")
        recorder.step(600, {}, processing={"a": False})
        schedule = plan_remaining_downlinks(recorder.mission)
        for row in schedule["plan"]["steps"]:
            recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
        recorder.save(operation_run)
        processing_replay = replay_reference_operations(operation_run)
        assert processing_replay["ok"] and recorder.mission.report()["on_time_bits"] == 2e6
        assert recorder.mission.processing_seconds["a"] == 2
        assert recorder.mission.processing_completed["scene"] == 602
        choice_spec = json.loads(json.dumps(operations_spec))
        choice_spec["vehicles"]["a"]["processing_power_w"] = 18
        choice_spec["products"] = [
            {"id": "long", "actor": "a", "size_bits": 1e6, "ready_sec": 600, "deadline_sec": 1200,
             "processing_sec": 10, "priority": 10},
            {"id": "short", "actor": "a", "size_bits": 1e6, "ready_sec": 600, "deadline_sec": 603,
             "processing_sec": 1}]
        recorder = ReferenceOperationsRecorder(choice_spec, lineage="installed-cpu-choice")
        recorder.step(600, {}, processing={"a": False})
        recorder.step(2, {"a": "ReferenceGround"}, processing_jobs={"a": "short"})
        recorder.step(11, {"a": "ReferenceGround"}, processing_jobs={"a": "long"})
        recorder.save(operation_run)
        cpu_choice_replay = replay_reference_operations(operation_run)
        assert cpu_choice_replay["ok"] and recorder.mission.report()["on_time_bits"] == 2e6
        assert recorder.mission.processing_completed["short"] == 601
        dependency_spec = json.loads(json.dumps(processing_spec))
        dependency_spec["products"][0]["processing_sec"] = 1
        dependency_spec["products"].append({"id": "dependent", "actor": "a", "size_bits": .5e6,
            "deadline_sec": 1200, "processing_sec": 1, "workspace_bits": .25e6,
            "priority": 9, "depends_on": ["scene"]})
        recorder = ReferenceOperationsRecorder(dependency_spec, lineage="installed-dependency-reference")
        recorder.step(600, {}, processing={"a": False})
        schedule = plan_remaining_downlinks(recorder.mission)
        for row in schedule["plan"]["steps"]:
            recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
        recorder.save(operation_run)
        dependency_replay = replay_reference_operations(operation_run)
        assert dependency_replay["ok"] and recorder.mission.report()["on_time_bits"] == 2.5e6
        assert recorder.mission.processing_completed["scene"] == 601
        assert recorder.mission.processing_completed["dependent"] == 602
        assert min(r["start_sec"] for r in recorder.mission.receipts if r["product_id"] == "scene") >= 602
        oracle = solve_small_downlinks(operations_spec)
        assert oracle["optimality_proven_in_scope"] and oracle["report"]["on_time_bits"] == 2e6
        calendar = evaluate_calendar_suite([{"id": "installed", "lineage": "installed-calendar",
                                           "split": "test", "spec": operations_spec}])
        calendar_path = Path(tmp) / "calendar.json"
        save_calendar_benchmark(calendar_path, calendar)
        calendar_verification = verify_calendar_benchmark(calendar_path)
        assert calendar_verification["ok"]
        console_env = dict(os.environ)
        console_env.pop("PYTHONPATH", None)
        console_output = subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
            "reference-verify-benchmark", str(calendar_path)], cwd=tmp, env=console_env, text=True)
        console_verification = json.loads(console_output)
        assert console_verification["ok"]
        from space_env.runtime import AccessPolicy, MissionRuntime, reference_scenario
        unified = MissionRuntime(reference_scenario(operations_spec, id="installed", lineage="B01"),
                                 access=AccessPolicy(truth=True))
        unified.advance(600, expected_revision=0, idempotency_key="coast")
        command = {"command_id": "radio", "actor_id": "a", "kind": "downlink",
                   "duration_sec": 5, "destination_id": "ReferenceGround"}
        admitted = unified.submit([command], expected_revision=1, idempotency_key="order")
        unified.advance(1, expected_revision=2, idempotency_key="prefix")
        unified_path = Path(tmp) / "unified.json"
        unified.save(unified_path)
        continued = MissionRuntime.load(unified_path, access=AccessPolicy(truth=True))
        assert continued.submit([command], expected_revision=1, idempotency_key="order") == admitted
        continued.advance(4, expected_revision=3, idempotency_key="continue")
        assert continued.report()["metrics"]["on_time_bits"] == 2e6
        continued.save(unified_path)
        unified_console = json.loads(subprocess.check_output(
            [str(args.expected_prefix / "bin" / "space-env"), "mission-replay", str(unified_path)],
            cwd=tmp, env=console_env, text=True))
        assert unified_console["ok"]
        from space_env.runtime import dynamic_reference_scenario
        dynamic_spec = {"schema": "space-env.reference-dynamic-spec/1", "horizon_sec": 1200.,
            "eop": reference_result["manifest"]["eop"],
            "orbits": {"a": {"initial": {"frame": "GCRS", "epoch_utc": reference_result["epoch_utc"],
                "r_m": reference_result["states_si"][0][:3], "v_m_s": reference_result["states_si"][0][3:]},
                "propulsion": {"dry_mass_kg": 90., "propellant_kg": 10., "isp_s": 300., "max_thrust_n": 2.}}},
            "operations": {k: v for k, v in operations_spec.items() if k != "ephemerides"}}
        dynamic_spec["acquisition_targets"] = [{"id": "site", "lat_deg": access["station"]["lat_deg"],
                                                 "lon_deg": access["station"]["lon_deg"]}]
        dynamic_spec["acquisition_sensors"] = {"a": {"max_off_nadir_deg": 45.}}
        from space_env.reference.covariance import NavigationSettings
        dynamic_spec["orbits"]["a"]["navigation"] = NavigationSettings(
            initial_covariance_si=np.diag([100., 100., 100., .01, .01, .01]).tolist(),
            acceleration_psd_m2_s3=(np.eye(3)*1e-6).tolist(),
            impulse_velocity_cov_m2_s2=(np.eye(3)*.0025).tolist()).model_dump(mode="json")
        dynamic_runtime = MissionRuntime(dynamic_reference_scenario(dynamic_spec, id="installed-dynamic", lineage="B02"),
                                         access=AccessPolicy(truth=True, forecast=True))
        assert "state.covariance_propagation" in dynamic_runtime.describe_capabilities()["supported"]
        dynamic_runtime.advance(600, expected_revision=0, idempotency_key="coast")
        dynamic_runtime.submit([{"command_id": "kick", "actor_id": "a", "kind": "impulse", "duration_sec": 0.,
            "impulse": {"frame": "GCRS", "delta_v_m_s": [0., 0., 1.]}}], expected_revision=1, idempotency_key="kick")
        dynamic_runtime.submit([{"command_id": "burn", "actor_id": "a", "kind": "finite_burn", "duration_sec": 2.,
            "burn": {"frame": "GCRS", "direction_gcrs": [0., 1., 0.], "thrust_n": 2.}}], expected_revision=2, idempotency_key="burn")
        dynamic_runtime.advance(1, expected_revision=3, idempotency_key="prefix")
        dynamic_path = Path(tmp) / "dynamic-runtime.json"
        dynamic_runtime.save(dynamic_path)
        navigation_before = dynamic_runtime.observe("a")["actors"]["a"]["orbit"]["covariance_si"]
        dynamic_runtime = MissionRuntime.load(dynamic_path, access=AccessPolicy(truth=True, forecast=True))
        assert dynamic_runtime.observe("a")["actors"]["a"]["orbit"]["covariance_si"] == navigation_before
        dynamic_runtime.advance(1, expected_revision=4, idempotency_key="finish")
        opportunity = dynamic_runtime.forecast_acquisition("a", "site")
        assert opportunity["result"]["opportunities"]
        eclipses = dynamic_runtime.forecast_eclipses("a")
        assert eclipses["result"]["model"] == "core.spherical-conical-linear-penumbra/1"
        binding = dynamic_runtime.forecast("a", "ReferenceGround")["binding"]
        dynamic_runtime.submit([{**command, "command_id": "new-radio", "duration_sec": 3., "forecast": binding}],
                               expected_revision=5, idempotency_key="downlink")
        dynamic_runtime.advance(3, expected_revision=6, idempotency_key="delivery")
        assert dynamic_runtime.report()["metrics"]["on_time_bits"] == 2e6
        dynamic_runtime.save(dynamic_path)
        dynamic_console = json.loads(subprocess.check_output(
            [str(args.expected_prefix / "bin" / "space-env"), "mission-replay", str(dynamic_path)],
            cwd=tmp, env=console_env, text=True))
        assert dynamic_console["ok"]
        qualification_path = Path(tmp) / "physics-qualification.json"
        subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"), "reference-qualify-orbits",
            str(qualification_path)], cwd=tmp, env=console_env, text=True)
        physics_qualification = json.loads(qualification_path.read_text())
        assert physics_qualification["passed"]
        assert physics_qualification["covariance_prediction"]["passed"]
        assert physics_qualification["geometry_events"]["passed"]
        geometry_smoke = {"verified": True, "opportunities": len(opportunity["result"]["opportunities"]),
                          "binding_checked": True, "model": opportunity["result"]["model"]}
        from space_env.runtime import lifecycle_scenario
        lifecycle_spec = json.loads(json.dumps(dynamic_spec))
        lifecycle_spec["operations"]["products"] = []
        lifecycle_spec["lifecycle"] = {"payloads": {"a": {"power_w": 1.}}, "execution_step_sec": 5.,
            "requests": [{"id": "customer", "target_id": "site", "eligible_actors": ["a"],
                "release_sec": 600., "deadline_sec": 620., "acquisition_sec": 1., "raw_product_id": "raw",
                "raw_size_bits": 1000000, "delivery_product_id": "final", "destination_id": "ReferenceGround"}],
            "jobs": [{"id": "features", "inputs": ["raw"], "outputs": [{"id": "final", "size_bits": 500000}],
                "work_sec": 1., "workspace_bits": 100000, "power_w": 1., "transformer": "geometry-features/1"}]}
        eo = MissionRuntime(lifecycle_scenario(lifecycle_spec, id="installed-eo", lineage="B03"), access=AccessPolicy(truth=True))
        assert eo.checkpoint()["runtime"]["frame_contract"] == "GCRS/ITRS"
        eo.advance(600, expected_revision=0, idempotency_key="coast")
        eo.submit([{"command_id": "image", "actor_id": "a", "kind": "imaging", "duration_sec": 1.,
            "request_id": "customer"}], expected_revision=1, idempotency_key="image")
        eo.advance(1, expected_revision=2, idempotency_key="acquire")
        eo.submit([{"command_id": "transform", "actor_id": "a", "kind": "transform", "duration_sec": 1.,
            "job_id": "features"}], expected_revision=3, idempotency_key="transform")
        eo.advance(.5, expected_revision=4, idempotency_key="partial")
        lifecycle_path = Path(tmp) / "eo.json"
        eo.save(lifecycle_path)
        eo = MissionRuntime.load(lifecycle_path, access=AccessPolicy(truth=True))
        eo.advance(.5, expected_revision=5, idempotency_key="finish")
        eo.submit([{"command_id": "delivery", "actor_id": "a", "kind": "downlink", "duration_sec": .5,
            "product_id": "final", "destination_id": "ReferenceGround"}], expected_revision=6, idempotency_key="delivery")
        eo.advance(.5, expected_revision=7, idempotency_key="radio")
        eo.save(lifecycle_path)
        eo_console = json.loads(subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
            "mission-replay", str(lifecycle_path)], cwd=tmp, env=console_env, text=True))
        assert eo_console["ok"] and eo_console["report"]["metrics"]["fulfilled_requests"] == 1
        eo_report = eo._backend.report()["lifecycle"]
        assert eo_report["ledger"]["a"]["balance_error_bits"] == 0
        lifecycle_smoke = {"verified": True, "metrics": eo_report["metrics"],
                           "ledger": eo_report["ledger"], "acknowledgements": eo_report["acknowledgements"]}
        from space_env.runtime import spacecraft_scenario
        from space_env.reference.spacecraft_bus import bus_preset, pointing_quaternion
        bus_spec = json.loads(json.dumps(lifecycle_spec))
        bus_spec["operations"]["vehicles"]["a"]["solar_mode"] = "body_fixed_bus"
        config = bus_preset("synthetic-agile/1")
        goal = pointing_quaternion(eo._backend.mission.ephemerides["a"],
                                   eo._backend.acquisition_targets["site"], 600.)
        config["attitude"]["initial_quaternion_wxyz"] = goal.tolist()
        bus_spec["spacecraft_bus"] = {"a": config}
        target = next(row for row in bus_spec["acquisition_targets"] if row["id"] == "site")
        lon, lat = target["lon_deg"], target["lat_deg"]
        def wrap_longitude(value):
            return (value + 180) % 360 - 180
        shell = [[wrap_longitude(lon + dx), lat + dy] for dx, dy in
                 ((-.005, -.005), (.005, -.005), (.005, .005), (-.005, .005), (-.005, -.005))]
        bus_spec["lifecycle"]["areas_of_interest"] = [{"id": "pilot-aoi", "anchor_lon_deg": lon,
            "anchor_lat_deg": lat, "shell": shell}]
        bus_spec["lifecycle"]["requests"][0]["coverage"] = {"aoi_id": "pilot-aoi", "min_fraction": .95}
        bus_spec["lifecycle"]["payloads"]["a"]["footprint"] = {
            "half_angle_x_deg": 1., "half_angle_y_deg": 1.}
        spacecraft = MissionRuntime(spacecraft_scenario(bus_spec, id="installed-bus", lineage="B04-A"),
                                    access=AccessPolicy(truth=True))
        spacecraft.advance(600., expected_revision=0, idempotency_key="coast")
        spacecraft.submit([{"command_id": "slew", "actor_id": "a", "kind": "slew", "duration_sec": 1.,
                            "slew": {"frame": "GCRS", "quaternion_wxyz": goal.tolist()}}],
                          expected_revision=1, idempotency_key="slew")
        spacecraft.advance(.5, expected_revision=2, idempotency_key="partial-slew")
        bus_path = Path(tmp) / "bus.json"
        spacecraft.save(bus_path)
        spacecraft = MissionRuntime.load(bus_path, access=AccessPolicy(truth=True))
        spacecraft.advance(.5, expected_revision=3, idempotency_key="settled")
        for kind, fields, duration in (
            ("imaging", {"request_id": "customer"}, 1.),
            ("transform", {"job_id": "features"}, 1.),
            ("downlink", {"product_id": "final", "destination_id": "ReferenceGround"}, .5),
        ):
            revision = spacecraft.revision
            spacecraft.submit([{"command_id": kind, "actor_id": "a", "kind": kind,
                                "duration_sec": duration, **fields}],
                              expected_revision=revision, idempotency_key=kind)
            if kind == "imaging":
                spacecraft.advance(duration / 2, expected_revision=revision + 1, idempotency_key="area-prefix")
                spacecraft.save(bus_path)
                spacecraft = MissionRuntime.load(bus_path, access=AccessPolicy(truth=True))
                spacecraft.advance(duration / 2, expected_revision=spacecraft.revision, idempotency_key="area-finish")
            else:
                spacecraft.advance(duration, expected_revision=revision + 1, idempotency_key=kind + "-time")
        spacecraft.save(bus_path)
        bus_report = spacecraft.observe(view="truth")["backend"]["report"]
        assert bus_report["lifecycle"]["metrics"]["fulfilled_requests"] == 1
        for row in bus_report["spacecraft_bus"]["ledgers"].values():
            assert all(abs(value) < 1e-7 for key, value in row.items() if "error" in key)
        bus_console = json.loads(subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
            "mission-replay", str(bus_path)], cwd=tmp, env=console_env, text=True))
        assert bus_console["ok"]
        area = bus_report["lifecycle"]["area_coverage"]
        assert area["delivered"]["pilot-aoi"]["fraction"] >= .95
        assert area["delivered"]["pilot-aoi"]["covered_area_m2"] <= area["areas"]["pilot-aoi"]["area_m2"] * (1 + 1e-12)
        area_smoke = {"verified": True, "acquired_fraction": area["acquired"]["pilot-aoi"]["fraction"],
                      "delivered_fraction": area["delivered"]["pilot-aoi"]["fraction"],
                      "area_m2": area["areas"]["pilot-aoi"]["area_m2"], "partial_acquisition_checkpoint": True,
                      "geometry_manifest": spacecraft._backend.lifecycle.coverage_geometry_manifest}
        spacecraft_smoke = {"verified": True, "metrics": bus_report["lifecycle"]["metrics"],
                            "ledgers": bus_report["spacecraft_bus"]["ledgers"], "checkpoint_replay": True}
        import importlib.util
        from space_env.runtime.cli import apply_plan
        quality_path = Path(__file__).resolve().parents[1] / "examples/payload_quality/demo.py"
        quality_loader = importlib.util.spec_from_file_location("installed_quality_demo", quality_path)
        quality_module = importlib.util.module_from_spec(quality_loader)
        quality_loader.loader.exec_module(quality_module)
        payload_quality_smoke = {}
        for kind in ("optical", "sar"):
            quality_scenario, (quality_prefix, quality_resume) = quality_module.build_scenario(kind)
            image_runtime = MissionRuntime(quality_scenario, access=AccessPolicy(truth=True))
            apply_plan(image_runtime, quality_prefix)
            quality_record = Path(tmp) / ("quality-" + kind + ".json")
            image_runtime.save(quality_record)
            image_runtime = MissionRuntime.load(quality_record, access=AccessPolicy(truth=True))
            apply_plan(image_runtime, quality_resume)
            image_runtime.save(quality_record)
            image_report = image_runtime.observe(view="truth")["backend"]["report"]
            image_quality = image_report["lifecycle"]["payload_quality"]
            assert image_quality["requests"]["customer-a"]["delivered_and_fulfilled"]
            assert image_report["lifecycle"]["metrics"]["fulfilled_requests"] == 1
            image_console = json.loads(subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
                "mission-replay", str(quality_record)], cwd=tmp, env=console_env, text=True))
            assert image_console["ok"]
            payload_quality_smoke[kind] = {"verified": True,
                "metrics": image_report["lifecycle"]["metrics"], "request": image_quality["requests"]["customer-a"],
                "sar_intervals": image_quality["sar_intervals"], "mid_acquisition_replay": True}
        observation_path = Path(__file__).resolve().parents[1] / "examples/observations/demo.py"
        observation_loader = importlib.util.spec_from_file_location("installed_observation_demo", observation_path)
        observation_module = importlib.util.module_from_spec(observation_loader)
        observation_loader.loader.exec_module(observation_module)
        observation_smoke = {}
        for mode in ("revisit", "stereo-fleet"):
            observation_scenario, (observation_prefix, observation_resume) = observation_module.build_scenario(mode)
            observation_runtime = MissionRuntime(observation_scenario, access=AccessPolicy(truth=True))
            apply_plan(observation_runtime, observation_prefix)
            observation_record = Path(tmp) / ("observations-" + mode + ".json")
            observation_runtime.save(observation_record)
            observation_runtime = MissionRuntime.load(observation_record, access=AccessPolicy(truth=True))
            for operation in observation_resume["operations"]:
                apply_plan(observation_runtime, {"operations": [operation]})
                if operation["idempotency_key"] == "partial-radio":
                    observation_runtime.save(observation_record)
                    observation_runtime = MissionRuntime.load(observation_record, access=AccessPolicy(truth=True))
            observation_runtime.save(observation_record)
            observation_report = observation_runtime.observe(view="truth")["backend"]["report"]["lifecycle"]
            obligation = observation_report["observation_requirements"]["customer-a"]
            assert obligation["delivered_and_fulfilled"]
            assert observation_report["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
            observation_console = json.loads(subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
                "mission-replay", str(observation_record)], cwd=tmp, env=console_env, text=True))
            assert observation_console["ok"]
            observation_smoke[mode] = {"verified": True, "metrics": observation_report["metrics"],
                "joint_fraction": obligation["joint_fraction"], "distinct_acquisitions": obligation["distinct_acquisitions"],
                "full_delivery": obligation["full_delivery"], "mid_acquisition_and_partial_radio_replay": True}
        lanes_path = Path(__file__).resolve().parents[1] / "examples/resource_lanes/demo.py"
        lanes_loader = importlib.util.spec_from_file_location("installed_lanes_demo", lanes_path)
        lanes_module = importlib.util.module_from_spec(lanes_loader)
        lanes_loader.loader.exec_module(lanes_module)
        lanes_smoke = {}
        for mode in ("wide", "constrained"):
            lanes_scenario, (lanes_prefix, lanes_resume) = lanes_module.build_scenario(mode)
            lanes_runtime = MissionRuntime(lanes_scenario, access=AccessPolicy(truth=True))
            apply_plan(lanes_runtime, lanes_prefix)
            lanes_record = Path(tmp) / ("lanes-" + mode + ".json")
            lanes_runtime.save(lanes_record)
            lanes_runtime = MissionRuntime.load(lanes_record, access=AccessPolicy(truth=True))
            for operation in lanes_resume["operations"]:
                apply_plan(lanes_runtime, {"operations":[operation]})
                if operation["idempotency_key"] == "partial-lanes":
                    lanes_runtime.save(lanes_record)
                    lanes_runtime = MissionRuntime.load(lanes_record, access=AccessPolicy(truth=True))
            lanes_runtime.save(lanes_record)
            lanes_report = lanes_runtime.observe(view="truth")["backend"]["report"]["lifecycle"]
            assert lanes_report["metrics"] == {"fulfilled_requests":3,"total_requests":3}
            overlap = sum(len(row["actors"]["a"]["managed"]) == 2 and row["actors"]["a"]["radio_bits"] > 0
                          for row in lanes_report["resource_lanes"]["intervals"])
            assert overlap > 0
            lanes_console = json.loads(subprocess.check_output([str(args.expected_prefix / "bin" / "space-env"),
                "mission-replay",str(lanes_record)],cwd=tmp,env=console_env,text=True))
            assert lanes_console["ok"]
            lanes_smoke[mode] = {"verified":True,"metrics":lanes_report["metrics"],"overlap_intervals":overlap,
                "memory_io_bits":sum(row["actors"]["a"]["memory_io_bits"] for row in lanes_report["resource_lanes"]["intervals"]),
                "three_unfinished_lanes_replay":True}
        actuation_path = Path(__file__).resolve().parents[1] / "examples/actuation/demo.py"
        actuation_loader = importlib.util.spec_from_file_location("installed_actuation_demo", actuation_path)
        actuation_module = importlib.util.module_from_spec(actuation_loader)
        actuation_loader.loader.exec_module(actuation_module)
        actuation_smoke = {}
        for kind in ("chemical", "electric"):
            actuation_scenario, (actuation_prefix, actuation_continuation) = actuation_module.build_scenario(kind)
            actuation_runtime = MissionRuntime(actuation_scenario, access=AccessPolicy(truth=True))
            apply_plan(actuation_runtime, actuation_prefix)
            actuation_record = Path(tmp) / ("actuation-" + kind + ".json")
            actuation_runtime.save(actuation_record)
            actuation_runtime = MissionRuntime.load(actuation_record, access=AccessPolicy(truth=True))
            for operation in actuation_continuation["operations"]:
                apply_plan(actuation_runtime, {"operations": [operation]})
                if operation["idempotency_key"] == "partial-firing":
                    actuation_runtime.save(actuation_record)
                    actuation_runtime = MissionRuntime.load(actuation_record, access=AccessPolicy(truth=True))
            actuation_runtime.save(actuation_record)
            actuation_life = actuation_runtime.observe(view="truth")["backend"]["report"]["lifecycle"]
            engine = actuation_life["actuation"]
            assert actuation_life["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
            assert engine["operations"]["engine"]["status"] == "completed"
            assert engine["operations"]["engine"]["firing_sec"] == 1.0
            assert abs(engine["fuel_balance_error_kg"]["a"]) < 1e-12
            assert abs(engine["electrical_heat_balance_error_j"]["engine"]) < 1e-9
            actuation_replay = json.loads(subprocess.check_output([
                str(args.expected_prefix / "bin" / "space-env"), "mission-replay", str(actuation_record)
            ], cwd=tmp, env=console_env, text=True))
            assert actuation_replay["ok"]
            actuation_smoke[kind] = {"verified": True, "metrics": actuation_life["metrics"],
                "mid_gimbal_and_firing_replay": True,
                "fuel_kg": engine["states"]["a"]["consumed_propellant_kg"]}
        current_covariance = np.asarray(dynamic_runtime.observe("a")["actors"]["a"]["orbit"]["covariance_si"])
        assert np.linalg.eigvalsh(current_covariance).min() >= -1e-9
        covariance_smoke = {"verified": True, "frame": "GCRS",
                            "position_stddev_rss_m": float(np.sqrt(np.trace(current_covariance[:3, :3])))}


    assert physics["passed"], physics
    cli = subprocess.check_output(
        [sys.executable, "-I", "-m", "space_env.cli", "validate", str(scenario)], text=True,
    )
    print(json.dumps({
        "package": str(root), "dict_steps": len(rows), "suite_passed": suite["passed"],
        "cases": len(suite["cases"]), "reference_access_windows": len(access["windows"]),
        "reference_reward_mbit": reference_reward, "reference_operations_replay": reference_operations_replay,
        "dynamic_reference_replay": dynamic_replay, "dynamic_rl_steps": len(dynamic_rows),
        "lifecycle_reference_replay": lifecycle_replay,
        "booking_reference_replay": booking_replay,
        "thermal_reference_replay": thermal_replay,
        "processing_reference_replay": processing_replay,
        "cpu_choice_replay": cpu_choice_replay,
        "dependency_reference_replay": dependency_replay,
        "calendar_verification": calendar_verification,
        "console_verification": console_verification,
        "unified_runtime": unified_console,
        "dynamic_reference_runtime": dynamic_console,
        "navigation_covariance": covariance_smoke,
        "geometry_forecast": geometry_smoke,
        "product_lifecycle": lifecycle_smoke,
        "spacecraft_bus": spacecraft_smoke,
        "area_coverage": area_smoke,
        "payload_quality": payload_quality_smoke,
        "observation_requirements": observation_smoke,
        "resource_lanes": lanes_smoke,
        "body_actuation": actuation_smoke,
        "dynamic_physics_qualification": {"passed": physics_qualification["passed"],
            "status": physics_qualification["qualification_status"],
            "fixture_cases": len(physics_qualification["orbit_fixtures"]["cases"]),
            "covariance_passed": physics_qualification["covariance_prediction"]["passed"],
            "geometry_passed": physics_qualification["geometry_events"]["passed"]},
        "external_force_crosscheck": physics["force_crosscheck"]["passed"], "gcrs_reference": {"passed": physics["passed"], "checks": len(physics["checks"]), "scope": "upstream rotation and circular two-body 600/24h/72h"}, "studio_status": 200,
        "restored_time_sec": restored_time, "portable_replay": portable, "agent_replay": agent_replay, "dialogue_messages": dialogue["messages"], "transition_rows": dataset["rows"], "cli_validate": json.loads(cli),
        "dependencies": {name: version(name) for name in ("numpy", "boinor", "gymnasium", "astropy", "pydantic")},
    }))


if __name__ == "__main__":
    main()
