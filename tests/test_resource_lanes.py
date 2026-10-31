"""Concurrent execution, aggregate ledgers, arbitration and atomic ownership."""

from __future__ import annotations

import copy

import pytest

from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from tests import test_reference_lifecycle as life
from tests import test_spacecraft_bus as bus_fixtures


@pytest.fixture(scope="module")
def scenario():
    data = copy.deepcopy(bus_fixtures.scenario.__wrapped__())
    definition = data["definition"]
    definition["execution"] = {
        "mode": "resource-lanes/1",
        "actors": {
            a: {"memory_bandwidth_bps": 10000000.0} for a in data["actor_order"]
        },
    }
    definition["operations"]["products"] = [
        {
            "id": "archive",
            "actor": "a",
            "size_bits": 1000000,
            "ready_sec": 0.0,
            "deadline_sec": 20.0,
            "destination": "Ground",
        }
    ]
    definition["operations"]["vehicles"]["a"].update(
        memory_bits=8000000.0, base_power_w=2.0, tx_power_w=8.0
    )
    spec = definition["lifecycle"]
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2", delivery_product_id="raw-2")
    spec["requests"].append(second)
    return data


def report(model):
    return life.lifecycle(model)["resource_lanes"]


def command(kind, id, **fields):
    return {
        "command_id": id,
        "actor_id": "a",
        "kind": kind,
        "duration_sec": 4.0,
        **fields,
    }


def submit(model, commands, key="batch"):
    return model.submit(commands, expected_revision=model.revision, idempotency_key=key)


def ready(data):
    model = life.runtime(data)
    life.acquire(model)
    return model


def workload(model, duration=4.0, order=("imaging", "transform", "downlink")):
    rows = {
        "imaging": command("imaging", "image-2", request_id="second"),
        "transform": command("transform", "cpu", job_id="compress"),
        "downlink": command(
            "downlink", "radio", product_id="archive", destination_id="Ground"
        ),
    }
    for row in rows.values():
        row["duration_sec"] = duration
    submit(model, [rows[k] for k in order])


def test_three_lanes_really_progress_in_one_wall_clock_interval(scenario):
    model = ready(scenario)
    bus_before = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"][
        "states"
    ]["a"]
    initial = model._backend.mission.energy_wh["a"]
    workload(model, duration=2.0)
    life.advance(model, 2.0, "parallel-time")
    state = life.lifecycle(model)
    assert state["jobs"]["compress"]["status"] == "completed"
    assert state["artifacts"]["raw-2"]["created_sec"] == 4.0
    assert model._backend.mission.remaining["archive"] == 0
    assert state["artifacts"]["final"]["created_sec"] == 4.0
    bus = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]
    adcs = bus["states"]["a"]["slew_energy_wh"] - bus_before["slew_energy_wh"]
    expected_useful = (2 * 2 + 10 * 2 + 5 * 2 + 8 * 1) / 3600 + adcs
    assert initial - model._backend.mission.energy_wh["a"] == pytest.approx(
        expected_useful / 0.9, abs=1e-12
    )
    assert abs(bus["ledgers"]["a"]["battery_balance_error_wh"]) < 1e-10
    assert abs(bus["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-8
    intervals = [r for r in report(model)["intervals"] if r["start_sec"] >= 2]
    assert any(
        len(r["actors"]["a"]["managed"]) == 2 and r["actors"]["a"]["radio_bits"] > 0
        for r in intervals
    )


def test_memory_bandwidth_throttles_cpu_and_radio_without_fabricating_work(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["execution"]["actors"]["a"]["memory_bandwidth_bps"] = 1000000.0
    model = ready(data)
    workload(model, duration=6.0)
    life.advance(model, 2.0, "first-two")
    state = life.lifecycle(model)
    assert state["jobs"]["compress"]["remaining_sec"] == pytest.approx(2 / 3)
    assert "final" not in state["artifacts"]
    assert model._backend.mission.remaining["archive"] == 1000000
    life.advance(model, 4.0, "finish")
    assert life.lifecycle(model)["jobs"]["compress"]["status"] == "completed"
    assert model._backend.mission.remaining["archive"] == 0
    rows = [r for r in report(model)["intervals"] if r["start_sec"] >= 2]
    assert sum(r["actors"]["a"]["memory_io_bits"] for r in rows) == pytest.approx(
        3500000.0
    )
    for row in rows:
        assert (
            row["actors"]["a"]["memory_io_bits"]
            <= 1000000 * (row["end_sec"] - row["start_sec"]) + 1e-6
        )


def test_shared_peak_limit_pauses_lower_priority_lanes(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 19.0
    model = ready(data)
    workload(model)
    life.advance(model, 2.0, "limited")
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 0
    assert model._backend.mission.remaining["archive"] == 1000000
    assert "raw-2" in life.lifecycle(model)["artifacts"]
    assert any(r["reason"] == "shared_peak_power" for r in report(model)["blocked"])
    life.advance(model, 2.0, "resources-free")
    assert life.lifecycle(model)["jobs"]["compress"]["status"] == "completed"
    assert model._backend.mission.remaining["archive"] == 0


@pytest.mark.parametrize(
    "which", ["two-cpu", "two-payload", "attitude", "wait", "directed-radio"]
)
def test_conflicting_control_claims_reject_whole_batch(scenario, which):
    data = copy.deepcopy(scenario)
    if which == "directed-radio":
        data["definition"]["spacecraft_bus"]["a"]["attitude"]["radio_tolerance_deg"] = (
            5.0
        )
    model = ready(data)
    cpu = command("transform", "cpu", job_id="compress")
    image = command("imaging", "image-2", request_id="second")
    slew = command(
        "slew", "slew", slew={"frame": "GCRS", "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    )
    radio = command("downlink", "radio", product_id="archive", destination_id="Ground")
    rows = {
        "two-cpu": [cpu, {**cpu, "command_id": "duplicate"}],
        "two-payload": [image, {**image, "command_id": "duplicate"}],
        "attitude": [image, slew],
        "wait": [cpu, command("wait", "wait")],
        "directed-radio": [image, radio],
    }[which]
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        submit(model, rows)
    assert error.value.code == "resource_busy"
    assert model.checkpoint() == before


def test_cpu_and_omnidirectional_radio_can_overlap_slew(scenario):
    model = ready(scenario)
    submit(
        model,
        [
            command("transform", "cpu", job_id="compress"),
            command("downlink", "radio", product_id="archive", destination_id="Ground"),
            command(
                "slew",
                "slew",
                slew={"frame": "GCRS", "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            ),
        ],
    )
    life.advance(model, 0.5, "overlap")
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 0.5
    assert model._backend.mission.remaining["archive"] < 1000000


@pytest.mark.parametrize("lane", ["cpu", "payload", "radio"])
def test_disabled_lane_and_legacy_processing_are_explicit(scenario, lane):
    data = copy.deepcopy(scenario)
    data["definition"]["execution"]["actors"]["a"][lane] = False
    model = ready(data) if lane != "payload" else life.runtime(data)
    row = {
        "cpu": command("transform", "cpu", job_id="compress"),
        "payload": command("imaging", "image-2", request_id="second"),
        "radio": command(
            "downlink", "radio", product_id="archive", destination_id="Ground"
        ),
    }[lane]
    with pytest.raises(RuntimeContractError, match="lane is disabled"):
        submit(model, [row])
    assert "commands.processing" not in model.describe_capabilities()["supported"]


def test_serial_mode_keeps_one_active_window_per_actor(scenario):
    data = copy.deepcopy(scenario)
    del data["definition"]["execution"]
    model = ready(data)
    with pytest.raises(RuntimeContractError) as error:
        workload(model)
    assert error.value.code == "actor_busy"
    assert (
        model.describe_capabilities()["concurrency"]
        == "one active control window per actor"
    )


def test_station_outage_releases_bandwidth_and_does_not_charge_radio(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["outages"] = [
        {"station": "Ground", "start_sec": 2.0, "end_sec": 4.0}
    ]
    data["definition"]["execution"]["actors"]["a"]["memory_bandwidth_bps"] = 1000000.0
    model = ready(data)
    workload(model)
    life.advance(model, 2.0, "closed")
    assert model._backend.mission.tx_seconds["a"] == 0
    assert any(
        r["actors"]["a"]["managed"]
        for r in report(model)["intervals"]
        if r["start_sec"] >= 2
    )


def test_shared_soc_floor_stops_cpu_but_does_not_freeze_lower_floor_work(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["initial_soc"] = 0.5
    data["definition"]["lifecycle"]["jobs"][0]["min_soc"] = 0.48
    model = ready(data)
    workload(model)
    life.advance(model, 2.0, "reserve")
    state = life.lifecycle(model)
    assert 0 < state["jobs"]["compress"]["work_sec"] < 2
    assert "raw-2" in state["artifacts"]
    assert model._backend.mission.remaining["archive"] == 0
    assert "final" not in state["artifacts"]


def test_checkpoint_with_three_unfinished_lanes_replays_and_retains_aliases(
    scenario, tmp_path
):
    model = ready(scenario)
    workload(model)
    life.advance(model, 0.25, "partial")
    model.save(tmp_path / "lanes.json")
    other = MissionRuntime.load(
        tmp_path / "lanes.json", access=AccessPolicy(truth=True)
    )
    assert other._backend.mission.bus is other._backend.bus
    for item in (model, other):
        life.advance(item, 3.75, "finish")
    assert model.checkpoint() == other.checkpoint()


def test_submit_order_does_not_override_declared_arbitration_priority(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 20.0
    first, second = ready(data), ready(data)
    workload(first, order=("imaging", "transform", "downlink"))
    workload(second, order=("downlink", "transform", "imaging"))
    for item in (first, second):
        life.advance(item, 2.0, "time")
    assert first._backend.mission.energy_wh == second._backend.mission.energy_wh
    assert life.lifecycle(first)["jobs"] == life.lifecycle(second)["jobs"]


@pytest.mark.parametrize("invalid", ["profile", "actor", "priority", "bandwidth"])
def test_invalid_parallel_configuration_has_no_fallback(scenario, invalid):
    data = copy.deepcopy(scenario)
    if invalid == "actor":
        del data["definition"]["execution"]["actors"]["b"]
    elif invalid == "priority":
        data["definition"]["execution"]["actors"]["a"]["priority"] = [
            "cpu",
            "cpu",
            "radio",
        ]
    elif invalid == "bandwidth":
        data["definition"]["execution"]["actors"]["a"]["memory_bandwidth_bps"] = 0.0
    else:
        del data["definition"]["spacecraft_bus"]
        data.update(
            backend="reference-lifecycle",
            physics_profile="reference-lifecycle/1",
            fidelity={
                "trajectory": "controlled-gcrs",
                "resources": "explicit-surrogate",
            },
        )
        for vehicle in data["definition"]["operations"]["vehicles"].values():
            vehicle["solar_mode"] = "ideal_sun_tracking"
    with pytest.raises(ValueError):
        life.runtime(data)


def test_failed_imaging_does_not_remove_concurrent_cpu_energy(scenario):
    data = copy.deepcopy(scenario)
    target = data["definition"]["acquisition_targets"][0]
    data["definition"]["acquisition_targets"].append(
        {
            "id": "far",
            "lat_deg": -target["lat_deg"],
            "lon_deg": (target["lon_deg"] + 360) % 360 - 180,
        }
    )
    data["definition"]["lifecycle"]["requests"][1]["target_id"] = "far"
    model = ready(data)
    initial = model._backend.mission.energy_wh["a"]
    workload(model, duration=2.0)
    life.advance(model, 2.0, "bad-image")
    state = life.lifecycle(model)
    assert "raw-2" not in state["artifacts"]
    assert state["jobs"]["compress"]["status"] == "completed"
    assert initial - model._backend.mission.energy_wh["a"] == pytest.approx(
        (2 * 2 + 5 * 2 + 8 * 1) / 3600 / 0.9, abs=1e-12
    )


def test_persistent_memory_reservations_prevent_output_overcommit(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["memory_bits"] = 3500000.0
    model = ready(data)
    workload(model)
    life.advance(model, 0.25, "peak-memory")
    assert life.lifecycle(model)["jobs"]["compress"]["reason"] == "memory"
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 0
    life.advance(model, 3.75, "finish")
    assert "final" in life.lifecycle(model)["artifacts"]
    for row in report(model)["intervals"]:
        actor = row["actors"]["a"]
        assert (
            actor["inventory_bits"] + actor["reserved_bits"]
            <= actor["memory_capacity_bits"]
        )


def test_cpu_fault_leaves_payload_and_radio_running_and_resume_keeps_work(scenario):
    model = ready(scenario)
    workload(model)
    life.advance(model, 0.5, "partial")
    model.inject(
        [
            life.event(
                model, "spacecraft_fault", {"actor": "a", "subsystem": "cpu"}, "fault"
            )
        ],
        expected_revision=model.revision,
        idempotency_key="fault",
    )
    life.advance(model, 1.5, "other-lanes")
    state = life.lifecycle(model)
    assert state["jobs"]["compress"]["work_sec"] == 0.5
    assert "raw-2" in state["artifacts"]
    assert model._backend.mission.remaining["archive"] == 0
    model.inject(
        [
            life.event(
                model,
                "spacecraft_fault_cleared",
                {"actor": "a", "subsystem": "cpu"},
                "repair",
            )
        ],
        expected_revision=model.revision,
        idempotency_key="repair",
    )
    life.advance(model, 1.5, "resume")
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 2.0


def test_cancel_one_lane_does_not_cancel_other_work_or_erase_cpu_progress(scenario):
    model = ready(scenario)
    workload(model)
    life.advance(model, 0.5, "partial")
    model.cancel("cpu", expected_revision=model.revision, idempotency_key="cancel-cpu")
    life.advance(model, 1.5, "remaining-lanes")
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 0.5
    assert "raw-2" in life.lifecycle(model)["artifacts"]
    assert model._backend.mission.remaining["archive"] == 0
    submit(
        model,
        [{**command("transform", "cpu-again", job_id="compress"), "duration_sec": 1.5}],
        "resume",
    )
    life.advance(model, 1.5, "cpu-finish")
    assert life.lifecycle(model)["jobs"]["compress"]["work_sec"] == 2.0


@pytest.mark.parametrize("kind", ["imaging", "transform"])
def test_cross_actor_duplicate_request_or_job_cannot_own_same_output(scenario, kind):
    data = copy.deepcopy(scenario)
    for request in data["definition"]["lifecycle"]["requests"]:
        request["eligible_actors"] = ["a", "b"]
    data["definition"]["acquisition_sensors"]["b"] = copy.deepcopy(
        data["definition"]["acquisition_sensors"]["a"]
    )
    model = life.runtime(data)
    fields = {"request_id": "request"} if kind == "imaging" else {"job_id": "compress"}
    rows = [
        command(kind, "one", **fields),
        {**command(kind, "two", **fields), "actor_id": "b"},
    ]
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="managed request/job already"):
        submit(model, rows)
    assert model.checkpoint() == before
    submit(model, [rows[0]], "first")
    before = model.checkpoint()
    with pytest.raises(
        RuntimeContractError,
        match="managed request/job already|request is not released",
    ):
        submit(model, [rows[1]], "second")
    assert model.checkpoint() == before


def test_station_terminal_capacity_is_shared_across_parallel_actors(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["products"].append(
        {
            "id": "archive-b",
            "actor": "b",
            "size_bits": 1000000,
            "ready_sec": 0.0,
            "deadline_sec": 20.0,
            "destination": "Ground",
        }
    )
    model = life.runtime(data)
    submit(
        model,
        [
            command(
                "downlink", "radio-a", product_id="archive", destination_id="Ground"
            ),
            {
                **command(
                    "downlink",
                    "radio-b",
                    product_id="archive-b",
                    destination_id="Ground",
                ),
                "actor_id": "b",
            },
        ],
    )
    life.advance(model, 2.0, "terminals")
    assert model._backend.mission.remaining["archive"] == 0
    assert model._backend.mission.remaining["archive-b"] == 0
    for row in report(model)["intervals"]:
        assert sum(actor["radio_bits"] > 0 for actor in row["actors"].values()) <= 1


def test_unexpected_exception_rolls_back_all_three_lanes_and_scheduler(
    scenario, monkeypatch
):
    model = ready(scenario)
    workload(model)
    before = model.checkpoint()
    mission_type = type(model._backend.mission)
    original = mission_type.step

    def fail_after_step(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected after native integration")

    with monkeypatch.context() as patch:
        patch.setattr(mission_type, "step", fail_after_step)
        with pytest.raises(RuntimeError, match="injected"):
            life.advance(model, 1.0, "failure")
    assert model.checkpoint() == before
    assert model._backend.bus is model._backend.mission.bus
    life.advance(model, 2.0, "success")
    assert life.lifecycle(model)["jobs"]["compress"]["status"] == "completed"


def test_local_lane_observation_exposes_own_grants_without_peer_commands(scenario):
    model = ready(scenario)
    workload(model)
    life.advance(model, 0.25, "time")
    own = model.observe("a")["actors"]["a"]["lifecycle"]["resource_lanes"]
    peer = model.observe("b")["actors"]["b"]["lifecycle"]["resource_lanes"]
    assert len(own["last_interval"]["managed"]) == 2
    assert peer["last_interval"]["managed"] == []
    own["last_interval"]["managed"].clear()
    assert (
        len(
            model.observe("a")["actors"]["a"]["lifecycle"]["resource_lanes"][
                "last_interval"
            ]["managed"]
        )
        == 2
    )


def test_parallel_sar_keeps_rf_dc_heat_and_session_duty_coherent(scenario):
    from tests import test_payload_quality as payload

    data = copy.deepcopy(payload.sar.__wrapped__())
    data["definition"]["execution"] = copy.deepcopy(scenario["definition"]["execution"])
    data["definition"]["operations"]["products"] = copy.deepcopy(
        scenario["definition"]["operations"]["products"]
    )
    data["definition"]["operations"]["vehicles"]["a"].update(
        memory_bits=8000000.0, base_power_w=2.0, tx_power_w=8.0
    )
    spec = data["definition"]["lifecycle"]
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2", delivery_product_id="raw-2")
    spec["requests"].append(second)
    model = ready(data)
    initial = model._backend.mission.energy_wh["a"]
    before = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"][
        "states"
    ]["a"]["slew_energy_wh"]
    workload(model, duration=2.0)
    life.advance(model, 2.0, "sar-parallel")
    state = life.lifecycle(model)
    assert (
        state["jobs"]["compress"]["status"] == "completed"
        and "raw-2" in state["artifacts"]
    )
    intervals = state["payload_quality"]["sar_intervals"]["a"]
    assert sum(r["dc_input_j"] for r in intervals) == 80.0
    assert sum(r["radiated_rf_j"] for r in intervals) == 40.0
    bus = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]
    adcs = bus["states"]["a"]["slew_energy_wh"] - before
    expected = (2 * 2 + 30 * 2 + 5 * 2 + 8 * 1) / 3600 + adcs
    assert initial - model._backend.mission.energy_wh["a"] == pytest.approx(
        expected / 0.9, abs=1e-12
    )


def test_higher_priority_cpu_interrupts_continuous_capture_without_raw(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["execution"]["actors"]["a"].update(
        memory_bandwidth_bps=1000000.0, priority=["cpu", "payload", "radio"]
    )
    model = ready(data)
    submit(
        model, [command("imaging", "image-2", request_id="second")], "new-image-start"
    )
    life.advance(model, 0.5, "started-capture")
    submit(
        model, [command("transform", "cpu", job_id="compress")], "higher-priority-cpu"
    )
    life.advance(model, 0.5, "preempt")
    state = life.lifecycle(model)
    assert (
        state["requests"]["second"]["reason"] == "resource_preempted:memory_bandwidth"
    )
    assert "raw-2" not in state["artifacts"]
    assert state["jobs"]["compress"]["work_sec"] == 0.5


def test_cpu_switching_uses_power_but_not_work_io_bandwidth(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["execution"]["actors"]["a"]["memory_bandwidth_bps"] = 1000000.0
    data["definition"]["lifecycle"]["jobs"][0]["switching_sec"] = 1.0
    model = ready(data)
    workload(model, duration=6.0)
    life.advance(model, 1.0, "switching")
    state = life.lifecycle(model)["jobs"]["compress"]
    assert state["switching_sec"] == 1.0 and state["work_sec"] == 0
    assert model._backend.mission.remaining["archive"] == pytest.approx(500000.0)
    life.advance(model, 5.0, "finish")
    state = life.lifecycle(model)["jobs"]["compress"]
    assert state["switching_sec"] == 1.0 and state["work_sec"] == pytest.approx(2.0)
    assert model._backend.mission.transmitter_rates["a"] == 1000000.0


def test_native_in_place_processing_is_rejected_in_parallel_mode(scenario):
    model = ready(scenario)
    with pytest.raises(RuntimeContractError) as error:
        submit(model, [command("processing", "legacy-cpu", product_id="archive")])
    assert error.value.code == "unsupported_capability"
