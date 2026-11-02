"""B04-E: independent thermal balances, radiation, routing and mission control."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.reference.thermal_network import (
    HEAT_SOURCES,
    SIGMA_SB,
    ThermalNetworkSettings,
    advance_network,
    projected_earth_view_factor,
)
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_reference_lifecycle as life
from tests import test_spacecraft_bus as fixtures


def node(**updates):
    return {
        "initial_k": 285.0,
        "capacity_j_per_k": 10.0,
        "trip_k": 330.0,
        "resume_k": 310.0,
        **updates,
    }


def network(**updates):
    data = {
        "model_id": "synthetic-network/1",
        "parameter_status": "synthetic",
        "source": "declared analytic engineering assumptions",
        "nodes": {"hull": node(), "cpu": node(), "panel": node(), "camera": node()},
        "links": [{"a": "hull", "b": "cpu", "conductance_w_per_k": 0.1}],
        "routing": {k: {"hull": 1.0} for k in HEAT_SOURCES},
        "earth_environment": False,
    }
    data["routing"].update(
        cpu={"cpu": 1.0},
        payload={"camera": 1.0},
        radio={"cpu": 1.0},
        pv_extraction={"panel": 1.0},
    )
    return data | updates


def step(config, temps, duration, internal=None, external=None):
    names = config.nodes
    return advance_network(
        config,
        temps,
        duration,
        internal_w=internal or dict.fromkeys(names, 0.0),
        external_w=external or dict.fromkeys(names, 0.0),
    )


def two_nodes(**kwargs):
    data = network(
        nodes={
            "a": node(initial_k=300.0, capacity_j_per_k=2.0),
            "b": node(initial_k=280.0, capacity_j_per_k=3.0),
        },
        links=[{"a": "a", "b": "b", "conductance_w_per_k": 0.5}],
        routing={k: {"a": 1.0} for k in HEAT_SOURCES},
    )
    data.update(kwargs)
    return ThermalNetworkSettings(**data)


@pytest.mark.parametrize("solver", ["Radau", "DOP853"])
def test_two_node_closed_system_matches_analytic_exchange(solver):
    c = two_nodes(solver=solver)
    r = step(c, {"a": 300.0, "b": 280.0}, 3.0)
    equilibrium = (2 * 300 + 3 * 280) / 5
    difference = 20 * math.exp(-0.5 * (1 / 2 + 1 / 3) * 3)
    assert r["temperatures_k"]["a"] == pytest.approx(
        equilibrium + 3 / 5 * difference, abs=1e-8
    )
    assert r["temperatures_k"]["b"] == pytest.approx(
        equilibrium - 2 / 5 * difference, abs=1e-8
    )
    assert abs(sum(r["stored_j"].values())) < 1e-9
    assert max(abs(v) for v in r["node_balance_error_j"].values()) < 1e-8


def test_isolated_node_constant_power_linear_rise():
    c = two_nodes(links=[])
    r = step(c, {"a": 300.0, "b": 280.0}, 2.0, internal={"a": 8.0, "b": 0.0})
    assert r["temperatures_k"]["a"] == pytest.approx(308.0)
    assert r["temperatures_k"]["b"] == 280.0


def test_bath_matches_independent_rc_closed_form():
    c = two_nodes(links=[])
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(bath_k=270.0, bath_conductance_w_per_k=0.4)
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 300.0, "b": 280.0}, 3.0, internal={"a": 8.0, "b": 0.0})
    expected = 290.0 + (300.0 - 290.0) * math.exp(-0.4 / 2 * 3)
    assert r["temperatures_k"]["a"] == pytest.approx(expected, abs=1e-8)


def test_radiative_cooling_matches_exact_cold_space_solution():
    c = two_nodes(links=[])
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(emission_area_m2=0.1, emissivity=0.8)
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 300.0, "b": 280.0}, 2.0)
    expected = (300.0**-3 + 3 * 0.8 * SIGMA_SB * 0.1 / 2 * 2) ** (-1 / 3)
    assert r["temperatures_k"]["a"] == pytest.approx(expected, abs=1e-8)
    assert r["radiation_out_j"]["a"] == pytest.approx(-r["stored_j"]["a"], abs=1e-8)


def test_radiative_equilibrium_with_declared_incoming_flux():
    c = two_nodes(links=[])
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(emission_area_m2=0.1, emissivity=0.8)
    c = ThermalNetworkSettings(**d)
    r = step(
        c,
        {"a": 300.0, "b": 280.0},
        2.0,
        external={"a": 0.8 * SIGMA_SB * 0.1 * 300**4, "b": 0.0},
    )
    assert r["temperatures_k"]["a"] == pytest.approx(300.0, abs=1e-10)


def test_stiff_symmetric_conduction_uses_implicit_solver():
    c = two_nodes()
    d = c.model_dump(mode="json")
    d["links"][0]["conductance_w_per_k"] = 1e6
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 300.0, "b": 280.0}, 0.1)
    assert r["temperatures_k"]["a"] == pytest.approx(288.0, abs=1e-6)
    assert r["temperatures_k"]["b"] == pytest.approx(288.0, abs=1e-6)


def test_hot_and_cold_crossings_are_recorded():
    c = two_nodes(links=[])
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(trip_k=301.0, resume_k=299.0)
    d["nodes"]["b"].update(cold_trip_k=279.0, cold_resume_k=280.0)
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 300.0, "b": 280.0}, 1.0, internal={"a": 4.0, "b": -6.0})
    assert {(e["node"], e["kind"]) for e in r["trips"]} == {("a", "hot"), ("b", "cold")}
    assert all(e["time_sec"] == pytest.approx(0.5, abs=1e-8) for e in r["trips"])


def test_constant_threshold_does_not_spam_crossings():
    c = two_nodes(links=[])
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(trip_k=300.0, resume_k=299.0)
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 300.0, "b": 280.0}, 1.0)
    assert not r["trips"]


def test_kernel_budget_and_invalid_inputs_do_not_mutate_inputs():
    c = two_nodes(max_evaluations=1)
    t = {"a": 300.0, "b": 280.0}
    with pytest.raises(ValueError, match="budget"):
        step(c, t, 1.0)
    assert t == {"a": 300.0, "b": 280.0}
    with pytest.raises(ValueError):
        step(two_nodes(), t, -1.0)
    with pytest.raises(ValueError):
        step(two_nodes(), {"a": 0.0, "b": 280.0}, 1.0)
    with pytest.raises(ValueError):
        advance_network(
            two_nodes(), t, 1.0, internal_w={"a": 1.0}, external_w={"a": 0.0, "b": 0.0}
        )


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "self",
        "unknown",
        "routing_missing",
        "routing_fraction",
        "routing_node",
        "normal",
        "limits",
        "heater",
        "nan",
    ],
)
def test_strict_topology_and_parameter_validation(kind):
    d = network()
    if kind == "duplicate":
        d["links"].append({"a": "cpu", "b": "hull", "conductance_w_per_k": 1.0})
    elif kind == "self":
        d["links"][0]["b"] = "hull"
    elif kind == "unknown":
        d["links"][0]["a"] = "missing"
    elif kind == "routing_missing":
        d["routing"].pop("radio")
    elif kind == "routing_fraction":
        d["routing"]["cpu"]["cpu"] = 0.5
    elif kind == "routing_node":
        d["routing"]["cpu"] = {"missing": 1.0}
    elif kind == "normal":
        d["nodes"]["hull"]["normal_body"] = [2.0, 0.0, 0.0]
    elif kind == "limits":
        d["nodes"]["hull"]["resume_k"] = 350.0
    elif kind == "heater":
        d["nodes"]["hull"]["heater_power_w"] = 10.0
    elif kind == "nan":
        d["nodes"]["hull"]["emissivity"] = float("nan")
    with pytest.raises(ValueError):
        ThermalNetworkSettings(**d)


def test_view_factors_nadir_away_and_limb():
    radius = 6378137.0
    r = np.array([7000000.0, 0.0, 0.0])
    assert projected_earth_view_factor(r, [-1.0, 0.0, 0.0], radius) == pytest.approx(
        (radius / r[0]) ** 2
    )
    assert projected_earth_view_factor(r, [1.0, 0.0, 0.0], radius) == 0.0
    limb = projected_earth_view_factor(r, [0.0, 1.0, 0.0], radius)
    theta = math.asin(radius / r[0])
    expected = (theta - math.sin(theta) * math.cos(theta)) / math.pi
    assert limb == pytest.approx(expected, abs=3e-4)
    with pytest.raises(ValueError):
        projected_earth_view_factor([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], radius)


@pytest.fixture(scope="module")
def scenario():
    d = copy.deepcopy(fixtures.scenario.__wrapped__()["definition"])
    d["spacecraft_bus"]["a"]["thermal_network"] = network()
    return spacecraft_scenario(d, id="thermal-network", lineage="B04-E")


def runtime(s):
    return MissionRuntime(s, access=AccessPolicy(truth=True, forecast=True))


def view(m):
    return m.observe("a")["actors"]["a"]["spacecraft_bus"]["thermal_network"]


def test_actual_capture_cpu_heat_are_routed_and_conserved(scenario):
    m = runtime(scenario)
    life.acquire(m)
    life.transform(m)
    v = view(m)
    assert v["source_heat_j"]["payload"] == pytest.approx(16.0)
    assert v["source_heat_j"]["cpu"] == pytest.approx(8.0)
    assert v["temperatures_k"]["camera"] > v["temperatures_k"]["hull"]
    assert v["temperatures_k"]["cpu"] > 285.0
    assert max(abs(x) for x in v["node_balance_error_j"].values()) < 1e-7
    assert abs(fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-7


def test_hot_payload_fails_capture_and_latches_node_reason(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["camera"] = node(
        capacity_j_per_k=0.2, trip_k=286.0, resume_k=284.0
    )
    m = runtime(d)
    life.acquire(m)
    assert not life.lifecycle(m)["artifacts"]
    assert m._backend.bus.states["a"]["safe"]
    assert "overtemperature:camera" in m._backend.bus.states["a"]["safe_reasons"]
    assert view(m)["events"]
    assert view(m)["source_heat_j"]["payload"] > 0


def test_cpu_overheat_does_not_materialize_final_product(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["cpu"] = node(
        capacity_j_per_k=0.1, trip_k=286.0, resume_k=284.0
    )
    m = runtime(d)
    life.acquire(m)
    life.transform(m)
    assert "final" not in life.lifecycle(m)["artifacts"]
    assert m._backend.bus.states["a"]["safe"]


def test_checkpoint_and_atomic_thermal_rollback(scenario, tmp_path, monkeypatch):
    m = runtime(scenario)
    life.submit(m, "imaging", 2.0, "image", request_id="request")
    life.advance(m, 0.5, "part")
    p = tmp_path / "thermal.json"
    m.save(p)
    restored = MissionRuntime.load(p, access=AccessPolicy(truth=True, forecast=True))
    life.advance(m, 0.5, "more")
    life.advance(restored, 0.5, "more")
    assert restored.checkpoint() == m.checkpoint()
    assert restored._backend.lifecycle.thermal_network is restored._backend.bus.network
    before = m.checkpoint()
    from space_env.reference.thermal_bus import CoupledThermalNetwork

    original = CoupledThermalNetwork.integrate

    def broken(self, *a, **kw):
        original(self, *a, **kw)
        raise ValueError("after thermal integration")

    monkeypatch.setattr(CoupledThermalNetwork, "integrate", broken)
    with pytest.raises(RuntimeContractError):
        life.advance(m, 0.5, "failure")
    assert m.checkpoint() == before
    assert m._backend.lifecycle.thermal_network is m._backend.bus.network


def test_lifetime_thermal_budget_is_atomic(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"][
        "max_total_evaluations"
    ] = 1
    m = runtime(d)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        life.advance(m, 1.0, "budget")
    assert m.checkpoint() == before


def heater_scenario(scenario, **updates):
    d = copy.deepcopy(scenario)
    n = node(
        initial_k=260.0,
        capacity_j_per_k=1.0,
        cold_trip_k=270.0,
        cold_resume_k=280.0,
        resume_k=300.0,
        trip_k=310.0,
        heater_power_w=20.0,
        heater_on_below_k=280.0,
        heater_off_above_k=285.0,
    )
    n.update(updates)
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["panel"] = n
    return d


def test_paid_heater_warms_cold_node_but_recovery_is_explicit(scenario):
    m = runtime(heater_scenario(scenario))
    assert m._backend.bus.states["a"]["safe"]
    before = m._backend.mission.energy_wh["a"]
    life.advance(m, 1.5, "warm")
    assert view(m)["temperatures_k"]["panel"] >= 280.0
    assert m._backend.mission.energy_wh["a"] < before
    assert m._backend.bus.states["a"]["safe"]
    m.inject(
        [life.event(m, "spacecraft_recover", {"actor": "a"})],
        expected_revision=m.revision,
        idempotency_key="recover",
    )
    assert not m._backend.bus.states["a"]["safe"]
    assert abs(fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-7


def test_heater_peak_and_soc_do_not_create_free_heat(scenario):
    for kind in ("peak", "soc"):
        d = heater_scenario(scenario)
        if kind == "peak":
            d["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 10.0
        m = runtime(d)
        if kind == "soc":
            m._backend.mission.energy_wh["a"] = (
                0.1 * m._backend.mission.vehicles["a"].battery_wh
            )
        life.advance(m, 0.5, "blocked")
        assert view(m)["temperatures_k"]["panel"] == 260.0
        assert view(m)["heater_power_w"]["panel"] == 0


def test_bus_fault_blocks_heater_and_clearing_does_not_heal_temperature(scenario):
    m = runtime(heater_scenario(scenario))
    m.inject(
        [life.event(m, "spacecraft_fault", {"actor": "a", "subsystem": "bus"})],
        expected_revision=m.revision,
        idempotency_key="fault",
    )
    life.advance(m, 0.5, "faulted")
    assert view(m)["temperatures_k"]["panel"] == 260.0


def test_heater_reserve_cuts_actual_heating(scenario):
    m = runtime(heater_scenario(scenario))
    v = m._backend.mission.vehicles["a"]
    m._backend.mission.energy_wh["a"] = 0.1 * v.battery_wh + 10 / (3600 * 0.9)
    life.advance(m, 1.0, "reserve")
    assert view(m)["temperatures_k"]["panel"] == pytest.approx(270.0, abs=1e-6)


def test_recovery_checks_each_node_not_summary_only(scenario):
    m = runtime(
        heater_scenario(
            scenario,
            heater_power_w=0.0,
            heater_on_below_k=None,
            heater_off_above_k=None,
        )
    )
    with pytest.raises(RuntimeContractError):
        m.inject(
            [life.event(m, "spacecraft_recover", {"actor": "a"})],
            expected_revision=m.revision,
            idempotency_key="bad-recover",
        )


def test_network_actor_isolation_and_legacy_actor_unchanged(scenario):
    m = runtime(scenario)
    assert "thermal_network" not in m.observe("b")["actors"]["b"]["spacecraft_bus"]
    assert "b" not in m.observe("a")["actors"]


def test_pv_mapping_inconsistent_with_electrical_panel_rejected(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["operations"]["vehicles"]["a"].update(
        solar_area_m2=1.0, solar_efficiency=0.2
    )
    with pytest.raises(RuntimeContractError, match="PV thermal surface"):
        runtime(d)


def solar_case(scenario, initial_soc):
    from space_env.core.attitude import align_z
    from space_env.reference.orbit import _ephemerides

    d = copy.deepcopy(scenario)
    for orbit in d["definition"]["orbits"].values():
        orbit["initial"]["r_m"] = [-x for x in orbit["initial"]["r_m"]]
        orbit["initial"]["v_m_s"] = [-x for x in orbit["initial"]["v_m_s"]]
    template = runtime(d)
    p = template._backend.mission.ephemerides["a"]
    sun, _ = _ephemerides(p.epoch)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(sun - p.state(0)[:3]).tolist()
    )
    d["definition"]["operations"]["vehicles"]["a"].update(
        solar_area_m2=1.0, solar_efficiency=0.2, initial_soc=initial_soc
    )
    n = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]
    n["nodes"]["panel"].update(
        solar_area_m2=1.0, absorptivity=0.8, capacity_j_per_k=100.0
    )
    return d


@pytest.mark.parametrize("initial_soc", [0.5, 1.0])
def test_pv_extraction_and_curtailment_close_combined_thermal_battery_energy(
    scenario, initial_soc
):
    m = runtime(solar_case(scenario, initial_soc))
    before = m._backend.mission.energy_wh["a"]
    life.advance(m, 0.1, "solar")
    v = view(m)
    assert v["source_heat_j"]["pv_extraction"] < 0
    absorbed = v["sampled_environment"]["panel"]["solar_w"] * 0.1
    stored = m._backend.bus.network.stored("a")
    battery = (m._backend.mission.energy_wh["a"] - before) * 3600
    assert stored + battery == pytest.approx(absorbed, abs=1e-7)
    assert abs(v["ledgers"]["routing_error_j"]) < 1e-8
    if initial_soc == 1.0:
        assert m._backend.mission.curtailed_wh["a"] > 0


def test_panel_turn_away_removes_solar_heating_and_generation(scenario):
    from space_env.core.attitude import align_z
    from space_env.reference.orbit import _ephemerides

    d = solar_case(scenario, 0.5)
    template = runtime(d)
    p = template._backend.mission.ephemerides["a"]
    sun, _ = _ephemerides(p.epoch)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(p.state(0)[:3] - sun).tolist()
    )
    m = runtime(d)
    life.advance(m, 0.1, "away")
    assert abs(view(m)["source_heat_j"]["pv_extraction"]) < 1e-10
    assert view(m)["sampled_environment"]["panel"]["solar_w"] == 0.0


def test_uniform_earth_ir_survives_earth_night_eclipse(scenario):
    from space_env.core.attitude import align_z
    from space_env.reference.orbit import _ephemerides

    d = copy.deepcopy(scenario)
    template = runtime(d)
    p = template._backend.mission.ephemerides["a"]
    sun, _ = _ephemerides(p.epoch)
    initial = d["definition"]["orbits"]["a"]["initial"]
    radius = np.linalg.norm(initial["r_m"])
    r = -sun / np.linalg.norm(sun) * radius
    axis = np.cross(r, [0, 0, 1.0])
    axis = axis / np.linalg.norm(axis) * 7500.0
    initial.update(r_m=r.tolist(), v_m_s=axis.tolist())
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(-r).tolist()
    )
    config = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]
    config["earth_environment"] = True
    config["nodes"]["panel"].update(
        solar_area_m2=0.01, absorptivity=0.8, emission_area_m2=0.01, emissivity=0.8
    )
    m = runtime(d)
    life.advance(m, 0.1, "night")
    env = view(m)["sampled_environment"]["panel"]
    assert env["shadow_fraction"] == 0
    assert env["solar_w"] == 0
    assert env["earth_ir_w"] > 0
    assert abs(env["albedo_w"]) < 1e-6


def test_heater_hysteresis_stops_without_chattering_or_free_temperature_reset(scenario):
    m = runtime(heater_scenario(scenario))
    life.advance(m, 2.0, "warm")
    assert not view(m)["heater_requested"]["panel"]
    t = view(m)["temperatures_k"]["panel"]
    life.advance(m, 0.5, "off")
    assert view(m)["heater_power_w"]["panel"] == 0
    assert view(m)["temperatures_k"]["panel"] == pytest.approx(t)


def test_multiple_heaters_respect_peak_priority(scenario):
    d = heater_scenario(scenario)
    config = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]
    config["nodes"]["camera"] = copy.deepcopy(config["nodes"]["panel"])
    config["nodes"]["camera"]["heater_priority"] = 1
    d["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 25.0
    m = runtime(d)
    life.advance(m, 0.1, "priority")
    assert view(m)["heater_power_w"]["panel"] == 20
    assert view(m)["heater_power_w"]["camera"] == 0


def test_engine_wall_and_electric_heat_are_not_double_counted(scenario):
    from tests.test_actuation import burn, report
    from tests.test_actuation import scenario as engine_scenario

    d = copy.deepcopy(engine_scenario.__wrapped__())
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"] = network()
    m = runtime(d)
    burn(m, firing=0.5, duration=1.5)
    life.advance(m, 1.5, "engine")
    expected = report(m)["operations"]["burn"]["engine_heat_j"]
    assert view(m)["source_heat_j"]["propulsion"] == pytest.approx(expected, abs=1e-8)
    assert abs(fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-7


def test_rcs_wheel_braking_heat_and_valve_heat_have_separate_sources(scenario):
    from tests.test_momentum_unloading import op, unload
    from tests.test_momentum_unloading import scenario as rcs_scenario

    d = copy.deepcopy(rcs_scenario.__wrapped__())
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"] = network()
    m = runtime(d)
    unload(m)
    life.advance(m, 3.0, "rcs")
    assert view(m)["source_heat_j"]["propulsion"] == pytest.approx(op(m)["rcs_heat_j"])
    assert (
        view(m)["source_heat_j"]["adcs"] > m._backend.bus.rigid.states["a"].dc_energy_j
    )
    assert abs(fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-7


def test_radio_heat_uses_actual_service_time(scenario):
    m = runtime(scenario)
    life.acquire(m)
    life.transform(m)
    m._backend.mission.vehicles["a"] = m._backend.mission.vehicles["a"].model_copy(
        update={"tx_power_w": 8.0}
    )
    life.submit(
        m, "downlink", 1.0, "radio", destination_id="Ground", product_id="final"
    )
    life.advance(m, 1.0, "ack")
    seconds = m._backend.mission.tx_seconds["a"]
    assert view(m)["source_heat_j"]["radio"] == pytest.approx(
        seconds * 8 * 0.8, abs=1e-8
    )


def test_routing_and_source_models_are_sealed_in_checkpoint(scenario, tmp_path):
    m = runtime(scenario)
    p = tmp_path / "sealed.json"
    m.save(p)
    assert (
        "thermal_network"
        in m.checkpoint()["effective_scenario"]["definition"]["spacecraft_bus"]["a"]
    )


def test_thermal_event_budget_rejects_transaction(scenario):
    d = copy.deepcopy(scenario)
    c = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]
    c["max_events"] = 1
    c["nodes"]["hull"].update(initial_k=285.0, trip_k=286.0, resume_k=284.0)
    c["nodes"]["camera"].update(initial_k=285.0, trip_k=286.0, resume_k=284.0)
    # Route equal payload heat to two nodes; both cross together.
    c["routing"]["payload"] = {"hull": 0.5, "camera": 0.5}
    c["nodes"]["hull"]["capacity_j_per_k"] = 0.1
    c["nodes"]["camera"]["capacity_j_per_k"] = 0.1
    m = runtime(d)
    life.submit(m, "imaging", 2.0, "image", request_id="request")
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="event history"):
        life.advance(m, 0.1, "events")
    assert m.checkpoint() == before


def test_transient_hot_crossing_is_recorded_even_after_cooling_below_limit():
    d = two_nodes().model_dump(mode="json")
    d["nodes"]["a"].update(
        initial_k=280.0,
        capacity_j_per_k=1.0,
        trip_k=300.0,
        resume_k=290.0,
        bath_conductance_w_per_k=1.0,
        bath_k=280.0,
    )
    d["nodes"]["b"].update(
        initial_k=500.0,
        capacity_j_per_k=1.0,
        trip_k=600.0,
        resume_k=550.0,
        bath_conductance_w_per_k=1.0,
        bath_k=280.0,
    )
    d["links"][0]["conductance_w_per_k"] = 10.0
    c = ThermalNetworkSettings(**d)
    r = step(c, {"a": 280.0, "b": 500.0}, 10.0)
    assert r["temperatures_k"]["a"] < 290.0
    assert any(x["node"] == "a" and x["kind"] == "hot" for x in r["trips"])


def test_kernel_partition_consistency_for_constant_sources():
    c = two_nodes()
    start = {"a": 300.0, "b": 280.0}
    whole = step(c, start, 1.0)
    part = step(c, start, 0.4)
    end = step(c, part["temperatures_k"], 0.6)
    assert end["temperatures_k"] == pytest.approx(whole["temperatures_k"], abs=1e-8)


def test_parallel_cpu_camera_radio_sources_match_paid_service():
    from tests import test_resource_lanes as lanes

    d = copy.deepcopy(lanes.scenario.__wrapped__()["definition"])
    d["spacecraft_bus"]["a"]["thermal_network"] = network()
    m = runtime(spacecraft_scenario(d, id="thermal-parallel", lineage="B04-E"))
    life.acquire(m)
    lanes.workload(m)
    life.advance(m, 4.0, "parallel")
    sources = view(m)["source_heat_j"]
    assert sources["cpu"] == pytest.approx(8.0, abs=1e-8)
    assert sources["payload"] == pytest.approx(32.0, abs=1e-8)
    assert sources["radio"] == pytest.approx(
        m._backend.mission.tx_seconds["a"] * 8 * 0.8, abs=1e-8
    )
    assert abs(fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-7


def test_sar_rf_energy_leaves_thermal_network_instead_of_becoming_heat():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/payload_quality/demo.py"
    loader = importlib.util.spec_from_file_location("thermal_sar_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    d, _ = module.build_scenario("sar")
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"] = network()
    m = runtime(d)
    life.submit(m, "imaging", 2.0, "image", request_id="customer-a")
    life.advance(m, 2.0, "sar")
    rows = m._backend.lifecycle.quality.sar_intervals["a"]
    expected = sum(r["amplifier_heat_j"] for r in rows)
    assert expected > 0
    electronics = (
        m._backend.lifecycle.spec.payloads["a"].power_w
        * 2
        * m._backend.bus.models["a"].electrical_heat_fraction
    )
    assert view(m)["source_heat_j"]["payload"] == pytest.approx(
        expected + electronics, abs=1e-7
    )
    assert expected < sum(r["dc_input_j"] for r in rows)


def test_thermal_heater_competes_with_payload_in_shared_peak_admission(scenario):
    d = heater_scenario(
        scenario,
        initial_k=285.0,
        heater_power_w=10.0,
        heater_on_below_k=286.0,
        heater_off_above_k=290.0,
    )
    d["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 15.0
    m = runtime(d)
    life.acquire(m)
    assert not life.lifecycle(m)["artifacts"]
    assert view(m)["temperatures_k"]["panel"] > 285.0
    assert any(r["reason"] == "bus_power_capacity" for r in m._backend.bus.denials)


def test_short_lived_thermal_derivative_budget_rolls_back_all_resources(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["max_evaluations"] = 1
    m = runtime(d)
    life.submit(m, "imaging", 2.0, "image", request_id="request")
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        life.advance(m, 0.1, "reject")
    assert m.checkpoint() == before


def test_declared_nonthermal_electrical_export_is_explicit(scenario):
    m = runtime(scenario)
    life.acquire(m)
    life.transform(m)
    budget = view(m)["ledgers"]
    expected = 30.0 + fixtures.bus(m)["states"]["a"]["slew_energy_wh"] * 3600
    assert budget["dc_input_j"] == pytest.approx(expected)
    assert budget["dc_heat_j"] == pytest.approx(0.8 * expected)
    assert budget["declared_nonthermal_export_j"] == pytest.approx(
        0.2 * expected, abs=1e-8
    )
    assert budget["dc_input_j"] - budget["dc_heat_j"] - budget["wheel_work_j"] - budget[
        "declared_nonthermal_export_j"
    ] == pytest.approx(0.0, abs=1e-8)


def test_heater_cold_start_checkpoint_is_numerically_canonical(scenario, tmp_path):
    m = runtime(heater_scenario(scenario))
    life.advance(m, 1.5, "warm")
    p = tmp_path / "cold.json"
    m.save(p)
    restored = MissionRuntime.load(p, access=AccessPolicy(truth=True, forecast=True))
    assert restored.checkpoint() == m.checkpoint()
    assert view(m)["ledgers"]["heater_dc_energy_j"] > 0


def test_node_route_and_edge_order_are_canonical():
    first = network()
    second = copy.deepcopy(first)
    second["nodes"] = dict(reversed(list(second["nodes"].items())))
    second["routing"] = {
        k: dict(reversed(list(v.items())))
        for k, v in reversed(list(second["routing"].items()))
    }
    second["links"] = [
        {**e, "a": e["b"], "b": e["a"]} for e in reversed(second["links"])
    ]
    a, b = ThermalNetworkSettings(**first), ThermalNetworkSettings(**second)
    assert a.model_dump() == b.model_dump()
    ta = {k: n.initial_k for k, n in a.nodes.items()}
    ta["cpu"] = 300.0
    assert step(a, ta, 1.0) == step(b, dict(reversed(list(ta.items()))), 1.0)


def test_lambert_phase_albedo_cannot_reflect_more_than_incident_planet_power():
    with pytest.raises(ValueError):
        ThermalNetworkSettings(**network(earth_geometric_albedo=0.9))
