"""Material arithmetic, energy conservation and transactional native integration."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from space_env.reference.materials import (
    MaterialHeatCapacity,
    material_evidence,
)
from space_env.reference.materials import (
    specific_enthalpy_delta_j_per_kg as enthalpy,
)
from space_env.reference.materials import (
    specific_heat_j_per_kg_k as cp,
)
from space_env.reference.thermal_network import ThermalNetworkSettings, ThermalNode
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.validation.materials import (
    FIXTURE,
    FIXTURE_SHA256,
    material_validation_report,
)
from tests import test_reference_lifecycle as life
from tests import test_thermal_network as thermal


def material_node(**updates):
    return {
        "initial_k": 200.0,
        "trip_k": 295.0,
        "resume_k": 290.0,
        "material_capacity": {
            "mass_kg": 0.01,
            "assumption": "synthetic homogeneous alloy host",
        },
        **updates,
    }


def config(**updates):
    data = thermal.network(
        nodes={"a": material_node(), "b": material_node(initial_k=100.0)},
        links=[{"a": "a", "b": "b", "conductance_w_per_k": 0.1}],
        routing={k: {"a": 1.0} for k in thermal.HEAT_SOURCES},
    )
    data.update(updates)
    return ThermalNetworkSettings(**data)


@pytest.mark.parametrize(
    "t", [3.999, 300.001, 0, -1, float("inf"), float("nan"), True, "200"]
)
def test_outside_material_domain_rejected(t):
    with pytest.raises(ValueError, match="no extrapolation"):
        cp(t)
    with pytest.raises(ValueError):
        enthalpy(200, t)


@pytest.mark.parametrize(
    "updates",
    [
        {"capacity_j_per_k": 2.0},
        {"material_capacity": None},
        {"initial_k": 301.0},
        {"material_capacity": {"mass_kg": -1.0, "assumption": "x"}},
        {"material_capacity": {"mass_kg": True, "assumption": "x"}},
        {"material_capacity": {"mass_kg": 1.0, "assumption": "   "}},
        {
            "material_capacity": {
                "mass_kg": 1.0,
                "assumption": "x",
                "material_id": "unknown",
            }
        },
    ],
)
def test_ambiguous_or_invalid_material_contract_rejected(updates):
    with pytest.raises(ValueError):
        ThermalNode(**material_node(**updates))


def test_evidence_owned_and_error_not_sigma():
    evidence = material_evidence()
    evidence["coefficients_ascending"][0] = 100
    assert material_evidence()["coefficients_ascending"][0] == 46.6467
    assert "not measurement sigma" in evidence["error_interpretation"]
    assert not evidence["hardware_calibration_complete"]


def test_enthalpy_antisymmetry_additivity_and_not_endpoint_capacity():
    h = enthalpy(4, 300)
    assert enthalpy(300, 4) == -h
    assert enthalpy(77, 77) == 0.0
    assert enthalpy(4, 77) + enthalpy(77, 300) == pytest.approx(h, abs=1e-5)
    assert abs(cp(300) * 296 - h) > 10000
    m = MaterialHeatCapacity(mass_kg=2.0, assumption="synthetic specimen")
    assert m.stored(4, 300) == 2 * h
    assert m.capacity(200) == 2 * cp(200)


@pytest.mark.parametrize("solver", ["DOP853", "Radau"])
def test_closed_variable_network_conserves_enthalpy_and_partition(solver):
    c = config(solver=solver)
    start = {"a": 200.0, "b": 100.0}
    before = copy.deepcopy(start)
    whole = thermal.step(c, start, 2.0)
    assert start == before
    assert abs(sum(whole["stored_j"].values())) < 1e-5
    assert max(abs(v) for v in whole["node_balance_error_j"].values()) < 1e-5
    part = thermal.step(c, start, 0.7)
    end = thermal.step(c, part["temperatures_k"], 1.3)
    assert end["temperatures_k"] == pytest.approx(whole["temperatures_k"], abs=1e-7)
    assert whole["temperatures_k"]["a"] < 200
    assert whole["temperatures_k"]["b"] > 100


@pytest.mark.parametrize("solver", ["DOP853", "Radau"])
def test_variable_node_radiation_bath_and_internal_energy_ledgers(solver):
    c = config(solver=solver)
    d = c.model_dump(mode="json")
    d["nodes"]["a"].update(
        emission_area_m2=0.02,
        emissivity=0.8,
        bath_conductance_w_per_k=0.05,
        bath_k=150.0,
    )
    result = thermal.step(
        ThermalNetworkSettings(**d),
        {"a": 200.0, "b": 100.0},
        1.0,
        internal={"a": 10.0, "b": 0.0},
        external={"a": 1.0, "b": 0.0},
    )
    assert max(abs(v) for v in result["node_balance_error_j"].values()) < 1e-5
    assert result["radiation_out_j"]["a"] > 0
    assert result["bath_out_j"]["a"] > 0


def test_domain_escape_does_not_return_clipped_or_partial_solution():
    c = config(links=[])
    start = {"a": 299.9, "b": 100.0}
    with pytest.raises(ValueError, match="no extrapolation"):
        thermal.step(c, start, 1.0, internal={"a": 1e4, "b": 0.0})
    assert start == {"a": 299.9, "b": 100.0}


def native_scenario():
    d = copy.deepcopy(thermal.scenario.__wrapped__())
    d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["hull"] = (
        material_node()
    )
    return d


def test_native_storage_observation_checkpoint_and_replay(tmp_path):
    m = thermal.runtime(native_scenario())
    life.advance(m, 0.3, "first")
    m.save(tmp_path / "prefix.json")
    restored = MissionRuntime.load(
        tmp_path / "prefix.json", access=AccessPolicy(truth=True, forecast=True)
    )
    assert restored.checkpoint() == m.checkpoint()
    life.advance(m, 0.4, "second")
    life.advance(restored, 0.4, "second")
    assert restored.checkpoint() == m.checkpoint()
    assert max(abs(x) for x in thermal.view(m)["node_balance_error_j"].values()) < 1e-5
    net = m._backend.bus.network
    assert net.stored("a") == pytest.approx(
        sum(
            n.stored(n.initial_k, net.temperatures["a"][k])
            for k, n in net.models["a"].nodes.items()
        ),
        abs=1e-8,
    )


def test_native_material_domain_escape_rolls_back_entire_transaction():
    d = native_scenario()
    c = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]
    c["nodes"]["camera"] = material_node(
        initial_k=299.9,
        trip_k=310.0,
        resume_k=305.0,
        material_capacity={
            "mass_kg": 1e-5,
            "assumption": "intentional domain escape negative case",
        },
    )
    m = thermal.runtime(d)
    life.submit(m, "imaging", 2.0, "image", request_id="request")
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="no extrapolation"):
        life.advance(m, 0.1, "domain-escape")
    assert m.checkpoint() == before
    assert m._backend.lifecycle.thermal_network is m._backend.bus.network


def test_frozen_decimal_oracle_and_generator_hash():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    d = json.loads(FIXTURE.read_text())
    generator = (
        Path(__file__).resolve().parents[1] / "scripts/generate_material_vectors.py"
    )
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == d["generator_sha256"]
    loader = importlib.util.spec_from_file_location(
        "independent_material_oracle", generator
    )
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    assert module.generate() == d


def test_validation_pinned_vectors_and_actual_native_integration():
    report = material_validation_report()
    assert report["passed"]
    assert len(report["checks"]) == 34
    assert not report["hardware_calibration_complete"]
    assert not report["industrial_readiness"]


def test_legacy_constant_node_keeps_exact_storage():
    n = ThermalNode(**thermal.node(capacity_j_per_k=20.0))
    assert n.material_capacity is None
    assert n.stored(280.0, 300.0) == 400.0
    assert n.capacity(250.0) == 20.0


def test_variable_material_paid_heater_uses_enthalpy_and_battery():
    d = thermal.heater_scenario(thermal.scenario.__wrapped__())
    n = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["panel"]
    n.pop("capacity_j_per_k")
    n["material_capacity"] = {
        "mass_kg": 0.001,
        "assumption": "synthetic heated alloy node",
    }
    m = thermal.runtime(d)
    before = m._backend.mission.energy_wh["a"]
    life.advance(m, 0.5, "heat")
    v = thermal.view(m)
    assert v["temperatures_k"]["panel"] > 260
    stored = 0.001 * enthalpy(260.0, v["temperatures_k"]["panel"])
    assert stored == pytest.approx(v["ledgers"]["heater_dc_energy_j"], abs=1e-6)
    assert m._backend.mission.energy_wh["a"] < before
    assert max(abs(x) for x in v["node_balance_error_j"].values()) < 1e-6


@pytest.mark.parametrize("initial_soc", [0.5, 1.0])
def test_variable_material_pv_battery_combined_energy(initial_soc):
    d = thermal.solar_case(thermal.scenario.__wrapped__(), initial_soc)
    n = d["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"]["panel"]
    n.pop("capacity_j_per_k")
    n["material_capacity"] = {
        "mass_kg": 0.01,
        "assumption": "synthetic panel lump; geometry and material assignment assumed",
    }
    m = thermal.runtime(d)
    before = m._backend.mission.energy_wh["a"]
    life.advance(m, 0.1, "solar")
    v = thermal.view(m)
    absorbed = v["sampled_environment"]["panel"]["solar_w"] * 0.1
    delta_battery = (m._backend.mission.energy_wh["a"] - before) * 3600
    assert m._backend.bus.network.stored("a") + delta_battery == pytest.approx(
        absorbed, abs=1e-5
    )
    assert v["source_heat_j"]["pv_extraction"] < 0


def test_material_cli_writes_full_numerical_report(tmp_path, capsys):
    from space_env.cli import main

    target = tmp_path / "material.json"
    main(["reference-qualify-materials", str(target)])
    assert json.loads(target.read_text())["passed"]
    assert json.loads(capsys.readouterr().out)["checks"] == 34
