"""Конечное включение отдельно от мгновенного импульса."""
import math

import numpy as np
import pytest

from space_env.core.orbit import Orbit
from space_env.core.propulsion import G0_M_S2, mass_after_thrust
from space_env.core.spacecraft import PropulsionConfig, Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")
_ISP_S = 200.0
_THRUST_N = 10.0
_MASS_KG = 100.0


def _engine(*, thrust_n: float = 0.0, isp_s: float = 0.0, dry_mass_kg: float = 0.0, dv_budget: float = 50.0) -> SimulationEngine:
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, EPOCH)
    craft = Spacecraft(
        config=SpacecraftConfig(
            propulsion=PropulsionConfig(
                delta_v_total_m_s=dv_budget,
                mass_kg=_MASS_KG,
                dry_mass_kg=dry_mass_kg,
                thrust_N=thrust_n,
                isp_s=isp_s,
                drag_area_m2=0.0,
                srp_area_m2=0.0,
            )
        )
    )
    craft.reset_state()
    return SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))


def test_an_unconfigured_thruster_still_applies_the_impulse_at_once():
    engine = _engine()
    speed_before = float(np.linalg.norm(engine.orbit.v))
    snap, events = engine.step(10.0, "maneuver", {"dv_m_s": 10.0, "direction": "prograde"})
    assert engine.last_command_ok is True
    assert snap.delta_v_used_m_s == pytest.approx(10.0)
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(_MASS_KG)
    assert float(np.linalg.norm(engine.orbit.v)) > speed_before + 5.0
    assert events[-1].payload == {"dv_m_s": 10.0, "direction": "prograde"}
    assert engine._burn is None


def test_a_partial_thruster_card_does_not_fall_back_to_an_impulse():
    engine = _engine(thrust_n=_THRUST_N)
    engine.step(10.0, "maneuver", {"dv_m_s": 5.0, "direction": "prograde"})
    assert engine.last_command_ok is False
    assert engine.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(50.0)
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(_MASS_KG)
    assert engine._burn is None


def test_finite_burn_spends_propellant_across_steps_and_then_stops():
    coast = _engine(thrust_n=_THRUST_N, isp_s=_ISP_S, dry_mass_kg=90.0)
    coast.step(10.0, "wait")
    engine = _engine(thrust_n=_THRUST_N, isp_s=_ISP_S, dry_mass_kg=90.0)
    engine.step(10.0, "maneuver", {"dv_m_s": 2.0, "direction": "prograde"})
    assert engine.last_command_ok is True
    assert engine._burn is not None
    mass_10, dv_10 = mass_after_thrust(_MASS_KG, 10.0, _THRUST_N, _ISP_S, 90.0)
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(mass_10, rel=1e-9)
    assert engine.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(50.0 - dv_10, abs=1e-6)
    speed_gap = float(np.linalg.norm(engine.orbit.v) - np.linalg.norm(coast.orbit.v))
    assert speed_gap == pytest.approx(dv_10, abs=0.05)
    held = engine.capture()
    engine.step(10.0, "wait")
    engine.restore(held)
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(mass_10, rel=1e-9)
    assert engine._burn is not None
    assert float(engine._burn["dv_left"]) == pytest.approx(2.0 - dv_10, abs=1e-6)
    engine.step(10.0, "maneuver", {"dv_m_s": 1.0, "direction": "prograde"})
    assert engine.last_command_ok is False
    engine.step(10.0, "wait")
    ve = _ISP_S * G0_M_S2
    assert engine._burn is None
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(_MASS_KG * math.exp(-2.0 / ve), rel=1e-6)
    assert engine.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(48.0, abs=1e-3)


def test_a_burn_stops_when_the_propellant_cannot_supply_the_request():
    engine = _engine(thrust_n=_THRUST_N, isp_s=_ISP_S, dry_mass_kg=99.9, dv_budget=50.0)
    engine.step(10.0, "maneuver", {"dv_m_s": 5.0, "direction": "prograde"})
    assert engine.last_command_ok is False
    assert engine._burn is None
    assert engine.spacecraft.config.propulsion.mass_kg == pytest.approx(_MASS_KG)
    assert engine.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(50.0)


def test_safe_mode_drops_the_rest_of_a_finite_burn():
    engine = _engine(thrust_n=_THRUST_N, isp_s=_ISP_S, dry_mass_kg=90.0)
    engine.step(10.0, "maneuver", {"dv_m_s": 2.0, "direction": "prograde"})
    _mass, dv_10 = mass_after_thrust(_MASS_KG, 10.0, _THRUST_N, _ISP_S, 90.0)
    engine.step(10.0, "safe_mode")
    assert engine._burn is None
    assert engine.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(50.0 - dv_10, abs=1e-6)
