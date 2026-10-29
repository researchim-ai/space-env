"""Гистерезис питания: безопасный режим держится, пока заряд заметно не вырастет."""
import pytest

from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PowerConfig, Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.events import EventKind
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")


def _engine(*, enter: float = 0.5, leave: float = 0.8) -> SimulationEngine:
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, EPOCH)
    craft = Spacecraft(
        config=SpacecraftConfig(
            power=PowerConfig(
                solar_panel_area_m2=0.0,
                consumption_nominal_W=0.0,
                consumption_peak_W=0.0,
                safe_soc_enter=enter,
                safe_soc_exit=leave,
            )
        )
    )
    craft.reset_state()
    return SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))


def _actions(events) -> list[str]:
    return [str(event.payload.get("action")) for event in events if event.payload.get("action")]


def test_disabled_thresholds_do_not_enter_safe_mode():
    engine = _engine(enter=0.0, leave=0.0)
    engine.spacecraft.state.power.battery_soc = 0.05
    engine.step(5.0, "wait")
    assert engine._safe_latched is False
    assert engine._load_mode == "nominal"
    assert engine.spacecraft.state.pointing == "nadir"


def test_low_charge_latches_until_the_exit_threshold():
    engine = _engine()
    engine.step(1.0, "reduced_power")
    engine.spacecraft.state.power.battery_soc = 0.4
    engine._downlink_armed = True
    engine._burn = {"direction": "prograde", "dv_left": 1.0}
    _snap, entered = engine.step(5.0, "imaging")
    assert engine._safe_latched is True
    assert engine._load_mode == "safe"
    assert engine._burn is None
    assert engine._downlink_armed is False
    assert engine.spacecraft.state.pointing == "sun"
    assert engine.last_command_ok is False
    assert _actions(entered) == ["power_safe"]
    engine.spacecraft.state.power.battery_soc = 0.7
    _snap, held = engine.step(5.0, "nominal_power")
    assert engine._safe_latched is True
    assert engine._load_mode == "safe"
    assert engine.last_command_ok is False
    assert _actions(held) == []
    held_shot = engine.capture()
    engine.spacecraft.state.power.battery_soc = 0.9
    _snap, cleared = engine.step(5.0, "wait")
    assert engine._safe_latched is False
    assert engine._load_mode == "reduced"
    assert any(event.kind == EventKind.RECOVERY and event.payload["action"] == "power_safe_cleared" for event in cleared)
    engine.restore(held_shot)
    assert engine._safe_latched is True
    assert engine._load_mode == "safe"
    assert engine.spacecraft.state.power.battery_soc == pytest.approx(0.7)


def test_a_reversed_band_is_rejected():
    engine = _engine(enter=0.8, leave=0.5)
    with pytest.raises(ValueError, match="порог выхода"):
        engine.step(1.0, "wait")
