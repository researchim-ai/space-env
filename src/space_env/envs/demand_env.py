"""
Заявка из сценария: снять место и сдать биты до срока.
Наблюдение — доля времени, буфер, линия на станцию заявки и видимость цели.
Кода правильной команды здесь нет. Успех — уникальные биты доехали до срока.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from space_env.catalog.build import mean_orbit, spacecraft_config
from space_env.core.space_weather import indices_for
from space_env.core.access import subsatellite_latlon, target_access
from space_env.core.environment import in_eclipse
from space_env.core.ground import GroundStation, station_by_name
from space_env.core.spacecraft import Spacecraft
from space_env.core.sun import sun_position_m
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.mission.shift_scenario import load_mission_scenario
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

_STEP_SEC = 30.0


class DemandShiftEnv(gym.Env):
    """Один аппарат. Заявка задаёт место, срок и станцию сдачи."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: str | None = None, max_episode_steps: int = 40):
        super().__init__()
        self._scenario_path = str(packaged_demand("demand_deliver.yaml") if scenario is None else scenario)
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._engine: SimulationEngine | None = None
        self._demand: dict | None = None
        self._weights = load_rewards()["operator_shift"]

    def _require(self) -> tuple[SimulationEngine, dict]:
        if self._engine is None or self._demand is None:
            raise RuntimeError("сначала reset")
        return self._engine, self._demand

    def _visible(self, engine: SimulationEngine, demand: dict) -> bool:
        if engine._slew_remaining_deg > 1.0 or engine.spacecraft.state.pointing != "nadir":
            return False
        moment = engine.now()
        access = target_access(
            engine.orbit.r,
            float(demand["lat_deg"]),
            float(demand["lon_deg"]),
            moment,
            half_fov_deg=engine.spacecraft.config.payload.half_fov_deg,
            frame=engine.orbit.frame,
            images_in_eclipse=engine.spacecraft.config.payload.images_in_eclipse,
            satellite_in_umbra=in_eclipse(engine.orbit.r, sun_position_m(moment)),
        )
        return access.visible

    def _linked(self, engine: SimulationEngine, demand: dict) -> bool:
        snap = engine.get_state_snapshot()
        return bool(snap.in_contact and snap.contact_station == demand["deliver_to"] and snap.link_ok)

    def _obs(self) -> np.ndarray:
        engine, demand = self._require()
        snap = engine.get_state_snapshot()
        remaining = max(0.0, float(demand["deadline_sec"]) - engine.time_sec) / float(demand["deadline_sec"])
        frame = 0.0 if snap.stored_bits <= 0.0 else 1.0
        delivered = 0.0 if snap.delivered_bits <= 0.0 else 1.0
        return np.array(
            [
                remaining,
                frame,
                1.0 if self._linked(engine, demand) else 0.0,
                1.0 if self._visible(engine, demand) else 0.0,
                delivered,
            ],
            dtype=np.float32,
        )

    def _success(self) -> bool:
        engine, demand = self._require()
        snap = engine.get_state_snapshot()
        size = demand.get("size_bits")
        enough = snap.delivered_bits > 0.0 if size is None else snap.delivered_bits + 1e-6 >= float(size)
        return bool(
            snap.generated_bits > 0.0
            and enough
            and engine.time_sec <= float(demand["deadline_sec"]) + 1e-6
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self._scenario_path is None:
            raise ValueError("DemandShift нужен файл сценария с заявкой")
        loaded = load_mission_scenario(self._scenario_path)
        demand = loaded["demand"]
        if demand is None:
            raise ValueError("в сценарии нет заявки")
        if len(loaded["keys"]) != 1:
            raise ValueError("заявка этой среды ведёт один аппарат")
        key = loaded["keys"][0]
        orbit = mean_orbit(key, loaded["epoch"], nu_deg=float(loaded["nu_deg"]))
        demand = dict(demand)
        if demand["place"] == "subsatellite":
            lat, lon = subsatellite_latlon(orbit.r, orbit.epoch)
            demand["lat_deg"] = lat
            demand["lon_deg"] = lon
        weather = indices_for(orbit.epoch)
        station = self._delivery_station(demand)
        target = GroundTarget(name=str(demand["name"]), lat_deg=float(demand["lat_deg"]), lon_deg=float(demand["lon_deg"]))
        craft = Spacecraft(config=spacecraft_config(key))
        craft.reset_state()
        self._demand = demand
        self._engine = SimulationEngine(
            orbit,
            craft,
            scenario=ScenarioConfig(
                ground_stations=[station],
                ground_targets=[target],
                f107=150.0 if weather is None else weather.f107,
                f107a=150.0 if weather is None else weather.f107a,
                ap=4.0 if weather is None else weather.ap,
            ),
        )
        return self._obs(), self._info(False)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        engine, demand = self._require()
        choice = int(np.asarray(action).flat[0])
        command = {0: "wait", 1: "imaging", 2: "comms"}.get(choice, "wait")
        engine.step(_STEP_SEC, command)
        success = self._success()
        finished = success or engine.time_sec + 1e-6 >= float(demand["deadline_sec"])
        reward = float(self._weights["task_done"]) if success else 0.0
        return self._obs(), reward, finished, False, self._info(success)

    def _delivery_station(self, demand: dict) -> GroundStation:
        deliver = str(demand["deliver_to"])
        if deliver == "overhead":
            return GroundStation(
                name="overhead",
                lat_deg=float(demand["lat_deg"]),
                lon_deg=float(demand["lon_deg"]),
                alt_m=0.0,
                min_elevation_deg=5.0,
            )
        try:
            return station_by_name(deliver)
        except KeyError as exc:
            raise ValueError(f"станция сдачи не из каталога: {deliver}") from exc

    def _info(self, success: bool) -> dict:
        engine, demand = self._require()
        snap = engine.get_state_snapshot()
        return {
            "success": success,
            "scenario": str(self._scenario_path),
            "demand": demand["name"],
            "deliver_to": demand["deliver_to"],
            "deadline_sec": float(demand["deadline_sec"]),
            "generated_bits": float(snap.generated_bits),
            "delivered_bits": float(snap.delivered_bits),
            "stored_bits": float(snap.stored_bits),
            "time_sec": float(engine.time_sec),
        }


def packaged_demand(name: str) -> Path:
    """Файл заявки из репозитория. Сначала каталог запуска, затем пакет."""
    cwd = Path.cwd() / "configs" / "scenarios" / name
    if cwd.is_file():
        return cwd
    return Path(__file__).resolve().parents[1] / "configs" / name
