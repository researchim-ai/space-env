"""
Съёмка и сдача данных аппарата из каталога.
Награда — только биты, ушедшие на станцию при закрытой линии.
Запись в память сама по себе очки не даёт.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine, mean_orbit, spacecraft_config
from space_env.catalog.references import reference
from space_env.catalog.traffic import dump_bits, store_bits
from space_env.core.access import ground_track_latlon, off_nadir_deg, subsatellite_latlon
from space_env.core.link import stepped_rate_bps
from space_env.envs.base import format_ansi
from space_env.mission.scenarios import GroundTarget
from space_env.simulation.engine import SimulationEngine

_COMMANDS = {0: "wait", 1: "imaging", 2: "comms"}
_OPEN_CAP_S = 1.0e12


class CatalogAcquisitionEnv(gym.Env):
    """
    Действия: ожидание, съёмка, сеанс.
    По умолчанию эпизод начинается за три минуты до ближайшего AOS на горизонте 6 ч,
    чтобы сдача попала в окно, а не в случайную точку витка.
    Длина по умолчанию — чуть больше витка при шаге 60 с.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        vehicle: str = "sentinel-2a",
        step_sec: float = 60.0,
        max_episode_steps: int = 110,
        with_stations: bool = True,
        start_at_pass: bool = True,
        lead_sec: float = 180.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.vehicle = vehicle
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.with_stations = with_stations
        self.start_at_pass = start_at_pass
        self.lead_sec = lead_sec
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        spec = reference(vehicle)
        self._title = spec.title
        self._rate_bps = float(spec.store_rate_bps or spec.downlink_bps)
        self._memory_bits = float(spec.memory_bits or 1.0)
        self._imaging_cap_s = float(spec.max_imaging_s) if spec.max_imaging_s is not None else _OPEN_CAP_S
        self._downlink_bps = spec.downlink_bps
        self._engine: SimulationEngine | None = None
        self._buffer_bits = 0.0
        self._delivered_bits = 0.0
        self._imaging_sec = 0.0
        self._steps = 0

    def _targets_on_track(self) -> list[GroundTarget]:
        orbit = mean_orbit(self.vehicle, nu_deg=180.0)
        prop = spacecraft_config(self.vehicle).propulsion
        points = ground_track_latlon(
            orbit,
            [0.0, 600.0, 1500.0],
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
        )
        prefix = {"landsat-9": "L9", "cubesat-6u": "C6"}.get(self.vehicle, "S2")
        return [
            GroundTarget(name=f"{prefix}-{index + 1}", lat_deg=lat, lon_deg=lon)
            for index, (lat, lon) in enumerate(points)
        ]

    def _park_before_pass(self) -> None:
        engine = self._engine
        if engine is None or not self.start_at_pass:
            return
        aos = engine.seconds_to_next_aos()
        if aos is None or aos <= self.lead_sec:
            return
        engine.step(aos - self.lead_sec, "wait")
        lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
        engine.scenario.ground_targets = [GroundTarget(name="у сеанса", lat_deg=lat, lon_deg=lon)]
        engine.clear_captured_targets()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._buffer_bits = 0.0
        self._delivered_bits = 0.0
        self._imaging_sec = 0.0
        self._engine = make_engine(
            self.vehicle,
            nu_deg=180.0,
            with_stations=self.with_stations,
            ground_targets=self._targets_on_track(),
        )
        self._park_before_pass()
        names = [item.name for item in self._engine.scenario.ground_targets]
        return self._obs(), {"targets": names, "vehicle": self.vehicle}

    def _nearest_off_nadir_fraction(self) -> float:
        engine = self._engine
        if engine is None:
            return 1.0
        half = engine.spacecraft.config.payload.half_fov_deg
        if not engine.scenario.ground_targets or half <= 0.0:
            return 1.0
        moment = engine.now()
        captured = engine.captured_targets()
        angles = [
            off_nadir_deg(engine.orbit.r, target.lat_deg, target.lon_deg, moment)
            for target in engine.scenario.ground_targets
            if target.name not in captured
        ]
        if not angles:
            return 0.0
        return float(np.clip(min(angles) / half, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(7, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        aos = 1.0 if snap.seconds_to_aos is None else float(np.clip(snap.seconds_to_aos / (6.0 * 3600.0), 0.0, 1.0))
        imaging = (
            float(np.clip(self._imaging_sec / self._imaging_cap_s, 0.0, 1.0))
            if self._imaging_cap_s < _OPEN_CAP_S
            else 0.0
        )
        values = [
            snap.battery_soc,
            float(snap.in_eclipse),
            float(snap.in_contact),
            float(np.clip(self._buffer_bits / self._memory_bits, 0.0, 1.0)),
            imaging,
            aos,
            self._nearest_off_nadir_fraction(),
        ]
        return np.asarray(values, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        command = _COMMANDS.get(int(action), "wait")
        before = self._engine.get_state_snapshot().imaging_count
        _state, _events = self._engine.step(self.step_sec, None if command == "wait" else command)
        snap = self._engine.get_state_snapshot()
        stored = 0.0
        if command == "imaging" and snap.imaging_count > before:
            self._buffer_bits, self._imaging_sec, stored = store_bits(
                self._buffer_bits,
                self.step_sec,
                rate_bps=self._rate_bps,
                memory_bits=self._memory_bits,
                imaging_sec=self._imaging_sec,
                imaging_cap_sec=self._imaging_cap_s,
            )
        dumped = 0.0
        if command == "comms" and snap.link_ok and snap.link_margin_db >= 0.0:
            rate = stepped_rate_bps(snap.link_margin_db, self._downlink_bps)
            self._buffer_bits, dumped = dump_bits(self._buffer_bits, self.step_sec, rate)
            self._delivered_bits += dumped
        reward = dumped / self._memory_bits
        if snap.battery_soc < 0.2:
            reward -= 1.0
        self._steps += 1
        terminated = snap.battery_soc < 0.2
        truncated = self._steps >= self.max_episode_steps
        info = {
            "buffer_bits": self._buffer_bits,
            "delivered_bits": self._delivered_bits,
            "stored_bits": stored,
            "imaging_count": snap.imaging_count,
            "station": snap.contact_station,
            "success": truncated and self._delivered_bits > 0.0 and not terminated,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._engine is None:
            return None
        snap = self._engine.get_state_snapshot()
        rows = [
            ("SoC", f"{snap.battery_soc:.2f}"),
            ("память, Гбит", f"{self._buffer_bits / 1e9:.1f}"),
            ("сдано, Гбит", f"{self._delivered_bits / 1e9:.1f}"),
            ("кадры", str(snap.imaging_count)),
        ]
        return format_ansi(self._title, rows)


class SentinelAcquisitionEnv(CatalogAcquisitionEnv):
    """Sentinel-2A: 450 Мбит/с в 2.4 Тбит, не дольше 40 минут на виток, сброс до 560 Мбит/с."""

    def __init__(
        self,
        step_sec: float = 60.0,
        max_episode_steps: int = 110,
        with_stations: bool = True,
        start_at_pass: bool = True,
        lead_sec: float = 180.0,
        render_mode: str | None = None,
    ):
        super().__init__(
            "sentinel-2a",
            step_sec=step_sec,
            max_episode_steps=max_episode_steps,
            with_stations=with_stations,
            start_at_pass=start_at_pass,
            lead_sec=lead_sec,
            render_mode=render_mode,
        )


class CubesatAcquisitionEnv(CatalogAcquisitionEnv):
    """Опорный 6U: та же задача сдачи, что у Sentinel и Landsat, на допущениях класса."""

    def __init__(
        self,
        step_sec: float = 60.0,
        max_episode_steps: int = 110,
        with_stations: bool = True,
        start_at_pass: bool = True,
        lead_sec: float = 180.0,
        render_mode: str | None = None,
    ):
        super().__init__(
            "cubesat-6u",
            step_sec=step_sec,
            max_episode_steps=max_episode_steps,
            with_stations=with_stations,
            start_at_pass=start_at_pass,
            lead_sec=lead_sec,
            render_mode=render_mode,
        )


class SentinelPassFaultEnv(CatalogAcquisitionEnv):
    """
    Сеанс Sentinel-2A уже выбран, радио в защёлке.
    Действия: 0 ждать, 1 сдавать, 2 перезагрузка.
    Успех — радио готово и на станции есть биты до конца эпизода.
    """

    def __init__(self, step_sec: float = 60.0, max_episode_steps: int = 12, render_mode: str | None = None):
        super().__init__(
            "sentinel-2a",
            step_sec=step_sec,
            max_episode_steps=max_episode_steps,
            with_stations=True,
            start_at_pass=True,
            lead_sec=180.0,
            render_mode=render_mode,
        )
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._rebooted = False

    def _fault_obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(4, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        window = float(np.clip(1.0 - self._steps / self.max_episode_steps, 0.0, 1.0))
        return np.array(
            [
                float(self._engine.radio_ready()),
                float(snap.in_contact),
                1.0 if self._rebooted else 0.0,
                window,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        _obs, info = super().reset(seed=seed, options=options)
        assert self._engine is not None
        self._engine.arm_fault("radio_latch")
        self._rebooted = False
        self._buffer_bits = 8.0e9
        info["fault"] = "radio_latch"
        return self._fault_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        choice = int(action)
        command = {1: "comms", 2: "reboot"}.get(choice)
        if choice == 2:
            self._rebooted = True
        self._engine.step(self.step_sec, command)
        snap = self._engine.get_state_snapshot()
        if choice == 1 and snap.link_ok and snap.link_margin_db >= 0.0:
            rate = stepped_rate_bps(snap.link_margin_db, self._downlink_bps)
            self._buffer_bits, dumped = dump_bits(self._buffer_bits, self.step_sec, rate)
            self._delivered_bits += dumped
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        success = truncated and self._delivered_bits > 0.0 and self._engine.radio_ready()
        info = {
            "delivered_bits": self._delivered_bits,
            "radio_ready": self._engine.radio_ready(),
            "in_contact": snap.in_contact,
            "success": success,
        }
        return self._fault_obs(), float(self._delivered_bits > 0.0), False, truncated, info


class LandsatAcquisitionEnv(CatalogAcquisitionEnv):
    """Landsat 9: регистратор 3.14 Тбит, сброс 384 Мбит/с на Су-Фолс и Шпицберген."""

    def __init__(
        self,
        step_sec: float = 60.0,
        max_episode_steps: int = 110,
        with_stations: bool = True,
        start_at_pass: bool = True,
        lead_sec: float = 180.0,
        render_mode: str | None = None,
    ):
        super().__init__(
            "landsat-9",
            step_sec=step_sec,
            max_episode_steps=max_episode_steps,
            with_stations=with_stations,
            start_at_pass=start_at_pass,
            lead_sec=lead_sec,
            render_mode=render_mode,
        )
