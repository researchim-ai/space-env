"""
Задачи, которые в эксплуатации стоят отдельно от одной съёмки.
Удержание дистанции, выбор цели при одном слоте памяти, отказ радио в окне,
уклонение от неманеврирующего фрагмента и увод с орбиты.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine
from space_env.catalog.phasing import add_prograde_dv
from space_env.catalog.references import reference
from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.core.environment import in_eclipse
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.relative import eci_to_rtn, rtn_basis
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.horizon import PC_THRESHOLD
from space_env.mission.scenarios import ScenarioConfig
from space_env.rl.rewards import load_rewards
from space_env.simulation.fleet import Fleet, _make_vehicle

_HOLD_DV_M_S = 0.05
_DISPOSAL_DV_M_S = 0.05
_DISPOSAL_HORIZON_S = 3.0 * 3600.0


def _along_m(leader, follower) -> float:
    return float(eci_to_rtn(follower.orbit.r - leader.orbit.r, leader.orbit.r, leader.orbit.v)[1])


def _sma_m(position: np.ndarray, velocity: np.ndarray) -> float:
    radius = float(np.linalg.norm(position))
    energy = 0.5 * float(np.dot(velocity, velocity)) - MU_EARTH / radius
    return -MU_EARTH / (2.0 * energy)


def _coast_sma_m(position: np.ndarray, velocity: np.ndarray) -> float:
    end_r, end_v, _samples = propagate_state(
        position,
        velocity,
        _DISPOSAL_HORIZON_S,
        mass_kg=100.0,
        drag_area_m2=0.9,
        drag_coefficient=2.2,
        include_drag=True,
        include_j2=True,
        include_third_body=False,
        substep_sec=60.0,
    )
    return _sma_m(end_r, end_v)


class RendezvousHoldEnv(gym.Env):
    """
    Ведомый стоит на ~2 км впереди и уже получил +0.05 м/с.
    Действия: 0 ждать, 1 встречные 0.05 м/с.
    Успех — скорость расхождения вдоль трека за 300 с меньше 0.05 м/с по модулю.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._weights = load_rewards()["rendezvous_hold"]

    def _obs(self, along_m: float) -> np.ndarray:
        return np.array([float(np.clip(along_m / 10_000.0, -1.0, 1.0)), 1.0, 0.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        fleet = Fleet.string_of_pearls(2, separation_deg=0.02, scenario=ScenarioConfig(ground_stations=[]))
        names = list(fleet.vehicles)
        along = _along_m(fleet.vehicles[names[0]], fleet.vehicles[names[1]])
        return self._obs(along), {"hold": "погасить +0.05 м/с вдоль трека", "along_m": along}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        burn = int(np.asarray(action).flat[0]) == 1
        fleet = Fleet.string_of_pearls(2, separation_deg=0.02, scenario=ScenarioConfig(ground_stations=[]))
        names = list(fleet.vehicles)
        leader, follower = fleet.vehicles[names[0]], fleet.vehicles[names[1]]
        add_prograde_dv(follower, _HOLD_DV_M_S)
        if burn:
            fleet.step(
                1.0,
                {names[1]: "maneuver"},
                {names[1]: {"dv_m_s": _HOLD_DV_M_S, "direction": "retrograde"}},
            )
        before = _along_m(leader, follower)
        fleet.step(300.0)
        after = _along_m(leader, follower)
        rate = (after - before) / 300.0
        success = burn and abs(rate) < 0.05
        reward = self._weights["held"] if success else -self._weights["drift"]
        info = {"rate_m_s": rate, "along_m": after, "burned": burn, "success": success}
        return self._obs(after), float(reward), True, False, info


class PriorityCoverageEnv(gym.Env):
    """
    В полосе две цели и один свободный слот памяти опорного 6U.
    Действия: 0 пропустить, 1 снять приоритетную, 2 снять запасную.
    Успех на свету — в слоте только приоритетная цель.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._sunlit = True
        self._weights = load_rewards()["priority_coverage"]

    def _obs(self) -> np.ndarray:
        return np.array([1.0 if self._sunlit else 0.0, 1.0, 1.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        engine = make_engine("cubesat-6u", with_stations=False, nu_deg=180.0)
        self._sunlit = not in_eclipse(engine.orbit.r, sun_position_m(engine.now()))
        return self._obs(), {"priority": "P1", "filler": "F1", "sunlit": self._sunlit, "vehicle": "cubesat-6u"}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        stored = None
        if self._sunlit and choice == 1:
            stored = "P1"
        elif self._sunlit and choice == 2:
            stored = "F1"
        success = (stored == "P1") if self._sunlit else stored is None
        reward = self._weights["priority"] if success else -self._weights["missed"]
        info = {"stored": stored, "sunlit": self._sunlit, "success": success}
        return self._obs(), float(reward), True, False, info


class PassFaultEnv(gym.Env):
    """
    Окно 6 минут, радио опорного 6U в защёлке.
    Действия: 0 ждать, 1 сдавать, 2 перезагрузка. Перезагрузка занимает 3 шага движка.
    Успех — после восстановления в окне сдана порция.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 6, step_sec: float = 60.0):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.step_sec = step_sec
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._engine = None
        self._steps = 0
        self._rebooted = False
        self._delivered = False
        self._weights = load_rewards()["pass_fault"]

    def _obs(self) -> np.ndarray:
        ready = 0.0 if self._engine is None else float(self._engine.radio_ready())
        window = float(np.clip(1.0 - self._steps / self.max_episode_steps, 0.0, 1.0))
        return np.array([ready, window, 1.0 if self._rebooted else 0.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._rebooted = False
        self._delivered = False
        self._engine = make_engine("cubesat-6u", with_stations=False)
        self._engine.arm_fault("radio_latch")
        return self._obs(), {"fault": "radio_latch", "window_steps": self.max_episode_steps}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        choice = int(np.asarray(action).flat[0])
        command = {1: "comms", 2: "reboot"}.get(choice)
        if choice == 2:
            self._rebooted = True
        self._engine.step(self.step_sec, command)
        if choice == 1 and self._engine.radio_ready() and self._steps < self.max_episode_steps:
            self._delivered = True
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        success = self._delivered and self._engine.radio_ready()
        reward = self._weights["recovered"] if success and truncated else 0.0
        if truncated and not success:
            reward = -self._weights["lost_pass"]
        info = {
            "radio_ready": self._engine.radio_ready(),
            "delivered": self._delivered,
            "success": success and truncated,
        }
        return self._obs(), float(reward), False, truncated, info


class DebrisAvoidEnv(gym.Env):
    """
    Фрагмент без топлива стоит в 40 м по нормали. Импульс ему не исполняется.
    Действия: 0 ждать, 1 импульс 1 м/с по нормали своему аппарату, 2 импульс фрагменту.
    Успех — через 1200 с Pc ниже 10⁻⁴ и двигатель фрагмента не трогали.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._weights = load_rewards()["debris_avoid"]

    def _fleet(self) -> Fleet:
        moment = Time("2024-01-01T00:00:00Z")
        scenario = ScenarioConfig(ground_stations=[])
        own = _make_vehicle("OWN", 6_800_000.0, 51.6, 0.0, 0.0, moment, scenario)
        debris_cfg = reference("debris-fragment").spacecraft_config("DEBRIS")
        debris = _make_vehicle("DEBRIS", 6_800_000.0, 51.6, 0.0, 10.0, moment, scenario, debris_cfg)
        basis = rtn_basis(own.orbit.r, own.orbit.v)
        position = own.orbit.r + basis @ np.array([0.0, 0.0, 40.0])
        velocity = own.orbit.v + basis @ np.array([0.0, -5.0, 0.0])
        debris.orbit = Orbit.from_vectors(position, velocity, moment)
        return Fleet({"OWN": own, "DEBRIS": debris})

    def _pc(self, fleet: Fleet) -> float:
        for hit in fleet.screen(range_limit_m=200_000.0):
            if {hit.leader, hit.follower} == {"OWN", "DEBRIS"}:
                return hit.probability
        return 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        fleet = self._fleet()
        probability = self._pc(fleet)
        obs = np.array([float(np.clip(probability / 1.0e-2, 0.0, 1.0)), 0.0, 1.0], dtype=np.float32)
        return obs, {"pc": probability, "debris_dv_m_s": 0.0}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        fleet = self._fleet()
        debris_rejected = False
        if choice == 1:
            fleet.step(1200.0, {"OWN": "maneuver"}, {"OWN": {"dv_m_s": 1.0, "direction": "normal"}})
        elif choice == 2:
            fleet.step(1200.0, {"DEBRIS": "maneuver"}, {"DEBRIS": {"dv_m_s": 1.0, "direction": "normal"}})
            debris_rejected = fleet.vehicles["DEBRIS"].last_command_ok is False
        else:
            fleet.step(1200.0)
        probability = self._pc(fleet)
        success = choice == 1 and probability < PC_THRESHOLD and not debris_rejected
        if choice == 2:
            success = False
        reward = self._weights["avoided"] if success else -self._weights["missed"]
        obs = np.array([float(np.clip(probability / 1.0e-2, 0.0, 1.0)), 0.0, 1.0], dtype=np.float32)
        info = {"pc": probability, "debris_rejected": debris_rejected, "success": success}
        return obs, float(reward), True, False, info


class DisposalEnv(gym.Env):
    """
    Увод с 780 км: импульс 0.05 м/с против скорости.
    Успех — через 3 часа большая полуось ниже, чем у того же прогона без импульса, больше чем на 10 м.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._weights = load_rewards()["disposal"]

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return np.array([1.0, 780.0 / 1000.0], dtype=np.float32), {"ordered": True, "altitude_km": 780.0}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        burn = int(np.asarray(action).flat[0]) == 1
        moment = Time("2024-01-01T00:00:00Z")
        orbit = Orbit.from_keplerian(R_EARTH_EQUATOR + 780_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, moment)
        waited = _coast_sma_m(orbit.r, orbit.v)
        if burn:
            retro = orbit.v - orbit.v / np.linalg.norm(orbit.v) * _DISPOSAL_DV_M_S
            done = _coast_sma_m(orbit.r, retro)
        else:
            done = waited
        drop = waited - done
        success = burn and drop > 10.0
        reward = self._weights["lowered"] if success else -self._weights["left"]
        info = {"drop_m": drop, "burned": burn, "success": success}
        return np.array([1.0, 780.0 / 1000.0], dtype=np.float32), float(reward), True, False, info
