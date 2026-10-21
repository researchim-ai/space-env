"""
Группировка: несколько полноценных аппаратов на общих часах.
Каждый ведёт свой SimulationEngine (батарея, тепло, связь, съёмка).
Между парами считается геометрия RTN и скрининг Pc, если дистанция меньше порога.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.core.orbit import Orbit
from space_env.core.probability import bplane_probability
from space_env.core.relative import eci_to_rtn
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine


@dataclass(frozen=True)
class PairGeometry:
    leader: str
    follower: str
    range_m: float
    range_rate_m_s: float
    radial_m: float
    along_m: float
    cross_m: float


@dataclass(frozen=True)
class ScreenHit:
    leader: str
    follower: str
    range_m: float
    probability: float


class Fleet:
    """Именованные аппараты. Шаг один и тот же для всех."""

    def __init__(self, vehicles: dict[str, SimulationEngine]) -> None:
        if len(vehicles) < 2:
            raise ValueError("в группировке минимум два аппарата")
        self.vehicles = dict(vehicles)

    @classmethod
    def string_of_pearls(
        cls,
        count: int = 3,
        separation_deg: float = 10.0,
        *,
        epoch: str = "2024-01-01T00:00:00Z",
        a_m: float = 6800e3,
        inc_deg: float = 51.6,
        scenario: ScenarioConfig | None = None,
        names: list[str] | None = None,
    ) -> "Fleet":
        """Одна плоскость, истинная аномалия с шагом separation_deg. Это цепочка, не полный Walker."""
        if count < 2:
            raise ValueError("в цепочке минимум два аппарата")
        template = scenario or ScenarioConfig()
        moment = Time(epoch)
        vehicles: dict[str, SimulationEngine] = {}
        for index in range(count):
            name = names[index] if names else f"SC-{index + 1}"
            orbit = Orbit.from_keplerian(
                a_m, 0.0, np.deg2rad(inc_deg), 0.0, 0.0, np.deg2rad(separation_deg * index), moment
            )
            craft = Spacecraft(config=SpacecraftConfig(name=name))
            craft.reset_state()
            vehicles[name] = SimulationEngine(orbit, craft, scenario=template.model_copy(deep=True))
        return cls(vehicles)

    def step(
        self,
        dt_sec: float,
        commands: dict[str, str | None] | None = None,
        payloads: dict[str, dict] | None = None,
    ) -> dict[str, tuple]:
        """Команда адресуется по имени. Аппарат без команды просто летит."""
        commands = commands or {}
        payloads = payloads or {}
        unknown = set(commands) - set(self.vehicles)
        if unknown:
            raise KeyError(f"нет аппарата {sorted(unknown)[0]}")
        results = {}
        for name, engine in self.vehicles.items():
            results[name] = engine.step(dt_sec, commands.get(name), payloads.get(name))
        times = {engine.time_sec for engine in self.vehicles.values()}
        if len(times) != 1:
            raise RuntimeError("часы аппаратов разъехались")
        return results

    @property
    def time_sec(self) -> float:
        return next(iter(self.vehicles.values())).time_sec

    def pairs(self) -> list[PairGeometry]:
        """Все неупорядоченные пары. Ось RTN — у первого имени в алфавитном порядке."""
        names = sorted(self.vehicles)
        found: list[PairGeometry] = []
        for i, leader in enumerate(names):
            eng_a = self.vehicles[leader]
            for follower in names[i + 1 :]:
                eng_b = self.vehicles[follower]
                dr = eng_b.orbit.r - eng_a.orbit.r
                dv = eng_b.orbit.v - eng_a.orbit.v
                distance = float(np.linalg.norm(dr))
                rate = float(np.dot(dr, dv) / distance) if distance > 1.0 else 0.0
                rtn = eci_to_rtn(dr, eng_a.orbit.r, eng_a.orbit.v)
                found.append(
                    PairGeometry(leader, follower, distance, rate, float(rtn[0]), float(rtn[1]), float(rtn[2]))
                )
        return found

    def screen(
        self,
        *,
        hard_body_radius_m: float = 15.0,
        sigma_m: float = 200.0,
        range_limit_m: float = 50_000.0,
    ) -> list[ScreenHit]:
        """
        Pc для пар ближе порога. Ковариация диагональная: σ по радиусу и нормали,
        вдоль трека в 8 раз больше. Это допущение скрининга, не CDM конкретной пары.
        """
        hits: list[ScreenHit] = []
        covariance = np.diag([sigma_m**2, (8.0 * sigma_m) ** 2, sigma_m**2, 0.0, 0.0, 0.0])
        for pair in self.pairs():
            if pair.range_m > range_limit_m:
                continue
            leader = self.vehicles[pair.leader]
            follower = self.vehicles[pair.follower]
            dr = eci_to_rtn(follower.orbit.r - leader.orbit.r, leader.orbit.r, leader.orbit.v)
            dv = eci_to_rtn(follower.orbit.v - leader.orbit.v, leader.orbit.r, leader.orbit.v)
            probability = bplane_probability(dr, dv, covariance, hard_body_radius_m)
            hits.append(ScreenHit(pair.leader, pair.follower, pair.range_m, probability))
        return hits
