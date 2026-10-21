"""
План цепочки на общий горизонт.

Шлюз — аппарат с наибольшим углом места над Фэрбанксом на эпоху старта.
Пакет стоит на аппарате с наименьшим углом. За шаг открывается один луч:
пересылка вызывается тем же relay, что и среда, после шага Кауэлла.
Если Земля закрывает хорду или запас линии отрицательный, дело в план
входит как невыполненное, а не как успешная сдача.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.core.frames import elevation_deg
from space_env.core.ground import FAIRBANKS
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet

_PACKET_MBIT = 8.0


@dataclass(frozen=True)
class FleetTask:
    """Одно дело одного аппарата. done — механика шага это дело пропустила."""

    start_sec: float
    end_sec: float
    actor: str
    kind: str
    destination: str
    done: bool
    bits_mbit: float
    reason: str


@dataclass(frozen=True)
class FleetSchedule:
    """Список дел цепочки. Строки можно отдать в SFT без отдельной разметки."""

    count: int
    step_sec: float
    source: str
    gateway: str
    arrived: bool
    tasks: tuple[FleetTask, ...]

    def lines(self) -> list[str]:
        return [item.reason for item in self.tasks]


def _elevations(fleet: Fleet, names: list[str]) -> list[float]:
    values = []
    for name in names:
        engine = fleet.vehicles[name]
        values.append(
            elevation_deg(
                engine.orbit.r,
                FAIRBANKS.lat_deg,
                FAIRBANKS.lon_deg,
                FAIRBANKS.alt_m,
                engine.orbit.epoch,
            )
        )
    return values


def _pair(fleet: Fleet, left: str, right: str):
    return next(
        (item for item in fleet.pairs() if {item.leader, item.follower} == {left, right}),
        None,
    )


def build_relay_schedule(
    *,
    count: int = 4,
    steps: int = 8,
    step_sec: float = 60.0,
    separation_deg: float = 1.5,
) -> FleetSchedule:
    """
    План доставки 8 Мбит по цепочке.
    separation_deg оставляют около 1.5°: линия 1 Вт / 6 дБи на большем шаге уже не закрывается.
    """
    if count < 3:
        raise ValueError("в цепочке плана минимум три аппарата")
    if steps < 1 or step_sec <= 0.0:
        raise ValueError("горизонт плана должен быть положительным")
    fleet = Fleet.string_of_pearls(
        count,
        separation_deg,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    names = list(fleet.vehicles)
    elevations = _elevations(fleet, names)
    gateway_index = int(np.argmax(elevations))
    source_index = int(np.argmin(elevations))
    if source_index == gateway_index:
        source_index = 0 if gateway_index != 0 else count - 1
    source = names[source_index]
    gateway = names[gateway_index]
    fleet.collect(source, _PACKET_MBIT)
    tasks = [
        FleetTask(
            0.0,
            0.0,
            source,
            "assign",
            gateway,
            True,
            _PACKET_MBIT,
            (
                f"0 с: пакет {_PACKET_MBIT:.0f} Мбит на {source}. "
                f"Шлюз — {gateway}, у него наибольший угол места над Фэрбанксом. "
                "Это геометрия антенны, фидер не замкнут."
            ),
        )
    ]
    holder = source_index
    arrived = holder == gateway_index
    for step in range(steps):
        if arrived:
            break
        start = step * step_sec
        end = start + step_sec
        nxt = holder + (1 if gateway_index > holder else -1)
        actor = names[holder]
        destination = names[nxt]
        fleet.step(step_sec)
        pair = _pair(fleet, actor, destination)
        moved = fleet.relay(actor, destination, step_sec)
        if moved > 0.0:
            holder = nxt
            arrived = holder == gateway_index and fleet.buffer_mb[gateway] >= _PACKET_MBIT - 1e-6
            tasks.append(
                FleetTask(
                    start,
                    end,
                    actor,
                    "relay",
                    destination,
                    True,
                    moved,
                    (
                        f"{start:.0f}–{end:.0f} с: {actor} пересылает на {destination}. "
                        f"Луч 1 Вт / 6 дБи открыт, Земля не закрывает, перешло {moved:.1f} Мбит. "
                        "Остальные аппараты ждут: за шаг один луч."
                    ),
                )
            )
            continue
        if pair is None:
            why = "между аппаратами нет пары."
        elif pair.earth_blocked:
            why = "луч пересекает Землю."
        else:
            why = f"запас линии {pair.isl_margin_db:.1f} дБ."
        tasks.append(
            FleetTask(
                start,
                end,
                actor,
                "relay",
                destination,
                False,
                0.0,
                (
                    f"{start:.0f}–{end:.0f} с: {actor} не пересылает на {destination}. {why} "
                    "Пакет остаётся на месте."
                ),
            )
        )
        break
    if arrived:
        tasks.append(
            FleetTask(
                tasks[-1].end_sec,
                tasks[-1].end_sec,
                gateway,
                "arrive",
                gateway,
                True,
                fleet.buffer_mb[gateway],
                (
                    f"{tasks[-1].end_sec:.0f} с: пакет на {gateway}. "
                    "Дальше по цепочке его не несут. Сдача на Землю этим планом не считается."
                ),
            )
        )
    return FleetSchedule(count, step_sec, source, gateway, arrived, tuple(tasks))
