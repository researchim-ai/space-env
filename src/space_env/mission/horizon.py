"""
Планы на горизонт, в которых манёвр появляется только из механики шага.

Сближение: Pc из скрининга пары, порог 10⁻⁴, направление — большая компонента
промаха в радиусе или по нормали. Бюджет Δv режет очередь.
Фаза: второй импульс через один начальный период ведомого снимает +0.5 м/с.
Накопленная за виток дуга при этом остаётся.
Снижение: два прогона с J2, с сопротивлением и без. Импульс есть, если потеря
большой полуоси больше 5 м. Третье тело в разность не входит.
Обход: следующий скачок берётся, только если пара не в списке закрытых и relay
после шага Кауэлла переносит биты.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.catalog.build import sentinel2_design_fleet
from space_env.catalog.phasing import add_prograde_dv, phase_error_m
from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.relative import eci_to_rtn, rtn_basis
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet, _make_vehicle

PC_THRESHOLD = 1.0e-4
_DECAY_LOSS_M = 5.0
_CONJUNCTION_COAST_SEC = 1200.0
_PHASE_DV_M_S = 0.5
_PACKET_MBIT = 8.0


@dataclass(frozen=True)
class HorizonTask:
    """Одно дело плана. required=False — строка объясняет, почему манёвра нет."""

    kind: str
    name: str
    start_sec: float
    required: bool
    reason: str
    dv_m_s: float = 0.0
    direction: str = ""
    probability: float = 0.0


@dataclass(frozen=True)
class HorizonPlan:
    title: str
    tasks: tuple[HorizonTask, ...]
    metric: float = 0.0
    reference: float = 0.0

    def lines(self) -> list[str]:
        return [item.reason for item in self.tasks]

    def required_items(self) -> tuple[HorizonTask, ...]:
        return tuple(item for item in self.tasks if item.required)


def _direction(radial_m: float, cross_m: float) -> str:
    return "prograde" if abs(radial_m) >= abs(cross_m) else "normal"


def schedule_conjunctions(
    rows: list[tuple[str, float, float, float]],
    *,
    budget_m_s: float,
    burn_m_s: float = 1.0,
    threshold: float = PC_THRESHOLD,
    when_sec: list[float] | None = None,
) -> list[HorizonTask]:
    """
    rows — (имя объекта, Pc, радиальный промах, боковой промах), по убыванию Pc.
    Объект ниже порога остаётся без импульса. Объект выше порога получает импульс,
    пока хватает бюджета. when_sec — секунды этих Pc от эпохи; без списка все метки в нуле.
    """
    if burn_m_s <= 0.0 or budget_m_s < 0.0:
        raise ValueError("бюджет и импульс должны быть неотрицательными, импульс — положительным")
    if when_sec is not None and len(when_sec) != len(rows):
        raise ValueError("меток времени должно быть столько же, сколько объектов")
    remaining = budget_m_s
    tasks: list[HorizonTask] = []
    for index, (name, probability, radial_m, cross_m) in enumerate(rows):
        moment = 0.0 if when_sec is None else float(when_sec[index])
        if probability < threshold:
            tasks.append(
                HorizonTask(
                    "wait",
                    name,
                    moment,
                    False,
                    (
                        f"{moment:.0f} с: {name} не маневрировать. Pc {probability:.3e} ниже порога {threshold:.0e}."
                    ),
                    probability=probability,
                )
            )
            continue
        direction = _direction(radial_m, cross_m)
        if remaining + 1e-9 < burn_m_s:
            tasks.append(
                HorizonTask(
                    "wait",
                    name,
                    moment,
                    False,
                    (
                        f"{moment:.0f} с: {name} не маневрировать. Pc {probability:.3e} выше порога, "
                        f"бюджет Δv исчерпан."
                    ),
                    probability=probability,
                )
            )
            continue
        remaining -= burn_m_s
        tasks.append(
            HorizonTask(
                "maneuver",
                name,
                moment,
                True,
                (
                    f"{moment:.0f} с: {name} импульс {burn_m_s:.1f} м/с {direction}. "
                    f"Pc {probability:.3e}, промах по радиусу {radial_m:.0f} м, "
                    f"по нормали {cross_m:.0f} м."
                ),
                dv_m_s=burn_m_s,
                direction=direction,
                probability=probability,
            )
        )
    return tasks


def _secondary(name: str, primary, radial_m: float, cross_m: float, approach_m_s: float):
    moment = primary.orbit.epoch
    basis = rtn_basis(primary.orbit.r, primary.orbit.v)
    position = primary.orbit.r + basis @ np.array([radial_m, 0.0, cross_m])
    velocity = primary.orbit.v + basis @ np.array([0.0, -approach_m_s, 0.0])
    engine = _make_vehicle(name, 6_800_000.0, 51.6, 0.0, 10.0, moment, ScenarioConfig(ground_stations=[]))
    engine.orbit = engine.orbit.with_state(position, velocity, moment)
    return engine


def _conjunction_fleet() -> Fleet:
    """HOT и WARM выше порога на старте, COLD с промахом 4 км по нормали — ниже."""
    moment = Time("2024-01-01T00:00:00Z")
    scenario = ScenarioConfig(ground_stations=[])
    primary = _make_vehicle("OWN", 6_800_000.0, 51.6, 0.0, 0.0, moment, scenario)
    return Fleet(
        {
            "OWN": primary,
            "HOT": _secondary("HOT", primary, 0.0, 40.0, 5.0),
            "WARM": _secondary("WARM", primary, 0.0, 250.0, 5.0),
            "COLD": _secondary("COLD", primary, 0.0, 4_000.0, 5.0),
        }
    )


def _own_rows(fleet: Fleet) -> list[tuple[str, float, float, float]]:
    primary = fleet.vehicles["OWN"]
    rows = []
    for hit in fleet.screen(range_limit_m=200_000.0):
        if "OWN" not in (hit.leader, hit.follower):
            continue
        other = hit.follower if hit.leader == "OWN" else hit.leader
        relative = eci_to_rtn(
            fleet.vehicles[other].orbit.r - primary.orbit.r,
            primary.orbit.r,
            primary.orbit.v,
        )
        rows.append((other, hit.probability, float(relative[0]), float(relative[2])))
    rows.sort(key=lambda item: -item[1])
    return rows


def _pc_against_own(fleet: Fleet) -> dict[str, float]:
    return {name: probability for name, probability, _radial, _cross in _own_rows(fleet)}


def build_conjunction_plan(*, budget_m_s: float = 1.0, burn_m_s: float = 1.0) -> HorizonPlan:
    """Три объекта рядом с OWN. Два выше порога, один с промахом 4 км по нормали — ниже."""
    tasks = schedule_conjunctions(_own_rows(_conjunction_fleet()), budget_m_s=budget_m_s, burn_m_s=burn_m_s)
    return HorizonPlan("conjunction", tuple(tasks))


def coast_conjunction(direction: str | None, *, dv_m_s: float = 1.0) -> dict[str, float]:
    """1200 с тем же шагом группировки. direction=None — только ожидание."""
    fleet = _conjunction_fleet()
    if direction:
        fleet.step(
            _CONJUNCTION_COAST_SEC,
            {"OWN": "maneuver"},
            {"OWN": {"dv_m_s": dv_m_s, "direction": direction}},
        )
    else:
        fleet.step(_CONJUNCTION_COAST_SEC)
    return _pc_against_own(fleet)


def execute_conjunction_plan(*, budget_m_s: float = 1.0, burn_m_s: float = 1.0) -> HorizonPlan:
    """Ставит один импульс из очереди и сравнивает Pc с прогоном без импульса."""
    planned = build_conjunction_plan(budget_m_s=budget_m_s, burn_m_s=burn_m_s)
    burn = next(item for item in planned.required_items() if item.kind == "maneuver")
    after = coast_conjunction(burn.direction, dv_m_s=burn.dv_m_s)
    waited = coast_conjunction(None)
    measure = HorizonTask(
        "measure",
        "HOT",
        _CONJUNCTION_COAST_SEC,
        True,
        (
            f"{_CONJUNCTION_COAST_SEC:.0f} с: после импульса Pc HOT {after['HOT']:.3e}, "
            f"без импульса {waited['HOT']:.3e}. COLD {after['COLD']:.3e}. "
            "Отдельный импульс на WARM не ставился."
        ),
        probability=after["HOT"],
    )
    return HorizonPlan("conjunction", planned.tasks + (measure,), after["HOT"], waited["HOT"])


def build_two_burn_phase_plan(*, burn: bool = True) -> HorizonPlan:
    """
    Ведомый начинает с +0.5 м/с. Второй импульс −0.5 м/с ставится через один
    начальный период его орбиты, если burn. Скорость фазы считается по дуге за 300 с.
    """
    fleet = sentinel2_design_fleet()
    follower = fleet.vehicles["Sentinel-2B"]
    leader = fleet.vehicles["Sentinel-2A"]
    add_prograde_dv(follower, _PHASE_DV_M_S)
    semimajor = follower.orbit.keplerian().a
    period = float(2.0 * np.pi * np.sqrt(semimajor**3 / MU_EARTH))
    elapsed = 0.0
    while elapsed < period - 1e-6:
        chunk = min(60.0, period - elapsed)
        fleet.step(chunk)
        elapsed += chunk
    before = phase_error_m(leader.orbit.r, leader.orbit.v, follower.orbit.r)
    if burn:
        fleet.step(
            1.0,
            {"Sentinel-2B": "maneuver"},
            {"Sentinel-2B": {"dv_m_s": _PHASE_DV_M_S, "direction": "retrograde"}},
        )
        at_mark = phase_error_m(leader.orbit.r, leader.orbit.v, follower.orbit.r)
    else:
        at_mark = before
    fleet.step(300.0)
    after = phase_error_m(leader.orbit.r, leader.orbit.v, follower.orbit.r)
    rate = (after - at_mark) / 300.0
    measure = (
        (
            f"Через 300 с после импульса скорость фазы {rate:.4f} м/с, "
            f"дуга {after / 1000.0:.1f} км. Дрейф остановлен, накопленная фаза осталась."
        )
        if burn
        else (
            f"Без второго импульса скорость фазы {rate:.4f} м/с. "
            f"Дуга {after / 1000.0:.1f} км продолжает расти."
        )
    )
    tasks = (
        HorizonTask(
            "inject",
            "Sentinel-2B",
            0.0,
            True,
            (
                "0 с: у Sentinel-2B уже +0.5 м/с по скорости. "
                "Большая полуось выше, пара начинает уходить от антипода."
            ),
            dv_m_s=_PHASE_DV_M_S,
            direction="prograde",
        ),
        HorizonTask(
            "maneuver",
            "Sentinel-2B",
            elapsed,
            burn,
            (
                f"{elapsed:.0f} с: импульс 0.5 м/с ретроградно. "
                "Это один начальный период ведомого, аргумент широты снова у точки первого импульса. "
                f"Дуга к этому моменту {before / 1000.0:.1f} км и вторым импульсом не стирается."
            )
            if burn
            else f"{elapsed:.0f} с: второй импульс не ставится.",
            dv_m_s=_PHASE_DV_M_S if burn else 0.0,
            direction="retrograde" if burn else "",
        ),
        HorizonTask(
            "measure",
            "Sentinel-2B",
            elapsed + 300.0,
            True,
            measure,
        ),
    )
    return HorizonPlan("phasing", tasks, rate)


def _sma_m(position: np.ndarray, velocity: np.ndarray) -> float:
    radius = float(np.linalg.norm(position))
    speed2 = float(np.dot(velocity, velocity))
    energy = 0.5 * speed2 - MU_EARTH / radius
    return -MU_EARTH / (2.0 * energy)


def _coast_sma_m(position: np.ndarray, velocity: np.ndarray, horizon_sec: float, *, drag: bool) -> float:
    end_r, end_v, _samples = propagate_state(
        position,
        velocity,
        horizon_sec,
        mass_kg=100.0,
        drag_area_m2=0.9,
        drag_coefficient=2.2,
        include_drag=drag,
        include_j2=True,
        include_third_body=False,
        substep_sec=60.0,
    )
    return _sma_m(end_r, end_v)


def build_decay_plan(*, altitude_m: float = 400_000.0, horizon_sec: float = 3.0 * 3600.0) -> HorizonPlan:
    """
    Опорный борт 100 кг, мидель 0.9 м², Cd 2.2.
    Потеря — разность больших полуосей двух прогонов с J2. Импульс Δa · v / (2a).
    """
    if horizon_sec <= 0.0 or altitude_m <= 0.0:
        raise ValueError("высота и горизонт должны быть положительными")
    moment = Time("2024-01-01T00:00:00Z")
    semimajor = R_EARTH_EQUATOR + altitude_m
    orbit = Orbit.from_keplerian(semimajor, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, moment)
    with_drag = _coast_sma_m(orbit.r, orbit.v, horizon_sec, drag=True)
    without_drag = _coast_sma_m(orbit.r, orbit.v, horizon_sec, drag=False)
    loss_m = without_drag - with_drag
    speed = float(np.sqrt(MU_EARTH / semimajor))
    if loss_m > _DECAY_LOSS_M:
        dv = loss_m * speed / (2.0 * semimajor)
        task = HorizonTask(
            "maneuver",
            "SC",
            0.0,
            True,
            (
                f"0 с: импульс {dv:.3f} м/с по скорости. За {horizon_sec / 3600.0:.0f} ч на высоте "
                f"{altitude_m / 1000.0:.0f} км сопротивление забирает {loss_m:.0f} м большой полуоси. "
                "Оценка импульса — Δa · v / (2a) для опорного борта 100 кг."
            ),
            dv_m_s=dv,
            direction="prograde",
        )
    else:
        dv = 0.0
        task = HorizonTask(
            "wait",
            "SC",
            0.0,
            False,
            (
                f"0 с: перигей не поднимать. За {horizon_sec / 3600.0:.0f} ч на высоте "
                f"{altitude_m / 1000.0:.0f} км потеря большой полуоси {loss_m:.1f} м, порог {_DECAY_LOSS_M:.0f} м."
            ),
        )
    return HorizonPlan("decay", (task,), loss_m)


def _decay_orbit(altitude_m: float) -> Orbit:
    moment = Time("2024-01-01T00:00:00Z")
    semimajor = R_EARTH_EQUATOR + altitude_m
    return Orbit.from_keplerian(semimajor, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, moment)


def execute_decay_plan(*, altitude_m: float = 400_000.0, horizon_sec: float = 3.0 * 3600.0) -> HorizonPlan:
    """Если план велит жечь Δv, сравнивает большую полуось с прогоном без импульса."""
    planned = build_decay_plan(altitude_m=altitude_m, horizon_sec=horizon_sec)
    orbit = _decay_orbit(altitude_m)
    task = planned.tasks[0]
    if not task.required:
        return planned
    waited = _coast_sma_m(orbit.r, orbit.v, horizon_sec, drag=True)
    burned_v = orbit.v + orbit.v / np.linalg.norm(orbit.v) * task.dv_m_s
    raised = _coast_sma_m(orbit.r, burned_v, horizon_sec, drag=True)
    gain = raised - waited
    measure = HorizonTask(
        "measure",
        "SC",
        horizon_sec,
        True,
        (
            f"{horizon_sec / 3600.0:.0f} ч: с импульсом большая полуось на {gain:.0f} м выше, "
            "чем у того же прогона без манёвра."
        ),
        dv_m_s=task.dv_m_s,
        direction="prograde",
    )
    return HorizonPlan("decay", planned.tasks + (measure,), gain, waited)


def _open_neighbors(fleet: Fleet, name: str, closed: set[frozenset[str]]) -> list[str]:
    found = []
    for pair in fleet.pairs():
        if name not in (pair.leader, pair.follower):
            continue
        other = pair.follower if pair.leader == name else pair.leader
        if frozenset((name, other)) in closed or pair.earth_blocked or pair.isl_margin_db < 0.0:
            continue
        found.append(other)
    return found


def _path(fleet: Fleet, source: str, goal: str, closed: set[frozenset[str]]) -> list[str] | None:
    if source == goal:
        return [source]
    seen = {source}
    queue = [[source]]
    while queue:
        path = queue.pop(0)
        for other in _open_neighbors(fleet, path[-1], closed):
            if other in seen:
                continue
            nxt = [*path, other]
            if other == goal:
                return nxt
            seen.add(other)
            queue.append(nxt)
    return None


def build_reroute_schedule(*, close_direct: bool = True, use_detour: bool = True) -> HorizonPlan:
    """
    Три аппарата: B в 1.5° по аномалии от A, C в 1.5° по узлу от A.
    Пакет на B, цель A. При закрытом прямом луче путь идёт через C, если relay переносит биты.
    """
    moment = Time("2024-01-01T00:00:00Z")
    scenario = ScenarioConfig(ground_stations=[])
    fleet = Fleet(
        {
            "A": _make_vehicle("A", 6_800_000.0, 51.6, 0.0, 0.0, moment, scenario),
            "B": _make_vehicle("B", 6_800_000.0, 51.6, 0.0, 1.5, moment, scenario),
            "C": _make_vehicle("C", 6_800_000.0, 51.6, 1.5, 0.0, moment, scenario),
        }
    )
    closed = {frozenset(("A", "B"))} if close_direct else set()
    fleet.collect("B", _PACKET_MBIT)
    if close_direct and not use_detour:
        return HorizonPlan(
            "reroute",
            (
                HorizonTask(
                    "blocked",
                    "B",
                    0.0,
                    False,
                    "0 с: луч B–A закрыт. Прямой скачок не вызывается, биты остаются на B.",
                ),
            ),
            0.0,
        )
    route = _path(fleet, "B", "A", closed)
    tasks: list[HorizonTask] = []
    if close_direct:
        tasks.append(
            HorizonTask(
                "blocked",
                "B",
                0.0,
                False,
                "0 с: луч B–A закрыт по условию задачи. Прямой скачок в план не ставится.",
            )
        )
    if route is None:
        tasks.append(
            HorizonTask(
                "wait",
                "B",
                0.0,
                False,
                "0 с: открытого пути от B до A нет. Пакет остаётся на B.",
            )
        )
        return HorizonPlan("reroute", tuple(tasks), 0.0)
    holder = "B"
    elapsed = 0.0
    for destination in route[1:]:
        start = elapsed
        fleet.step(60.0)
        elapsed += 60.0
        moved = fleet.relay(holder, destination, 60.0)
        done = moved > 0.0
        if done:
            tasks.append(
                HorizonTask(
                    "relay",
                    holder,
                    start,
                    True,
                    (
                        f"{start:.0f}–{elapsed:.0f} с: {holder} пересылает на {destination}. "
                        f"Луч открыт, перешло {moved:.1f} Мбит."
                    ),
                )
            )
            holder = destination
            continue
        tasks.append(
            HorizonTask(
                "relay",
                holder,
                start,
                False,
                f"{start:.0f}–{elapsed:.0f} с: {holder} не пересылает на {destination}. Линия биты не перенесла.",
            )
        )
        break
    arrived = fleet.buffer_mb["A"] >= _PACKET_MBIT - 1e-6
    return HorizonPlan("reroute", tuple(tasks), 1.0 if arrived else 0.0)
