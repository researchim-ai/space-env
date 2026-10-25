"""
Сутки смешанного флота: оптика, 6U и фрагмент на одних часах.

Заявки и сеансы ищутся Кауэллом на весь горизонт, по одной цели на виток.
Сближения — тот же Кауэлл. Популяция задана Δa, Δi и Δν, час встречи находит дуга.
Импульс по скорости ставится в начале дуги, боковой — за четверть периода до
минимальной дистанции. Сеть зеркал разбирает пересекающиеся окна до сборки дел.
Импульс фрагменту не ставится, пока его Pc ниже 10⁻⁴.
Баллистический коэффициент соседей совпадает с первым аппаратом: промах не
должен вырасти из разницы миделя.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from space_env.catalog.build import mean_orbit, spacecraft_config
from space_env.catalog.geometry import mean_semimajor_axis_m
from space_env.catalog.references import reference
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.probability import bplane_probability
from space_env.core.propagate import propagate_state
from space_env.core.relative import eci_to_rtn, inertial_delta_v_m_s
from space_env.core.space_weather import indices_for
from space_env.mission.horizon import HorizonTask, PC_THRESHOLD, schedule_conjunctions
from space_env.mission.workplan import (
    Opportunity,
    WorkItem,
    WorkPlan,
    _stations_for,
    collect_opportunities,
    schedule_acquisition,
)

DAY_SEC = 86_400.0
MISSION_KEYS = ("sentinel-2a", "landsat-9", "sar-small", "cubesat-6u")
_BURN_M_S = 1.0
_HARD_BODY_M = 15.0
_SIGMA_M = 200.0
# Имя, Δa в метрах, Δi и Δν в градусах. Часа встречи в таблице нет.
# У HOT и WARM Δν — начальный отставание по аномалии при более низкой орбите.
_POPULATION = (
    ("HOT", -40.0, 0.0, -0.032455),
    ("WARM", -400.0, 0.0, -0.39667),
    ("debris-fragment", 0.0, 0.2, 2.0),
    ("plane-offset", 0.0, 0.05, 20.0),
)
# Имя в очереди и ключ карточки. У каждого своя орбита, не промах относительно первого аппарата.
_CATALOG_SCREEN = (
    ("sentinel-2b", "sentinel-2b"),
    ("landsat-9", "landsat-9"),
    ("cubesat-6u", "cubesat-6u"),
    ("debris-class", "debris-fragment"),
)


@dataclass(frozen=True)
class DayCampaign:
    """Горизонт миссии. Планы съёмки — по аппаратам, очередь импульсов — одна."""

    epoch: str
    horizon_sec: float
    dv_budget_m_s: float
    plans: tuple[WorkPlan, ...]
    conjunctions: tuple[HorizonTask, ...]
    burn_at_sec: float | None = None
    coast_pc: tuple[tuple[str, float], ...] = ()
    flown_pc: tuple[tuple[str, float], ...] = ()
    population: tuple[tuple[str, float, float, float], ...] = ()
    nu_deg: float = 180.0
    sample_sec: float = 60.0

    @property
    def vehicles(self) -> tuple[str, ...]:
        return tuple(plan.vehicle for plan in self.plans)

    @property
    def dv_spent_m_s(self) -> float:
        return float(sum(item.dv_m_s for item in self.conjunctions if item.required))

    def lines(self) -> list[str]:
        hours = self.horizon_sec / 3600.0
        text = [
            (
                f"Миссия {hours:.0f} ч с {self.epoch}. Аппараты: {', '.join(self.vehicles)}. "
                f"Бюджет Δv {self.dv_budget_m_s:.1f} м/с, в очередь ушло {self.dv_spent_m_s:.1f} м/с. "
                + (
                    f"План {self.vehicles[0]} построен с этим импульсом."
                    if self.burn_at_sec is not None and self.vehicles
                    else "Импульса нет, план идёт по дуге без манёвра."
                )
            )
        ]
        for plan in self.plans:
            required = plan.required_items()
            shots = sum(1 for item in required if item.kind == "imaging")
            dumps = sum(1 for item in required if item.kind == "comms")
            text.append(
                f"{plan.vehicle}: заявок {len(plan.targets)}, съёмок {shots}, сеансов {dumps}."
            )
            text.extend(plan.lines())
        text.extend(item.reason for item in self.conjunctions)
        if self.burn_at_sec is not None and self.flown_pc:
            coast = dict(self.coast_pc)
            flown = dict(self.flown_pc)
            bits = [
                (
                    f"Импульс исполнен в {self.burn_at_sec:.0f} с. "
                    f"Бюджет {self.dv_spent_m_s:.1f} м/с."
                )
            ]
            for name, _probability in self.flown_pc:
                bits.append(f"{name}: Pc без импульса {coast.get(name, 0.0):.3e}, после {flown[name]:.3e}.")
            text.extend(bits)
        return text


def orbital_period_s(key: str) -> float:
    """Период круговой орбиты карточки, с."""
    altitude_m = reference(key).altitude_m
    semimajor_m = mean_semimajor_axis_m(altitude_m)
    return 2.0 * math.pi * math.sqrt(semimajor_m**3 / MU_EARTH)


@dataclass(frozen=True)
class EncounterPeak:
    """Pc в момент минимальной дистанции за сутки."""

    name: str
    t_sec: float
    probability: float
    radial_m: float
    cross_m: float


def _force(prop, f107: float, f107a: float, ap: float) -> dict:
    return {
        "mass_kg": prop.mass_kg,
        "drag_area_m2": prop.drag_area_m2,
        "drag_coefficient": prop.drag_coefficient,
        "include_drag": True,
        "include_j2": True,
        "include_third_body": True,
        "atmosphere": "table",
        "f107": f107,
        "f107a": f107a,
        "ap": ap,
        "srp_area_m2": prop.srp_area_m2,
        "srp_coefficient": prop.srp_coefficient,
        "substep_sec": 10.0,
    }


def _screen_covariance() -> np.ndarray:
    along = 8.0 * _SIGMA_M
    return np.diag([_SIGMA_M**2, along**2, _SIGMA_M**2, 0.0, 0.0, 0.0])


def _grid(
    epoch,
    position: np.ndarray,
    velocity: np.ndarray,
    horizon_sec: float,
    sample_sec: float,
    force: dict,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    samples: list[tuple[float, np.ndarray, np.ndarray]] = [(0.0, position.copy(), velocity.copy())]
    elapsed = 0.0
    while elapsed < horizon_sec - 1e-9:
        step = min(sample_sec, horizon_sec - elapsed)
        when = (epoch + elapsed).datetime.replace(tzinfo=None)
        position, velocity, _trace = propagate_state(position, velocity, step, when=when, **force)
        elapsed += step
        samples.append((elapsed, position.copy(), velocity.copy()))
    return samples


def _closest(
    primary: list[tuple[float, np.ndarray, np.ndarray]],
    track: list[tuple[float, np.ndarray, np.ndarray]],
    covariance: np.ndarray,
) -> tuple[float, float, float, float]:
    """Момент минимальной дистанции, Pc и промах в RTN."""
    closest = 1e18
    close_t = 0.0
    close_pc = 0.0
    close_radial = 0.0
    close_cross = 0.0
    for (t_sec, r_primary, v_primary), (_t_other, r_other, v_other) in zip(primary, track):
        distance = float(np.linalg.norm(r_other - r_primary))
        if distance < closest:
            probability = 0.0
            relative = eci_to_rtn(r_other - r_primary, r_primary, v_primary)
            if distance < 100_000.0:
                relative_v = eci_to_rtn(v_other - v_primary, r_primary, v_primary)
                probability = bplane_probability(relative, relative_v, covariance, _HARD_BODY_M)
            closest = distance
            close_t = t_sec
            close_pc = probability
            close_radial = float(relative[0])
            close_cross = float(relative[2])
    return close_t, close_pc, close_radial, close_cross


def _track_from_keplerian(
    a_m: float,
    ecc: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: float,
    epoch,
    horizon_sec: float,
    sample_sec: float,
    force: dict,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Дуга из собственных элементов. Дальше Кауэлл, без повторной вставки промаха."""
    orb = Orbit.from_keplerian(a_m, ecc, inc_rad, raan_rad, argp_rad, nu_rad, epoch)
    return _grid(epoch, orb.r.copy(), orb.v.copy(), horizon_sec, sample_sec, force)


def _with_burn(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    burn_at_sec: float,
    dv_m_s: float,
    direction: str,
    epoch,
    force: dict,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Копия дуги своего аппарата: до импульса она совпадает с прогоном без манёвра."""
    index = min(range(len(samples)), key=lambda item: abs(samples[item][0] - burn_at_sec))
    flown: list[tuple[float, np.ndarray, np.ndarray]] = [
        (t_sec, position.copy(), velocity.copy()) for t_sec, position, velocity in samples[:index]
    ]
    t_sec, position, velocity = samples[index]
    position = position.copy()
    velocity = velocity.copy() + inertial_delta_v_m_s(position, velocity, dv_m_s, direction)
    flown.append((t_sec, position, velocity))
    previous = t_sec
    for step_t, _position, _velocity in samples[index + 1 :]:
        when = (epoch + previous).datetime.replace(tzinfo=None)
        position, velocity, _trace = propagate_state(position, velocity, step_t - previous, when=when, **force)
        previous = step_t
        flown.append((step_t, position.copy(), velocity.copy()))
    return flown


@lru_cache(maxsize=4)
def _mission_arc(
    primary_key: str,
    horizon_sec: float,
    epoch: str,
    nu_deg: float,
    sample_sec: float,
) -> tuple[
    tuple[EncounterPeak, ...],
    tuple[tuple[float, np.ndarray, np.ndarray], ...],
    tuple[tuple[str, tuple[tuple[float, np.ndarray, np.ndarray], ...]], ...],
    tuple[tuple[str, float, float, float], ...],
]:
    """Дуга своего аппарата и соседей без импульса. Сетка sample_sec, шаг интегратора 10 с."""
    orbit = mean_orbit(primary_key, epoch, nu_deg=nu_deg)
    own = orbit.keplerian()
    prop = spacecraft_config(primary_key).propulsion
    weather = indices_for(orbit.epoch)
    f107 = 150.0 if weather is None else weather.f107
    f107a = 150.0 if weather is None else weather.f107a
    ap = 4.0 if weather is None else weather.ap
    force = _force(prop, f107, f107a, ap)
    samples = _grid(orbit.epoch, orbit.r.copy(), orbit.v.copy(), horizon_sec, sample_sec, force)
    covariance = _screen_covariance()
    peaks: list[EncounterPeak] = []
    tracks: list[tuple[str, tuple[tuple[float, np.ndarray, np.ndarray], ...]]] = []
    population: list[tuple[str, float, float, float]] = []
    for name, delta_a_m, delta_inc_deg, delta_nu_deg in _POPULATION:
        track = _track_from_keplerian(
            own.a + delta_a_m,
            own.ecc,
            own.inc + float(np.deg2rad(delta_inc_deg)),
            own.raan,
            own.argp,
            own.nu + float(np.deg2rad(delta_nu_deg)),
            orbit.epoch,
            horizon_sec,
            sample_sec,
            force,
        )
        close_t, close_pc, close_radial, close_cross = _closest(samples, track, covariance)
        peaks.append(EncounterPeak(name, close_t, float(close_pc), close_radial, close_cross))
        tracks.append((name, tuple(track)))
        population.append((name, delta_a_m, delta_inc_deg, delta_nu_deg))
    for label, key in _CATALOG_SCREEN:
        partner_nu = (nu_deg + 180.0) % 360.0 if key == "sentinel-2b" else 180.0
        if key == "debris-fragment":
            partner_nu = 0.0
        other = mean_orbit(key, epoch, nu_deg=partner_nu)
        other_prop = spacecraft_config(key).propulsion
        other_weather = indices_for(other.epoch)
        other_force = _force(
            other_prop,
            150.0 if other_weather is None else other_weather.f107,
            150.0 if other_weather is None else other_weather.f107a,
            4.0 if other_weather is None else other_weather.ap,
        )
        other_samples = _grid(other.epoch, other.r.copy(), other.v.copy(), horizon_sec, sample_sec, other_force)
        close_t, close_pc, close_radial, close_cross = _closest(samples, other_samples, covariance)
        peaks.append(EncounterPeak(label, close_t, float(close_pc), close_radial, close_cross))
        tracks.append((label, tuple(other_samples)))
    peaks.sort(key=lambda item: -item.probability)
    return tuple(peaks), tuple(samples), tuple(tracks), tuple(population)


@dataclass(frozen=True)
class _Draft:
    """План и окна, из которых он собран. Окна нужны, чтобы пересобрать дела без отданного зеркала."""

    plan: WorkPlan
    opportunities: tuple[Opportunity, ...]
    imaging_cap_s: float
    downlink_bps: float


def _ready_bits(plan: WorkPlan, at_sec: float) -> float:
    bits = 0.0
    for item in plan.items:
        if not item.required or item.end_sec > at_sec + 1e-9:
            continue
        if item.kind == "imaging":
            bits += item.bits
        elif item.kind == "comms":
            bits -= item.bits
    return max(0.0, bits)


def _share_stations(plans: tuple[WorkPlan, ...]) -> tuple[WorkPlan, ...]:
    """Одно зеркало. Из двух обязательных сеансов на одной станции остаётся аппарат с большим буфером."""
    chosen: list[tuple[int, int, str, float, float]] = []
    for plan_index, plan in enumerate(plans):
        for item_index, item in enumerate(plan.items):
            if item.required and item.kind == "comms":
                chosen.append((plan_index, item_index, item.name, item.start_sec, item.end_sec))
    lost: dict[tuple[int, int], str] = {}
    ordered = sorted(chosen, key=lambda row: row[3])
    for index, left in enumerate(ordered):
        if (left[0], left[1]) in lost:
            continue
        for right in ordered[index + 1 :]:
            if (right[0], right[1]) in lost or left[0] == right[0] or left[2] != right[2]:
                continue
            if left[4] <= right[3] or right[4] <= left[3]:
                continue
            left_bits = _ready_bits(plans[left[0]], left[3])
            right_bits = _ready_bits(plans[right[0]], right[3])
            drop = right if left_bits > right_bits or (left_bits == right_bits and left[3] <= right[3]) else left
            keep = left if drop is right else right
            lost[(drop[0], drop[1])] = plans[keep[0]].vehicle
            if drop is left:
                break
    if not lost:
        return plans
    rebuilt: list[WorkPlan] = []
    for plan_index, plan in enumerate(plans):
        items = []
        for item_index, item in enumerate(plan.items):
            holder = lost.get((plan_index, item_index))
            if holder is None:
                items.append(item)
                continue
            items.append(
                replace(
                    item,
                    required=False,
                    reason=(
                        f"{item.start_sec:.0f}–{item.end_sec:.0f} с: не сдавать на {item.name}. "
                        f"Зеркало занято аппаратом {holder}."
                    ),
                )
            )
        rebuilt.append(WorkPlan(plan.vehicle, plan.horizon_sec, tuple(items), plan.memory_bits, plan.targets))
    return tuple(rebuilt)


def _same_pass(opportunity: Opportunity, other: Opportunity) -> bool:
    return (
        opportunity.name == other.name
        and abs(opportunity.start_sec - other.start_sec) < 1.0
        and abs(opportunity.end_sec - other.end_sec) < 1.0
    )


def _assign_dishes(drafts: tuple[_Draft, ...]) -> tuple[WorkPlan, ...]:
    """
    Сеть зеркал. Пересекающиеся окна одной станции отдаются аппарату,
    у которого к началу окна в отдельно собранном плане больше данных.
    При равном буфере окно остаётся у того, кто начинает раньше.
    Проигравшее окно в план не входит, список дел собирается заново.
    """
    windows: list[tuple[int, Opportunity]] = []
    for index, draft in enumerate(drafts):
        for opportunity in draft.opportunities:
            if opportunity.kind == "comms":
                windows.append((index, opportunity))
    windows.sort(
        key=lambda row: (
            -_ready_bits(drafts[row[0]].plan, row[1].start_sec),
            row[1].start_sec,
            row[0],
        )
    )
    granted: list[tuple[str, float, float, int]] = []
    denied: list[list[tuple[Opportunity, str]]] = [[] for _ in drafts]
    for index, opportunity in windows:
        holder = next(
            (
                span
                for span in granted
                if span[3] != index
                and span[0] == opportunity.name
                and span[1] < opportunity.end_sec
                and opportunity.start_sec < span[2]
            ),
            None,
        )
        if holder is None:
            granted.append((opportunity.name, opportunity.start_sec, opportunity.end_sec, index))
            continue
        denied[index].append((opportunity, drafts[holder[3]].plan.vehicle))
    assigned: list[WorkPlan] = []
    for index, draft in enumerate(drafts):
        blocked = denied[index]
        if not blocked:
            assigned.append(draft.plan)
            continue
        remaining = [
            item
            for item in draft.opportunities
            if not any(_same_pass(item, opportunity) for opportunity, _holder in blocked)
        ]
        items = schedule_acquisition(
            remaining,
            memory_bits=draft.plan.memory_bits,
            imaging_cap_s=draft.imaging_cap_s,
            downlink_bps=draft.downlink_bps,
        )
        for opportunity, holder in blocked:
            items.append(
                WorkItem(
                    kind="comms",
                    name=opportunity.name,
                    start_sec=opportunity.start_sec,
                    end_sec=opportunity.end_sec,
                    required=False,
                    reason=(
                        f"{opportunity.start_sec:.0f}–{opportunity.end_sec:.0f} с: не сдавать на {opportunity.name}. "
                        f"Зеркало занято аппаратом {holder}."
                    ),
                )
            )
        items.sort(key=lambda item: (item.start_sec, item.required))
        assigned.append(
            WorkPlan(
                draft.plan.vehicle,
                draft.plan.horizon_sec,
                tuple(items),
                draft.plan.memory_bits,
                draft.plan.targets,
            )
        )
    return tuple(assigned)


@dataclass(frozen=True)
class SharedWindow:
    """Два аппарата над одной станцией в пересекающихся окнах."""

    station: str
    vehicles: tuple[str, str]
    starts: tuple[float, float]
    ends: tuple[float, float]
    ready_bits: tuple[float, float]

    def keeper_index(self) -> int:
        """Зеркало у большего буфера. При равенстве — у того, кто начинает раньше."""
        return min(range(2), key=lambda index: (-self.ready_bits[index], self.starts[index], index))


def campaign_drafts(campaign: DayCampaign) -> tuple[_Draft, ...]:
    """Окна суток до сети зеркал. Ключ совпадает с тем, по которому собран план."""
    burn = next((item for item in campaign.conjunctions if item.required), None)
    return _day_plans(
        campaign.vehicles,
        campaign.horizon_sec,
        campaign.epoch,
        campaign.nu_deg,
        campaign.sample_sec,
        campaign.vehicles[0] if campaign.vehicles else "",
        -1.0 if campaign.burn_at_sec is None else float(campaign.burn_at_sec),
        0.0 if burn is None else float(burn.dv_m_s),
        "prograde" if burn is None else (burn.direction or "prograde"),
    )


def shared_windows(drafts: tuple[_Draft, ...]) -> tuple[SharedWindow, ...]:
    """Физические пересечения сеансов разных аппаратов на одном имени станции."""
    rows: list[tuple[int, Opportunity]] = []
    for index, draft in enumerate(drafts):
        for opportunity in draft.opportunities:
            if opportunity.kind == "comms":
                rows.append((index, opportunity))
    found: list[SharedWindow] = []
    for left in range(len(rows)):
        left_index, left_pass = rows[left]
        for right in range(left + 1, len(rows)):
            right_index, right_pass = rows[right]
            if left_index == right_index or left_pass.name != right_pass.name:
                continue
            if left_pass.end_sec <= right_pass.start_sec or right_pass.end_sec <= left_pass.start_sec:
                continue
            ordered = tuple(sorted(((left_index, left_pass), (right_index, right_pass)), key=lambda item: item[0]))
            found.append(
                SharedWindow(
                    station=left_pass.name,
                    vehicles=(drafts[ordered[0][0]].plan.vehicle, drafts[ordered[1][0]].plan.vehicle),
                    starts=(ordered[0][1].start_sec, ordered[1][1].start_sec),
                    ends=(ordered[0][1].end_sec, ordered[1][1].end_sec),
                    ready_bits=(
                        _ready_bits(drafts[ordered[0][0]].plan, ordered[0][1].start_sec),
                        _ready_bits(drafts[ordered[1][0]].plan, ordered[1][1].start_sec),
                    ),
                )
            )
    found.sort(key=lambda item: (min(item.starts), item.station, item.vehicles))
    return tuple(found)


def award_window(
    drafts: tuple[_Draft, ...],
    window: SharedWindow,
    *,
    holder: str | None = None,
) -> tuple[WorkPlan, ...]:
    """
    Отдаёт одно окно выбранному аппарату.
    Чужие сеансы той же станции, которые его пересекают, из плана выпадают.
    holder=None оставляет зеркало у большего буфера.
    """
    if holder is None:
        slot = window.keeper_index()
        holder = window.vehicles[slot]
    elif holder not in window.vehicles:
        raise ValueError("зеркало можно отдать только аппарату из этого окна")
    else:
        slot = window.vehicles.index(holder)
    keep_start = window.starts[slot]
    keep_end = window.ends[slot]
    assigned: list[WorkPlan] = []
    for draft in drafts:
        if draft.plan.vehicle == holder or draft.plan.vehicle not in window.vehicles:
            assigned.append(draft.plan)
            continue
        blocked = [
            opportunity
            for opportunity in draft.opportunities
            if opportunity.kind == "comms"
            and opportunity.name == window.station
            and opportunity.start_sec < keep_end
            and keep_start < opportunity.end_sec
        ]
        if not blocked:
            assigned.append(draft.plan)
            continue
        remaining = [
            item
            for item in draft.opportunities
            if not any(_same_pass(item, opportunity) for opportunity in blocked)
        ]
        items = schedule_acquisition(
            remaining,
            memory_bits=draft.plan.memory_bits,
            imaging_cap_s=draft.imaging_cap_s,
            downlink_bps=draft.downlink_bps,
        )
        for opportunity in blocked:
            items.append(
                WorkItem(
                    kind="comms",
                    name=opportunity.name,
                    start_sec=opportunity.start_sec,
                    end_sec=opportunity.end_sec,
                    required=False,
                    reason=(
                        f"{opportunity.start_sec:.0f}–{opportunity.end_sec:.0f} с: не сдавать на {opportunity.name}. "
                        f"Зеркало занято аппаратом {holder}."
                    ),
                )
            )
        items.sort(key=lambda item: (item.start_sec, item.required))
        assigned.append(
            WorkPlan(
                draft.plan.vehicle,
                draft.plan.horizon_sec,
                tuple(items),
                draft.plan.memory_bits,
                draft.plan.targets,
            )
        )
    return tuple(assigned)


def commit_award(
    drafts: tuple[_Draft, ...],
    window: SharedWindow,
    *,
    holder: str | None = None,
) -> tuple[_Draft, ...]:
    """Закрепляет окно за аппаратом и убирает его из чужих сеансов. holder=None — больший буфер."""
    if holder is None:
        holder = window.vehicles[window.keeper_index()]
    elif holder not in window.vehicles:
        raise ValueError("зеркало можно отдать только аппарату из этого окна")
    plans = award_window(drafts, window, holder=holder)
    slot = window.vehicles.index(holder)
    keep_start = window.starts[slot]
    keep_end = window.ends[slot]
    updated: list[_Draft] = []
    for draft, plan in zip(drafts, plans):
        if draft.plan.vehicle == holder or draft.plan.vehicle not in window.vehicles:
            updated.append(_Draft(plan, draft.opportunities, draft.imaging_cap_s, draft.downlink_bps))
            continue
        remaining = tuple(
            opportunity
            for opportunity in draft.opportunities
            if not (
                opportunity.kind == "comms"
                and opportunity.name == window.station
                and opportunity.start_sec < keep_end
                and keep_start < opportunity.end_sec
            )
        )
        updated.append(_Draft(plan, remaining, draft.imaging_cap_s, draft.downlink_bps))
    return tuple(updated)


def award_day(drafts: tuple[_Draft, ...]) -> tuple[WorkPlan, ...]:
    """Все пересечения суток по порядку. Каждое окно отдаётся большему буферу."""
    current = drafts
    for _ in range(64):
        windows = shared_windows(current)
        if not windows:
            return tuple(draft.plan for draft in current)
        current = commit_award(current, windows[0])
    raise RuntimeError("сеть зеркал не сошлась")


@lru_cache(maxsize=4)
def _day_plans(
    keys: tuple[str, ...],
    horizon_sec: float,
    epoch: str,
    nu_deg: float,
    sample_sec: float,
    burn_key: str,
    burn_at_sec: float,
    burn_dv_m_s: float,
    burn_direction: str,
) -> tuple[_Draft, ...]:
    drafts: list[_Draft] = []
    for key in keys:
        spec = reference(key)
        period = orbital_period_s(key)
        cap = float(spec.max_imaging_s) * (horizon_sec / period) if spec.max_imaging_s is not None else 1.0e12
        use_burn = key == burn_key and burn_at_sec >= 0.0 and burn_dv_m_s > 0.0
        opportunities, target_rows = collect_opportunities(
            key,
            horizon_sec=horizon_sec,
            epoch=epoch,
            nu_deg=nu_deg,
            sample_sec=sample_sec,
            order_every_sec=period,
            burn_at_sec=burn_at_sec if use_burn else None,
            burn_dv_m_s=burn_dv_m_s if use_burn else 0.0,
            burn_direction=burn_direction,
        )
        items = schedule_acquisition(
            opportunities,
            memory_bits=float(spec.memory_bits),
            imaging_cap_s=cap,
            downlink_bps=float(spec.downlink_bps),
        )
        plan = WorkPlan(key, horizon_sec, tuple(items), float(spec.memory_bits), target_rows)
        stations = _stations_for(key)
        if stations and not any(item.kind == "comms" for item in plan.items):
            names = ", ".join(station.name for station in stations)
            note = WorkItem(
                kind="comms",
                name=stations[0].name,
                start_sec=0.0,
                end_sec=horizon_sec,
                required=False,
                reason=(
                    f"0–{horizon_sec:.0f} с: сеанса на {names} нет. "
                    f"За {horizon_sec / 3600.0:.0f} ч угол места и запас линии вместе не держались."
                ),
            )
            plan = WorkPlan(plan.vehicle, plan.horizon_sec, (*plan.items, note), plan.memory_bits, plan.targets)
        drafts.append(_Draft(plan, tuple(opportunities), cap, float(spec.downlink_bps)))
    return tuple(drafts)


def build_day_campaign(
    keys: tuple[str, ...] = MISSION_KEYS,
    *,
    horizon_sec: float = DAY_SEC,
    epoch: str = "2026-10-20T00:00:00Z",
    nu_deg: float = 180.0,
    sample_sec: float = 60.0,
    dv_budget_m_s: float = 1.0,
    burn_m_s: float = _BURN_M_S,
) -> DayCampaign:
    """
    Горизонт по карточкам. По умолчанию сутки Sentinel-2A, Landsat 9 и 6U.
    Скрининг — на первом аппарате по всей дуге. Импульс по скорости ставится
    в начале дуги, боковой — за четверть периода до сближения. Порог Pc — 10⁻⁴.
    Пересекающиеся окна одной станции разбирает сеть зеркал до сборки дел.
    """
    if horizon_sec <= 0.0:
        raise ValueError("горизонт миссии должен быть положительным")
    if not keys:
        raise ValueError("в миссии нужен хотя бы один аппарат")
    peaks, samples, tracks, population = _mission_arc(keys[0], horizon_sec, epoch, nu_deg, sample_sec)
    rows = [(item.name, item.probability, item.radial_m, item.cross_m) for item in peaks]
    times = [item.t_sec for item in peaks]
    conjunctions = tuple(
        schedule_conjunctions(
            rows,
            budget_m_s=dv_budget_m_s,
            burn_m_s=burn_m_s,
            threshold=PC_THRESHOLD,
            when_sec=times,
        )
    )
    coast_pc = tuple((item.name, item.probability) for item in peaks)
    required = [item for item in conjunctions if item.required]
    direction = "normal"
    burn_at: float | None = None
    flown_pc = coast_pc
    if required:
        orbit = mean_orbit(keys[0], epoch, nu_deg=nu_deg)
        weather = indices_for(orbit.epoch)
        prop = spacecraft_config(keys[0]).propulsion
        force = _force(
            prop,
            150.0 if weather is None else weather.f107,
            150.0 if weather is None else weather.f107a,
            4.0 if weather is None else weather.ap,
        )
        quarter = orbital_period_s(keys[0]) / 4.0
        flown_samples = list(samples)
        for index, item in enumerate(sorted(required, key=lambda task: task.start_sec)):
            # Боковой промах уходит за четверть периода. Проградный меняет энергию
            # и ставится в начале дуги: к четверти периода радиальный зазор уже мал.
            if (item.direction or "prograde") == "normal":
                moment = max(0.0, round((item.start_sec - quarter) / sample_sec) * sample_sec)
            else:
                moment = 0.0
            if index == 0:
                burn_at = moment
                direction = item.direction or "prograde"
            flown_samples = _with_burn(
                flown_samples,
                moment,
                item.dv_m_s,
                item.direction or "prograde",
                orbit.epoch,
                force,
            )
        covariance = _screen_covariance()
        flown_pc = tuple(
            (name, _closest(flown_samples, list(track), covariance)[1]) for name, track in dict(tracks).items()
        )
    plans = award_day(
        _day_plans(
            tuple(keys),
            horizon_sec,
            epoch,
            nu_deg,
            sample_sec,
            keys[0],
            -1.0 if burn_at is None else burn_at,
            0.0 if burn_at is None else burn_m_s,
            direction,
        )
    )
    return DayCampaign(
        epoch,
        horizon_sec,
        dv_budget_m_s,
        plans,
        conjunctions,
        burn_at,
        coast_pc,
        flown_pc,
        population,
        nu_deg=nu_deg,
        sample_sec=sample_sec,
    )
