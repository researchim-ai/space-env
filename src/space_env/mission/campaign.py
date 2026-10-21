"""
Сутки смешанного флота: оптика, 6U и фрагмент на одних часах.

Заявки и сеансы ищутся Кауэллом на весь горизонт, по одной цели на виток.
Сближения — тот же Кауэлл на сутках. Объект ставится на заданный промах в свой
момент наибольшего сближения, состояние уносится назад тем же RK4 и снова
интегрируется вперёд. В очередь берётся момент минимальной дистанции и Pc в этот момент. Один бюджет Δv
режет очередь. Назначенный импульс исполняется за четверть периода до этого момента
и пара считается ещё раз. Импульс фрагменту не ставится, пока его Pc ниже 10⁻⁴.
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
from space_env.core.probability import bplane_probability
from space_env.core.propagate import _rk4, propagate_state
from space_env.core.relative import eci_to_rtn, inertial_delta_v_m_s, rtn_basis
from space_env.core.space_weather import indices_for
from space_env.mission.horizon import HorizonTask, PC_THRESHOLD, schedule_conjunctions
from space_env.mission.workplan import WorkItem, WorkPlan, _stations_for, build_acquisition_plan

DAY_SEC = 86_400.0
MISSION_KEYS = ("sentinel-2a", "landsat-9", "cubesat-6u")
_BURN_M_S = 1.0
_APPROACH_M_S = 5.0
_HARD_BODY_M = 15.0
_SIGMA_M = 200.0
# Имя, радиальный промах, боковой промах, секунда наибольшего сближения.
_ENCOUNTERS = (
    ("HOT", 0.0, 40.0, 3_600.0),
    ("WARM", 0.0, 250.0, 21_600.0),
    ("debris-fragment", 0.0, 4_000.0, 43_200.0),
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
                    f"Импульс исполнен в {self.burn_at_sec:.0f} с, за четверть периода до сближения. "
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


def _rewind(position: np.ndarray, velocity: np.ndarray, dt_sec: float, moment, force: dict) -> tuple[np.ndarray, np.ndarray]:
    """Тот же RK4 с отрицательным шагом. Вперёд из этой точки возвращает промах на момент сближения."""
    elapsed = 0.0
    step = 10.0
    while elapsed < dt_sec - 1e-9:
        h = min(step, dt_sec - elapsed)
        position, velocity = _rk4(
            position,
            velocity,
            -h,
            mass_kg=force["mass_kg"],
            drag_area_m2=force["drag_area_m2"],
            drag_coefficient=force["drag_coefficient"],
            include_j2=True,
            include_drag=True,
            atmosphere="table",
            f107=force["f107"],
            f107a=force["f107a"],
            ap=force["ap"],
            density_scale=1.0,
            lat_deg=0.0,
            lon_deg=0.0,
            when=moment.datetime.replace(tzinfo=None),
            srp_area_m2=force["srp_area_m2"],
            srp_coefficient=force["srp_coefficient"],
            include_third_body=True,
        )
        elapsed += h
        moment = moment - h
    return position, velocity


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


def _secondary_track(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    radial_m: float,
    cross_m: float,
    t_ca: float,
    epoch,
    force: dict,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    index = min(range(len(samples)), key=lambda item: abs(samples[item][0] - t_ca))
    t_hit, r_hit, v_hit = samples[index]
    basis = rtn_basis(r_hit, v_hit)
    r_other = r_hit + basis @ np.array([radial_m, 0.0, cross_m])
    v_other = v_hit + basis @ np.array([0.0, -_APPROACH_M_S, 0.0])
    if t_hit > 1e-6:
        r_other, v_other = _rewind(r_other, v_other, t_hit, epoch + t_hit, force)
    track: list[tuple[float, np.ndarray, np.ndarray]] = [(0.0, r_other.copy(), v_other.copy())]
    previous = 0.0
    for t_sec, _r_primary, _v_primary in samples:
        if t_sec <= previous + 1e-9:
            continue
        when = (epoch + previous).datetime.replace(tzinfo=None)
        r_other, v_other, _trace = propagate_state(r_other, v_other, t_sec - previous, when=when, **force)
        previous = t_sec
        track.append((t_sec, r_other.copy(), v_other.copy()))
    return track


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
) -> tuple[tuple[EncounterPeak, ...], tuple[tuple[float, np.ndarray, np.ndarray], ...], tuple[tuple[str, tuple[tuple[float, np.ndarray, np.ndarray], ...]], ...]]:
    """Дуга своего аппарата и соседей без импульса. Сетка sample_sec, шаг интегратора 10 с."""
    orbit = mean_orbit(primary_key, epoch, nu_deg=nu_deg)
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
    for name, radial_m, cross_m, t_ca in _ENCOUNTERS:
        if t_ca > horizon_sec + 1e-9:
            continue
        track = _secondary_track(samples, radial_m, cross_m, t_ca, orbit.epoch, force)
        close_t, close_pc, close_radial, close_cross = _closest(samples, track, covariance)
        peaks.append(EncounterPeak(name, close_t, float(close_pc), close_radial, close_cross))
        tracks.append((name, tuple(track)))
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
    return tuple(peaks), tuple(samples), tuple(tracks)


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
) -> tuple[WorkPlan, ...]:
    plans: list[WorkPlan] = []
    for key in keys:
        spec = reference(key)
        period = orbital_period_s(key)
        cap = None
        if spec.max_imaging_s is not None:
            cap = float(spec.max_imaging_s) * (horizon_sec / period)
        use_burn = key == burn_key and burn_at_sec >= 0.0 and burn_dv_m_s > 0.0
        plan = build_acquisition_plan(
            key,
            horizon_sec=horizon_sec,
            epoch=epoch,
            nu_deg=nu_deg,
            sample_sec=sample_sec,
            order_every_sec=period,
            imaging_cap_s=cap,
            burn_at_sec=burn_at_sec if use_burn else None,
            burn_dv_m_s=burn_dv_m_s if use_burn else 0.0,
            burn_direction=burn_direction,
        )
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
        plans.append(plan)
    return tuple(plans)


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
    Скрининг — на первом аппарате по всей дуге. Назначенный импульс прогоняется
    за четверть периода до сближения. Порог Pc — 10⁻⁴.
    """
    if horizon_sec <= 0.0:
        raise ValueError("горизонт миссии должен быть положительным")
    if not keys:
        raise ValueError("в миссии нужен хотя бы один аппарат")
    peaks, samples, tracks = _mission_arc(keys[0], horizon_sec, epoch, nu_deg, sample_sec)
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
            moment = max(0.0, round((item.start_sec - quarter) / sample_sec) * sample_sec)
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
    plans = _share_stations(
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
    )
