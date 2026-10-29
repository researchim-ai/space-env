"""
Короткие эталоны ядра. Допуск записан до сравнения и не подгоняется под ответ.

Двухтелая задача: относительный уход энергии и момента за один оборот при шаге 10 с
не больше 1e-8. Это запас на метод RK4, не измеренный остаток конкретного прогона.
J2: оскулирующий узел за 6 ч против вековой формулы Кёртиса, относительный допуск 0.25.
Короткопериодическая часть в вековой формуле отсутствует, поэтому допуск шире.
Сопротивление: с таблицей плотности радиус через час меньше, чем без неё.
Это знак эффекта, не эталон US Standard.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import numpy as np

from space_env import __version__
from space_env.core.constants import (
    AU,
    J2_EARTH,
    MU_EARTH,
    R_EARTH_EQUATOR,
    SOLAR_PRESSURE_1AU_N_M2,
)
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.time import TIME_TOLERANCE_SEC, Time
from space_env.validation.profiles import REFERENCE_PROFILE, TRAINING_PROFILE, optional_backends

_TWO_BODY_TOLERANCE = 1.0e-8
_J2_RELATIVE = 0.25


def _energy(r: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * float(np.dot(v, v)) - MU_EARTH / float(np.linalg.norm(r))


def _momentum(r: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(r, v)))


def _raan_deg(r: np.ndarray, v: np.ndarray) -> float:
    angular = np.cross(r, v)
    node = np.cross(np.array([0.0, 0.0, 1.0]), angular)
    return float(np.rad2deg(math.atan2(node[1], node[0])))


def _case(name: str, measured: float, tolerance: float, applicability: str, source: str) -> dict[str, Any]:
    return {
        "name": name,
        "measured": measured,
        "tolerance": tolerance,
        "passed": abs(measured) <= tolerance,
        "applicability": applicability,
        "source": source,
        "status": "verified" if abs(measured) <= tolerance else "experimental",
    }


def core_report() -> dict[str, Any]:
    """Считает эталоны ядра и возвращает отчёт с измеренной ошибкой."""
    a = 6_800_000.0
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(a, 0.001, math.radians(51.6), 0.2, 0.1, 0.3, epoch)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        period,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=False,
        include_drag=False,
        substep_sec=10.0,
    )
    energy_error = abs(_energy(r, v) - _energy(orbit.r, orbit.v)) / abs(_energy(orbit.r, orbit.v))
    momentum_error = abs(_momentum(r, v) - _momentum(orbit.r, orbit.v)) / _momentum(orbit.r, orbit.v)
    inclined = Orbit.from_keplerian(a, 0.0, math.radians(51.6), math.radians(40.0), 0.0, 0.0, epoch)
    hours = 6.0
    node_r, node_v, _samples = propagate_state(
        inclined.r,
        inclined.v,
        hours * 3600.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=True,
        include_drag=False,
        substep_sec=20.0,
    )
    delta = _raan_deg(node_r, node_v) - 40.0
    if delta > 180.0:
        delta -= 360.0
    if delta < -180.0:
        delta += 360.0
    mean_motion = math.sqrt(MU_EARTH / a**3)
    analytical = math.degrees(
        -1.5 * mean_motion * J2_EARTH * (R_EARTH_EQUATOR / a) ** 2 * math.cos(math.radians(51.6)) * hours * 3600.0
    )
    j2_error = abs(delta - analytical) / abs(analytical)
    coast = Orbit.from_keplerian(a, 0.0, math.radians(51.6), 0.0, 0.0, 0.0, epoch)
    common = dict(mass_kg=100.0, drag_area_m2=1.0, drag_coefficient=2.2, include_j2=True, substep_sec=30.0)
    dragged, _vd, _samples = propagate_state(coast.r, coast.v, 3600.0, include_drag=True, **common)
    clean, _vc, _samples = propagate_state(coast.r, coast.v, 3600.0, include_drag=False, **common)
    radius_drop_m = float(np.linalg.norm(clean) - np.linalg.norm(dragged))
    cases = [
        _case(
            "two_body_energy",
            energy_error,
            _TWO_BODY_TOLERANCE,
            "Один оборот, шаг 10 с, J2, сопротивление, давление и третье тело выключены.",
            "Удельная энергия v²/2 − μ/r сохраняется в задаче двух тел.",
        ),
        _case(
            "two_body_angular_momentum",
            momentum_error,
            _TWO_BODY_TOLERANCE,
            "Тот же оборот и тот же шаг.",
            "Модуль момента |r × v| сохраняется в задаче двух тел.",
        ),
        _case(
            "j2_node",
            j2_error,
            _J2_RELATIVE,
            "Круговая орбита 6800 км, 51.6°, 6 ч, без сопротивления. Оскулирующий узел против вековой формулы.",
            "Кёртис: dΩ/dt = −(3/2) n J2 (Re/a)² cos i.",
        ),
        {
            "name": "drag_lowers_radius",
            "measured": radius_drop_m,
            "tolerance": 0.0,
            "passed": radius_drop_m > 0.0,
            "applicability": "Таблица среднего Солнца, час, площадь 1 м², Cd 2.2. Не эталон плотности.",
            "source": "Сопротивление уменьшает большую полуось, поэтому радиус через час меньше, чем без него.",
            "status": "approximate" if radius_drop_m > 0.0 else "experimental",
        },
    ]
    return {
        "version": __version__,
        "profile": "core",
        "cases": cases,
        "passed": all(item["passed"] for item in cases),
    }


def _two_body(orbit: Orbit, seconds: float, substep_sec: float) -> tuple[np.ndarray, np.ndarray]:
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        seconds,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=False,
        include_drag=False,
        substep_sec=substep_sec,
    )
    return r, v


def _cannonball_m_s2(r_m: np.ndarray, when: datetime, mass_kg: float, area_m2: float, coefficient: float) -> np.ndarray:
    """Формула пушечного ядра, записанная отдельно от реализации ускорения."""
    from space_env.core.environment import shadow_state
    from space_env.core.sun import sun_position_m

    sun = sun_position_m(Time(when))
    distance = float(np.linalg.norm(sun))
    fraction = shadow_state(r_m, sun).flux_fraction
    if distance < 1.0 or fraction <= 0.0 or mass_kg <= 0.0:
        return np.zeros(3)
    return -coefficient * (area_m2 / mass_kg) * SOLAR_PRESSURE_1AU_N_M2 * (AU / distance) ** 2 * fraction * (sun / distance)


def _contact_boundary_sec() -> float | None:
    """Разница AOS поиска и сетки 0.25 с той же модели. None — сеанс не найден."""
    from space_env.core.frames import elevation_deg
    from space_env.core.ground import GroundStation
    from space_env.core.link import link_margin_db
    from space_env.core.passes import find_passes

    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, epoch)
    station = GroundStation(name="equator", lat_deg=0.0, lon_deg=0.0, alt_m=0.0, min_elevation_deg=5.0)
    radio = dict(
        frequency_hz=2.2e9,
        tx_power_w=10.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        system_loss_db=0.0,
        required_power_dbw=-300.0,
    )
    windows = find_passes(
        orbit.r,
        orbit.v,
        epoch,
        [station],
        horizon_sec=7000.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        sample_sec=60.0,
        refine_sec=REFERENCE_PROFILE.contact_tolerance_sec or 1.0,
        **radio,
    )
    if not windows:
        return None
    aos = float(windows[0].aos_sec)
    start = max(0.0, aos - 2.0)
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        start,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        when=epoch.datetime.replace(tzinfo=None),
    )
    seen_at: float | None = None
    clock = start
    while clock <= aos + 2.0 + 1e-9:
        moment = epoch + clock
        elevation = elevation_deg(r, station.lat_deg, station.lon_deg, station.alt_m, moment)
        slant = float(np.linalg.norm(r))  # запас заведомо закрыт требуемой мощностью −300 дБВт
        margin = link_margin_db(max(slant, 1.0), **radio)
        if elevation >= station.min_elevation_deg and margin >= 0.0:
            seen_at = clock
            break
        step = min(0.25, aos + 2.0 - clock)
        if step <= 1e-9:
            break
        r, v, _samples = propagate_state(
            r,
            v,
            step,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            when=moment.datetime.replace(tzinfo=None),
        )
        clock += step
    if seen_at is None:
        return REFERENCE_PROFILE.contact_tolerance_sec + 1.0 if REFERENCE_PROFILE.contact_tolerance_sec else 2.0
    return aos - seen_at


def reference_report() -> dict[str, Any]:
    """Короткий эталонный набор. Допуски берутся из REFERENCE_PROFILE до сравнения."""
    from space_env.core.bodies import moon_position_m
    from space_env.core.environment import shadow_state
    from space_env.core.gravity import _srp_acceleration_m_s2
    from space_env.core.relative import inertial_delta_v_m_s

    profile = REFERENCE_PROFILE
    arc = profile.max_arc_sec
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.001, 0.0, 0.2, 0.1, 0.3, epoch)
    kepler = orbit.propagate(arc)
    fine_r, fine_v = _two_body(orbit, arc, profile.substep_sec)
    coarse_r, _coarse_v = _two_body(orbit, arc, profile.substep_sec * 2.0)
    position_error_m = float(np.linalg.norm(fine_r - kepler.r))
    velocity_error_m_s = float(np.linalg.norm(fine_v - kepler.v))
    coarse_error_m = float(np.linalg.norm(coarse_r - kepler.r))
    ratio = coarse_error_m / max(position_error_m, 1e-12)
    noon = Time("2000-01-01T12:00:00Z")
    jd_error_sec = (noon.jd - 2451545.0) * 86400.0
    sun = np.array([AU, 0.0, 0.0])
    radius = R_EARTH_EQUATOR + 400_000.0
    lit = shadow_state(np.array([radius, 0.0, 0.0]), sun)
    dark = shadow_state(np.array([-radius, 0.0, 0.0]), sun)
    shadow_miss = float(abs(lit.flux_fraction - 1.0) + abs(dark.flux_fraction - 0.0) + (0.0 if dark.kind == "umbra" else 1.0))
    moon_m = float(np.linalg.norm(moon_position_m(epoch)))
    moon_lo, moon_hi = 3.50e8, 4.10e8
    when = datetime(2024, 6, 21, 12, 0, 0)
    from space_env.core.sun import sun_position_m

    sun_now = sun_position_m(Time(when))
    sun_hat = sun_now / np.linalg.norm(sun_now)
    lit_r = sun_hat * radius
    expected = _cannonball_m_s2(lit_r, when, 100.0, 1.0, 1.2)
    got = _srp_acceleration_m_s2(lit_r, when, 100.0, 1.0, 1.2)
    srp_scale = max(float(np.linalg.norm(expected)), 1e-20)
    srp_error = float(np.linalg.norm(got - expected) / srp_scale)
    burn = inertial_delta_v_m_s(orbit.r, orbit.v, 10.0, "prograde")
    v_burn = orbit.v + burn
    radius_now = float(np.linalg.norm(orbit.r))
    speed_now = float(np.linalg.norm(v_burn))
    a_vis = 1.0 / (2.0 / radius_now - speed_now * speed_now / MU_EARTH)
    a_external = Orbit.from_vectors(orbit.r, v_burn, epoch).keplerian().a
    burn_error = abs(a_external - a_vis) / abs(a_vis)
    boundary = _contact_boundary_sec()
    position_tol = float(profile.position_tolerance_m or 0.0)
    velocity_tol = float(profile.velocity_tolerance_m_s or 0.0)
    contact_tol = float(profile.contact_tolerance_sec or 0.0)
    ratio_floor = float(profile.convergence_ratio_min or 0.0)
    cases: list[dict[str, Any]] = [
        _case(
            "reference_position",
            position_error_m,
            position_tol,
            "Дуга 600 с, шаг 5 с, только два тела. Допуск 50 м взят из уже зафиксированного короткого сравнения, не из этого остатка.",
            "Двухтелая дуга boinor на те же элементы.",
        ),
        _case(
            "reference_velocity",
            velocity_error_m_s,
            velocity_tol,
            "Та же дуга и тот же шаг.",
            "Скорость той же двухтелой дуги boinor. Допуск 1 м/с задан до прогона.",
        ),
        {
            "name": "rk4_step_convergence",
            "measured": ratio,
            "tolerance": ratio_floor,
            "passed": ratio >= ratio_floor,
            "applicability": "Ошибка шага 10 с к ошибке шага 5 с на той же дуге. Порог 8 ниже теоретических 16 для метода четвёртого порядка.",
            "source": "Та же двухтелая дуга boinor. Порог не подгоняется под измеренное отношение.",
            "status": "verified" if ratio >= ratio_floor else "experimental",
        },
        _case(
            "utc_noon_j2000",
            jd_error_sec,
            TIME_TOLERANCE_SEC,
            "Календарный UTC без високосных секунд. Допуск модуля — 1 мс.",
            "JD полудня 2000-01-01 равен 2451545.0.",
        ),
        _case(
            "conical_shadow_interior",
            shadow_miss,
            0.0,
            "Две внутренние точки конуса на высоте 400 км: солнечная сторона и антисолнечная. Это не кривая блеска.",
            "Полный свет на солнечной стороне и umbra с нулевым потоком на антисолнечной.",
        ),
        {
            "name": "moon_distance_envelope",
            "measured": moon_m,
            "tolerance": moon_hi - moon_lo,
            "passed": moon_lo <= moon_m <= moon_hi,
            "applicability": "Один момент 2024-01-01. Ряд Миуса обещает сотни километров, не метры.",
            "source": "Геоцентрическое расстояние Луны лежит между 350 и 410 тыс. км. Границы шире перигея и апогея и заданы до прогона.",
            "status": "approximate" if moon_lo <= moon_m <= moon_hi else "experimental",
        },
        _case(
            "cannonball_srp",
            srp_error,
            1.0e-9,
            "Площадь 1 м², масса 100 кг, Cr 1.2. Не паспорт реального аппарата.",
            "a = −Cr (A/m) P (1 а.е. / R)² ν r̂. Допуск 1e-9 — согласие формул в float64.",
        ),
        _case(
            "impulsive_prograde_semimajor_axis",
            burn_error,
            1.0e-6,
            "Импульс 10 м/с вдоль скорости на той же орбите. Это не конечное включение двигателя.",
            "Большая полуось из vis-viva против элементов boinor. Допуск 1e-6 задан до прогона.",
        ),
    ]
    if boundary is None:
        cases.append(
            {
                "name": "contact_boundary",
                "measured": None,
                "tolerance": contact_tol,
                "passed": False,
                "applicability": "Экваториальная дуга 7000 с над экваториальной станцией с маской 5°.",
                "source": "Поиск не нашёл сеанс, поэтому границу сверять не с чем.",
                "status": "experimental",
            }
        )
    else:
        cases.append(
            _case(
                "contact_boundary",
                boundary,
                contact_tol,
                "Экваториальная дуга, маска 5°, грубая сетка 60 с и деление до 1 с. Сравнение с сеткой 0.25 с той же модели, не с внешним каталогом.",
                "Допуск события равен refine_sec профиля и задан до прогона.",
            )
        )
    return {
        "version": __version__,
        "profile": profile.to_dict(),
        "training_profile": TRAINING_PROFILE.to_dict(),
        "backends": optional_backends(),
        "cases": cases,
        "passed": all(item["passed"] for item in cases),
    }


def applicability_limits() -> list[dict[str, str]]:
    """Граница профиля. Эти пункты не выдаются за проверенный эталон."""
    return [
        {
            "name": "pc",
            "status": "checked_against_disk_integral",
            "note": "Приближение Чена сверено с интегралом по диску в этом пакете. Внешнего каталога Pc нет, это не эксплуатационная оценка.",
        },
        {
            "name": "rk4_stage_epoch",
            "status": "stage_epoch",
            "note": "Четыре стадии RK4 берут моменты 0, h/2, h/2 и h. Широта стадии пересчитывается, когда модель плотности её читает.",
        },
        {
            "name": "teme_to_gcrf",
            "status": "vallado_2006_itrs",
            "note": "TEME переводится в ITRS через GMST 1982 и pom00, затем в численный кадр пакета. Отдельный вектор GCRS строится c2t06a и в движок не подставляется.",
        },
        {
            "name": "iau_2000a",
            "status": "training_frame",
            "note": "Переход ECEF использует IAU 2000A через ERFA. Десять членов IAU 1980 сохранены для сравнения и не крутят учебный кадр.",
        },
        {
            "name": "outcome_error",
            "status": "mission_contract",
            "note": "Исход миссии сверяется списком сдачи и причин. Метры reference-дуги на этот список не переносятся.",
        },
        {
            "name": "slew_accel",
            "status": "configured",
            "note": "Нулевое угловое ускорение по-прежнему набирает предельную скорость сразу. Положительное разгоняет с покоя и тормозит к цели. Это не паспорт маховика.",
        },
        {
            "name": "memory_queue",
            "status": "configured",
            "note": "По умолчанию лишний объём отбрасывается, а продукт доступен сразу. Вытеснение, очередь обработки и доля сжатия включаются полями полезной нагрузки.",
        },
        {
            "name": "power_hysteresis",
            "status": "configured",
            "note": "Оба порога заряда по нулям — автоматического безопасного режима нет. Иначе вход ниже выхода, и работа не возобновляется в промежутке.",
        },
        {
            "name": "flux_along_step",
            "status": "sample_average",
            "note": "Выработка батарей и внешний нагрев — средние по отсчётам шага, с косинусом в каждой точке. Конец шага не подменяет освещённое начало.",
        },
        {
            "name": "sun_ephemeris",
            "status": "epv00_shared",
            "note": "Тень, световое давление, освещённость и прилив читают ERFA epv00. Узел каталога остаётся на ряде Миуса: записанные цели лежат на этой трассе.",
        },
        {
            "name": "centered_f107",
            "status": "inside_snapshot",
            "note": "Если в снимке есть 40 суток до даты и 40 после, f107a — это центрированное среднее. На 20 октября 2026 хвоста нет, поэтому остаётся скользящее.",
        },
        {
            "name": "star_1w",
            "status": "evaluated_open",
            "note": "Оболочка 66/6/2 на 780 км построена. Линия 1 Вт / 6 дБи на ближайшей паре разомкнута. Пороги линии не увеличивались.",
        },
        {
            "name": "disposal_25y",
            "status": "scale_height_estimate",
            "note": "Срок круговой орбиты — одна местная шкала высот экспоненциальной атмосферы. Это не 25-летний Кауэлл и не сертификат сведения.",
        },
        {
            "name": "finite_burn",
            "status": "separate_profile",
            "note": "Конечное включение считается только при заданных тяге, удельном импульсе и сухой массе. Иначе манёвр остаётся мгновенным и массу не меняет.",
        },
        {
            "name": "external_backend",
            "status": "does_not_replace_validation",
            "note": "boinor задаёт двухтелую дугу эталона. Orekit, SGP4 и SPICE этот отчёт не заменяют и TEME в GCRF не переводят.",
        },
    ]


def main() -> None:
    print(json.dumps({"core": core_report(), "reference": reference_report()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
