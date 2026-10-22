"""
План работы аппарата на горизонте: что снять, куда сдавать и какое окно не занимать.

Съёмка берётся, пока цель в полосе, вне полной тени и в регистраторе есть место.
Сеанс со слабой ступенью модема откладывается, если до более быстрого окна память
вмещает всё, что будет снято. Скорость сеанса — ступень по лучшему запасу этого окна.
Один пункт занимает шину: съёмка, попавшая в выбранный сеанс, из плана уходит.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from space_env.catalog.build import mean_orbit, spacecraft_config
from space_env.catalog.references import CUBESAT_STATIONS, LANDSAT_STATIONS, SENTINEL_STATIONS, reference
from space_env.core.access import ground_track_latlon, off_nadir_deg, subsatellite_latlon
from space_env.core.environment import in_eclipse
from space_env.core.frames import eci_to_ecef, elevation_deg, geodetic_to_ecef
from space_env.core.relative import inertial_delta_v_m_s
from space_env.core.ground import GroundStation
from space_env.core.link import link_margin_db, stepped_rate_bps
from space_env.core.propagate import propagate_state
from space_env.core.space_weather import indices_for
from space_env.core.sun import sun_position_m
from space_env.mission.events import EventKind, PlannedEvent
from space_env.mission.scenarios import GroundTarget


@dataclass(frozen=True)
class Opportunity:
    """Окно, которое физика уже подтвердила. Планировщик решает, входит ли оно в дела."""

    kind: str
    name: str
    start_sec: float
    end_sec: float
    rate_bps: float
    margin_db: float = 0.0


@dataclass(frozen=True)
class WorkItem:
    """Одно дело. required=False — пояснение, почему окно в список не попало."""

    kind: str
    name: str
    start_sec: float
    end_sec: float
    required: bool
    reason: str
    rate_bps: float = 0.0
    bits: float = 0.0
    direction: str = ""
    dv_m_s: float = 0.0

    def as_planned_event(self) -> PlannedEvent | None:
        if not self.required:
            return None
        if self.kind == "imaging":
            return PlannedEvent(
                kind=EventKind.IMAGING,
                scheduled_at_sec=self.start_sec,
                duration_sec=self.end_sec - self.start_sec,
                payload={"target": self.name},
            )
        if self.kind == "comms":
            return PlannedEvent(
                kind=EventKind.COMMS_PASS,
                scheduled_at_sec=self.start_sec,
                duration_sec=self.end_sec - self.start_sec,
                payload={"station": self.name},
            )
        if self.kind == "maneuver":
            return PlannedEvent(
                kind=EventKind.MANEUVER,
                scheduled_at_sec=self.start_sec,
                duration_sec=0.0,
                payload={"dv_m_s": self.dv_m_s, "direction": self.direction or "prograde"},
            )
        return None


@dataclass(frozen=True)
class WorkPlan:
    """Список дел на горизонт. Строки — то, что можно отдать оператору или языковой модели."""

    vehicle: str
    horizon_sec: float
    items: tuple[WorkItem, ...]
    memory_bits: float
    targets: tuple[tuple[str, float, float], ...] = ()

    def required_items(self) -> tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.required)

    def lines(self) -> list[str]:
        if not self.items:
            return ["На этом горизонте съёмки и сеанса нет."]
        return [item.reason for item in self.items]

    def as_planned_events(self) -> list[PlannedEvent]:
        events = []
        for item in self.required_items():
            event = item.as_planned_event()
            if event is not None:
                events.append(event)
        return events


def maneuver_item(at_sec: float, dv_m_s: float, direction: str, reason: str) -> WorkItem:
    """Дело-манёвр. Планировщик съёмки сам импульс не выдумывает: его добавляет постановка."""
    return WorkItem(
        kind="maneuver",
        name=direction,
        start_sec=at_sec,
        end_sec=at_sec,
        required=True,
        reason=reason,
        direction=direction,
        dv_m_s=dv_m_s,
    )


def _bits_text(bits: float) -> str:
    if abs(bits) >= 1e9:
        return f"{bits / 1e9:.1f} Гбит"
    return f"{bits / 1e6:.1f} Мбит"


def _span(start: float, end: float) -> str:
    return f"{start:.0f}–{end:.0f} с"


def _cut_out(start: float, end: float, blocks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for block_start, block_end in blocks:
        nxt: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if block_end <= piece_start or block_start >= piece_end:
                nxt.append((piece_start, piece_end))
                continue
            if piece_start < block_start:
                nxt.append((piece_start, block_start))
            if block_end < piece_end:
                nxt.append((block_end, piece_end))
        pieces = nxt
    return [(left, right) for left, right in pieces if right - left >= 1.0]


def _wanted_bits(
    imaging: list[Opportunity],
    until_sec: float,
    *,
    initial_bits: float,
    imaging_cap_s: float,
) -> float:
    total = initial_bits
    used = 0.0
    for opp in imaging:
        if opp.start_sec >= until_sec:
            break
        duration = min(opp.end_sec, until_sec) - opp.start_sec
        take = min(max(duration, 0.0), max(0.0, imaging_cap_s - used))
        total += opp.rate_bps * take
        used += take
    return total


def schedule_acquisition(
    opportunities: list[Opportunity],
    *,
    memory_bits: float,
    imaging_cap_s: float,
    downlink_bps: float,
    initial_bits: float = 0.0,
) -> list[WorkItem]:
    """
    Из окон собирает дела. Слабый сеанс пропускается, только если более быстрый
    успевает принять всё, что съёмка хочет положить в память до его начала.
    """
    if memory_bits <= 0.0:
        raise ValueError("объём памяти должен быть положительным")
    imaging = sorted((item for item in opportunities if item.kind == "imaging"), key=lambda item: item.start_sec)
    passes = sorted((item for item in opportunities if item.kind == "comms"), key=lambda item: item.start_sec)
    skipped: list[WorkItem] = []
    kept: list[Opportunity] = []
    for index, current in enumerate(passes):
        later = next(
            (item for item in passes[index + 1 :] if item.rate_bps > current.rate_bps * 1.5),
            None,
        )
        pending = _wanted_bits(imaging, later.start_sec, initial_bits=initial_bits, imaging_cap_s=imaging_cap_s) if later else 0.0
        if later is not None and pending <= memory_bits:
            current_step = current.rate_bps / downlink_bps if downlink_bps else 0.0
            later_step = later.rate_bps / downlink_bps if downlink_bps else 0.0
            skipped.append(
                WorkItem(
                    kind="comms",
                    name=current.name,
                    start_sec=current.start_sec,
                    end_sec=current.end_sec,
                    required=False,
                    rate_bps=current.rate_bps,
                    reason=(
                        f"{_span(current.start_sec, current.end_sec)}: не занимать {current.name}. "
                        f"Ступень {current_step:.2f}, окно {later.name} с {later.start_sec:.0f} с "
                        f"идёт на ступени {later_step:.2f}, память до него вмещает {_bits_text(pending)}."
                    ),
                )
            )
            continue
        kept.append(current)

    blocks = [(item.start_sec, item.end_sec) for item in kept]
    pieces: list[tuple[str, str, float, float, float]] = []
    seen_targets: set[str] = set()
    blocked: dict[str, Opportunity] = {}
    for opp in imaging:
        if opp.name in seen_targets:
            continue
        free = _cut_out(opp.start_sec, opp.end_sec, blocks)
        if not free:
            blocked[opp.name] = opp
            continue
        left, right = free[0]
        pieces.append(("imaging", opp.name, left, right, opp.rate_bps))
        seen_targets.add(opp.name)
    for name, opp in blocked.items():
        if name in seen_targets:
            continue
        skipped.append(
            WorkItem(
                kind="imaging",
                name=name,
                start_sec=opp.start_sec,
                end_sec=opp.end_sec,
                required=False,
                reason=(
                    f"{_span(opp.start_sec, opp.end_sec)}: не снимать {name}. "
                    "Всё окно занято выбранным сеансом."
                ),
            )
        )
    for opp in kept:
        pieces.append(("comms", opp.name, opp.start_sec, opp.end_sec, opp.rate_bps))
    pieces.sort(key=lambda item: (item[2], 0 if item[0] == "comms" else 1))

    buffer = initial_bits
    imaging_used = 0.0
    required: list[WorkItem] = []
    for kind, name, start, end, rate in pieces:
        duration = end - start
        if kind == "imaging":
            take = min(duration, max(0.0, imaging_cap_s - imaging_used))
            room = max(0.0, memory_bits - buffer)
            if rate > 0.0:
                take = min(take, room / rate)
            if take < 1.0:
                skipped.append(
                    WorkItem(
                        kind="imaging",
                        name=name,
                        start_sec=start,
                        end_sec=end,
                        required=False,
                        reason=(
                            f"{start:.0f} с: не снимать {name}. "
                            "Регистратор заполнен, сначала сдача."
                        ),
                    )
                )
                continue
            bits = rate * take
            buffer += bits
            imaging_used += take
            finish = start + take
            required.append(
                WorkItem(
                    kind="imaging",
                    name=name,
                    start_sec=start,
                    end_sec=finish,
                    required=True,
                    rate_bps=rate,
                    bits=bits,
                    reason=(
                        f"{_span(start, finish)}: снять {name}. "
                        f"Цель в полосе и вне полной тени, в регистратор помещается {_bits_text(bits)}."
                    ),
                )
            )
            continue
        dumped = min(buffer, rate * duration)
        if dumped <= 0.0:
            skipped.append(
                WorkItem(
                    kind="comms",
                    name=name,
                    start_sec=start,
                    end_sec=end,
                    required=False,
                    rate_bps=rate,
                    reason=(
                        f"{_span(start, end)}: не сдавать на {name}. "
                        "К началу окна в памяти нет данных."
                    ),
                )
            )
            continue
        buffer -= dumped
        step = rate / downlink_bps if downlink_bps else 0.0
        required.append(
            WorkItem(
                kind="comms",
                name=name,
                start_sec=start,
                end_sec=end,
                required=True,
                rate_bps=rate,
                bits=dumped,
                reason=(
                    f"{_span(start, end)}: сдать на {name}. "
                    f"Ступень модема {step:.2f}, за окно уходит до {_bits_text(dumped)}."
                ),
            )
        )
    return sorted([*required, *skipped], key=lambda item: (item.start_sec, item.required))


def _stations_for(key: str) -> list[GroundStation]:
    if key.startswith("sentinel-2"):
        return list(SENTINEL_STATIONS)
    if key == "landsat-9":
        return list(LANDSAT_STATIONS)
    if key == "cubesat-6u":
        return list(CUBESAT_STATIONS)
    if key == "sar-small":
        return [station for station in SENTINEL_STATIONS if station.name == "Svalbard"]
    return []


def _target_prefix(key: str) -> str:
    if key == "landsat-9":
        return "L9"
    if key == "cubesat-6u":
        return "C6"
    if key == "sar-small":
        return "RS"
    return "S2"


def _group_open(open_runs: dict[str, list[float]], kind: str, sample_sec: float, rate_of) -> list[Opportunity]:
    found = []
    for name, samples in open_runs.items():
        if not samples:
            continue
        block = [samples[0]]
        for moment in samples[1:]:
            if moment - block[-1] > sample_sec * 1.5:
                found.append(_block_to_opportunity(kind, name, block, sample_sec, rate_of))
                block = [moment]
            else:
                block.append(moment)
        found.append(_block_to_opportunity(kind, name, block, sample_sec, rate_of))
    return found


def _block_to_opportunity(kind: str, name: str, samples: list[float], sample_sec: float, rate_of) -> Opportunity:
    margin, rate = rate_of(name, samples)
    return Opportunity(kind, name, samples[0], samples[-1] + sample_sec, rate, margin)


def collect_opportunities(
    key: str,
    *,
    horizon_sec: float,
    epoch: str = "2026-01-01T00:00:00Z",
    nu_deg: float = 180.0,
    sample_sec: float = 60.0,
    targets: list[GroundTarget] | None = None,
    order_every_sec: float | None = None,
    burn_at_sec: float | None = None,
    burn_dv_m_s: float = 0.0,
    burn_direction: str = "normal",
) -> tuple[list[Opportunity], tuple[tuple[str, float, float], ...]]:
    """Окна съёмки и сеансов тем же Кауэллом, что и шаг среды. Шаг сетки — sample_sec.

    order_every_sec ставит заявку в подспутниковую точку раз в этот интервал, на свету.
    Так сутки получают отдельную цель на виток, а не три точки короткого плана.
    """
    spec = reference(key)
    if spec.memory_bits is None or spec.store_rate_bps is None:
        raise ValueError(f"у {key} нет опубликованных памяти и скорости записи")
    stations = _stations_for(key)
    orbit = mean_orbit(key, epoch, nu_deg=nu_deg)
    craft = spacecraft_config(key)
    prop = craft.propulsion
    weather = indices_for(orbit.epoch)
    f107 = 150.0 if weather is None else weather.f107
    f107a = 150.0 if weather is None else weather.f107a
    ap = 4.0 if weather is None else weather.ap
    if order_every_sec is not None and order_every_sec <= 0.0:
        raise ValueError("шаг заявок должен быть положительным")
    dynamic_orders = targets is None and order_every_sec is not None
    prefix = _target_prefix(key)
    if targets is None and not dynamic_orders:
        points = ground_track_latlon(
            orbit,
            [600.0, 1500.0, 2400.0],
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
        )
        targets = [
            GroundTarget(name=f"{prefix}-{index + 1}", lat_deg=lat, lon_deg=lon)
            for index, (lat, lon) in enumerate(points)
        ]
    elif targets is None:
        targets = []
    comms = craft.comms
    half_fov = craft.payload.half_fov_deg
    position = orbit.r.copy()
    velocity = orbit.v.copy()
    imaging_hits: dict[str, list[float]] = {item.name: [] for item in targets}
    comms_hits: dict[str, list[tuple[float, float]]] = {station.name: [] for station in stations}
    elapsed = 0.0
    burned = False
    while elapsed <= horizon_sec + 1e-9:
        if (
            burn_at_sec is not None
            and not burned
            and burn_dv_m_s > 0.0
            and elapsed + 1e-9 >= burn_at_sec
        ):
            velocity = velocity + inertial_delta_v_m_s(position, velocity, burn_dv_m_s, burn_direction)
            burned = True
        moment = orbit.epoch + elapsed
        sunlit = not in_eclipse(position, sun_position_m(moment))
        if dynamic_orders and sunlit and elapsed + 1e-9 >= len(targets) * float(order_every_sec):
            lat, lon = subsatellite_latlon(position, moment)
            name = f"{prefix}-{len(targets) + 1}"
            targets.append(GroundTarget(name=name, lat_deg=lat, lon_deg=lon))
            imaging_hits[name] = []
        if sunlit:
            visible = [
                (off_nadir_deg(position, item.lat_deg, item.lon_deg, moment), item.name)
                for item in targets
            ]
            if visible:
                angle, name = min(visible)
                if angle <= half_fov:
                    imaging_hits[name].append(elapsed)
        for station in stations:
            elevation = elevation_deg(position, station.lat_deg, station.lon_deg, station.alt_m, moment)
            if elevation < station.min_elevation_deg:
                continue
            slant = float(
                np.linalg.norm(
                    eci_to_ecef(position, moment)
                    - geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
                )
            )
            margin = link_margin_db(
                slant,
                frequency_hz=comms.frequency_Hz,
                tx_power_w=comms.tx_power_W,
                tx_gain_dbi=comms.tx_gain_dBi,
                rx_gain_dbi=comms.rx_gain_dBi,
                system_loss_db=comms.system_loss_dB,
                required_power_dbw=comms.required_power_dBW,
            )
            if margin >= comms.min_margin_dB:
                comms_hits[station.name].append((elapsed, margin))
        if elapsed >= horizon_sec:
            break
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            sample_sec,
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
            include_drag=True,
            include_j2=True,
            include_third_body=True,
            atmosphere="table",
            f107=f107,
            f107a=f107a,
            ap=ap,
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
            when=moment.datetime.replace(tzinfo=None),
            substep_sec=10.0,
        )
        elapsed += sample_sec
    downlink = spec.downlink_bps

    def imaging_rate(_name: str, _samples: list[float]) -> tuple[float, float]:
        return 0.0, float(spec.store_rate_bps or 0.0)

    def comms_rate(name: str, samples: list[float]) -> tuple[float, float]:
        block = set(samples)
        chosen = [margin for moment, margin in comms_hits[name] if moment in block]
        best = max(chosen) if chosen else 0.0
        return best, stepped_rate_bps(best, downlink)

    imaging_runs = {name: times for name, times in imaging_hits.items() if times}
    comms_runs = {name: [moment for moment, _margin in hits] for name, hits in comms_hits.items() if hits}
    opportunities = [
        *_group_open(imaging_runs, "imaging", sample_sec, imaging_rate),
        *_group_open(comms_runs, "comms", sample_sec, comms_rate),
    ]
    clipped = []
    for item in opportunities:
        end = min(item.end_sec, horizon_sec)
        if end - item.start_sec >= 1.0:
            clipped.append(replace(item, end_sec=end))
    target_rows = tuple((item.name, item.lat_deg, item.lon_deg) for item in targets)
    return clipped, target_rows


def build_acquisition_plan(
    key: str,
    *,
    horizon_sec: float = 5400.0,
    epoch: str = "2026-01-01T00:00:00Z",
    nu_deg: float = 180.0,
    sample_sec: float = 60.0,
    initial_bits: float = 0.0,
    targets: list[GroundTarget] | None = None,
    extra_items: list[WorkItem] | None = None,
    order_every_sec: float | None = None,
    imaging_cap_s: float | None = None,
    burn_at_sec: float | None = None,
    burn_dv_m_s: float = 0.0,
    burn_direction: str = "normal",
) -> WorkPlan:
    """План съёмки и сдачи для карточки с памятью. Дополнительные дела, например импульс, добавляются как есть."""
    spec = reference(key)
    opportunities, target_rows = collect_opportunities(
        key,
        horizon_sec=horizon_sec,
        epoch=epoch,
        nu_deg=nu_deg,
        sample_sec=sample_sec,
        targets=targets,
        order_every_sec=order_every_sec,
        burn_at_sec=burn_at_sec,
        burn_dv_m_s=burn_dv_m_s,
        burn_direction=burn_direction,
    )
    if imaging_cap_s is None:
        imaging_cap_s = float(spec.max_imaging_s) if spec.max_imaging_s is not None else 1.0e12
    items = schedule_acquisition(
        opportunities,
        memory_bits=float(spec.memory_bits),
        imaging_cap_s=imaging_cap_s,
        downlink_bps=float(spec.downlink_bps),
        initial_bits=initial_bits,
    )
    if extra_items:
        items = sorted([*items, *extra_items], key=lambda item: (item.start_sec, item.required))
    return WorkPlan(key, horizon_sec, tuple(items), float(spec.memory_bits), target_rows)
