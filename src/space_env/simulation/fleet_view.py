"""Снимок группировки на эпоху: дистанции и замкнутые межспутниковые линии."""
from __future__ import annotations

import math
from functools import lru_cache

from space_env.catalog.build import make_engine, mean_orbit, spacecraft_config
from space_env.core.ground import DEFAULT_GROUND_STATIONS
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft
from space_env.mission.campaign import build_day_campaign
from space_env.mission.horizon import PC_THRESHOLD
from space_env.simulation.engine import SimulationEngine
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.shift_scenario import load_shift_scenario
from space_env.simulation.fleet import Fleet
from space_env.viz.export import export_trajectory

_STAR_ALTITUDE_M = 780_000.0
_STAR_RADIUS_M = 6_378_137.0
_STAR_INC_DEG = 86.4
_MAX_COUNT = 66
_SHIFT_DT_SEC = 60.0
_DIRECTION_RU = {
    "prograde": "по скорости",
    "retrograde": "против скорости",
    "normal": "по нормали",
    "radial": "по радиусу",
}
_STATION_RU = {
    "Svalbard": "Шпицберген",
    "Moscow": "Москву",
    "Kiruna": "Кируну",
    "Matera": "Матеру",
    "Sioux Falls": "Су-Фолс",
    "Inuvik": "Инувик",
}
_STATION_NAME = {
    "Svalbard": "Шпицбергена",
    "Moscow": "Москвы",
    "Kiruna": "Кируны",
    "Matera": "Матеры",
    "Sioux Falls": "Су-Фолс",
    "Inuvik": "Инувика",
}


def describe_fleet(
    count: int = 6,
    layout: str = "chain",
    *,
    planes: int = 6,
    phasing: int = 2,
) -> dict:
    """
    chain — цепочка с шагом 1.5°.
    star — Walker-star: при count=66, planes=6, phasing=2 это схема 86.4°:66/6/2.
    Линия 1 Вт / 6 дБи. Замкнутой считается пара с неотрицательным запасом и без лимба.
    """
    if layout not in {"chain", "star"}:
        raise ValueError("схема группировки: chain или star")
    if count < 2 or count > _MAX_COUNT:
        raise ValueError("в снимке от 2 до 66 аппаратов")
    scenario = ScenarioConfig(ground_stations=[])
    if layout == "chain":
        fleet = Fleet.string_of_pearls(count, separation_deg=1.5, scenario=scenario)
        names = list(fleet.vehicles)
        neighbor_pairs = [(names[index], names[index + 1]) for index in range(len(names) - 1)]
    else:
        if count % planes != 0:
            raise ValueError("число аппаратов звезды должно делиться на число плоскостей")
        fleet = Fleet.walker_star(
            count,
            planes,
            phasing,
            a_m=_STAR_RADIUS_M + _STAR_ALTITUDE_M,
            inc_deg=_STAR_INC_DEG,
            scenario=scenario,
        )
        names = list(fleet.vehicles)
        per_plane = count // planes
        neighbor_pairs = []
        for plane in range(planes):
            slots = [f"P{plane + 1}-S{slot + 1}" for slot in range(per_plane)]
            neighbor_pairs.extend((slots[index], slots[(index + 1) % per_plane]) for index in range(per_plane))
    indexed = {}
    for pair in fleet.pairs():
        indexed[(pair.leader, pair.follower)] = pair
        indexed[(pair.follower, pair.leader)] = pair
    neighbor_ranges = []
    closed_neighbors = 0
    for left, right in neighbor_pairs:
        pair = indexed[(left, right)]
        neighbor_ranges.append(pair.range_m)
        if not pair.earth_blocked and pair.isl_margin_db >= 0.0:
            closed_neighbors += 1
    pairs = fleet.pairs()
    nearest = min(pairs, key=lambda item: item.range_m)
    closed_links = sum(1 for item in pairs if not item.earth_blocked and item.isl_margin_db >= 0.0)
    mean_neighbor_m = sum(neighbor_ranges) / len(neighbor_ranges) if neighbor_ranges else 0.0
    return {
        "layout": layout,
        "count": len(names),
        "planes": planes if layout == "star" else 1,
        "closed_links": closed_links,
        "closed_neighbor_links": closed_neighbors,
        "neighbor_km": mean_neighbor_m / 1000.0,
        "nearest_km": nearest.range_m / 1000.0,
        "nearest_margin_db": nearest.isl_margin_db,
        "nearest_pair": [nearest.leader, nearest.follower],
        "link": "1 Вт / 6 дБи, порог 0 дБ",
    }


def _mark_neighbors(document: dict, neighbor_pairs: list[tuple[str, str]]) -> None:
    wanted = {frozenset(pair) for pair in neighbor_pairs}
    for frame in document["frames"]:
        kept = []
        for pair in frame["pairs"]:
            if frozenset((pair["leader"], pair["follower"])) not in wanted:
                continue
            pair["neighbor"] = True
            pair["open"] = (not pair["earth_blocked"]) and pair["isl_margin_db"] >= 0.0
            kept.append(pair)
        frame["pairs"] = kept


def _arc_events(day, until_sec: float) -> list[dict]:
    """Дела плана, которые попадают в показанную дугу, и импульс, если он в ней."""
    events: list[dict] = []
    burn = next((item for item in day.conjunctions if item.required), None)
    if (
        burn is not None
        and day.burn_at_sec is not None
        and day.vehicles
        and 0.0 <= float(day.burn_at_sec) <= until_sec
    ):
        direction = burn.direction or "prograde"
        events.append(
            {
                "vehicle": day.vehicles[0],
                "kind": "maneuver",
                "name": burn.name,
                "start_sec": float(day.burn_at_sec),
                "end_sec": float(day.burn_at_sec),
                "required": True,
                "dv_m_s": float(burn.dv_m_s),
                "direction": direction,
                "note": (
                    f"Импульс {burn.dv_m_s:.1f} м/с {_DIRECTION_RU.get(direction, direction)} "
                    f"в {day.burn_at_sec:.0f} с. Пара {burn.name}."
                ),
            }
        )
    for plan in day.plans:
        for item in plan.items:
            if item.start_sec > until_sec or item.end_sec <= 0.0:
                continue
            if item.kind == "imaging" and not item.required:
                continue
            if item.kind == "comms" and not item.required and "занято" not in item.reason:
                continue
            if item.kind not in {"imaging", "comms"}:
                continue
            events.append(
                {
                    "vehicle": plan.vehicle,
                    "kind": item.kind,
                    "name": item.name,
                    "start_sec": float(item.start_sec),
                    "end_sec": float(item.end_sec),
                    "required": bool(item.required),
                    "dv_m_s": float(item.dv_m_s),
                    "direction": item.direction,
                    "note": item.reason,
                }
            )
    events.sort(key=lambda row: (row["start_sec"], not row["required"], row["vehicle"], row["kind"]))
    return events


def _conjunction_rows(day) -> list[dict]:
    """Объекты скрининга суток: Pc без импульса, Pc после дуги с импульсом, час встречи."""
    coast = dict(day.coast_pc)
    flown = dict(day.flown_pc)
    rows = []
    for item in day.conjunctions:
        rows.append(
            {
                "name": item.name,
                "t_sec": float(item.start_sec),
                "required": bool(item.required),
                "coast_pc": float(coast.get(item.name, item.probability)),
                "flown_pc": float(flown.get(item.name, item.probability)),
                "direction": item.direction or "",
                "dv_m_s": float(item.dv_m_s),
            }
        )
    rows.sort(key=lambda row: (-row["coast_pc"], row["t_sec"], row["name"]))
    return rows


def _screening_companions(day) -> dict[str, SimulationEngine]:
    """Соседи с Pc без импульса не ниже порога. Элементы те же, что у скрининга суток."""
    if not day.vehicles:
        return {}
    primary = day.vehicles[0]
    orbit = mean_orbit(primary, day.epoch, nu_deg=day.nu_deg)
    own = orbit.keplerian()
    coast = dict(day.coast_pc)
    engines: dict[str, SimulationEngine] = {}
    for name, delta_a_m, delta_inc_deg, delta_nu_deg in day.population:
        if coast.get(name, 0.0) < PC_THRESHOLD:
            continue
        partner = Orbit.from_keplerian(
            own.a + delta_a_m,
            own.ecc,
            own.inc + math.radians(delta_inc_deg),
            own.raan,
            own.argp,
            own.nu + math.radians(delta_nu_deg),
            orbit.epoch,
        )
        craft = Spacecraft(config=spacecraft_config(primary))
        craft.reset_state()
        engines[name] = SimulationEngine(
            partner,
            craft,
            scenario=ScenarioConfig(ground_stations=[]),
        )
    return engines


def _shift_span(minutes: int) -> str:
    if minutes == 1440:
        return "сутки"
    if minutes > 1440 and minutes % 1440 == 0:
        return f"{minutes // 1440} сут"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60} ч"
    return f"{minutes} мин"


def _shift_brief(events: list[dict], minutes: int, conjunctions: list[dict]) -> str:
    burns = [item for item in events if item["kind"] == "maneuver" and item["required"]]
    dumps = [item for item in events if item["kind"] == "comms" and item["required"]]
    denied = [item for item in events if item["kind"] == "comms" and not item["required"]]
    parts = [f"На экране {_shift_span(minutes)} сценария, не учебная цепочка."]
    if burns:
        parts.append(burns[0]["note"])
    hot = next((item for item in conjunctions if item["name"] == "HOT"), None)
    if hot is not None and hot["coast_pc"] >= 1.0e-4 and hot["flown_pc"] < 1.0e-4:
        parts.append(
            f"Пара HOT без импульса Pc {hot['coast_pc']:.3e}, после импульса {hot['flown_pc']:.3e}. "
            f"Минимальная дистанция в {hot['t_sec']:.0f} с."
        )
    marked = [item["name"] for item in conjunctions if item["coast_pc"] >= 1.0e-4]
    if marked:
        parts.append(
            "На глобусе и на карте "
            + " и ".join(marked)
            + " — соседи скрининга на той же дуге. Импульс меняет только свой аппарат."
        )
    if dumps:
        parts.append(f"Назначенных сдач: {len(dumps)}.")
        svalbard = [item for item in dumps if item["name"] == "Svalbard"]
        if svalbard:
            names = ", ".join(dict.fromkeys(item["vehicle"] for item in svalbard))
            first = svalbard[0]
            parts.append(
                f"Шпицберген в плане у {names}. "
                f"Первая такая сдача — {first['vehicle']} {first['start_sec']:.0f}–{first['end_sec']:.0f} с."
            )
        moscow = next((item for item in dumps if item["name"] == "Moscow"), None)
        if moscow:
            parts.append(
                f"Сдача на Москву: {moscow['vehicle']} {moscow['start_sec']:.0f}–{moscow['end_sec']:.0f} с."
            )
    else:
        parts.append("Назначенной сдачи в этой дуге нет.")
    if denied:
        parts.append(denied[0]["note"])
        if len(denied) > 1:
            parts.append(f"Отказов зеркала в дуге: {len(denied)}.")
    gaps = []
    for item in denied:
        if "аппаратом " not in item["note"]:
            continue
        holder = item["note"].split("аппаратом ", 1)[1].rstrip(".")
        holder_dumps = any(
            other["required"]
            and other["kind"] == "comms"
            and other["vehicle"] == holder
            and other["name"] == item["name"]
            and other["start_sec"] < item["end_sec"]
            and item["start_sec"] < other["end_sec"]
            for other in dumps
        )
        if not holder_dumps:
            gaps.append((item, holder))
    if gaps:
        item, holder = gaps[0]
        station = _STATION_NAME.get(item["name"], item["name"])
        parts.append(
            f"Окно {station} в {item['start_sec']:.0f}–{item['end_sec']:.0f} с закреплено за {holder}, "
            "но назначенной сдачи этого аппарата на это окно в дуге нет."
        )
        if len(gaps) > 1:
            parts.append(f"Таких пустых закреплений: {len(gaps)}.")
    parts.append("Таблица показывает, закрыта ли линия в кадре. Журнал — дела плана на всю дугу.")
    return " ".join(parts)


def _shift_episode() -> tuple[Fleet, list[tuple[dict[str, str], dict[str, dict]]], dict, dict | None, dict, dict]:
    """Сутки сценария: импульс в дуге и журнал дел того же плана, что у смены."""
    cfg = load_shift_scenario()
    horizon = float(cfg["horizon_sec"])
    steps = int(round(horizon / _SHIFT_DT_SEC))
    if steps < 1 or abs(steps * _SHIFT_DT_SEC - horizon) > 1.0:
        raise ValueError("горизонт студии должен укладываться в шаг кадра")
    day = build_day_campaign(
        tuple(cfg["keys"]),
        horizon_sec=horizon,
        epoch=str(cfg["epoch"]),
        nu_deg=float(cfg["nu_deg"]),
        dv_budget_m_s=float(cfg["dv_budget_m_s"]),
    )
    fleet = Fleet(
        {
            key: make_engine(key, epoch=cfg["epoch"], nu_deg=float(cfg["nu_deg"]), with_stations=True)
            for key in cfg["keys"]
        }
    )
    events = _arc_events(day, horizon)
    burn = next((item for item in events if item["kind"] == "maneuver" and item["required"]), None)
    commands: list[tuple[dict[str, str], dict[str, dict]]] = [({}, {}) for _ in range(steps)]
    burn_state = None
    if burn is not None:
        step_index = int(round(burn["start_sec"] / _SHIFT_DT_SEC))
        vehicle = burn["vehicle"]
        if 0 <= step_index < steps and vehicle in fleet.vehicles:
            burn_state = {
                "before": fleet.vehicles[vehicle].spacecraft.state.propulsion.delta_v_remaining_m_s,
                "vehicle": vehicle,
            }
            commands[step_index] = (
                {vehicle: "maneuver"},
                {vehicle: {"dv_m_s": burn["dv_m_s"], "direction": burn["direction"] or "prograde"}},
            )
    conjunctions = _conjunction_rows(day)
    coast: dict[str, SimulationEngine] = {}
    if burn is not None:
        coast[burn["vehicle"]] = make_engine(
            burn["vehicle"],
            epoch=cfg["epoch"],
            nu_deg=float(cfg["nu_deg"]),
            with_stations=False,
        )
    summary = {
        "layout": "shift",
        "count": len(cfg["keys"]),
        "closed_links": 0,
        "neighbor_km": 0.0,
        "nearest_km": 0.0,
        "nearest_margin_db": 0.0,
        "minutes": steps,
        "events": events,
        "conjunctions": conjunctions,
        "burn_spent_m_s": 0.0,
        "brief": _shift_brief(events, steps, conjunctions),
    }
    return fleet, commands, summary, burn_state, _screening_companions(day), coast


@lru_cache(maxsize=4)
def fleet_episode(layout: str) -> dict:
    """Дуга для студии. Повторный запрос берёт уже посчитанные кадры."""
    summary = None
    commands = None
    burn_state = None
    companions = None
    coast = None
    if layout == "shift":
        fleet, commands, summary, burn_state, companions, coast = _shift_episode()
        neighbors: list[tuple[str, str]] = []
        steps, dt_sec = int(summary["minutes"]), _SHIFT_DT_SEC
    elif layout == "chain":
        fleet = Fleet.string_of_pearls(
            6,
            separation_deg=1.5,
            scenario=ScenarioConfig(ground_stations=list(DEFAULT_GROUND_STATIONS)),
        )
        names = list(fleet.vehicles)
        neighbors = [(names[index], names[index + 1]) for index in range(len(names) - 1)]
        steps, dt_sec = 30, 60.0
    elif layout == "star":
        fleet = Fleet.walker_star(
            66,
            6,
            2,
            a_m=_STAR_RADIUS_M + _STAR_ALTITUDE_M,
            inc_deg=_STAR_INC_DEG,
            scenario=ScenarioConfig(ground_stations=[]),
        )
        per_plane = 11
        neighbors = []
        for plane in range(6):
            slots = [f"P{plane + 1}-S{slot + 1}" for slot in range(per_plane)]
            neighbors.extend((slots[index], slots[(index + 1) % per_plane]) for index in range(per_plane))
        steps, dt_sec = 8, 180.0
    else:
        raise ValueError("схема группировки: shift, chain или star")
    if layout != "shift":
        summary = describe_fleet(6 if layout == "chain" else 66, layout, planes=6, phasing=2)
    document = export_trajectory(
        fleet, steps, dt_sec, commands=commands, companions=companions, coast=coast
    )
    if layout == "shift":
        if burn_state is not None:
            after = fleet.vehicles[burn_state["vehicle"]].spacecraft.state.propulsion.delta_v_remaining_m_s
            summary["burn_spent_m_s"] = float(burn_state["before"] - after)
        station_names = []
        for station in document["frames"][0]["stations"]:
            if station["name"] not in station_names:
                station_names.append(station["name"])
        summary["stations"] = station_names
    _mark_neighbors(document, neighbors)
    document["layout"] = layout
    document["summary"] = summary
    return document
