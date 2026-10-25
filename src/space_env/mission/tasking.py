"""Заявки на общем мире: одна станция в один момент, квитанция или причина отказа."""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from space_env.core.access import subsatellite_latlon, target_access
from space_env.core.environment import in_eclipse
from space_env.core.sun import sun_position_m
from space_env.mission.scenario_v2 import load_scenario_v2, scenario_manifest, world_from_scenario
from space_env.mission.scenarios import GroundTarget
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from space_env.simulation.world import MissionWorld

POLICIES = ("fifo", "deadline", "priority")


@dataclass(frozen=True)
class TaskRequest:
    request_id: str
    lat_deg: float
    lon_deg: float
    release_sec: float
    deadline_sec: float
    size_bits: float
    priority: int
    payload: str


@dataclass(frozen=True)
class StationOutage:
    """Станция не принимает в полуинтервале [start_sec, end_sec)."""

    station: str
    start_sec: float
    end_sec: float


def _better(left: TaskRequest, right: TaskRequest, policy: str) -> bool:
    if policy == "fifo":
        return (left.release_sec, left.request_id) < (right.release_sec, right.request_id)
    if policy == "deadline":
        return (left.deadline_sec, -left.priority, left.request_id) < (
            right.deadline_sec,
            -right.priority,
            right.request_id,
        )
    return (-left.priority, left.deadline_sec, left.request_id) < (
        -right.priority,
        right.deadline_sec,
        right.request_id,
    )


def _rank(request: TaskRequest, policy: str, name: str) -> tuple:
    if policy == "fifo":
        return (request.release_sec, request.request_id, name)
    if policy == "deadline":
        return (request.deadline_sec, -request.priority, request.request_id, name)
    return (-request.priority, request.deadline_sec, request.request_id, name)


def _onboard_request(engine: SimulationEngine, requests: dict[str, TaskRequest]) -> TaskRequest | None:
    owner = engine.spacecraft.config.name
    found: TaskRequest | None = None
    for product in engine._products:
        if product.get("custodian") != owner or product.get("delivered_to"):
            continue
        if float(product.get("onboard_bits", 0.0)) <= 0.0:
            continue
        request = requests.get(str(product["request_id"]))
        if request is None:
            continue
        if found is None or _better(request, found, "priority"):
            found = request
    return found


def _payload_matches(engine: SimulationEngine, request: TaskRequest) -> bool:
    radar = bool(engine.spacecraft.config.payload.images_in_eclipse)
    return radar if request.payload == "sar" else not radar


def _visible_state(engine: SimulationEngine, request: TaskRequest, position, moment) -> bool:
    if request.payload == "optical" and in_eclipse(position, sun_position_m(moment)):
        return False
    access = target_access(
        position,
        request.lat_deg,
        request.lon_deg,
        moment,
        half_fov_deg=engine.spacecraft.config.payload.half_fov_deg,
        frame=engine.orbit.frame,
        images_in_eclipse=engine.spacecraft.config.payload.images_in_eclipse,
    )
    return access.visible


def _visible(engine: SimulationEngine, request: TaskRequest) -> bool:
    return _visible_state(engine, request, engine.orbit.r, engine.now())


def _visible_during(engine: SimulationEngine, request: TaskRequest, dt: float) -> bool:
    """Цель видна сейчас или на отсчётах предстоящего шага. Часы не сдвигаются."""
    if _visible(engine, request):
        return True
    if dt <= 0.0:
        return False
    lat, _lon = subsatellite_latlon(engine.orbit.r, engine.now())
    reach = engine.spacecraft.config.payload.half_fov_deg + 0.08 * dt
    if abs(lat - request.lat_deg) > reach:
        return False
    origin = engine.now()
    for offset, position in engine.preview_positions(dt):
        if offset <= 1e-6:
            continue
        if _visible_state(engine, request, position, origin + offset):
            return True
    return False


def _first_visible_offset(
    engine: SimulationEngine,
    requests: list[TaskRequest],
    now: float,
    dt: float,
) -> float | None:
    """Первый отсчёт внутри шага, где цель уже видна. None, если она видна сразу или не видна."""
    pending = [
        item
        for item in requests
        if item.release_sec <= now < item.deadline_sec and _payload_matches(engine, item)
    ]
    if not pending or any(_visible(engine, item) for item in pending):
        return None
    origin = engine.now()
    for offset, position in engine.preview_positions(dt):
        if offset <= 1e-6:
            continue
        moment = origin + offset
        if any(_visible_state(engine, item, position, moment) for item in pending):
            return float(offset)
    return None


def _first_contact_offset(
    engine: SimulationEngine,
    now: float,
    dt: float,
    outages: list[StationOutage],
) -> float | None:
    """Первый отсчёт, где откроется станция. None, если линия уже есть или битов на борту нет."""
    if engine.stored_bits <= 0.0 or _can_dump(engine, now, outages):
        return None
    origin = engine.now()
    for offset, position in engine.preview_positions(dt):
        if offset <= 1e-6:
            continue
        station, _margin = engine.contact_at(position, origin + offset)
        if station and not _station_down(station, now + offset, outages):
            return float(offset)
    return None


def _cut_to_event(
    world: MissionWorld,
    requests: list[TaskRequest],
    commands: dict[str, str],
    now: float,
    dt: float,
    outages: list[StationOutage],
) -> tuple[float, dict[str, str]]:
    """Режет шаг до съёмки или открытия станции, чтобы сдача попала в тот же проход."""
    earliest = dt
    later: dict[str, float] = {}
    for name, command in commands.items():
        engine = world.engines[name]
        event = None
        if command == "imaging":
            event = _first_visible_offset(engine, requests, now, dt)
        elif command == "wait":
            event = _first_contact_offset(engine, now, dt, outages)
        if event is None:
            continue
        later[name] = event
        if event + 1e-6 < earliest:
            earliest = event
    if earliest >= dt - 1e-6:
        return dt, commands
    adjusted = dict(commands)
    for name, event in later.items():
        if event > earliest + 1e-6:
            adjusted[name] = "wait"
    return earliest, adjusted


def _ensure_target(engine: SimulationEngine, request: TaskRequest) -> None:
    if any(target.name == request.request_id for target in engine.scenario.ground_targets):
        return
    engine.scenario.ground_targets.append(
        GroundTarget(name=request.request_id, lat_deg=request.lat_deg, lon_deg=request.lon_deg)
    )


def _focus_target(engine: SimulationEngine, request: TaskRequest, request_ids: set[str]) -> None:
    """В этом шаге снимается выбранная заявка. Остальные имена заявок из списка целей уходят."""
    engine.scenario.ground_targets = [
        target for target in engine.scenario.ground_targets if target.name not in request_ids
    ]
    _ensure_target(engine, request)


def _station_down(station: str, now: float, outages: list[StationOutage]) -> bool:
    return any(item.station == station and item.start_sec <= now < item.end_sec for item in outages)


def _can_dump(engine: SimulationEngine, now: float, outages: list[StationOutage]) -> bool:
    snap = engine.get_state_snapshot()
    return bool(
        engine.radio_ready()
        and snap.in_contact
        and snap.contact_station
        and not _station_down(snap.contact_station, now, outages)
    )


def _relay(
    world: MissionWorld,
    requests: list[TaskRequest],
    now: float,
    dt: float,
    outages: list[StationOutage],
) -> list[dict]:
    """Передаёт борт соседу, который может сдать, если сам аппарат линию на Землю не откроет."""
    if len(world.engines) < 2 or dt <= 0.0:
        return []
    fleet = Fleet(world.engines)
    by_id = {item.request_id: item for item in requests}
    margins: dict[frozenset[str], float] = {}
    for pair in fleet.pairs():
        if pair.earth_blocked or pair.isl_margin_db < 0.0:
            continue
        margins[frozenset((pair.leader, pair.follower))] = pair.isl_margin_db
    moved = []
    for name, engine in world.engines.items():
        held = _onboard_request(engine, by_id)
        if held is None or _can_dump(engine, now, outages):
            continue
        candidates = []
        for other, destination in world.engines.items():
            if other == name:
                continue
            margin = margins.get(frozenset((name, other)))
            if margin is None or not _can_dump(destination, now, outages):
                continue
            candidates.append((margin, other))
        if not candidates:
            continue
        neighbor = max(candidates)[1]
        bits_mb = fleet.relay(name, neighbor, dt)
        if bits_mb > 0.0:
            moved.append(
                {
                    "request_id": held.request_id,
                    "source": name,
                    "destination": neighbor,
                    "bits": bits_mb * 1.0e6,
                }
            )
    return moved


def _commands(
    world: MissionWorld,
    requests: list[TaskRequest],
    policy: str,
    now: float,
    outages: list[StationOutage],
    dt: float,
) -> tuple[dict[str, str], set[str]]:
    blocked: set[str] = set()
    by_id = {item.request_id: item for item in requests}
    request_ids = set(by_id)
    commands = {name: "wait" for name in world.engines}
    claimants: dict[str, list[tuple[tuple, str]]] = {}
    for name, engine in world.engines.items():
        snap = engine.get_state_snapshot()
        held = _onboard_request(engine, by_id)
        if snap.in_contact and snap.contact_station and held is not None:
            if _station_down(snap.contact_station, now, outages):
                blocked.add(held.request_id)
            else:
                claimants.setdefault(snap.contact_station, []).append((_rank(held, policy, name), name))
    winners = set()
    for group in claimants.values():
        winners.add(min(group)[1])
    for name in winners:
        commands[name] = "comms"
    for name, engine in world.engines.items():
        if commands[name] != "wait":
            continue
        pending = [
            item
            for item in requests
            if item.release_sec <= now < item.deadline_sec and _payload_matches(engine, item) and _visible_during(engine, item, dt)
        ]
        if not pending:
            continue
        chosen = pending[0]
        for item in pending[1:]:
            if _better(item, chosen, policy):
                chosen = item
        _focus_target(engine, chosen, request_ids)
        commands[name] = "imaging"
    return commands, blocked


def requests_from_scenario(spec: dict) -> list[TaskRequest]:
    """Заявки файла v2 становятся теми же объектами, которые ведёт политика."""
    rows = spec.get("requests")
    if not isinstance(rows, list):
        raise ValueError("в сценарии нет списка заявок")
    return [
        TaskRequest(
            request_id=str(item["id"]),
            lat_deg=float(item["lat_deg"]),
            lon_deg=float(item["lon_deg"]),
            release_sec=float(item["release_sec"]),
            deadline_sec=float(item["deadline_sec"]),
            size_bits=float(item["size_bits"]),
            priority=int(item["priority"]),
            payload=str(item["payload"]),
        )
        for item in rows
    ]


def _weight(request: TaskRequest) -> float:
    return float(request.priority) if request.priority > 0 else 1.0


def _receipts_for(receipts: list[dict], request_id: str) -> list[dict]:
    mine = [item for item in receipts if item["request_id"] == request_id]
    mine.sort(key=lambda item: (float(item["time_sec"]), float(item.get("offset_bits", 0.0))))
    return mine


def _completion_time(mine: list[dict], size_bits: float) -> float | None:
    """Момент, когда сумма квитанций впервые закрывает объём заявки."""
    total = 0.0
    for item in mine:
        total += float(item["bits"])
        if total + 1e-6 >= size_bits:
            return float(item["time_sec"])
    return None


def delivery_score(requests: list[TaskRequest], receipts: list[dict], unmet: list[dict]) -> dict:
    """Доля приоритета, сданная целиком и в срок. Нулевой приоритет весит как единица."""
    on_time = []
    for request in requests:
        done_at = _completion_time(_receipts_for(receipts, request.request_id), request.size_bits)
        if done_at is not None and done_at <= request.deadline_sec + 1e-6:
            on_time.append(request.request_id)
    total = sum(_weight(item) for item in requests)
    hit = sum(_weight(item) for item in requests if item.request_id in set(on_time))
    reasons: dict[str, int] = {}
    for item in unmet:
        reason = str(item["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "on_time": on_time,
        "weighted_on_time": (hit / total) if total else 0.0,
        "unmet_by_reason": reasons,
    }


@dataclass(frozen=True)
class SlotOffer:
    """Одно допустимое назначение аппарата на слот станции. Вес — приоритет заявки."""

    vehicle: str
    slot: int
    weight: float


def exact_slot_plan(offers: list[SlotOffer]) -> list[SlotOffer]:
    """Перебор назначений. Один аппарат и один слот занимаются не больше одного раза.

    Размер ограничен: это эталон для проверяемого экземпляра, не решатель на 72 часа.
    """
    if not offers:
        return []
    slots = sorted({item.slot for item in offers})
    vehicles = sorted({item.vehicle for item in offers})
    if len(slots) * len(vehicles) > 12:
        raise ValueError("точный перебор только для проверяемого размера")
    table: dict[tuple[str, int], SlotOffer] = {}
    for item in offers:
        key = (item.vehicle, item.slot)
        current = table.get(key)
        if current is None or item.weight > current.weight:
            table[key] = item
    best: list[SlotOffer] = []
    best_weight = -1.0
    columns = []
    for slot in slots:
        columns.append([None, *[vehicle for vehicle in vehicles if (vehicle, slot) in table]])
    for combo in product(*columns):
        chosen_vehicles = [vehicle for vehicle in combo if vehicle is not None]
        if len(chosen_vehicles) != len(set(chosen_vehicles)):
            continue
        chosen = [table[(vehicle, slots[index])] for index, vehicle in enumerate(combo) if vehicle is not None]
        weight = sum(item.weight for item in chosen)
        if weight > best_weight:
            best_weight = weight
            best = chosen
    return best


def mission_kpis(
    world: MissionWorld,
    requests: list[TaskRequest],
    receipts: list[dict],
    score: dict,
    *,
    runtime_sec: float,
    replans: int,
    relay_bits: float,
) -> dict:
    """Срок, качество, топливо, энергия прибора, сданные биты, отказы и время счёта."""
    delays = []
    for request in requests:
        if request.request_id not in score["on_time"]:
            continue
        first = min(float(item["time_sec"]) for item in receipts if item["request_id"] == request.request_id)
        delays.append(first - request.release_sec)
    qualities = [
        1.0 if product.get("quality") == "full" else 0.0
        for engine in world.engines.values()
        for product in engine._products
    ]
    fuel = 0.0
    faults = 0
    for engine in world.engines.values():
        fuel += float(engine.get_state_snapshot().delta_v_used_m_s)
        if engine.faults_active():
            faults += 1
    energy = sum(float(engine._payload_energy_j) for engine in world.engines.values())
    ground = sum(float(engine.delivered_bits) for engine in world.engines.values())
    return {
        "weighted_on_time": score["weighted_on_time"],
        "mean_delay_sec": (sum(delays) / len(delays)) if delays else None,
        "quality_full_fraction": (sum(qualities) / len(qualities)) if qualities else None,
        "fuel_m_s": fuel,
        "payload_energy_j": energy,
        "ground_bits": ground,
        "relay_bits": relay_bits,
        "safety_violations": 0,
        "active_faults": faults,
        "runtime_sec": runtime_sec,
        "replans": replans,
    }


def run_policy(
    world: MissionWorld,
    requests: list[TaskRequest],
    *,
    step_sec: float = 10.0,
    policy: str = "priority",
    outages: list[StationOutage] | None = None,
) -> dict:
    """Ведёт заявки до горизонта. Каждый шаг — новый план: съёмка, ретрансляция или сдача."""
    if policy not in POLICIES:
        raise ValueError(f"политика {policy} не поддерживается")
    if step_sec <= 0.0:
        raise ValueError("шаг должен быть положительным")
    closed = list(outages or [])
    blocked: set[str] = set()
    relays: list[dict] = []
    elapsed = 0.0
    replans = 0
    started = time.perf_counter()
    while elapsed + 1e-9 < world.horizon_sec:
        dt = min(step_sec, world.horizon_sec - elapsed)
        relays.extend(_relay(world, requests, elapsed, dt, closed))
        commands, denied = _commands(world, requests, policy, elapsed, closed, dt)
        blocked.update(denied)
        dt, commands = _cut_to_event(world, requests, commands, elapsed, dt, closed)
        world.step(dt, commands)
        elapsed += dt
        replans += 1
    runtime_sec = time.perf_counter() - started
    receipts = [
        item
        for engine in world.engines.values()
        for item in engine.receipts
    ]
    unmet = []
    for request in requests:
        mine = _receipts_for(receipts, request.request_id)
        delivered = sum(float(item["bits"]) for item in mine)
        done_at = _completion_time(mine, request.size_bits)
        if done_at is not None and done_at <= request.deadline_sec + 1e-6:
            continue
        if request.request_id in blocked and delivered <= 1e-6:
            reason = "станция недоступна"
        elif done_at is not None or world.horizon_sec + 1e-6 >= request.deadline_sec:
            reason = "срок вышел"
        else:
            reason = "не сдано до горизонта"
        unmet.append({"request_id": request.request_id, "reason": reason, "delivered_bits": delivered})
    score = delivery_score(requests, receipts, unmet)
    relay_bits = sum(float(item["bits"]) for item in relays)
    return {
        "policy": policy,
        "receipts": receipts,
        "unmet": unmet,
        "relays": relays,
        "score": score,
        "kpis": mission_kpis(
            world,
            requests,
            receipts,
            score,
            runtime_sec=runtime_sec,
            replans=replans,
            relay_bits=relay_bits,
        ),
    }


def run_scenario(
    path: str | Path,
    *,
    step_sec: float = 60.0,
    policy: str = "priority",
    outages: list[StationOutage] | None = None,
) -> dict:
    """Исполняет файл v2 на его горизонте. Станции файла вешаются на каждый аппарат."""
    spec = load_scenario_v2(path)
    world = world_from_scenario(path, attach_stations=True)
    report = run_policy(
        world,
        requests_from_scenario(spec),
        step_sec=step_sec,
        policy=policy,
        outages=outages,
    )
    report["manifest"] = scenario_manifest(path)
    return report
