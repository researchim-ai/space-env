"""Заявки на общем мире: одна станция в один момент, квитанция или причина отказа."""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from space_env.core.access import subsatellite_latlon, target_access
from space_env.core.environment import in_eclipse
from space_env.core.sun import sun_position_m
from space_env.mission.events import EventKind
from space_env.mission.scenario_v2 import load_scenario_v2, scenario_manifest, world_from_scenario
from space_env.mission.scenarios import GroundTarget
from space_env.simulation.commands import MissionCommand
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from space_env.simulation.world import MissionWorld

POLICIES = ("fifo", "deadline", "priority", "rolling")


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
    destination: str = ""
    quality: str = "full"
    geometry: str = "point"
    length_km: float = 0.0


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
    if policy == "rolling":
        return (left.deadline_sec - left.release_sec, -left.priority, left.request_id) < (
            right.deadline_sec - right.release_sec,
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
    if policy == "rolling":
        return (request.deadline_sec - request.release_sec, -request.priority, request.request_id, name)
    return (-request.priority, request.deadline_sec, request.request_id, name)


def _onboard_request(engine: SimulationEngine, requests: dict[str, TaskRequest]) -> TaskRequest | None:
    owner = engine.spacecraft.config.name
    found: TaskRequest | None = None
    for product in engine.products():
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


def _captured(engine: SimulationEngine, request_id: str) -> bool:
    """Кадр этой заявки уже лежит в памяти. Повторно окно не занимаем."""
    owner = engine.spacecraft.config.name
    return any(
        product.get("request_id") == request_id
        and product.get("origin") == "imaging"
        and product.get("custodian") == owner
        and float(product.get("accepted_bits", 0.0)) > 0.0
        for product in engine.products()
    )


def _opening_offset(engine: SimulationEngine, request: TaskRequest, dt: float) -> float | None:
    """Первый отсчёт, где цель видна. 0 — видна уже сейчас."""
    if _visible(engine, request):
        return 0.0
    if dt <= 0.0:
        return None
    origin = engine.now()
    for offset, position in engine.preview_positions(dt):
        if offset <= 1e-6:
            continue
        if _visible_state(engine, request, position, origin + offset):
            return float(offset)
    return None


def _prefer_window(
    engine: SimulationEngine,
    candidate: TaskRequest,
    chosen: TaskRequest,
    policy: str,
    dt: float,
) -> bool:
    """Раньше закрывающееся окно важнее приоритета. При одном моменте решает политика."""
    left = _opening_offset(engine, candidate, dt)
    right = _opening_offset(engine, chosen, dt)
    if left is None:
        return False
    if right is None or left + 1e-6 < right:
        return True
    if right + 1e-6 < left:
        return False
    return _better(candidate, chosen, policy)


def _scarce_window(engine: SimulationEngine, request: TaskRequest, dt: float) -> bool:
    """Окно видно не на всём шаге. Его нельзя откладывать ради сдачи уже снятого кадра."""
    if dt <= 0.0 or not _visible_during(engine, request, dt):
        return False
    origin = engine.now()
    flags = [
        _visible_state(engine, request, position, origin + offset)
        for offset, position in engine.preview_positions(dt)
    ]
    return bool(flags) and any(flags) and not all(flags)


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
        if item.release_sec <= now < item.deadline_sec and _payload_matches(engine, item) and not _captured(engine, item.request_id)
    ]
    names = [target.name for target in engine.scenario.ground_targets]
    if names:
        focused = [item for item in pending if item.request_id == names[-1]]
        if focused:
            pending = focused
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
        station, _margin = engine.contact_at(position, origin + offset, time_sec=engine.time_sec + offset)
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
    downlink: set[str] | None = None,
) -> list[dict]:
    """Передаёт борт соседу, который может сдать, если сам аппарат линию на Землю не откроет.

    Полудуплекс не совмещает ретрансляцию и сдачу. full_duplex на борту это разрешает.
    """
    if len(world.engines) < 2 or dt <= 0.0:
        return []
    fleet = Fleet(world.engines)
    fleet.use_reservations(world.relay_used)
    busy = set(downlink or [])
    by_id = {item.request_id: item for item in requests}
    margins: dict[frozenset[str], float] = {}
    for pair in fleet.pairs():
        if pair.earth_blocked or pair.isl_margin_db < 0.0:
            continue
        margins[frozenset((pair.leader, pair.follower))] = pair.isl_margin_db
    moved = []
    for name, engine in world.engines.items():
        held = _onboard_request(engine, by_id)
        if held is None:
            continue
        if not engine.spacecraft.config.comms.full_duplex and (name in busy or _can_dump(engine, now, outages)):
            continue
        candidates = []
        for other, destination in world.engines.items():
            if other == name:
                continue
            if other in busy and not destination.spacecraft.config.comms.full_duplex:
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
    keep_window: set[str] = set()
    for name, engine in world.engines.items():
        held = _onboard_request(engine, by_id)
        if held is not None and held.deadline_sec <= now + dt:
            continue
        if any(
            item.release_sec <= now < item.deadline_sec
            and _payload_matches(engine, item)
            and not _captured(engine, item.request_id)
            and _scarce_window(engine, item, dt)
            for item in requests
        ):
            keep_window.add(name)
    for name in winners:
        if name in keep_window:
            continue
        commands[name] = "comms"
    for name, engine in world.engines.items():
        if commands[name] != "wait":
            continue
        pending = [
            item
            for item in requests
            if item.release_sec <= now < item.deadline_sec
            and _payload_matches(engine, item)
            and not _captured(engine, item.request_id)
            and _visible_during(engine, item, dt)
        ]
        if not pending:
            continue
        scarce = [item for item in pending if _scarce_window(engine, item, dt)]
        pool = scarce or pending
        chosen = pool[0]
        for item in pool[1:]:
            if _prefer_window(engine, item, chosen, policy, dt):
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
            destination=str(item.get("destination", "")),
            quality=str(item.get("quality", "full")),
            geometry=str(item.get("geometry", "point")),
            length_km=float(item.get("length_km", 0.0)),
        )
        for item in rows
    ]


def _weight(request: TaskRequest) -> float:
    return float(request.priority) if request.priority > 0 else 1.0


def _receipts_for(receipts: list[dict], request_id: str) -> list[dict]:
    mine = [item for item in receipts if item["request_id"] == request_id]
    mine.sort(key=lambda item: (float(item["time_sec"]), float(item.get("offset_bits", 0.0))))
    return mine


def _covered_union(items: list[dict]) -> float:
    """Уникальный объём по отрезкам [offset, offset+bits). Повтор того же отрезка не считается дважды."""
    spans = []
    for item in items:
        start = float(item.get("offset_bits", 0.0))
        stop = start + float(item["bits"])
        if stop > start:
            spans.append((start, stop))
    spans.sort()
    total = 0.0
    cursor: float | None = None
    for start, stop in spans:
        if cursor is None or start >= cursor:
            total += stop - start
            cursor = stop
        elif stop > cursor:
            total += stop - cursor
            cursor = stop
    return total


def _countable_receipts(receipts: list[dict], request: TaskRequest) -> list[dict]:
    mine = _receipts_for(receipts, request.request_id)
    kept = []
    for item in mine:
        if float(item["time_sec"]) + 1e-9 < request.release_sec:
            continue
        if request.destination and str(item.get("destination", "")) != request.destination:
            continue
        if request.quality == "full" and str(item.get("quality", "full")) != "full":
            continue
        kept.append(item)
    return kept


def _single_acquisition(items: list[dict]) -> list[dict]:
    """Одна съёмка. Куски разных product_id не складываются в один снимок."""
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(str(item.get("product_id", "")), []).append(item)
    if len(groups) <= 1:
        return items
    return max(groups.values(), key=_covered_union)


def _completion_time(mine: list[dict], size_bits: float) -> float | None:
    """Самый ранний момент, когда какая-либо одна съёмка закрывает объём."""
    groups: dict[str, list[dict]] = {}
    for item in mine:
        groups.setdefault(str(item.get("product_id", "")), []).append(item)
    earliest: float | None = None
    for group in groups.values():
        accepted: list[dict] = []
        for item in group:
            accepted.append(item)
            if _covered_union(accepted) + 1e-6 >= size_bits:
                moment = float(item["time_sec"])
                if earliest is None or moment < earliest:
                    earliest = moment
                break
    return earliest


def service_level(issued: list[dict], on_time_ids: set[str]) -> dict[str, float | int]:
    """
    Доля приоритета, сданная в срок.
    Недостижимые заявки остаются в знаменателе выданного спроса и не считаются успехом съёмки.
    """

    def weight(item: dict) -> int:
        return max(int(item.get("priority", 1)), 1)

    reachable = [item for item in issued if item.get("reachable")]
    def ratio(pool: list[dict]) -> float:
        total = sum(weight(item) for item in pool)
        if total <= 0:
            return 0.0
        hit = sum(weight(item) for item in pool if str(item.get("id")) in on_time_ids)
        return hit / total

    return {
        "issued": len(issued),
        "reachable": len(reachable),
        "unreachable": len(issued) - len(reachable),
        "on_time_of_reachable": ratio(reachable),
        "on_time_of_issued": ratio(issued),
    }


def delivery_score(requests: list[TaskRequest], receipts: list[dict], unmet: list[dict]) -> dict:
    """Доля приоритета, сданная целиком и в срок. Нулевой приоритет весит как единица."""
    on_time = []
    for request in requests:
        mine = _countable_receipts(receipts, request)
        done_at = _completion_time(mine, request.size_bits)
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


def request_lifecycle(request: TaskRequest, receipts: list[dict], now_sec: float) -> str:
    """unreleased → available → in_progress → delivered или expired."""
    if now_sec + 1e-9 < request.release_sec:
        return "unreleased"
    mine = _countable_receipts(receipts, request)
    done_at = _completion_time(mine, request.size_bits)
    if done_at is not None and done_at <= request.deadline_sec + 1e-6:
        return "delivered"
    if now_sec + 1e-9 >= request.deadline_sec:
        return "expired"
    if mine:
        return "in_progress"
    return "available"


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
        mine = _countable_receipts(receipts, request)
        done_at = _completion_time(mine, request.size_bits)
        delays.append(done_at - request.release_sec)
    qualities = [
        1.0 if product.get("quality") == "full" else 0.0
        for engine in world.engines.values()
        for product in engine.products()
    ]
    fuel = 0.0
    faults = 0
    violations = 0
    for engine in world.engines.values():
        fuel += float(engine.get_state_snapshot().delta_v_used_m_s)
        if engine.faults_active():
            faults += 1
        for _time, _snap, events in engine.log:
            violations += sum(event.kind == EventKind.CONSTRAINT_VIOLATION for event in events)
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
        "safety_violations": violations,
        "active_faults": faults,
        "runtime_sec": runtime_sec,
        "replans": replans,
    }


def _unmet_rows(world: MissionWorld, requests: list[TaskRequest], blocked: set[str]) -> list[dict]:
    """Причины несданных заявок. Закрытая станция не превращается в «срок вышел»."""
    receipts = [item for engine in world.engines.values() for item in engine.receipts]
    unmet = []
    for request in requests:
        mine = _countable_receipts(receipts, request)
        delivered = _covered_union(_single_acquisition(mine))
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
    return unmet


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
    world.set_outages([(item.station, item.start_sec, item.end_sec) for item in closed])
    world.task_requests = {item.request_id: item for item in requests}
    blocked: set[str] = set()
    relays: list[dict] = []
    rejected: list[dict] = []
    schedule: list[dict] = []
    elapsed = float(world.now_sec())
    replans = 0
    started = time.perf_counter()
    while elapsed + 1e-9 < world.horizon_sec:
        dt = min(step_sec, world.horizon_sec - elapsed)
        commands, denied = _commands(world, requests, policy, elapsed, closed, dt)
        blocked.update(denied)
        dt, commands = _cut_to_event(world, requests, commands, elapsed, dt, closed)
        downlink = set()
        for name, kind in commands.items():
            engine = world.engines[name]
            if kind == "comms" and _can_dump(engine, elapsed, closed):
                downlink.add(name)
            snap = engine.get_state_snapshot()
            if engine._downlink_armed and engine.radio_ready() and snap.in_contact:
                downlink.add(name)
        moved = _relay(world, requests, elapsed, dt, closed, downlink)
        relays.extend(moved)
        batch = []
        for name, kind in commands.items():
            targets = [target.name for target in world.engines[name].scenario.ground_targets]
            request_id = targets[-1] if targets else ""
            batch.append(
                MissionCommand(
                    command_id=f"{elapsed:.3f}:{name}:{kind}:{request_id}",
                    actor_id=name,
                    kind=kind,
                    request_id=request_id,
                    destination_id=world.engines[name].get_state_snapshot().contact_station,
                )
            )
        orders = world.accept(batch)
        issued = {world.commands.get(command_id)["actor_id"]: world.commands.get(command_id) for command_id in world._batch}
        for command in batch:
            if command.actor_id not in issued:
                try:
                    issued[command.actor_id] = world.commands.get(command.command_id)
                except KeyError:
                    continue
        schedule.append(
            {
                "dt": dt,
                "orders": orders,
                "commands": issued,
                "relays": moved,
                "relay_dt": dt,
                "targets": {
                    name: [target.name for target in engine.scenario.ground_targets]
                    for name, engine in world.engines.items()
                },
                "denied": sorted(denied),
            }
        )
        world.step(dt, orders)
        schedule[-1]["rejected"] = list(getattr(world, "last_rejected", []))
        rejected.extend(schedule[-1]["rejected"])
        world.settle()
        for name, record in list(issued.items()):
            issued[name] = world.commands.get(record["command_id"])
        schedule[-1]["commands"] = issued
        elapsed = float(world.now_sec())
        replans += 1
    runtime_sec = time.perf_counter() - started
    receipts = [item for engine in world.engines.values() for item in engine.receipts]
    unmet = _unmet_rows(world, requests, blocked)
    score = delivery_score(requests, receipts, unmet)
    relay_bits = sum(float(item["bits"]) for item in relays)
    return {
        "policy": policy,
        "receipts": receipts,
        "unmet": unmet,
        "relays": relays,
        "rejected": rejected,
        "schedule": schedule,
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
        "final": _final_rows(world),
    }


def _final_rows(world: MissionWorld) -> dict:
    rows = {}
    for name, engine in world.engines.items():
        snap = engine.get_state_snapshot()
        rows[name] = {
            "time_sec": round(float(engine.time_sec), 6),
            "soc": round(float(snap.battery_soc), 6),
            "dv_remaining_m_s": round(float(snap.delta_v_remaining_m_s), 6),
            "stored_bits": round(float(snap.stored_bits), 3),
            "delivered_bits": round(float(snap.delivered_bits), 3),
            "lost_bits": round(float(snap.lost_bits), 3),
        }
    return rows


def run_scenario(
    path: str | Path,
    *,
    step_sec: float = 60.0,
    policy: str = "priority",
    outages: list[StationOutage] | None = None,
    refresh_passes: bool = True,
) -> dict:
    """Исполняет файл v2 на его горизонте. Станции файла вешаются на каждый аппарат."""
    spec = load_scenario_v2(path)
    world = world_from_scenario(path, attach_stations=True, refresh_passes=refresh_passes)
    file_outages = [
        StationOutage(item["station"], item["start_sec"], item["end_sec"]) for item in spec.get("outages", [])
    ]
    report = run_policy(
        world,
        requests_from_scenario(spec),
        step_sec=step_sec,
        policy=policy,
        outages=[*file_outages, *(outages or [])],
    )
    report["manifest"] = scenario_manifest(path)
    return report
