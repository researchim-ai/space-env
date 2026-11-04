"""Operator-authorized value-only forecasts and bounded fixed-delivery scheduling.

No native rollout, admission oracle, RNG, truth report or ephemeris object enters
the scheduler. This first profile handles preloaded nonprocessed data and ideal
constant-rate radio; acquisition/ADCS and stochastic RF are explicitly excluded.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass

from pydantic import Field

from space_env.core.ground import GroundStation
from space_env.planning.mission_bundle import strict_json
from space_env.planning.mission_plan import make_mission_plan
from space_env.reference.operations import OnboardProduct, VehicleResources
from space_env.runtime import MissionRuntime
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec


class DeliverySchedulerSettings(StrictModel):
    edge_guard_sec: float = Field(default=0.002, ge=0, le=1, strict=True)
    max_slot_products: int = Field(default=640000, ge=1, le=640000, strict=True)
    max_window_checks: int = Field(default=2000000, ge=1, le=2000000, strict=True)


class _Vehicle(StrictModel):
    rate_bps: float = Field(gt=0, strict=True)
    battery_wh: float = Field(gt=0, strict=True)
    initial_soc: float = Field(ge=0, le=1, strict=True)
    min_tx_soc: float = Field(ge=0, lt=1, strict=True)
    base_power_w: float = Field(ge=0, strict=True)
    tx_power_w: float = Field(ge=0, strict=True)
    max_tx_sec: float | None = Field(ge=0, strict=True)


class _Product(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=128)
    station: str = Field(min_length=1, max_length=128)
    size_bits: float = Field(ge=1, strict=True)
    ready_sec: float = Field(ge=0, strict=True)
    deadline_sec: float = Field(gt=0, strict=True)
    priority: int = Field(ge=0, strict=True)
    action_id: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True)
class DeliveryForecast:
    payload: str

    def __post_init__(self):
        value = strict_json(self.payload)
        _validate_view(value)
        object.__setattr__(self, "payload", canonical_json(value))

    def view(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return digest(self.view())


def _profile(scenario, task):
    if scenario.get("backend") != "reference-fixed":
        raise ValueError("forecast scheduler supports reference-fixed delivery only")
    d = scenario["definition"]
    t = TaskSpec.model_validate(task)
    allowed = {
        "schema",
        "ephemerides",
        "vehicles",
        "stations",
        "products",
        "outages",
        "horizon_sec",
        "resource_step_sec",
    }
    if set(d) - allowed or d.get("schema") != "space-env.reference-downlink-spec/1":
        raise ValueError("unsupported forecast delivery physics/profile fields")
    if (
        not 1 <= len(d["vehicles"]) <= 8
        or not 1 <= len(d["products"]) <= 64
        or not 1 <= len(d["stations"]) <= 8
    ):
        raise ValueError("forecast delivery supports 8 actors/sites and 64 products")
    h = float(d["horizon_sec"])
    q = t.decision_sec
    if (
        math.ceil(h / q) > min(10000, t.max_decisions)
        or abs(h / q - round(h / q)) > 1e-8
    ):
        raise ValueError("forecast delivery requires bounded whole decision cells")
    vehicles = {a: VehicleResources.model_validate(v) for a, v in d["vehicles"].items()}
    products = {p["id"]: OnboardProduct.model_validate(p) for p in d["products"]}
    stations = {s["name"]: GroundStation.model_validate(s) for s in d["stations"]}
    if len(products) != len(d["products"]) or len(stations) != len(d["stations"]):
        raise ValueError("duplicate product/station identifiers")
    if any(
        v.thermal is not None or v.solar_mode != "ideal_sun_tracking"
        for v in vehicles.values()
    ):
        raise ValueError(
            "forecast delivery excludes thermal and body-fixed attitude models"
        )
    if any(s.share_bandwidth or s.band_hz for s in stations.values()):
        raise ValueError("forecast delivery requires ideal per-terminal radio")
    if (
        set(d["ephemerides"]) != set(vehicles)
        or any(
            p.actor not in vehicles
            or p.destination not in stations
            or p.processing_sec
            or p.depends_on
            for p in products.values()
        )
        or any(
            p.deadline_sec > h or p.ready_sec >= p.deadline_sec
            for p in products.values()
        )
    ):
        raise ValueError(
            "forecast delivery requires owned, fixed-destination nonprocessed products"
        )
    if any(
        sum(p.size_bits for p in products.values() if p.actor == a) > v.memory_bits
        for a, v in vehicles.items()
    ):
        raise ValueError("forecast initial memory budget exceeded")
    actions = {}
    for action in t.actions:
        if action.kind != "submit" or len(action.commands) != 1:
            raise ValueError(
                "forecast delivery requires one downlink per catalogue action"
            )
        c = action.commands[0]
        p = products.get(c.product_id)
        if (
            c.kind != "downlink"
            or p is None
            or c.actor_id != p.actor
            or c.destination_id != p.destination
            or c.duration_sec != q
            or action.role not in t.roles
            or t.roles[action.role].actors != (p.actor,)
        ):
            raise ValueError(
                "forecast task template differs from declared product/rate cell"
            )
        if p.id in actions:
            raise ValueError("duplicate downlink template for product")
        actions[p.id] = action.id
    if set(actions) != set(products) or t.max_joint_actions < len(vehicles):
        raise ValueError("forecast catalogue must cover every product and actor")
    return d, t, vehicles, products, stations, actions


def collect_delivery_forecast(runtime, scenario, task):
    """Trusted bridge with explicit forecast permission; never grants it itself."""
    d, t, vehicles, products, stations, actions = _profile(scenario, task)
    if not isinstance(runtime, MissionRuntime) or not runtime.access.forecast:
        raise PermissionError("operator-authorized forecast access is required")
    if (
        runtime.revision != 0
        or runtime.time_sec != 0
        or canonical_json(runtime._original) != canonical_json(scenario)
    ):
        raise ValueError("forecast snapshot must match the unchanged initial scenario")
    pairs = []
    for actor in sorted(vehicles):
        for station in sorted(stations):
            result = runtime.forecast(actor, station)
            pairs.append(
                {
                    "actor": actor,
                    "station": station,
                    "binding": copy.deepcopy(result["binding"]),
                    "windows": [
                        {
                            "start_sec": float(w["start_sec"]),
                            "end_sec": float(w["end_sec"]),
                        }
                        for w in result["windows"]
                    ],
                }
            )
    body = {
        "schema": "space-env.delivery-forecast-view/1",
        "scenario_sha256": digest(scenario),
        "task_sha256": digest(t.model_dump(mode="json", by_alias=True)),
        "horizon_sec": float(d["horizon_sec"]),
        "decision_sec": t.decision_sec,
        "vehicles": {
            a: {
                "rate_bps": v.rate_bps,
                "battery_wh": v.battery_wh,
                "initial_soc": v.initial_soc,
                "min_tx_soc": v.min_tx_soc,
                "base_power_w": v.base_power_w,
                "tx_power_w": v.tx_power_w,
                "max_tx_sec": v.max_tx_sec,
            }
            for a, v in vehicles.items()
        },
        "stations": {s: stations[s].terminals for s in stations},
        "products": [
            {
                "id": p.id,
                "actor": p.actor,
                "station": p.destination,
                "size_bits": p.size_bits,
                "ready_sec": p.ready_sec,
                "deadline_sec": p.deadline_sec,
                "priority": p.priority,
                "action_id": actions[p.id],
            }
            for p in products.values()
        ],
        "pairs": pairs,
        "scope": "operator-authorized initial geometry/known outages; ideal constant rate; no future operator events, energy forecast or stochastic RF",
    }
    return DeliveryForecast(canonical_json(body))


def _validate_view(value):
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "scenario_sha256",
            "task_sha256",
            "horizon_sec",
            "decision_sec",
            "vehicles",
            "stations",
            "products",
            "pairs",
            "scope",
        }
        or value["schema"] != "space-env.delivery-forecast-view/1"
    ):
        raise ValueError("invalid delivery forecast view")
    canonical_json(value)
    if any(
        not isinstance(value[k], str) or re.fullmatch(r"[a-f0-9]{64}", value[k]) is None
        for k in ("scenario_sha256", "task_sha256")
    ):
        raise ValueError("invalid forecast scenario/task checksums")
    if (
        not isinstance(value["horizon_sec"], (int, float))
        or isinstance(value["horizon_sec"], bool)
        or not 0 < value["horizon_sec"] <= 259200
        or not isinstance(value["decision_sec"], (int, float))
        or isinstance(value["decision_sec"], bool)
        or not 1e-9 < value["decision_sec"] <= 3600
    ):
        raise ValueError("invalid forecast horizon/decision interval")
    if (
        not 1 <= len(value["vehicles"]) <= 8
        or not 1 <= len(value["stations"]) <= 8
        or not 1 <= len(value["products"]) <= 64
    ):
        raise ValueError("delivery forecast view budget exceeded")
    for vehicle in value["vehicles"].values():
        _Vehicle.model_validate(vehicle)
    if any(type(c) is not int or not 1 <= c <= 4 for c in value["stations"].values()):
        raise ValueError("invalid forecast terminal capacity")
    products = [_Product.model_validate(p) for p in value["products"]]
    if (
        len({p.id for p in products}) != len(products)
        or len({p.action_id for p in products}) != len(products)
        or any(
            p.actor not in value["vehicles"]
            or p.station not in value["stations"]
            or not p.ready_sec < p.deadline_sec <= value["horizon_sec"]
            for p in products
        )
    ):
        raise ValueError("invalid forecast product ownership/deadline/catalogue")
    expected = {(a, s) for a in value["vehicles"] for s in value["stations"]}
    if (
        len(value["pairs"]) != len(expected)
        or {(p["actor"], p["station"]) for p in value["pairs"]} != expected
    ):
        raise ValueError("incomplete delivery forecast pair grid")
    scenario_bindings = set()
    trajectory_bindings = {}
    for pair in value["pairs"]:
        if (
            set(pair) != {"actor", "station", "binding", "windows"}
            or len(pair["windows"]) > 128
        ):
            raise ValueError("invalid forecast pair fields/window budget")
        binding = pair["binding"]
        if (
            set(binding)
            != {
                "scenario_sha256",
                "trajectory_sha256",
                "actor_id",
                "revision",
                "geometry_revision",
            }
            or binding["actor_id"] != pair["actor"]
            or type(binding["revision"]) is not int
            or binding["revision"] != 0
            or type(binding["geometry_revision"]) is not int
            or binding["geometry_revision"] != 0
        ):
            raise ValueError("invalid forecast initial binding")
        if any(
            not isinstance(binding[k], str)
            or re.fullmatch(r"[a-f0-9]{64}", binding[k]) is None
            for k in ("scenario_sha256", "trajectory_sha256")
        ):
            raise ValueError("invalid forecast binding checksums")
        scenario_bindings.add(binding["scenario_sha256"])
        previous = trajectory_bindings.setdefault(
            pair["actor"], binding["trajectory_sha256"]
        )
        if len(scenario_bindings) != 1 or previous != binding["trajectory_sha256"]:
            raise ValueError("inconsistent forecast pair bindings")
        last = 0.0
        for w in pair["windows"]:
            if (
                set(w) != {"start_sec", "end_sec"}
                or any(type(w[k]) not in (int, float) for k in ("start_sec", "end_sec"))
                or not last <= w["start_sec"] < w["end_sec"] <= value["horizon_sec"]
            ):
                raise ValueError("invalid/overlapping delivery forecast windows")
            last = w["end_sec"]


def schedule_delivery(view, settings=None):
    """Deterministic urgency/shortest-remaining greedy packing; no native objects."""
    if not isinstance(view, DeliveryForecast):
        raise TypeError("value-only DeliveryForecast required")
    cfg = DeliverySchedulerSettings.model_validate(settings or {})
    v = view.view()
    q = v["decision_sec"]
    n = round(v["horizon_sec"] / q)
    if (
        not 1 <= n <= 10000
        or abs(n * q - v["horizon_sec"]) > 1e-8
        or n * len(v["products"]) > cfg.max_slot_products
    ):
        raise ValueError("forecast scheduler slot-product budget exceeded")
    windows = {(p["actor"], p["station"]): p["windows"] for p in v["pairs"]}
    if (
        n * sum(len(windows[p["actor"], p["station"]]) for p in v["products"])
        > cfg.max_window_checks
    ):
        raise ValueError("forecast scheduler window-check budget exceeded")
    eligible = {}
    remaining = {p["id"]: p["size_bits"] for p in v["products"]}
    used_tx = dict.fromkeys(v["vehicles"], 0.0)
    for p in v["products"]:
        vehicle = v["vehicles"][p["actor"]]
        duration = min(q, p["size_bits"] / vehicle["rate_bps"])
        eligible[p["id"]] = [
            i
            for i in range(n)
            if p["ready_sec"] <= i * q
            and i * q + duration <= p["deadline_sec"]
            and any(
                w["start_sec"] + (cfg.edge_guard_sec if w["start_sec"] > 0 else 0)
                <= i * q
                and i * q + duration
                <= w["end_sec"]
                - (cfg.edge_guard_sec if w["end_sec"] < v["horizon_sec"] else 0)
                for w in windows[p["actor"], p["station"]]
            )
        ]
    suffix = {p["id"]: len(eligible[p["id"]]) for p in v["products"]}
    sets = {k: set(values) for k, values in eligible.items()}
    steps = []
    allocations = []
    for i in range(n):
        candidates = []
        for p in v["products"]:
            id = p["id"]
            vehicle = v["vehicles"][p["actor"]]
            if i not in sets[id]:
                continue
            if remaining[id] > 1e-6:
                needed = math.ceil(remaining[id] / (q * vehicle["rate_bps"]))
                if suffix[id] >= needed:
                    candidates.append(
                        (
                            suffix[id] - needed,
                            p["deadline_sec"],
                            -p["priority"],
                            remaining[id] / vehicle["rate_bps"],
                            id,
                            p,
                        )
                    )
            suffix[id] -= 1
        actors = set()
        occupied = dict.fromkeys(v["stations"], 0)
        chosen = []
        for *_, p in sorted(candidates, key=lambda row: row[:-1]):
            a, s, id = p["actor"], p["station"], p["id"]
            vehicle = v["vehicles"][a]
            duration = min(q, remaining[id] / vehicle["rate_bps"])
            energy = (
                v["horizon_sec"] * vehicle["base_power_w"]
                + (used_tx[a] + duration) * vehicle["tx_power_w"]
            ) / 3600.0
            if (
                a in actors
                or occupied[s] >= v["stations"][s]
                or vehicle["max_tx_sec"] is not None
                and used_tx[a] + duration > vehicle["max_tx_sec"]
                or vehicle["initial_soc"] * vehicle["battery_wh"] - energy
                < vehicle["min_tx_soc"] * vehicle["battery_wh"] + 1e-10
            ):
                continue
            actors.add(a)
            occupied[s] += 1
            chosen.append(p["action_id"])
            used_tx[a] += duration
            bits = min(remaining[id], q * vehicle["rate_bps"])
            remaining[id] = max(0.0, remaining[id] - bits)
            allocations.append(
                {
                    "step": i,
                    "product": id,
                    "actor": a,
                    "station": s,
                    "predicted_bits": bits,
                }
            )
        steps.append(
            {
                "at_sec": i * q,
                "advance_sec": q,
                "action_ids": sorted(chosen) or ["wait"],
            }
        )
    body = {
        "schema": "space-env.forecast-delivery-schedule/1",
        "forecast_sha256": view.sha256,
        "settings": cfg.model_dump(mode="json"),
        "steps": steps,
        "allocations": allocations,
        "predicted_fulfilled": sum(r <= 1e-6 for r in remaining.values()),
        "total": len(remaining),
        "remaining_bits": remaining,
        "reserved_tx_sec": used_tx,
        "unserved_reasons": {
            p["id"]: (
                "no_forecast_cell"
                if not eligible[p["id"]]
                else "insufficient_grid_capacity"
                if len(eligible[p["id"]]) * q * v["vehicles"][p["actor"]]["rate_bps"]
                < p["size_bits"]
                else "contention_or_conservative_resources"
            )
            for p in v["products"]
            if remaining[p["id"]] > 1e-6
        },
        "scope": "grid-conservative greedy proposal; energy debits ignore PV; no native feasibility/optimality claim",
    }
    return body | {"sha256": digest(body)}


def forecast_mission_plan(scenario, task, view, *, goals=None, settings=None):
    _profile(scenario, task)
    v = view.view()
    if v["scenario_sha256"] != digest(scenario) or v["task_sha256"] != digest(
        TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    ):
        raise ValueError("forecast view scenario/task binding mismatch")
    schedule = schedule_delivery(view, settings)
    plan = make_mission_plan(scenario, task, schedule["steps"], goals=goals)
    return {"plan": plan, "schedule": schedule, "forecast": v}
