"""Independent orbit/site inputs and conditional fixed-ephemeris delivery evidence."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import MU_EARTH
from space_env.core.ground import GroundStation
from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.access import ReferenceEphemeris
from space_env.runtime import reference_scenario
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.scaling.contact_capacity import (
    contact_capacity,
    verify_capacity_certificate,
)
from space_env.simulation.snapshot import digest


class OrbitPlane(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    radius_m: float = Field(
        default=7_000_000.0, ge=6_700_000, le=7_500_000, strict=True
    )
    inclination_deg: float = Field(ge=0, le=180, strict=True)
    raan_deg: float = Field(ge=-360, le=360, strict=True)
    phases_deg: tuple[FiniteScalar, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def phases(self):
        if any(
            not math.isfinite(x) or not 0 <= x < 360 for x in self.phases_deg
        ) or len(set(self.phases_deg)) != len(self.phases_deg):
            raise ValueError("phases must be unique finite angles in [0,360)")
        return self


class FixedSite(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    lat_deg: float = Field(ge=-90, le=90, strict=True)
    lon_deg: float = Field(ge=-180, le=180, strict=True)
    alt_m: float = Field(default=0.0, ge=-500, le=9000, strict=True)
    min_elevation_deg: float = Field(default=5.0, ge=0, le=90, strict=True)
    terminals: int = Field(default=1, ge=1, le=4, strict=True)


class IndependentFleetSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,60}$")
    epoch_utc: str = "2026-07-18T12:00:00Z"
    planes: tuple[OrbitPlane, ...] = Field(min_length=2, max_length=4)
    stations: tuple[FixedSite, ...] = Field(min_length=1, max_length=8)
    horizon_sec: float = Field(default=6000.0, ge=60, le=7200, strict=True)
    sample_sec: float = Field(default=30.0, ge=1, le=60, strict=True)
    decision_sec: float = Field(default=60.0, ge=5, le=120, strict=True)
    product_bits: float = Field(default=20_000_000.0, ge=1, le=1e10, strict=True)
    rate_bps: float = Field(default=2_000_000.0, ge=1, le=1e9, strict=True)
    mode: Literal["nominal", "all_outage"] = "nominal"
    min_delivered: int = Field(default=1, ge=0, le=64, strict=True)

    @model_validator(mode="after")
    def bounded(self):
        Epoch.utc(self.epoch_utc)
        if (
            sum(len(p.phases_deg) for p in self.planes) > 8
            or len({p.id for p in self.planes}) != len(self.planes)
            or len({s.name for s in self.stations}) != len(self.stations)
        ):
            raise ValueError(
                "independent fleet requires unique IDs and at most 8 actors"
            )
        if self.decision_sec > self.horizon_sec or self.sample_sec > self.horizon_sec:
            raise ValueError("sampling/decision period exceeds horizon")
        if self.min_delivered > sum(len(p.phases_deg) for p in self.planes) * len(
            self.stations
        ):
            raise ValueError("delivery gate exceeds declared product count")
        normals = []
        for plane in self.planes:
            a, i = np.deg2rad([plane.raan_deg, plane.inclination_deg])
            normal = np.array(
                [math.sin(a) * math.sin(i), -math.cos(a) * math.sin(i), math.cos(i)]
            )
            if any(abs(float(normal @ other)) > 1 - 1e-10 for other in normals):
                raise ValueError("declared planes must be geometrically distinct")
            normals.append(normal)
        return self


class GeometryDeliveryGates(StrictModel):
    native_spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    min_delivered: int = Field(ge=0, le=64, strict=True)
    require_lp_success: bool = Field(default=True, strict=True)


def circular_state(plane, phase):
    a, i, f = np.deg2rad([plane.raan_deg, plane.inclination_deg, phase])
    x = np.array([math.cos(a), math.sin(a), 0.0])
    y = np.array([-math.sin(a) * math.cos(i), math.cos(a) * math.cos(i), math.sin(i)])
    return plane.radius_m * (math.cos(f) * x + math.sin(f) * y), math.sqrt(
        MU_EARTH / plane.radius_m
    ) * (-math.sin(f) * x + math.cos(f) * y)


def independent_workload(value):
    cfg = IndependentFleetSpec.model_validate(value)
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc(cfg.epoch_utc)
    ephemerides = {}
    for plane in cfg.planes:
        for j, phase in enumerate(plane.phases_deg):
            r, v = circular_state(plane, phase)
            orbit = propagate(
                ReferenceState(epoch, r, v),
                cfg.horizon_sec,
                context,
                model=ForceModel(j2=True),
                sample_sec=cfg.sample_sec,
                max_step_sec=30.0,
            )
            ephemerides[f"{plane.id}-{j:02d}"] = {**orbit, "sha256": digest(orbit)}
    actors = sorted(ephemerides)
    stations = [s.model_dump(mode="json") for s in cfg.stations]
    products = [
        {
            "id": f"p-{a}-{s.name}",
            "actor": a,
            "destination": s.name,
            "size_bits": cfg.product_bits,
            "deadline_sec": cfg.horizon_sec,
        }
        for a in actors
        for s in cfg.stations
    ]
    native = {
        "schema": "space-env.reference-downlink-spec/1",
        "ephemerides": ephemerides,
        "vehicles": {
            a: {
                "memory_bits": cfg.product_bits * (len(stations) + 1),
                "battery_wh": 100.0,
                "initial_soc": 1.0,
                "base_power_w": 1.0,
                "tx_power_w": 20.0,
                "rate_bps": cfg.rate_bps,
            }
            for a in actors
        },
        "stations": stations,
        "products": products,
        "horizon_sec": cfg.horizon_sec,
        "outages": [
            {"station": s.name, "start_sec": 0.0, "end_sec": cfg.horizon_sec}
            for s in cfg.stations
        ]
        if cfg.mode == "all_outage"
        else [],
    }
    task = {
        "id": cfg.id,
        "runtime_audit": "compact",
        "roles": {a: {"actors": [a]} for a in actors},
        "decision_sec": cfg.decision_sec,
        "max_decision_sec": cfg.decision_sec,
        "max_decisions": math.ceil(cfg.horizon_sec / cfg.decision_sec) + 1,
        "max_joint_actions": len(actors),
        "max_actions": len(products) + 1,
        "max_entities": len(products) * 3 + len(actors) + len(stations) + 32,
        "max_relations": len(products) * 16 + 128,
        "actions": [
            {
                "id": "send-" + p["id"],
                "role": p["actor"],
                "kind": "submit",
                "commands": [
                    {
                        "command_id": p["id"],
                        "actor_id": p["actor"],
                        "kind": "downlink",
                        "duration_sec": cfg.decision_sec,
                        "destination_id": p["destination"],
                        "product_id": p["id"],
                    }
                ],
            }
            for p in products
        ],
    }
    return {
        "id": cfg.id,
        "family": "independent-fixed-site-delivery/1",
        "source": "explicit circular initial planes + J2; independent synthetic fixed WGS84 sites; no orbit-derived site placement or outcome filtering",
        "scenario": reference_scenario(
            native, id=cfg.id, lineage="D2a-independent-sites"
        ),
        "task": task,
        "policy": {"name": "priority"},
        "timeout_sec": 300,
        "min_completion_fraction": 0.0,
        **({"max_completion_fraction": 0.0} if cfg.mode == "all_outage" else {}),
        "geometry_gates": {
            "native_spec_sha256": digest(native),
            "min_delivered": cfg.min_delivered,
        },
    }


def assess_geometry(scenario):
    if scenario["backend"] != "reference-fixed":
        raise ValueError("geometry assessment supports reference-fixed delivery only")
    d = scenario["definition"]
    if (
        d.get("radio_network")
        or d.get("dtn_network")
        or set(d)
        - {
            "schema",
            "ephemerides",
            "vehicles",
            "stations",
            "products",
            "outages",
            "horizon_sec",
            "resource_step_sec",
        }
    ):
        raise ValueError("unsupported geometry capacity profile")
    if any(
        p.get("processing_sec", 0)
        or p.get("ready_sec", 0)
        or p.get("depends_on")
        or not p.get("destination")
        or p["deadline_sec"] != d["horizon_sec"]
        for p in d["products"]
    ):
        raise ValueError(
            "capacity profile requires all-ready nonprocessed fixed-destination full-horizon products"
        )
    sites = {s["name"]: s for s in d["stations"]}
    windows = {}
    pairs = []
    for actor, arc in sorted(d["ephemerides"].items()):
        e = ReferenceEphemeris(arc)
        for name, site in sorted(sites.items()):
            found = e.access(GroundStation.model_validate(site))
            spans = [(w["aos_sec"], w["los_sec"]) for w in found["windows"]]
            windows[actor, name] = spans
            pairs.append({"actor": actor, "station": name, "windows": found["windows"]})
    capacity = contact_capacity(
        d["products"],
        d["vehicles"],
        sites,
        windows,
        d["horizon_sec"],
        d.get("outages", []),
    )
    result = {
        "schema": "space-env.independent-geometry-assessment/1",
        "native_spec_sha256": digest(d),
        "pairs": pairs,
        "pairs_with_contacts": sum(bool(p["windows"]) for p in pairs),
        "pairs_without_contacts": sum(not p["windows"] for p in pairs),
        "capacity": capacity,
        "scope": "diagnostic/privileged fixed-ephemeris model assessment; shared native geometric access equations, no physical calibration or actor forecast",
    }
    result["sha256"] = digest(result)
    return result


def geometry_delivery_evidence(scenario, report, gates):
    if digest(scenario["definition"]) != gates.native_spec_sha256:
        raise ValueError("geometry gate spec binding mismatch")
    assessment = assess_geometry(scenario)
    native = scenario["definition"]
    products = {p["id"]: p for p in native["products"]}
    if (
        len(report["products"]) != len(products)
        or {p["id"] for p in report["products"]} != set(products)
        or set(report["inventory_ledger"]) != set(native["vehicles"])
        or set(report["energy_ledger"]) != set(native["vehicles"])
    ):
        raise ValueError("geometry report product/ledger identity mismatch")
    windows = {
        (p["actor"], p["station"]): [(w["aos_sec"], w["los_sec"]) for w in p["windows"]]
        for p in assessment["pairs"]
    }
    receipt_checks = True
    for r in report["receipts"]:
        p = products.get(r.get("product_id"))
        if p is None or r["actor"] != p["actor"] or r["station"] != p["destination"]:
            raise ValueError("geometry receipt ownership/destination mismatch")
        lo, hi, bits = r["start_sec"], r["end_sec"], r["bits"]
        if (
            not all(
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and math.isfinite(x)
                for x in [lo, hi, bits]
            )
            or not 0 <= lo < hi <= native["horizon_sec"] + 1e-6
            or bits < 0
        ):
            raise ValueError("geometry receipt outside finite time/bit domain")
        covered = sum(
            max(0.0, min(hi, b) - max(lo, a))
            for a, b in windows[r["actor"], r["station"]]
        )
        blocked = any(
            min(hi, o["end_sec"]) - max(lo, o["start_sec"]) > 1e-6
            for o in native.get("outages", [])
            if o["station"] == r["station"]
        )
        receipt_checks &= (
            covered >= hi - lo - 1e-6
            and not blocked
            and bits <= (hi - lo) * native["vehicles"][r["actor"]]["rate_bps"] + 1e-5
        )
    product_checks = True
    for outcome in report["products"]:
        p = products[outcome["id"]]
        slips = [r for r in report["receipts"] if r["product_id"] == outcome["id"]]
        delivered_bits = math.fsum(r["bits"] for r in slips)
        product_checks &= (
            abs(delivered_bits - outcome["delivered_bits"]) <= 1e-5
            and abs(outcome["remaining_bits"] + delivered_bits - p["size_bits"]) <= 1e-5
        )
        if outcome["status"] == "delivered_on_time":
            product_checks &= (
                bool(slips)
                and outcome["remaining_bits"] == 0
                and outcome["completed_sec"] is not None
                and outcome["completed_sec"] <= p["deadline_sec"] + 1e-9
            )
    delivered = sum(p["status"] == "delivered_on_time" for p in report["products"])
    upper = assessment["capacity"]["certificate"]["integer_upper_conservative_ceiling"]
    slots = {s["name"]: s["terminals"] for s in scenario["definition"]["stations"]}
    max_excess = 0
    for name, count in slots.items():
        events = {}
        for receipt in report["receipts"]:
            if receipt["station"] == name and receipt["end_sec"] > receipt["start_sec"]:
                events[receipt["start_sec"]] = events.get(receipt["start_sec"], 0) + 1
                events[receipt["end_sec"]] = events.get(receipt["end_sec"], 0) - 1
        occupied = 0
        for _, change in sorted(events.items()):
            occupied += change
            max_excess = max(max_excess, occupied - count)
    actor_excess = 0
    for actor in native["vehicles"]:
        events = {}
        for r in report["receipts"]:
            if r["actor"] == actor:
                events[r["start_sec"]] = events.get(r["start_sec"], 0) + 1
                events[r["end_sec"]] = events.get(r["end_sec"], 0) - 1
        occupied = 0
        for _, change in sorted(events.items()):
            occupied += change
            actor_excess = max(actor_excess, occupied - 1)
    checks = {
        "minimum_delivery": delivered >= gates.min_delivered,
        "capacity_upper_bound": delivered <= upper,
        "terminal_capacity": max_excess == 0,
        "actor_single_transmitter": actor_excess == 0,
        "receipt_contact_outage_rate": bool(receipt_checks),
        "product_receipt_accounting": bool(product_checks),
        "inventory_balance": all(
            abs(v["balance_error_bits"]) <= 1e-5
            for v in report["inventory_ledger"].values()
        ),
        "energy_balance": all(
            abs(v["balance_error_wh"]) <= 1e-7 for v in report["energy_ledger"].values()
        ),
        "lp_solver": not gates.require_lp_success
        or assessment["capacity"]["solver_success"],
    }
    return {
        "schema": "space-env.independent-geometry-evidence/1",
        "passed": all(checks.values()),
        "checks": checks,
        "delivered_products": delivered,
        "total_products": len(report["products"]),
        "integer_upper_bound": upper,
        "pairs_with_contacts": assessment["pairs_with_contacts"],
        "pairs_without_contacts": assessment["pairs_without_contacts"],
        "assessment_sha256": assessment["sha256"],
        "capacity_certificate": assessment["capacity"]["certificate"],
        "max_terminal_excess": max_excess,
        "scope": "fixed-rate delivery-only model witness; not pointing/EO/RF or arbitrary mission qualification",
    }


def independent_bundle(value):
    cfg = IndependentFleetSpec.model_validate(value)
    workload = independent_workload(cfg.model_dump(mode="json"))
    result = {
        "schema": "space-env.independent-delivery-bundle/1",
        "input_spec": cfg.model_dump(mode="json"),
        "workload": workload,
        "assessment": assess_geometry(workload["scenario"]),
        "industrial_readiness": False,
    }
    result["sha256"] = digest(result)
    return result


def verify_independent_bundle(value):
    if (
        set(value)
        != {
            "schema",
            "input_spec",
            "workload",
            "assessment",
            "industrial_readiness",
            "sha256",
        }
        or value.get("schema") != "space-env.independent-delivery-bundle/1"
        or value.get("industrial_readiness") is not False
    ):
        raise ValueError("unsupported independent bundle envelope")
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("independent bundle checksum mismatch")
    expected = independent_bundle(value["input_spec"])
    if (
        digest(value["workload"]) != digest(expected["workload"])
        or value["assessment"]["pairs"] != expected["assessment"]["pairs"]
        or value["assessment"]["capacity"]["problem"]
        != expected["assessment"]["capacity"]["problem"]
    ):
        raise ValueError(
            "independent bundle disagrees with input geometry/capacity model"
        )
    for key in [
        "schema",
        "native_spec_sha256",
        "pairs_with_contacts",
        "pairs_without_contacts",
        "scope",
    ]:
        if value["assessment"][key] != expected["assessment"][key]:
            raise ValueError("independent assessment metadata disagrees with geometry")
    assessment = value["assessment"]
    if assessment.get("sha256") != digest(
        {k: v for k, v in assessment.items() if k != "sha256"}
    ):
        raise ValueError("independent assessment checksum mismatch")
    verify_capacity_certificate(assessment["capacity"])
    return {
        "verified": True,
        "orbit_geometry_recomputed": True,
        "solver_reexecuted": True,
        "native_mission_reexecuted": False,
        "scope": "compiler/model/certificate audit on this build, not a fresh mission, physical calibration or author signature",
    }
