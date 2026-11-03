"""Bounded lumped thermal network: symmetric conduction, bath and radiation.

SI inputs, constant source powers during an execution interval. This is not FEA
or calibrated hardware; environment and electrical routing are separate adapters.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.reference.materials import MaterialHeatCapacity
from space_env.runtime.schema import FiniteScalar, StrictModel

SIGMA_SB = 5.670374419e-8
HEAT_SOURCES = frozenset(
    {
        "base",
        "cpu",
        "payload",
        "radio",
        "adcs",
        "propulsion",
        "conversion",
        "pv_extraction",
    }
)


class ThermalNode(StrictModel):
    initial_k: float = Field(gt=0, le=3000, strict=True)
    capacity_j_per_k: float | None = Field(default=None, gt=0, le=1e12, strict=True)
    material_capacity: MaterialHeatCapacity | None = None
    trip_k: float = Field(gt=0, le=3000, strict=True)
    resume_k: float = Field(gt=0, le=3000, strict=True)
    cold_trip_k: float = Field(default=1.0, gt=0, strict=True)
    cold_resume_k: float = Field(default=2.0, gt=0, strict=True)
    bath_k: float = Field(default=280.0, gt=0, le=3000, strict=True)
    bath_conductance_w_per_k: float = Field(default=0.0, ge=0, le=1e9, strict=True)
    emission_area_m2: float = Field(default=0.0, ge=0, le=1e4, strict=True)
    emissivity: float = Field(default=0.0, ge=0, le=1, strict=True)
    solar_area_m2: float = Field(default=0.0, ge=0, le=1e4, strict=True)
    absorptivity: float = Field(default=0.0, ge=0, le=1, strict=True)
    normal_body: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (0.0, 0.0, 1.0)
    heater_power_w: float = Field(default=0.0, ge=0, le=1e6, strict=True)
    heater_on_below_k: FiniteScalar | None = None
    heater_off_above_k: FiniteScalar | None = None
    heater_priority: int = Field(default=0, ge=0, le=16, strict=True)

    @model_validator(mode="after")
    def physical(self):
        if (self.capacity_j_per_k is None) == (self.material_capacity is None):
            raise ValueError("node requires exactly one constant or material heat capacity")
        if self.material_capacity is not None:
            self.material_capacity.capacity(self.initial_k)
        if not self.cold_trip_k < self.cold_resume_k < self.resume_k < self.trip_k:
            raise ValueError(
                "node limits require cold trip < cold resume < hot resume < hot trip"
            )
        if not math.isclose(
            math.hypot(*self.normal_body), 1.0, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("thermal surface normal must be unit BODY vector")
        if self.heater_power_w:
            if (
                self.heater_on_below_k is None
                or self.heater_off_above_k is None
                or not self.cold_resume_k
                <= self.heater_on_below_k
                < self.heater_off_above_k
                < self.resume_k
            ):
                raise ValueError("heater requires cold-resume <= on < off < hot-resume")
        elif self.heater_on_below_k is not None or self.heater_off_above_k is not None:
            raise ValueError("heater thresholds require nonzero heater power")
        return self

    def capacity(self, temperature_k):
        if self.material_capacity is not None:
            return self.material_capacity.capacity(temperature_k)
        return self.capacity_j_per_k

    def stored(self, start_k, end_k):
        if self.material_capacity is not None:
            return self.material_capacity.stored(start_k, end_k)
        return self.capacity_j_per_k * (end_k - start_k)


class ThermalLink(StrictModel):
    a: str
    b: str
    conductance_w_per_k: float = Field(gt=0, le=1e9, strict=True)


class ThermalNetworkSettings(StrictModel):
    solver: Literal["Radau", "DOP853"] = "Radau"
    model_id: str = Field(min_length=1)
    parameter_status: Literal["synthetic", "published", "calibrated"]
    source: str = Field(min_length=1)
    nodes: dict[str, ThermalNode] = Field(min_length=1, max_length=16)
    links: tuple[ThermalLink, ...] = Field(default=(), max_length=120)
    routing: dict[str, dict[str, FiniteScalar]]
    earth_environment: bool = Field(default=True, strict=True)
    earth_ir_w_m2: float = Field(default=237.0, ge=0, le=1000, strict=True)
    earth_geometric_albedo: float = Field(default=0.3, ge=0, le=2 / 3, strict=True)
    space_background_k: float = Field(default=2.725, ge=0, le=1000, strict=True)
    environment_step_sec: float = Field(default=0.1, ge=0.01, le=1, strict=True)
    max_step_sec: float = Field(default=0.05, gt=0, le=1, strict=True)
    rtol: float = Field(default=1e-9, ge=1e-12, le=1e-5, strict=True)
    atol_k: float = Field(default=1e-7, ge=1e-12, le=0.01, strict=True)
    energy_atol_j: float = Field(default=1e-7, ge=1e-12, le=0.01, strict=True)
    max_evaluations: int = Field(default=20000, ge=1, le=1000000, strict=True)
    max_total_evaluations: int = Field(default=1000000, ge=1, le=10000000, strict=True)
    max_events: int = Field(default=10000, ge=1, le=100000, strict=True)
    heater_min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)

    @model_validator(mode="after")
    def topology(self):
        if (
            not self.model_id.strip()
            or not self.source.strip()
            or any(not n.strip() for n in self.nodes)
        ):
            raise ValueError("thermal model requires nonblank ID/source/node names")
        seen = set()
        for edge in self.links:
            key = tuple(sorted((edge.a, edge.b)))
            if (
                edge.a == edge.b
                or edge.a not in self.nodes
                or edge.b not in self.nodes
                or key in seen
            ):
                raise ValueError("thermal links must join distinct known nodes once")
            seen.add(key)
        if set(self.routing) != HEAT_SOURCES:
            raise ValueError(
                "thermal routing must explicitly cover all heat sources and PV extraction"
            )
        for route in self.routing.values():
            if (
                not route
                or set(route) - set(self.nodes)
                or any(v < 0 for v in route.values())
                or not math.isclose(sum(route.values()), 1.0, abs_tol=1e-12, rel_tol=0)
            ):
                raise ValueError(
                    "routing fractions must be nonnegative, use known nodes and sum to one"
                )
        # Canonical ordering is a numerical contract: journal JSON sorts keys.
        object.__setattr__(self, "nodes", dict(sorted(self.nodes.items())))
        object.__setattr__(
            self,
            "routing",
            {
                label: dict(sorted(route.items()))
                for label, route in sorted(self.routing.items())
            },
        )
        object.__setattr__(
            self,
            "links",
            tuple(
                sorted(
                    (
                        edge.model_copy(
                            update={"a": min(edge.a, edge.b), "b": max(edge.a, edge.b)}
                        )
                        for edge in self.links
                    ),
                    key=lambda edge: (edge.a, edge.b),
                )
            ),
        )
        return self


def advance_network(config, temperatures_k, duration_sec, *, internal_w, external_w):
    """Pure bounded integration with heat-flow ledgers and enthalpy-based storage."""
    from scipy.integrate import solve_ivp

    names = list(config.nodes)
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("thermal duration must be positive finite")
    if (
        set(temperatures_k) != set(names)
        or set(internal_w) != set(names)
        or set(external_w) != set(names)
    ):
        raise ValueError("thermal temperatures and source maps must cover all nodes")
    t0 = np.array([temperatures_k[k] for k in names], dtype=float)
    q = np.array([internal_w[k] + external_w[k] for k in names], dtype=float)
    if not np.isfinite(t0).all() or np.any(t0 <= 0) or not np.isfinite(q).all():
        raise ValueError(
            "thermal inputs outside finite positive-temperature applicability"
        )
    c = np.array([config.nodes[k].capacity(float(t0[i])) for i, k in enumerate(names)])
    variable = [(i, config.nodes[k].material_capacity) for i, k in enumerate(names)
                if config.nodes[k].material_capacity is not None]
    g = np.array([config.nodes[k].bath_conductance_w_per_k for k in names])
    bath = np.array([config.nodes[k].bath_k for k in names])
    rad = np.array(
        [
            config.nodes[k].emissivity * SIGMA_SB * config.nodes[k].emission_area_m2
            for k in names
        ]
    )
    edges = [(index[e.a], index[e.b], e.conductance_w_per_k) for e in config.links]
    evaluations = 0

    def rhs(_clock, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > config.max_evaluations:
            raise ValueError("thermal derivative evaluation budget exhausted")
        t = y[:n]
        if not np.isfinite(t).all() or np.any(t <= 0) or np.any(t > 1e5):
            raise ValueError("thermal solver left temperature applicability")
        bath_out = g * (t - bath)
        radiation = rad * t**4
        net = q - bath_out - radiation
        flows = []
        for a, b, conductance in edges:
            flow = conductance * (t[a] - t[b])
            net[a] -= flow
            net[b] += flow
            flows.append(flow)
        capacities = c.copy() if variable else c
        for i, material in variable:
            capacities[i] = material.capacity(float(t[i]))
        return np.r_[net / capacities, bath_out, radiation, flows]

    events = []
    labels = []
    for i, name in enumerate(names):
        for kind, threshold, direction in [
            ("hot", config.nodes[name].trip_k, 1),
            ("cold", config.nodes[name].cold_trip_k, -1),
        ]:

            def event(_t, y, i=i, threshold=threshold):
                return y[i] - threshold

            event.direction = direction
            events.append(event)
            labels.append((name, kind))
    y0 = np.r_[t0, np.zeros(2 * n + len(edges))]
    result = solve_ivp(
        rhs,
        (0.0, duration_sec),
        y0,
        method=config.solver,
        dense_output=True,
        rtol=config.rtol,
        atol=np.r_[
            np.full(n, config.atol_k), np.full(2 * n + len(edges), config.energy_atol_j)
        ],
        max_step=config.max_step_sec,
        events=events,
    )
    if not result.success or not np.isfinite(result.y).all():
        raise ValueError("thermal integration failed")
    end = result.y[:, -1]
    stored = np.array([config.nodes[k].stored(float(t0[i]), float(end[i]))
                       for i, k in enumerate(names)])
    bath_j = end[n : 2 * n]
    rad_j = end[2 * n : 3 * n]
    link_j = end[3 * n :]
    exchange = np.zeros(n)
    for (a, b, _), value in zip(edges, link_j):
        exchange[a] -= value
        exchange[b] += value
    residual = stored - q * duration_sec - exchange + bath_j + rad_j
    trips = []
    for i, times in enumerate(result.t_events):
        node, kind = labels[i]
        threshold = (
            config.nodes[node].trip_k
            if kind == "hot"
            else config.nodes[node].cold_trip_k
        )
        for clock in times:
            if clock <= 0:
                continue
            left = float(
                result.sol(max(0.0, float(clock) - min(1e-6, duration_sec * 1e-3)))[
                    index[node]
                ]
            )
            if (kind == "hot" and left < threshold) or (
                kind == "cold" and left > threshold
            ):
                trips.append({"node": node, "kind": kind, "time_sec": float(clock)})
    return {
        "temperatures_k": dict(zip(names, end[:n].tolist())),
        "stored_j": dict(zip(names, stored.tolist())),
        "bath_out_j": dict(zip(names, bath_j.tolist())),
        "radiation_out_j": dict(zip(names, rad_j.tolist())),
        "link_out_j": link_j.tolist(),
        "node_balance_error_j": dict(zip(names, residual.tolist())),
        "internal_j": {k: internal_w[k] * duration_sec for k in names},
        "external_j": {k: external_w[k] * duration_sec for k in names},
        "evaluations": evaluations,
        "trips": sorted(trips, key=lambda r: (r["time_sec"], r["node"], r["kind"])),
    }


def projected_earth_view_factor(position_gcrs, normal_gcrs, radius_m):
    """Uniform radiance sphere; bounded 12x48 solid-angle quadrature at the limb."""
    r = np.asarray(position_gcrs, dtype=float)
    normal = np.asarray(normal_gcrs, dtype=float)
    distance = np.linalg.norm(r)
    if (
        r.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(r).all()
        or not np.isfinite(normal).all()
        or not math.isfinite(radius_m)
        or not 0 < radius_m < distance
        or not math.isclose(np.linalg.norm(normal), 1.0, abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError("invalid surface/Earth geometry")
    normal = normal / np.linalg.norm(normal)
    earth = -r / distance
    cosine = float(np.clip(normal @ earth, -1.0, 1.0))
    sine_theta = radius_m / distance
    cos_theta = math.sqrt(1 - sine_theta**2)
    if cosine >= sine_theta:
        return cosine * sine_theta**2
    if cosine <= -sine_theta:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(12)
    mu = (1 - cos_theta) * nodes / 2 + (1 + cos_theta) / 2
    azimuth = (np.arange(48) + 0.5) * (2 * math.pi / 48)
    dot = cosine * mu[:, None] + math.sqrt(max(0.0, 1 - cosine * cosine)) * np.sqrt(
        1 - mu[:, None] ** 2
    ) * np.cos(azimuth)
    return float(
        np.clip(
            (1 - cos_theta) * np.sum(weights[:, None] * np.maximum(0.0, dot)) / 48,
            0.0,
            1.0,
        )
    )
