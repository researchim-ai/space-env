"""Mission adapter: heat routing, sampled radiation, paid thermostat heaters."""

from __future__ import annotations

import copy
import math

import numpy as np

from space_env.core.constants import AU, R_EARTH_EQUATOR, SOLAR_CONSTANT
from space_env.core.environment import shadow_state
from space_env.reference.orbit import _ephemerides
from space_env.reference.thermal_network import (
    SIGMA_SB,
    advance_network,
    projected_earth_view_factor,
)


def rotate(q, v):
    q = np.asarray(q)
    v = np.asarray(v)
    return v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


class CoupledThermalNetwork:
    def __init__(self, models, backend):
        self.models = {
            a: m.thermal_network
            for a, m in models.items()
            if m.thermal_network is not None
        }
        for actor, config in self.models.items():
            vehicle = backend.mission.vehicles[actor]
            for node, fraction in config.routing["pv_extraction"].items():
                if not fraction or vehicle.solar_area_m2 == 0:
                    continue
                surface = config.nodes[node]
                if (
                    not np.allclose(
                        surface.normal_body,
                        models[actor].attitude.panel_normal_body,
                        atol=1e-9,
                        rtol=0,
                    )
                    or surface.absorptivity * surface.solar_area_m2 + 1e-12
                    < vehicle.solar_area_m2 * vehicle.solar_efficiency * fraction
                ):
                    raise ValueError(
                        "PV thermal surface must match panel normal and absorption capacity"
                    )
        self.temperatures = {
            a: {k: n.initial_k for k, n in c.nodes.items()}
            for a, c in self.models.items()
        }
        self.sources = {a: {} for a in self.models}
        self.environment = {}
        self.heater_requests = {
            a: dict.fromkeys(c.nodes, False) for a, c in self.models.items()
        }
        self.heaters = {a: dict.fromkeys(c.nodes, 0.0) for a, c in self.models.items()}
        self.totals = {
            a: {
                "internal_j": dict.fromkeys(c.nodes, 0.0),
                "external_j": dict.fromkeys(c.nodes, 0.0),
                "bath_out_j": dict.fromkeys(c.nodes, 0.0),
                "radiation_out_j": dict.fromkeys(c.nodes, 0.0),
                "link_out_j": [0.0] * len(c.links),
                "evaluations": 0,
                "routing_error_j": 0.0,
                "heater_dc_energy_j": 0.0,
                "heater_node_j": dict.fromkeys(c.nodes, 0.0),
                "dc_input_j": 0.0,
                "dc_heat_j": 0.0,
                "wall_input_j": 0.0,
                "wheel_work_j": 0.0,
                "declared_nonthermal_export_j": 0.0,
            }
            for a, c in self.models.items()
        }
        self.events = {a: [] for a in self.models}
        self.recent = {a: [] for a in self.models}

    def heater_power(self, a):
        return sum(self.heaters.get(a, {}).values())

    def heater_floor(self, a):
        return self.models[a].heater_min_soc if self.heater_power(a) > 0 else 0.0

    def violations(self, a):
        if a not in self.models:
            return []
        return [
            f"{kind}:{name}"
            for name, n in self.models[a].nodes.items()
            for kind, flag in [
                ("overtemperature", self.temperatures[a][name] >= n.trip_k),
                ("undertemperature", self.temperatures[a][name] <= n.cold_trip_k),
            ]
            if flag
        ]

    def recovered(self, a):
        return all(
            n.cold_resume_k <= self.temperatures[a][k] <= n.resume_k
            for k, n in self.models[a].nodes.items()
        )

    def stored(self, a):
        return sum(
            n.capacity_j_per_k * (self.temperatures[a][k] - n.initial_k)
            for k, n in self.models[a].nodes.items()
        )

    def prepare(self, backend, bus):
        for a, c in self.models.items():
            self.heaters[a] = dict.fromkeys(c.nodes, 0.0)
            base = bus.base_power(a, backend.mission.vehicles[a].base_power_w)
            remaining = max(0.0, bus.models[a].power.max_load_w - base)
            for k, n in sorted(
                c.nodes.items(), key=lambda r: (r[1].heater_priority, r[0])
            ):
                if not n.heater_power_w:
                    continue
                if self.temperatures[a][k] <= n.heater_on_below_k:
                    self.heater_requests[a][k] = True
                elif self.temperatures[a][k] >= n.heater_off_above_k:
                    self.heater_requests[a][k] = False
                if (
                    self.heater_requests[a][k]
                    and "bus" not in bus.states[a]["faults"]
                    and backend.mission.energy_wh[a]
                    > c.heater_min_soc * backend.mission.vehicles[a].battery_wh + 1e-12
                    and n.heater_power_w <= remaining
                ):
                    self.heaters[a][k] = n.heater_power_w
                    remaining -= n.heater_power_w
            provider = backend.mission.ephemerides[a]
            r = provider.state(backend.now())[:3]
            sun, _ = _ephemerides(provider.epoch.shifted(backend.now()))
            direction = sun - r
            distance = np.linalg.norm(direction)
            irradiance = SOLAR_CONSTANT * (AU / distance) ** 2
            flux = shadow_state(r, sun).flux_fraction
            phase_angle = math.acos(
                float(
                    np.clip(
                        (r @ sun) / (np.linalg.norm(r) * np.linalg.norm(sun)), -1, 1
                    )
                )
            )
            phase = (
                math.sin(phase_angle) + (math.pi - phase_angle) * math.cos(phase_angle)
            ) / math.pi
            env = {}
            for k, n in c.nodes.items():
                normal = rotate(bus.states[a]["quaternion_wxyz"], n.normal_body)
                normal = normal / np.linalg.norm(normal)
                vf = (
                    projected_earth_view_factor(r, normal, R_EARTH_EQUATOR)
                    if c.earth_environment
                    else 0.0
                )
                env[k] = {
                    "solar_w": n.absorptivity
                    * n.solar_area_m2
                    * irradiance
                    * flux
                    * max(0.0, float(normal @ direction / distance)),
                    "earth_ir_w": n.emissivity
                    * n.emission_area_m2
                    * c.earth_ir_w_m2
                    * vf,
                    "albedo_w": n.absorptivity
                    * n.solar_area_m2
                    * c.earth_geometric_albedo
                    * irradiance
                    * phase
                    * vf,
                    "sky_w": n.emissivity
                    * n.emission_area_m2
                    * SIGMA_SB
                    * c.space_background_k**4
                    * (1 - vf),
                    "earth_view_factor": vf,
                    "shadow_fraction": float(flux),
                }
            for k, fraction in c.routing["pv_extraction"].items():
                if bus.raw_generation_w[a] * fraction > env[k]["solar_w"] + 1e-8:
                    raise ValueError(
                        "PV extraction exceeds declared local absorbed solar power"
                    )
            self.environment[a] = env

    def boundary(self, backend, bus, right):
        for a, c in self.models.items():
            tick = (
                math.floor((backend.now() + 1e-9) / c.environment_step_sec) + 1
            ) * c.environment_step_sec
            right = min(right, tick)
            if not self.heater_power(a):
                continue
            v = backend.mission.vehicles[a]
            net = bus.generation_w[a] - bus.battery_load(
                a,
                bus.base_power(a, v.base_power_w)
                + bus.extra_power(a)
                + backend.lifecycle.actuation_power(a),
            )
            seconds = (
                (backend.mission.energy_wh[a] - self.heater_floor(a) * v.battery_wh)
                * 3600
                / -net
                if net < 0
                else math.inf
            )
            if seconds <= 1e-9:
                self.heaters[a] = dict.fromkeys(c.nodes, 0.0)
            else:
                right = min(right, backend.now() + seconds)
        return right

    def capture(self, backend, a, span, managed, payload_adjust):
        if a not in self.models:
            return None
        bus = backend.bus
        life = backend.lifecycle
        v = backend.mission.vehicles[a]
        fraction = bus.models[a].electrical_heat_fraction
        return {
            "managed": dict(managed),
            "payload_adjust": payload_adjust,
            "adcs_adjust": bus.rigid.heat_adjustment(a, span, fraction),
            "prop_heat": sum(
                m.plans.get(a, {}).get("heat_w", 0.0)
                for m in (life.actuation, life.unloader)
                if m is not None
            ),
            "motor_work_before": bus.rigid.states[a].motor_work_j
            if a in bus.rigid.models
            else 0.0,
            "wall_w": sum(
                m.plans.get(a, {}).get("wall_w", 0.0)
                for m in (life.actuation, life.unloader)
                if m is not None
            ),
            "tx_start": backend.mission.tx_seconds[a],
            "cpu_start": backend.mission.processing_seconds[a],
            "rx_energy_start": 0. if backend.mission.dtn is None else backend.mission.dtn.rx_energy_j[a],
            "curtailed_start": backend.mission.curtailed_wh[a],
            "tx_power": v.tx_power_w,
            "cpu_power": v.processing_power_w,
        }

    def integrate(
        self,
        backend,
        bus,
        a,
        start,
        span,
        before,
        legacy_heat_j,
        conversion_j,
        raw_pv_j,
    ):
        c = self.models[a]
        info = before[4]
        tot = self.totals[a]
        fraction = bus.models[a].electrical_heat_fraction
        mission = backend.mission
        curtailed = (mission.curtailed_wh[a] - info["curtailed_start"]) * 3600
        heater_j = self.heater_power(a) * span
        available = legacy_heat_j + (1 - fraction) * heater_j + curtailed
        source = {
            "cpu": info["managed"]["cpu"] * span * fraction
            + (mission.processing_seconds[a] - info["cpu_start"])
            * info["cpu_power"]
            * fraction,
            "payload": (info["managed"]["payload"] * fraction + info["payload_adjust"])
            * span,
            "radio": (mission.tx_seconds[a] - info["tx_start"])
            * info["tx_power"]
            * fraction
            + (0. if mission.dtn is None else mission.dtn.rx_energy_j[a]-info["rx_energy_start"])*fraction,
            "adcs": (before[2] * fraction + info["adcs_adjust"]) * span,
            "propulsion": info["prop_heat"] * span,
            "conversion": conversion_j + curtailed,
            "pv_extraction": -raw_pv_j,
        }
        source["base"] = (
            available
            - heater_j
            - sum(v for k, v in source.items() if k != "pv_extraction")
        )
        tolerance = 1e-8 + 1e-10 * max(1.0, abs(available))
        dc_input = (
            (mission.consumed_wh[a] - before[1])
            * bus.models[a].power.discharge_efficiency
            * 3600
        )
        wall = info["wall_w"] * span
        dc_heat = legacy_heat_j + (1 - fraction) * heater_j - conversion_j - wall
        motor_work = (
            bus.rigid.states[a].motor_work_j - info["motor_work_before"]
            if a in bus.rigid.models
            else 0.0
        )
        exported = dc_input - dc_heat - motor_work
        if exported < -tolerance:
            raise ValueError("thermal DC/work/heat accounting creates energy")
        for label, value in (
            ("dc_input_j", dc_input),
            ("dc_heat_j", dc_heat),
            ("wall_input_j", wall),
            ("wheel_work_j", motor_work),
            ("declared_nonthermal_export_j", exported),
        ):
            tot[label] += value

        if any(v < -tolerance for k, v in source.items() if k != "pv_extraction"):
            raise ValueError("thermal source accounting exceeds actual bus heat")
        incoming = {
            k: sum(
                v for label, v in self.environment[a][k].items() if label.endswith("_w")
            )
            for k in c.nodes
        }
        internal = {k: self.heaters[a][k] for k in c.nodes}
        for label, energy in source.items():
            self.sources[a][label] = self.sources[a].get(label, 0.0) + energy
            for node, f in c.routing[label].items():
                internal[node] += energy * f / span
        result = advance_network(
            c, self.temperatures[a], span, internal_w=internal, external_w=incoming
        )
        if tot["evaluations"] + result["evaluations"] > c.max_total_evaluations:
            raise ValueError("thermal lifetime evaluation budget exhausted")
        trips = [{**r, "time_sec": start + r["time_sec"]} for r in result["trips"]]
        if len(self.events[a]) + len(trips) > c.max_events:
            raise ValueError("thermal event history budget exhausted")
        tot["heater_dc_energy_j"] += heater_j
        for node, power in self.heaters[a].items():
            tot["heater_node_j"][node] += power * span
        self.temperatures[a] = result["temperatures_k"]
        self.events[a].extend(trips)
        self.recent[a] = trips
        for label in ("internal_j", "external_j", "bath_out_j", "radiation_out_j"):
            for k, value in result[label].items():
                tot[label][k] += value
        tot["link_out_j"] = (
            np.asarray(tot["link_out_j"]) + result["link_out_j"]
        ).tolist()
        tot["evaluations"] += result["evaluations"]
        tot["routing_error_j"] += (
            available
            + sum(incoming.values()) * span
            - raw_pv_j
            - (sum(internal.values()) + sum(incoming.values())) * span
        )
        heat = (sum(internal.values()) + sum(incoming.values())) * span
        rejected = sum(result["bath_out_j"].values()) + sum(
            result["radiation_out_j"].values()
        )
        for trip in trips:
            reason = (
                ("overtemperature" if trip["kind"] == "hot" else "undertemperature")
                + ":"
                + trip["node"]
            )
            state = bus.states[a]
            state["safe_reasons"] = sorted(set(state["safe_reasons"]) | {reason})
            state["safe"] = True
        return heat, rejected

    def observation(self, a):
        c = self.models[a]
        t = self.totals[a]
        exchange = dict.fromkeys(c.nodes, 0.0)
        for edge, energy in zip(c.links, t["link_out_j"]):
            exchange[edge.a] -= energy
            exchange[edge.b] += energy
        residual = {
            k: n.capacity_j_per_k * (self.temperatures[a][k] - n.initial_k)
            - t["internal_j"][k]
            - t["external_j"][k]
            - exchange[k]
            + t["bath_out_j"][k]
            + t["radiation_out_j"][k]
            for k, n in c.nodes.items()
        }
        return copy.deepcopy(
            {
                "schema": "space-env.thermal-network/1",
                "temperatures_k": self.temperatures[a],
                "source_heat_j": self.sources[a],
                "heater_requested": self.heater_requests[a],
                "heater_power_w": self.heaters[a],
                "sampled_environment": self.environment.get(a, {}),
                "ledgers": t,
                "node_balance_error_j": residual,
                "events": self.events[a],
                "temperature_summary": "maximum_node_kelvin",
                "scope": "lumped network, sampled uniform Earth/Lambert phase and ideal thermostats; no flight calibration",
            }
        )
