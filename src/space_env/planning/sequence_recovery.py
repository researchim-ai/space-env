"""Bounded residual-first native-prefix adaptation for fresh sequence scheduling."""

from __future__ import annotations

import copy
import math

from space_env.planning.joint_network import schedule_network
from space_env.planning.sequence_lanes import KINDS, schedule_sequence, seal
from space_env.simulation.snapshot import digest


class UnsupportedRecovery(ValueError):
    """A native-prefix profile lacks a supported suffix certificate."""


def collect_recovery_geometry(configuration, forecast, runtime, scenario):
    from space_env.api.network_workspace import (
        NetworkConfiguration,
        _hardware,
    )

    spec = NetworkConfiguration.model_validate(configuration)
    backend = runtime._backend
    targets = {
        (t["lat_deg"], t["lon_deg"]): t["id"]
        for t in scenario["definition"]["acquisition_targets"]
    }
    _, dtn = _hardware(spec, scenario["actor_order"])
    report = backend.report()["spacecraft_bus"]
    actors = {}
    for actor, bus in scenario["definition"]["spacecraft_bus"].items():
        state = report["states"][actor]
        bus = report["models"][actor]
        if (
            bus["attitude"].get("rigid") is not None
            or state["rate_scale"] != 1.0
            or state["safe"]
            or state["faults"]
            or bus["attitude"]["radio_tolerance_deg"] < 180
        ):
            raise UnsupportedRecovery("unsupported_current_attitude_or_bus_profile")
        claims = {}
        for kind in KINDS:
            command = {
                "command_id": "claim-probe",
                "actor_id": actor,
                "kind": kind,
                "duration_sec": 1.0,
            }
            if kind == "transform":
                command["job_id"] = "claim-probe"
            if kind == "slew":
                command["slew"] = {
                    "frame": "GCRS",
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            if kind == "relay":
                command.update(
                    destination_id="probe", product_id="probe", relay={"ttl_sec": 1.0}
                )
            claims[kind] = runtime.command_claims(command)["resources"]
        if (
            claims["transform"] != ["cpu"]
            or set(claims["imaging"]) != {"payload", "attitude"}
            or claims["slew"] != ["attitude"]
            or set(claims["relay"])
            != {"cpu", "payload", "radio", "attitude", "propulsion"}
            or set(claims["downlink"]) != {"radio"}
        ):
            raise UnsupportedRecovery("unsupported_native_lane_profile")
        vehicle = backend.mission.vehicles[actor]
        actors[actor] = {
            "initial_quaternion": state["quaternion_wxyz"],
            "rate_deg_s": bus["attitude"]["max_rate_deg_s"],
            "settling_sec": bus["attitude"]["settling_sec"],
            "claims": claims,
            "bandwidth_bps": scenario["definition"]["execution"]["actors"][actor][
                "memory_bandwidth_bps"
            ],
            "base_w": vehicle.base_power_w,
            "max_load_w": bus["power"]["max_load_w"],
            "slew_w": bus["attitude"]["slew_power_w"],
            "imaging_w": backend.lifecycle.spec.payloads[actor].power_w,
            "radio_w": vehicle.tx_power_w,
            "receive_w": dtn["endpoints"][actor]["power_w"],
        }
    requests = {r.id: r for r in spec.mission.requests}
    goals = {r["id"]: {} for r in forecast["demands"]}
    n = round(spec.mission.horizon_sec / spec.mission.decision_sec)
    for capture in forecast["captures"]:
        request = requests[capture["request"]]
        goals[request.id][capture["actor"]] = [
            runtime.forecast_pointing(
                capture["actor"],
                targets[request.lat_deg, request.lon_deg],
                i * spec.mission.decision_sec + request.acquisition_sec / 2,
            )["goal_quaternion_wxyz"]
            if valid
            else None
            for i, valid in enumerate(capture["cells"])
        ]
    demand_io = {
        r.id: {
            "imaging_bps": r.raw_bits / r.acquisition_sec,
            "cpu_bps": (r.raw_bits + r.output_bits) / r.processing_sec,
            "cpu_w": backend.lifecycle.jobs["cpu-" + r.id].power_w,
            "radio_bps": spec.mission.rate_bps,
        }
        for r in spec.mission.requests
        if r.id in goals
    }
    return seal(
        {
            "schema": "space-env.sequence-lane-geometry/1",
            "forecast_sha256": forecast["sha256"],
            "cells": n,
            "actors": actors,
            "goals": goals,
            "io": demand_io,
            "committed_time_sec": runtime.time_sec,
            "committed_physical_sha256": digest(backend.report()),
            "scope": "native committed attitude; fresh goals and native claims; conditional kinematic suffix, no future execution",
        }
    )


def schedule_recovery(spec, view, runtime, scenario):
    """Keep custody work serial; append a separately sealed fresh lane suffix.

    Residual commands never change attitude in the supported omnidirectional
    profile. Holding that native pose until the suffix therefore needs no
    simulated future or reuse of an initial geometry certificate.
    """
    residual_view = copy.deepcopy(view)
    fresh_ids = {rid for rid, s in view["recovery"].items() if s["phase"] == "fresh"}
    for rid in fresh_ids:
        residual_view["recovery"][rid].update(
            phase="blocked", reason="fresh_sequence_suffix_pending"
        )
    residual_view = seal({k: x for k, x in residual_view.items() if k != "sha256"})
    result = schedule_network(
        residual_view, max_nodes=spec.max_search_nodes, max_hops=spec.max_hops
    )
    detail = {
        "mode": "residual_only",
        "fallback_reason": None,
        "fresh_requests": sorted(fresh_ids),
        "optimal": False,
    }
    geometry = suffix_view = suffix = None
    if fresh_ids:
        try:
            if any(
                r["status"] in {"accepted", "running"} for r in runtime.command_status()
            ):
                raise UnsupportedRecovery("unsettled_native_commands")
            suffix_view = copy.deepcopy(view)
            suffix_view.update(at_sec=0.0, recovery={}, stored_bits={})
            suffix_view["demands"] = [
                r for r in suffix_view["demands"] if r["id"] in fresh_ids
            ]
            suffix_view["captures"] = [
                c for c in suffix_view["captures"] if c["request"] in fresh_ids
            ]
            start = max(
                [view["at_sec"]] + [a["finish_sec"] for a in result["allocations"]]
            )
            # Include unresolved settling at the committed boundary.
            bus = runtime._backend.report()["spacecraft_bus"]
            start = max(
                start,
                view["at_sec"]
                + max(s["settle_remaining_sec"] for s in bus["states"].values()),
            )
            for r in suffix_view["demands"]:
                r["release_sec"] = max(
                    r["release_sec"],
                    math.ceil(start / view["decision_sec"]) * view["decision_sec"],
                )
                if r["release_sec"] >= r["deadline_sec"]:
                    raise UnsupportedRecovery("no_fresh_suffix_before_deadline")
            for actor, limits in suffix_view["actors"].items():
                limits["memory_bits"] -= view["stored_bits"][actor] + sum(
                    a["memory_bits"][actor] for a in result["allocations"]
                )
                limits["usable_wh"] -= sum(
                    a["energy_wh"][actor] for a in result["allocations"]
                )
                if limits["memory_bits"] <= 0 or limits["usable_wh"] < 0:
                    raise UnsupportedRecovery("no_conservative_suffix_resources")
            suffix_view = seal({k: x for k, x in suffix_view.items() if k != "sha256"})
            geometry = collect_recovery_geometry(spec, suffix_view, runtime, scenario)
            budget = spec.max_search_nodes - result["search_nodes"]
            if budget <= 0:
                raise UnsupportedRecovery("shared_search_budget_exhausted")
            suffix = schedule_sequence(
                suffix_view, geometry, max_nodes=budget, max_hops=spec.max_hops
            )
            result["allocations"].extend(suffix["allocations"])
            for rid in fresh_ids:
                result["unserved"].pop(rid, None)
            result["unserved"].update(suffix["unserved"])
            result["search_nodes"] += suffix["search_nodes"]
            result["search_budget_exhausted"] |= suffix["search_budget_exhausted"]
            detail.update(mode="residual_then_sequence", suffix_start_sec=start)
        except UnsupportedRecovery as exc:
            detail.update(mode="conservative_fallback", fallback_reason=str(exc))
            # The original custody-aware forecast remains a valid serial input.
            # Do not turn a limitation of sequence admission into lost fresh work.
            used = result["search_nodes"]
            budget = spec.max_search_nodes - used
            if budget > 0:
                result = schedule_network(
                    view, max_nodes=budget, max_hops=spec.max_hops
                )
                result["search_nodes"] += used
                result["max_search_nodes"] = spec.max_search_nodes
            else:
                detail["serial_fallback_reason"] = "shared_search_budget_exhausted"
                for rid in fresh_ids:
                    result["unserved"][rid] = "search_budget_exhausted"
            geometry = suffix_view = suffix = None
    result.update(
        forecast_sha256=view["sha256"],
        predicted_fulfilled=len(result["completed"]) + len(result["allocations"]),
        recovery_sequence=detail,
        optimal=False,
        scope=(
            "bounded residual-first serial custody plus fresh native-prefix sequence suffix; "
            "no overlap between residual CPU and fresh work; conservative lifetime resources; "
            "no global optimum or guaranteed native success"
            if detail["mode"] == "residual_then_sequence"
            else "bounded whole-view serial custody recovery; no sequence certificate, "
            "global optimum or guaranteed native success"
        ),
    )
    result = seal({k: x for k, x in result.items() if k != "sha256"})
    return result, {"geometry": geometry, "forecast": suffix_view, "schedule": suffix}
