"""Conditional controller-window diagnostics; never a physical feasibility oracle."""

from __future__ import annotations

import math


def ground_window_diagnostic(workload):
    result = {
        "schema": "space-env.ground-window-diagnostic/1",
        "applicable": False,
        "scope": "conditional bound for the existing whole-window baseline; not physical infeasibility or arbitrary-agent capacity proof",
    }

    def unsupported(reason):
        return {**result, "reason": reason}

    try:
        s, task, policy = workload["scenario"], workload["task"], workload["policy"]
        if policy.get("name", "route_aware_rolling") not in {
            "priority",
            "earliest_deadline",
            "route_aware_rolling",
            "random_masked",
            "wait",
        }:
            return unsupported("unknown controller semantics")
        if policy.get("reclaim_acked_downlinks", False):
            return unsupported("early release enabled")
        if task["decision_sec"] != 2.0 or policy.get("decision_sec") not in {None, 2.0}:
            return unsupported("unsupported decision grid")
        if s.get("backend") != "reference-spacecraft":
            return unsupported("unknown native control semantics")
        d = s["definition"]
        ops, life = d["operations"], d["lifecycle"]
        if (
            ops.get("products")
            or ops.get("events")
            or life.get("events")
            or d.get("operator_events")
            or len(ops["stations"]) != 1
            or type(ops["stations"][0]["terminals"]) is not int
            or ops["stations"][0]["terminals"] != 1
        ):
            return unsupported(
                "preload, injected events or non-single-terminal topology"
            )
        station = ops["stations"][0]["name"]
        requests, jobs = life["requests"], life["jobs"]
        if not requests or len(requests) != len(jobs):
            return unsupported("unknown product pipeline")
        by_request = {r["id"]: r for r in requests}
        finals = {r["delivery_product_id"]: r for r in requests}
        raws = {r["raw_product_id"]: r for r in requests}
        if len(finals) != len(requests) or len(raws) != len(requests):
            return unsupported("shared products may fulfill multiple requests")
        by_job = {j["id"]: j for j in jobs}
        job_raws = []
        for j in jobs:
            if len(j["inputs"]) != 1 or len(j["outputs"]) != 1:
                return unsupported("unsupported processing graph")
            raw = j["inputs"][0]
            if (
                raw not in raws
                or j["outputs"][0]["id"] != raws[raw]["delivery_product_id"]
                or j["work_sec"] + j.get("switching_sec", 0) != 2.0
            ):
                return unsupported("unsupported processing graph")
            job_raws.append(raw)
        if len(set(job_raws)) != len(requests):
            return unsupported("duplicate or bypass processing chain")
        for r in requests:
            if (
                r["acquisition_sec"] != 2.0
                or len(r["eligible_actors"]) != 1
                or r["destination_id"] != station
                or r.get("observations")
            ):
                return unsupported("unsupported acquisition or delivery obligations")
        captures, computes, hops, grounds, durations = set(), set(), set(), set(), set()
        for action in task["actions"]:
            if action["kind"] != "submit" or len(action["commands"]) != 1:
                return unsupported("cancellation, injection or compound actions")
            c = action["commands"][0]
            kind = c["kind"]
            if kind == "imaging":
                r = by_request[c["request_id"]]
                if (
                    c["duration_sec"] != 2.0
                    or c["actor_id"] not in r["eligible_actors"]
                ):
                    return unsupported("alternative acquisition windows")
                captures.add(r["id"])
            elif kind == "transform":
                j = by_job[c["job_id"]]
                if (
                    c["duration_sec"] != 2.0
                    or c["actor_id"] not in raws[j["inputs"][0]]["eligible_actors"]
                ):
                    return unsupported("alternative processing windows")
                computes.add(j["id"])
            elif kind == "relay":
                r = finals[c["product_id"]]
                if (
                    c["duration_sec"] != 4.0
                    or c["actor_id"] not in r["eligible_actors"]
                ):
                    return unsupported("alternative relay windows")
                hops.add((c["product_id"], c["destination_id"]))
            elif kind == "downlink":
                if c["product_id"] not in finals or c["destination_id"] != station:
                    return unsupported("alternative delivery path")
                grounds.add((c["product_id"], c["actor_id"]))
                durations.add(c["duration_sec"])
            else:
                return unsupported("additional control kinds")
        if (
            captures != set(by_request)
            or computes != set(by_job)
            or len(durations) != 1
            or {p for p, _ in hops & grounds} != set(finals)
        ):
            return unsupported("incomplete or mixed pipeline")
        duration = next(iter(durations))
        if not math.isfinite(duration) or duration <= 0:
            return unsupported("invalid control window")
        horizon = d["horizon_sec"]
        if any(r["release_sec"] < 0 or r["deadline_sec"] != horizon for r in requests):
            return unsupported("unsupported release/deadline bounds")
        earliest = 8.0
        stride = math.ceil(duration / 2.0) * 2.0
        windows = max(
            0, math.floor((horizon - duration - earliest + 1e-9) / stride) + 1
        )
        bound = min(len(requests), windows)
        target = math.ceil(
            workload.get("min_completion_fraction", 0.0) * len(requests) - 1e-12
        )
        return {
            **result,
            "applicable": True,
            "reason": "recognized 2/2/4 chain and uniform whole-window reservation",
            "earliest_ground_control_sec": earliest,
            "ground_control_sec": duration,
            "decision_grid_sec": 2.0,
            "effective_window_stride_sec": stride,
            "full_control_windows_upper_bound": windows,
            "requests_upper_bound": bound,
            "requested_minimum": target,
            "target_exceeds_conditional_bound": target > bound,
            "assumptions": [
                "no autonomous early termination freeing a useful control before its window ends",
                "controller reserves one terminal for each entire active downlink window",
                "receiver actor remains reserved during the relay control window",
                "no outside intervention, preload, cancellation or injected products",
            ],
            "requires_native_validation": True,
        }
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return unsupported("unrecognized or incomplete workload shape")
