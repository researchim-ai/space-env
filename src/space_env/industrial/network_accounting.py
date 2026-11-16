"""Trainer diagnostics derived from native journals, never a second bit ledger."""

from __future__ import annotations

import copy
import math


def network_accounting(report, mission):
    dtn, radio = report.get("dtn"), report.get("radio_network")
    if dtn is None or radio is None:
        return None
    stations = {}
    for station, model in mission.stations.items():
        cap = mission.radio_network.config.receivers[station].aggregate_rate_bps
        maximum_grants, maximum_rate, excess_bits, contended_sec = 0, 0.0, 0.0, 0.0
        for interval in radio["intervals"]:
            span = interval["end_sec"] - interval["start_sec"]
            grants = [g for g in interval["grants"].values() if g["station"] == station]
            maximum_grants = max(maximum_grants, len(grants))
            maximum_rate = max(maximum_rate, math.fsum(g["rate_bps"] for g in grants))
            if cap is not None:
                excess_bits = max(
                    excess_bits, interval["station_bits"][station] - cap * span
                )
        lanes = (report.get("lifecycle") or {}).get("resource_lanes", {})
        for interval in lanes.get("intervals", []):
            offered = sum(
                s == station for s in interval.get("radio_offers", {}).values()
            )
            if offered > model.terminals:
                contended_sec += interval["end_sec"] - interval["start_sec"]
        stations[station] = {
            "terminal_limit": model.terminals,
            "aggregate_rate_limit_bps": cap,
            "maximum_active_grants": maximum_grants,
            "maximum_granted_rate_bps": maximum_rate,
            "terminal_excess": max(0, maximum_grants - model.terminals),
            "rate_excess_bps": 0.0 if cap is None else max(0.0, maximum_rate - cap),
            "interval_capacity_excess_bits": max(0.0, excess_bits),
            "contended_offer_sec": contended_sec,
            "transmitted_bits": radio["station_transmitted_bits"][station],
        }
    errors = [abs(v["balance_error_bits"]) for v in report["inventory_ledger"].values()]
    errors.extend(abs(v) for v in dtn["receiver_balance_error_bits"].values())
    errors.append(abs(dtn["wire_balance_error_bits"]))
    return {
        "custody_history": copy.deepcopy(dtn["custody_history"]),
        "completed_transfers": sum(
            t["status"] == "completed" for t in dtn["transfers"].values()
        ),
        "expired_transfers": sum(
            t["status"] == "expired" for t in dtn["transfers"].values()
        ),
        "retry_events": sum(e["kind"] == "retry" for e in dtn["history"]),
        "duplicate_received_bits": math.fsum(dtn["rx_duplicate_bits"].values()),
        "maximum_balance_error_bits": max(errors, default=0.0),
        "wire_balance_error_bits": dtn["wire_balance_error_bits"],
        "stations": stations,
        "scope": "native inventory/custody/wire and RF grant diagnostics; not independent protocol or hardware validation",
    }
