"""Three-actor EO/CPU/ISL/ground mission with custody, rerouting, loss and TTL."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.cli import apply_plan


def build_scenario(mode="nominal"):
    if mode not in {"nominal", "reroute", "loss", "ttl"}:
        raise ValueError("mode must be nominal, reroute, loss or ttl")
    path = Path(__file__).resolve().parents[1] / "network_mission/demo.py"
    loader = importlib.util.spec_from_file_location("relay_ground_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, _ = module.build_scenario("nominal")
    scenario.update(id="relay-" + mode, lineage="B05-B")
    scenario["actor_order"].append("c")
    d = scenario["definition"]
    ops = d["operations"]
    for key in ["orbits", "spacecraft_bus", "acquisition_sensors"]:
        d[key]["c"] = copy.deepcopy(d[key]["b"])
    d["lifecycle"]["payloads"]["c"] = copy.deepcopy(d["lifecycle"]["payloads"]["b"])
    d["execution"]["actors"]["c"] = copy.deepcopy(d["execution"]["actors"]["b"])
    ops["vehicles"]["c"] = copy.deepcopy(ops["vehicles"]["b"])
    ops["radio_network"]["transmitters"]["c"] = copy.deepcopy(
        ops["radio_network"]["transmitters"]["b"]
    )
    for actor, angle in [("b", 0.01), ("c", 0.02)]:
        orbit = d["orbits"][actor]["initial"]
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for name in ["r_m", "v_m_s"]:
            orbit[name] = (rot @ np.asarray(orbit[name])).tolist()
        ops["vehicles"][actor]["memory_bits"] = 2000000.0
        d["spacecraft_bus"][actor]["thermal_network"] = copy.deepcopy(
            d["spacecraft_bus"]["a"]["thermal_network"]
        )
    # Directed receiver RF settings do not inherit ground setup/aggregate policy.
    receiver = {
        "gt_db_k": 0.0,
        "min_frequency_hz": 2.1e9,
        "max_frequency_hz": 2.3e9,
        "bandwidth_hz": 2e6,
    }
    first = {
        "source": "a",
        "destination": "c",
        "receiver": copy.deepcopy(receiver),
        "acquisition_sec": 0.05,
    }
    second = {
        "source": "a",
        "destination": "b",
        "receiver": copy.deepcopy(receiver),
        "acquisition_sec": 0.05,
    }
    ops["dtn_network"] = {
        "source": "synthetic S-band store-and-forward packet/retention assumptions",
        "parameter_status": "synthetic",
        "endpoints": {a: {"power_w": 2.0} for a in scenario["actor_order"]},
        "links": [first, second],
        "fragment_bits": 250000,
        "sample_sec": 0.1,
        "max_retries": 5,
        "seed": 0,
    }
    if mode == "reroute":
        first["outages"] = [[4.4, 4.6]]
        ops["vehicles"]["b"]["memory_bits"] = 400000.0
        ops["outages"] = [{"station": "Ground", "start_sec": 8.0, "end_sec": 8.2}]
    if mode == "loss":
        second["ack_loss_probability"] = 0.4
        # Pick and publish one deterministic seed for a lost first ACK followed by
        # successful retransmission and remaining fragment ACKs. No state is patched.
        for seed in range(10000):
            samples = [
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:relay-second:{fragment}:{attempt}:ack".encode()
                    ).digest()[:8],
                    "big",
                )
                / 2**64
                for fragment, attempt in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]
            ]
            if samples[0] < 0.4 and all(v >= 0.4 for v in samples[1:]):
                ops["dtn_network"]["seed"] = seed
                break
        else:
            raise RuntimeError("synthetic loss seed not found")
    plan = []

    def add(op, key, **kw):
        plan.append(
            {"op": op, "expected_revision": len(plan), "idempotency_key": key, **kw}
        )

    def command(id, kind, duration, actor="a", **kw):
        return {
            "command_id": id,
            "actor_id": actor,
            "kind": kind,
            "duration_sec": duration,
            **kw,
        }

    def hop(id, product, destination, duration=2.0, ttl=15.0):
        return command(
            id,
            "relay",
            duration,
            product_id=product,
            destination_id=destination,
            relay={"ttl_sec": ttl},
        )

    add(
        "submit",
        "capture-first",
        commands=[command("capture-first", "imaging", 2.0, request_id="customer-a")],
    )
    add("advance", "captured", dt_sec=2.0)
    add(
        "submit",
        "compute-first",
        commands=[command("compute-first", "transform", 2.0, job_id="features-a")],
    )
    add("advance", "computed", dt_sec=2.0)
    add("submit", "relay-first", commands=[hop("relay-first", "final-a", "c")])
    add("advance", "mid-fragment", dt_sec=0.15)
    prefix = len(plan)
    add("advance", "inflight", dt_sec=0.1501)
    add("advance", "first-custody", dt_sec=1.6999)
    add(
        "submit",
        "capture-second",
        commands=[
            command("capture-second", "imaging", 2.0, request_id="after-thermal")
        ],
    )
    add("advance", "captured-second", dt_sec=2.0)
    add(
        "submit",
        "mixed-network",
        commands=[
            hop(
                "relay-second",
                "raw-after",
                "b",
                duration=4.0,
                ttl=0.4 if mode == "ttl" else 12.0,
            ),
            command(
                "direct-first",
                "downlink",
                2.0,
                actor="c",
                product_id="final-a",
                destination_id="Ground",
            ),
        ],
    )
    if mode == "reroute":
        add("advance", "buffer-pressure", dt_sec=1.0)
        add("cancel", "cancel-blocked-hop", command_id="relay-second")
        add(
            "submit",
            "reroute",
            commands=[hop("reroute", "raw-after", "c", duration=4.0, ttl=20.0)],
        )
        add("advance", "rerouted-custody", dt_sec=4.0)
    else:
        add("advance", "mixed-complete", dt_sec=4.0)
    if mode != "ttl":
        add(
            "submit",
            "ground-second",
            commands=[
                command(
                    "ground-second",
                    "downlink",
                    3.5,
                    actor="c" if mode == "reroute" else "b",
                    product_id="raw-after",
                    destination_id="Ground",
                )
            ],
        )
        add("advance", "ground-second-ack", dt_sec=3.5)
    add("advance", "drain", dt_sec=0.25)
    return scenario, ({"operations": plan[:prefix]}, {"operations": plan[prefix:]})


def verify(model, mode):
    r = model._backend.report()
    life = r["lifecycle"]
    n = r["dtn"]
    assert life["metrics"] == {
        "fulfilled_requests": 1 if mode == "ttl" else 2,
        "total_requests": 2,
    }
    assert abs(n["wire_balance_error_bits"]) < 1e-6
    assert max(abs(v) for v in n["receiver_balance_error_bits"].values()) < 1e-6
    for rows in [r["inventory_ledger"], life["ledger"]]:
        assert all(abs(v["balance_error_bits"]) < 1e-6 for v in rows.values())
    for values in r["spacecraft_bus"]["ledgers"].values():
        assert (
            abs(values["battery_balance_error_wh"]) < 1e-9
            and abs(values["thermal_balance_error_j"]) < 1e-6
        )
    if mode == "reroute":
        assert n["transfers"]["relay-second"]["status"] == "cancelled"
        assert n["transfers"]["reroute"]["status"] == "completed"
        assert any(v > 0 for v in n["wasted_partial_bits"].values())
    if mode == "loss":
        assert n["rx_duplicate_bits"]["b"] > 0
    if mode == "ttl":
        assert life["artifacts"]["raw-after"]["status"] == "expired"
    assert (
        life["artifacts"]["final-a"]["origin_actor"] == "a"
        and life["artifacts"]["final-a"]["actor"] == "c"
    )
    return {
        "ok": True,
        "mode": mode,
        "metrics": life["metrics"],
        "transfer_status": {k: v["status"] for k, v in n["transfers"].items()},
        "custody_history": n["custody_history"],
        "rx_duplicate_bits": n["rx_duplicate_bits"],
        "tx_serialized_bits": n["tx_serialized_bits"],
        "rx_dc_energy_j": n["rx_dc_energy_j"],
        "wire_balance_error_bits": n["wire_balance_error_bits"],
        "receiver_balance_error_bits": n["receiver_balance_error_bits"],
        "acknowledgements": life["acknowledgements"],
        "scope": "synthetic whole-product custody; ideal isolated RF/central control and out-of-band ACK; not BPv7/BIBE/LTP",
    }


def main(directory, mode="nominal"):
    scenario, (prefix, resume) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in [("scenario", scenario), ("plan", prefix), ("resume", resume)]:
        (directory / (name + ".json")).write_text(json.dumps(data, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for operation in resume["operations"]:
        apply_plan(model, {"operations": [operation]})
        if operation["idempotency_key"] == "inflight":
            assert (
                model._backend.mission.dtn.transfers["relay-first"]["state"]
                == "inflight"
            )
            model.save(directory / "inflight.json")
            model = MissionRuntime.load(
                directory / "inflight.json", access=AccessPolicy(truth=True)
            )
    model.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )
    print(json.dumps(verify(model, mode), allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "nominal")
