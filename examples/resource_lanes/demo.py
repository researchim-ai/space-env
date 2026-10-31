"""No preloaded data: capture, concurrent CPU/camera/radio, qualified delivery."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.cli import apply_plan


def build_scenario(mode="wide"):
    if mode not in {"wide", "constrained"}:
        raise ValueError("mode must be wide or constrained")
    path = Path(__file__).resolve().parents[1] / "payload_quality/demo.py"
    loader = importlib.util.spec_from_file_location("quality_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, _ = module.build_scenario("optical")
    scenario.update(id=f"resource-lanes-{mode}", lineage="B04-C")
    definition = scenario["definition"]
    definition["execution"] = {
        "mode": "resource-lanes/1",
        "actors": {
            a: {"memory_bandwidth_bps": 10000000.0 if mode == "wide" else 1000000.0}
            for a in scenario["actor_order"]
        },
    }
    first = definition["lifecycle"]["requests"][0]
    archive = copy.deepcopy(first)
    archive.update(
        id="archive-request", raw_product_id="archive", delivery_product_id="archive"
    )
    new = copy.deepcopy(first)
    new.update(
        id="new-request", raw_product_id="raw-new", delivery_product_id="raw-new"
    )
    definition["lifecycle"]["requests"] = [archive, first, new]
    return scenario, plans(mode)


def plans(mode):
    operations = []

    def add(op, key, **fields):
        operations.append(
            {
                "op": op,
                "expected_revision": len(operations),
                "idempotency_key": key,
                **fields,
            }
        )

    def image(id, request):
        return {
            "command_id": id,
            "actor_id": "a",
            "kind": "imaging",
            "duration_sec": 2.0,
            "request_id": request,
        }

    def radio(id, product, duration=1.0):
        return {
            "command_id": id,
            "actor_id": "a",
            "kind": "downlink",
            "duration_sec": duration,
            "product_id": product,
            "destination_id": "Ground",
        }

    add("submit", "archive-image", commands=[image("archive-image", "archive-request")])
    add("advance", "partial-first-image", dt_sec=1.0)
    add("advance", "finish-first-image", dt_sec=1.0)
    add("submit", "cpu-input", commands=[image("cpu-input", "customer-a")])
    add("advance", "finish-cpu-input", dt_sec=2.0)
    duration = 2.0 if mode == "wide" else 4.0
    add(
        "submit",
        "three-lanes",
        commands=[
            image("new-image", "new-request"),
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "transform",
                "job_id": "features-a",
                "duration_sec": duration,
            },
            radio("archive-radio", "archive", duration),
        ],
    )
    add("advance", "partial-lanes", dt_sec=0.25)
    add("advance", "finish-lanes", dt_sec=duration - 0.25)
    add("submit", "deliver-final", commands=[radio("final-radio", "final-a")])
    add("advance", "final-ack", dt_sec=1.0)
    add("submit", "deliver-new", commands=[radio("new-radio", "raw-new")])
    add("advance", "new-ack", dt_sec=1.0)
    return {"operations": operations[:2]}, {"operations": operations[2:]}


def main(directory, mode="wide"):
    scenario, (prefix, continuation) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for operation in continuation["operations"]:
        apply_plan(model, {"operations": [operation]})
        if operation["idempotency_key"] == "partial-lanes":
            model.save(directory / "partial-lanes.json")
            model = MissionRuntime.load(
                directory / "partial-lanes.json", access=AccessPolicy(truth=True)
            )
    model.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )
    report = model.observe(view="truth")["backend"]["report"]
    life = report["lifecycle"]
    assert life["metrics"] == {"fulfilled_requests": 3, "total_requests": 3}
    assert all(row["balance_error_bits"] == 0 for row in life["ledger"].values())
    rows = life["resource_lanes"]["intervals"]
    overlaps = [
        row
        for row in rows
        if len(row["actors"]["a"]["managed"]) == 2
        and row["actors"]["a"]["radio_bits"] > 0
    ]
    assert overlaps
    for row in rows:
        actor = row["actors"]["a"]
        assert (
            actor["memory_io_bits"]
            <= actor["memory_bandwidth_bps"] * (row["end_sec"] - row["start_sec"])
            + 1e-6
        )
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "metrics": life["metrics"],
                "time_sec": model.time_sec,
                "overlap_intervals": len(overlaps),
                "memory_io_bits": sum(
                    row["actors"]["a"]["memory_io_bits"] for row in rows
                ),
                "ledger": life["ledger"],
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "cpu_work": life["jobs"],
                "qualified_requests": life["payload_quality"]["requests"],
                "scope": "managed CPU/payload/radio, sampled resources, no measured hardware or SLA claim",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "wide")
