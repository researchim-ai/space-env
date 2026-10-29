"""Один прогон политики и повтор тех же действий."""
from __future__ import annotations

import json
from pathlib import Path

from space_env.mission.scenario_v2 import scenario_manifest
from space_env.mission.tasking import run_scenario


def seal_run(path: str | Path, scenario: str | Path, *, policy: str = "priority", step_sec: float = 30.0) -> dict:
    report = run_scenario(scenario, step_sec=step_sec, policy=policy, refresh_passes=False)
    record = {
        "schema": 2,
        "scenario": str(scenario),
        "sha256": scenario_manifest(scenario)["sha256"],
        "policy": policy,
        "step_sec": step_sec,
        "schedule": report["schedule"],
        "on_time": list(report["score"]["on_time"]),
        "unmet": report["unmet"],
        "receipts": [
            {
                "request_id": item["request_id"],
                "destination": item["destination"],
                "bits": item["bits"],
                "time_sec": item["time_sec"],
            }
            for item in report["receipts"]
        ],
        "final": report.get("final", {}),
    }
    Path(path).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def replay(path: str | Path) -> bool:
    """Повтор записанных команд. Совпадают квитанции, сданные заявки и причины."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(record.get("schema", 0)) != 2:
        raise ValueError("запись прогона имеет schema: 2")
    if scenario_manifest(record["scenario"])["sha256"] != record["sha256"]:
        return False
    from space_env.mission.scenarios import GroundTarget
    from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
    from space_env.mission.tasking import requests_from_scenario, run_policy
    from space_env.simulation.commands import MissionCommand

    spec = load_scenario_v2(record["scenario"])
    requests = requests_from_scenario(spec)
    by_id = {item.request_id: item for item in requests}
    world = world_from_scenario(record["scenario"], attach_stations=True, refresh_passes=False)
    world.set_outages([(item["station"], item["start_sec"], item["end_sec"]) for item in spec.get("outages", [])])
    blocked: set[str] = set()
    for item in record["schedule"]:
        blocked.update(str(name) for name in item.get("denied", []))
        for name, idents in item.get("targets", {}).items():
            engine = world.engines[name]
            engine.scenario.ground_targets = [
                GroundTarget(name=ident, lat_deg=by_id[ident].lat_deg, lon_deg=by_id[ident].lon_deg)
                for ident in idents
                if ident in by_id
            ]
        hops = item.get("relays") or []
        if hops:
            from space_env.simulation.fleet import Fleet

            fleet = Fleet(world.engines)
            fleet.use_reservations(world.relay_used)
            span = float(item.get("relay_dt", item["dt"]))
            for hop in hops:
                fleet.relay(hop["source"], hop["destination"], span)
        batch = []
        for name, record_command in item.get("commands", {}).items():
            batch.append(
                MissionCommand(
                    command_id=record_command["command_id"],
                    actor_id=record_command["actor_id"],
                    kind=record_command["kind"],
                    request_id=record_command.get("request_id", ""),
                    destination_id=record_command.get("destination_id", ""),
                )
            )
        orders = world.accept(batch)
        world.step(float(item["dt"]), orders)
        world.settle()
    from space_env.mission.tasking import _final_rows

    final_now = _final_rows(world)
    from space_env.mission.tasking import _unmet_rows, delivery_score

    if world.now_sec() + 1e-9 >= world.horizon_sec:
        unmet = _unmet_rows(world, requests, blocked)
        receipts_now = [item for engine in world.engines.values() for item in engine.receipts]
        scored = {
            "receipts": receipts_now,
            "unmet": unmet,
            "score": delivery_score(requests, receipts_now, unmet),
        }
    else:
        scored = run_policy(world, requests, step_sec=float(record["step_sec"]), policy=record["policy"])
    receipts = [
        {
            "request_id": item["request_id"],
            "destination": item["destination"],
            "bits": item["bits"],
            "time_sec": item["time_sec"],
        }
        for item in scored["receipts"]
    ]
    final_ok = "final" not in record or final_now == record["final"]
    return (
        receipts == record["receipts"]
        and list(scored["score"]["on_time"]) == list(record["on_time"])
        and scored["unmet"] == record["unmet"]
        and final_ok
    )
