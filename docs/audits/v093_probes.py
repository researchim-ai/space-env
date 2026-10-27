"""Read-only synthetic audit probes. Outputs observations, not acceptance results."""
import json
import runpy
import tempfile
from pathlib import Path

import yaml

from space_env import __version__
from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.mission.scenarios import AnomalyInject
from space_env.mission.tasking import TaskRequest, delivery_score


def main():
    helpers = runpy.run_path(str(Path(__file__).with_name("v086_audit_probes.py")))
    engine, world = helpers["engine"], helpers["world"]
    out = {"version": __version__}

    def contacted(name):
        item = engine(name)
        lat, lon = subsatellite_latlon(item.orbit.r, item.now())
        item.stations = [GroundStation(name="Shared", lat_deg=lat, lon_deg=lon, min_elevation_deg=5)]
        item._refresh_environment()
        item.accept_bits(100e6, origin="imaging", request_id=name)
        return item

    w = world({"a": contacted("a"), "b": contacted("b")})
    w.step(1, {"a": "comms"})
    before = {name: item.delivered_bits for name, item in w.engines.items()}
    w.step(1, {"b": "comms"})
    out["armed_station_collision"] = {
        name: item.delivered_bits - before[name] for name, item in w.engines.items()
    }
    item = engine()
    item.scenario.anomaly_injects = [AnomalyInject(subsystem="payload", at_sec=.5, severity="failed", duration_sec=10)]
    w = world({"a": item})
    before_health = str(item.spacecraft.state.health["payload"])
    w.forecast("a", 1)
    out["forecast_fault_leak"] = {
        "clock": item.time_sec, "before_health": before_health,
        "after_health": str(item.spacecraft.state.health["payload"]),
        "pending_restores": len(item._restores),
    }
    w = world(horizon=2)
    w.step(3)
    out["world_horizon"] = {"horizon": 2, "actual": w.now_sec()}
    request = TaskRequest("r", 0, 0, 0, 10, 100, 1, "optical")
    slips = [
        {"request_id": "r", "product_id": "capture-a", "offset_bits": 0, "bits": 50, "time_sec": 5},
        {"request_id": "r", "product_id": "capture-b", "offset_bits": 50, "bits": 50, "time_sec": 6},
    ]
    out["mixed_acquisition_score"] = delivery_score([request], slips, [])["on_time"]
    baseline = {"version": 2, "horizon_sec": 60, "instances": [{"id": "a", "model": "cubesat-6u"}], "stations": [], "requests": []}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.yaml"
        checks = {}
        for name, changes in {
            "unknown_model": {"instances": [{"id": "a", "model": "does-not-exist"}]},
            "nan_unknown_outage": {"outages": [{"station": "unknown", "start_sec": float("nan"), "end_sec": 5}]},
            "invalid_epoch": {"epoch": "not-a-date"},
        }.items():
            path.write_text(yaml.safe_dump(dict(baseline, **changes)))
            try:
                load_scenario_v2(path)
                checks[name] = "accepted_by_loader"
            except Exception as exc:
                checks[name] = type(exc).__name__
        out["scenario_validation"] = checks
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
