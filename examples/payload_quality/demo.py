"""Optical daylight/cloud masks or night-capable SAR: acquire to qualified ACK."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.area_coverage import AreaOfInterest
from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.cli import apply_plan
from space_env.simulation.snapshot import digest


def build_scenario(kind="optical"):
    if kind not in {"optical", "sar"}:
        raise ValueError("kind must be optical or sar")
    path = Path(__file__).resolve().parents[1] / "area_coverage/demo.py"
    loader = importlib.util.spec_from_file_location("area_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario = module.build_scenario()
    scenario.update(id=f"quality-{kind}", lineage="B04-B2")
    definition = scenario["definition"]
    if kind == "optical":
        # Explicit synthetic daylight variant, not a historical orbit import.
        for orbit in definition["orbits"].values():
            for key in ("r_m", "v_m_s"):
                orbit["initial"][key] = [-value for value in orbit["initial"][key]]
        for target in definition["acquisition_targets"]:
            target["lat_deg"] *= -1
            target["lon_deg"] = (target["lon_deg"] + 360) % 360 - 180
        for station in definition["operations"]["stations"]:
            station["lat_deg"] *= -1
            station["lon_deg"] = (station["lon_deg"] + 360) % 360 - 180
    else:
        next(t for t in definition["acquisition_targets"] if t["id"] == "site")[
            "lat_deg"
        ] += 1.0
    target = next(t for t in definition["acquisition_targets"] if t["id"] == "site")
    lon, lat = target["lon_deg"], target["lat_deg"]
    aoi = AreaOfInterest.model_validate(
        {
            "id": "customer-area",
            "anchor_lon_deg": lon,
            "anchor_lat_deg": lat,
            "shell": [
                [(lon + dx + 180) % 360 - 180, lat + dy]
                for dx, dy in (
                    (-0.005, -0.005),
                    (0.005, -0.005),
                    (0.005, 0.005),
                    (-0.005, 0.005),
                    (-0.005, -0.005),
                )
            ],
        }
    )
    spec = definition["lifecycle"]
    spec["areas_of_interest"] = [aoi.model_dump(mode="json")]
    template = MissionRuntime(scenario)
    for actor, model in definition["spacecraft_bus"].items():
        target = next(
            t
            for t in definition["acquisition_targets"]
            if t["id"] == ("site" if actor == "a" else "far-side")
        )
        model["attitude"]["initial_quaternion_wxyz"] = pointing_quaternion(
            template._backend.mission.ephemerides[actor],
            AcquisitionTarget.model_validate(target),
            0.0,
        ).tolist()
    attribution = {
        "kind": kind,
        "model_id": f"synthetic-{kind}/1",
        "source_status": "synthetic",
        "source": "declared demonstration parameters; no flight calibration",
    }
    if kind == "optical":
        spec["payloads"]["a"]["imaging_model"] = {
            **attribution,
            "instantaneous_fov_rad": 1e-5,
        }
        spec["requests"][0]["quality"] = {
            "kind": "optical",
            "min_sun_elevation_deg": 5.0,
            "max_gsd_m": 25.0,
        }
        spec["cloud_masks"] = [
            {
                "id": "synthetic-clear",
                "aoi_id": aoi.id,
                "aoi_sha256": digest(aoi.model_dump(mode="json")),
                "bounds_m": [-2000.0, -2000.0, 2000.0, 2000.0],
                "state": "clear",
                "valid_from_sec": 0.0,
                "valid_until_sec": 20.0,
                "source_status": "synthetic",
                "source": "scripted categorical cloud mask",
            }
        ]
    else:
        spec["payloads"]["a"]["imaging_model"] = {
            **attribution,
            "bandwidth_hz": 1e8,
            "antenna_length_m": 4.0,
            "min_incidence_deg": 5.0,
            "max_incidence_deg": 60.0,
            "peak_rf_power_w": 20.0,
            "pulse_duty_fraction": 0.5,
            "amplifier_efficiency": 0.5,
            "rolling_window_sec": 10.0,
            "max_on_sec": 4.0,
        }
        spec["requests"][0]["quality"] = {
            "kind": "sar",
            "max_ground_range_resolution_m": 25.0,
            "max_azimuth_resolution_m": 10.0,
        }
    return scenario, module.plans()


def main(directory, kind="optical"):
    scenario, (prefix, continuation) = build_scenario(kind)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    apply_plan(model, continuation)
    model.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )
    report = model.observe(view="truth")["backend"]["report"]
    lifecycle = report["lifecycle"]
    assert lifecycle["metrics"] == {"fulfilled_requests": 1, "total_requests": 2}
    assert lifecycle["payload_quality"]["requests"]["customer-a"][
        "delivered_and_fulfilled"
    ]
    assert all(
        abs(row[key]) < 1e-7
        for row in report["spacecraft_bus"]["ledgers"].values()
        for key in row
        if "error" in key
    )
    print(
        json.dumps(
            {
                "ok": True,
                "kind": kind,
                "metrics": lifecycle["metrics"],
                "payload_quality": lifecycle["payload_quality"],
                "ledger": lifecycle["ledger"],
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "scope": "sampled engineering quality/resource model; no calibrated science image",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "optical")
