"""Geometric opportunities, forecast-bound commands and stale-plan rejection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.core.frames import ecef_to_geodetic
from space_env.reference import EOPSeries, Epoch, FrameContext
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    dynamic_reference_scenario,
)
from space_env.runtime.cli import apply_plan


def main(directory: Path):
    path = Path(__file__).with_name("dynamic_demo.py")
    loader = importlib.util.spec_from_file_location("dynamic_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    definition = module.build_spec()
    orbit, eop = definition["orbits"]["a"]["initial"], definition["eop"]
    context = FrameContext(
        EOPSeries(
            eop["rows"],
            source=eop["source"],
            allow_predictions=eop["allow_predictions"],
        )
    )
    lat, lon, _ = ecef_to_geodetic(
        context.matrix(Epoch.utc(orbit["epoch_utc"])) @ orbit["r_m"]
    )
    definition["acquisition_targets"] = [{"id": "site", "lat_deg": lat, "lon_deg": lon}]
    definition["acquisition_sensors"] = {
        "a": {"max_off_nadir_deg": 35.0, "minimum_dwell_sec": 1.0}
    }
    scenario = dynamic_reference_scenario(
        definition, id="geometry-demo", lineage="B02-Q2"
    )
    scenario["required_capabilities"] = [
        "forecast.acquisition.geometry",
        "forecast.eclipse",
    ]
    access = AccessPolicy(truth=True, forecast=True)
    runtime = MissionRuntime(scenario, access=access)
    initial = runtime.forecast_acquisition("a", "site")
    assert initial["result"]["opportunities"]
    prefix = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "bound-wait",
                "commands": [
                    {
                        "command_id": "first-wait",
                        "actor_id": "a",
                        "kind": "wait",
                        "duration_sec": 1.0,
                        "forecast": initial["binding"],
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 1,
                "idempotency_key": "first-time",
                "dt_sec": 1.0,
            },
        ]
    }
    directory.mkdir(parents=True, exist_ok=True)
    apply_plan(runtime, prefix)
    runtime.save(directory / "prefix.json")
    runtime = MissionRuntime.load(directory / "prefix.json", access=access)
    before = runtime.forecast_acquisition("a", "site")
    impulse = {
        "op": "submit",
        "expected_revision": 2,
        "idempotency_key": "kick",
        "commands": [
            {
                "command_id": "kick",
                "actor_id": "a",
                "kind": "impulse",
                "duration_sec": 0.0,
                "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 0.0, 10.0]},
                "forecast": before["binding"],
            }
        ],
    }
    apply_plan(runtime, {"operations": [impulse]})
    checkpoint = runtime.checkpoint()
    try:
        runtime.submit(
            [
                {
                    "command_id": "stale",
                    "actor_id": "a",
                    "kind": "wait",
                    "duration_sec": 1.0,
                    "forecast": before["binding"],
                }
            ],
            expected_revision=3,
            idempotency_key="stale",
        )
    except RuntimeContractError as error:
        assert error.code == "stale_forecast"
    else:
        raise AssertionError("stale geometry was accepted")
    assert runtime.checkpoint() == checkpoint
    refreshed = runtime.forecast_acquisition("a", "site")
    continuation = [
        impulse,
        {
            "op": "submit",
            "expected_revision": 3,
            "idempotency_key": "fresh",
            "commands": [
                {
                    "command_id": "fresh",
                    "actor_id": "a",
                    "kind": "wait",
                    "duration_sec": 1.0,
                    "forecast": refreshed["binding"],
                }
            ],
        },
        {
            "op": "advance",
            "expected_revision": 4,
            "idempotency_key": "second-time",
            "dt_sec": 1.0,
        },
    ]
    apply_plan(runtime, {"operations": continuation[1:]})
    runtime.save(directory / "final.json")
    for name, value in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", {"operations": continuation}),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    restored = MissionRuntime.load(directory / "final.json", access=access)
    assert restored.checkpoint() == runtime.checkpoint()
    eclipses = runtime.forecast_eclipses("a")
    print(
        json.dumps(
            {
                "ok": True,
                "stale_forecast_rejected": True,
                "time_sec": runtime.time_sec,
                "opportunities": len(refreshed["result"]["opportunities"]),
                "eclipse_model": eclipses["result"]["model"],
                "creates_products": False,
                "qualification_status": "needs_external_evidence",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
