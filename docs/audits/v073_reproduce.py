"""Read-only diagnostic probes for the v0.73 audit; prints observations, not fixes.

Run from repository root: python docs/audits/v073_reproduce.py
Private helpers are used only to isolate accounting semantics. No files or spacecraft
are controlled; these observations are not a replacement for regression tests.
"""
from __future__ import annotations

import json

import numpy as np
from astropy.time import Time as ReferenceTime

from space_env import __version__
from space_env.core.access import off_nadir_deg, subsatellite_latlon
from space_env.core.frames import ecef_to_eci
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PayloadConfig, Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.envs import registered_env_ids
from space_env.envs.industry_envs import PassFaultEnv
from space_env.core.cdm import parse_cdm_kvn
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.rl.reinforce import train_linear_reinforce
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet


def main() -> None:
    epoch = Time("2026-10-20T00:00:00Z")
    output = {"version": __version__, "env_count": len(registered_env_ids())}
    output["time_roundtrip"] = {
        "input": "2026-10-20T00:00:00Z",
        "jd": epoch.jd,
        "astropy_utc": ReferenceTime(epoch.jd, format="jd", scale="utc").isot,
        "actual_datetime": epoch.datetime.isoformat(),
        "roundtrip_error_sec": Time(epoch.datetime).seconds_since(epoch),
    }
    r = ecef_to_eci(np.array([6_800_000.0, 0.0, 0.0]), epoch)
    output["off_nadir"] = {
        "near_deg": off_nadir_deg(r, 0.0, 0.0, epoch),
        "antipodal_deg": off_nadir_deg(r, 0.0, 180.0, epoch),
    }
    sun = sun_position_m(epoch)
    lit_r = sun / np.linalg.norm(sun) * 7_000_000.0
    tangent = np.cross(np.array([0.0, 0.0, 1.0]), lit_r)
    velocity = tangent / np.linalg.norm(tangent) * 7500.0
    lat, lon = subsatellite_latlon(lit_r, epoch)
    far_lon = (lon + 360.0) % 360.0 - 180.0
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(lit_r, velocity, epoch),
        craft,
        scenario=ScenarioConfig(ground_stations=[], ground_targets=[GroundTarget(name="far-side", lat_deg=-lat, lon_deg=far_lon)]),
    )
    snap, _events = engine.step(1.0, "imaging")
    output["antipodal_capture"] = {"count": snap.imaging_count, "command_ok": engine.last_command_ok}
    recorder = SimulationEngine.__new__(SimulationEngine)
    recorder.spacecraft = Spacecraft(config=SpacecraftConfig(payload=PayloadConfig(memory_bits=10.0, image_bits=8.0)))
    recorder._stored_bits = recorder._generated_bits = recorder._lost_bits = 0.0
    recorder._store_capture()
    recorder._store_capture()
    output["overflow"] = {
        "attempted_bits": 16.0,
        "generated_bits": recorder._generated_bits,
        "stored_bits": recorder._stored_bits,
        "lost_bits": recorder._lost_bits,
    }
    orbit = Orbit.from_vectors(np.array([6_800_000.0, 0.0, 0.0]), np.array([0.0, 7600.0, 0.0]), epoch, frame="TEME")
    output["frame_preservation"] = {
        "input": orbit.frame,
        "after_zero_burn": orbit.with_delta_v(np.zeros(3)).frame,
        "after_propagate": orbit.propagate(1.0).frame,
    }
    fleet = Fleet.string_of_pearls(2, 1.5, scenario=ScenarioConfig(ground_stations=[]))
    a, b = list(fleet.vehicles)
    fleet.collect(a, 8.0)
    moved = fleet.relay(a, b, 60.0)
    output["dual_ledgers"] = {
        "relay_mbit": moved,
        "fleet_receiver_mbit": fleet.buffer_mb[b],
        "engine_receiver_bits": fleet.vehicles[b].get_state_snapshot().stored_bits,
        "time_after_relay_sec": fleet.time_sec,
    }
    first = train_linear_reinforce("space_env/PriorityCoverage-v0", episodes=10, max_episode_steps=1, seed=10, lr=0.0)
    second = train_linear_reinforce("space_env/PriorityCoverage-v0", episodes=10, max_episode_steps=1, seed=10, lr=10.0)
    output["one_step_training"] = {
        "lr_zero_returns": first["returns"],
        "lr_ten_returns": second["returns"],
        "identical": first["returns"] == second["returns"],
        "reason": "A single gain minus its episode mean is zero, so every update is zero.",
    }
    fault = PassFaultEnv()
    fault.reset(seed=0)
    for action in (2, 0, 0, 1, 0, 0):
        _obs, _reward, _term, _trunc, fault_info = fault.step(action)
    output["pass_fault_delivery"] = {
        "info": fault_info,
        "physical_contact": fault._engine.get_state_snapshot().in_contact,
        "engine_delivered_bits": fault._engine.get_state_snapshot().delivered_bits,
    }
    cdm = parse_cdm_kvn("MISS_DISTANCE = 100 [m]\nRELATIVE_SPEED = 1 [m/s]\nOBJECT = OBJECT1\nCR_R = 1 [km**2]\nCT_T = 1 [km**2]\nCN_N = 1 [km**2]\nCRDOT_R = 1 [km**2/s]\nCRDOT_RDOT = 1 [km**2/s**2]")
    output["cdm_explicit_units"] = {
        "input_miss_m": 100.0,
        "actual_miss_m": cdm.miss_distance_m,
        "input_relative_speed_m_s": 1.0,
        "actual_relative_speed_m_s": cdm.relative_speed_m_s,
        "input_cross_km2_s": 1.0,
        "actual_cross_m2_s": float(cdm.objects[0].state_covariance_si[0, 3]),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
