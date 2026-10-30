"""Read-only synthetic observations for 0.134; not acceptance tests."""
import json
import runpy
from pathlib import Path

import gymnasium as gym
import numpy as np

import space_env
from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.core.frames import eci_to_ecef
from space_env.core.time import Time
from space_env.core.dut1 import dut1_seconds, pole_arcsec
from space_env.simulation.commands import MissionCommand
from space_env.simulation.parallel import ParallelFleet
from space_env.rl.trajectories import rollout_episode


def main():
    helper = runpy.run_path(str(Path(__file__).with_name("v086_audit_probes.py")))
    out = {"version": space_env.__version__}
    world = helper["world"]()
    env = ParallelFleet(world)
    env.reset(seed=1)
    env.step({"a": 0})
    env.reset(seed=1)
    out["parallel_reset_clock"] = world.now_sec()
    world = helper["world"]()
    world.note_truth()
    snap = world.snapshot()
    before = len(world._truth_log["a"])
    world.step(1)
    world.restore(snap)
    out["truth_log_restore"] = {
        "clock": world.now_sec(), "before": before,
        "after": len(world._truth_log["a"]),
        "last_sample_time": world._truth_log["a"][-1]["time_sec"],
    }
    item = helper["engine"]()
    lat, lon = subsatellite_latlon(item.orbit.r, item.now())
    item.stations = [GroundStation(name="Shared", lat_deg=lat, lon_deg=lon, min_elevation_deg=5)]
    item._refresh_environment()
    item.accept_bits(100e6, origin="imaging", request_id="r")
    world = helper["world"]({"a": item})
    orders = world.accept([MissionCommand("bounded", "a", "comms", duration_sec=2)])
    world.step(10, orders)
    world.settle()
    out["operation_duration"] = {
        "duration_sec": 2, "step_sec": 10,
        "command": world.commands.get("bounded"),
        "receipt_end_sec": max((r["end_sec"] for r in item.receipts), default=None),
        "delivered_bits": item.delivered_bits,
    }
    env = gym.make("space_env/MissionDict-v0", max_episode_steps=1)
    try:
        rollout_episode(env, seed=0, policy=lambda _obs: 0)
        out["dict_rollout"] = "succeeded"
    except Exception as exc:
        out["dict_rollout"] = {"error": type(exc).__name__, "message": str(exc)}
    finally:
        env.close()
    from erfa import c2t06a

    epoch = Time("2026-10-20T00:00:00Z")
    jd = float(epoch.jd)
    tt = jd + (37 + 32.184) / 86400
    ut1 = jd + dut1_seconds(epoch) / 86400
    xp, yp = pole_arcsec(epoch)
    arcsec = np.pi / (180 * 3600)
    matrix = c2t06a(2400000.5, tt - 2400000.5, 2400000.5, ut1 - 2400000.5, xp * arcsec, yp * arcsec)
    vector = np.array([7e6, 0., 0.])
    out["same_vector_frame_difference_m"] = float(np.linalg.norm(eci_to_ecef(vector, epoch) - matrix @ vector))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
