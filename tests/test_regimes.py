"""Радар снимает в тени. Геостационарный аппарат удерживает большую полуось."""
import gymnasium as gym

from space_env.catalog.build import make_engine
from space_env.catalog.references import reference
from space_env.core.access import subsatellite_latlon
from space_env.mission.scenarios import GroundTarget
from space_env.rl.expert import expert_action


def test_radar_captures_in_eclipse_and_optical_does_not():
    radar = gym.make("space_env/RadarPass-v0")
    obs, info = radar.reset(seed=0)
    assert info["eclipse"]
    _obs, _reward, _terminated, _truncated, step = radar.step(expert_action("space_env/RadarPass-v0", obs))
    assert step["success"] and step["captured"] and step["soc"] >= 0.2
    radar.close()
    skipped = gym.make("space_env/RadarPass-v0")
    skipped.reset(seed=0)
    _obs, _reward, _terminated, _truncated, step = skipped.step(0)
    assert step["success"] is False and step["captured"] is False
    skipped.close()

    optical = make_engine("sentinel-2a", epoch="2026-10-20T00:00:00Z", nu_deg=0.0, with_stations=False)
    assert reference("sentinel-2a").images_in_eclipse is False
    for _ in range(24):
        if optical.get_state_snapshot().in_eclipse:
            break
        optical.step(300.0, "wait")
    assert optical.get_state_snapshot().in_eclipse
    lat, lon = subsatellite_latlon(optical.orbit.r, optical.now())
    optical.scenario.ground_targets = [GroundTarget(name="scene", lat_deg=lat, lon_deg=lon)]
    optical.step(120.0, "imaging")
    assert optical.get_state_snapshot().imaging_count == 0


def test_geo_burn_pulls_the_semimajor_axis_back():
    kept = gym.make("space_env/LongitudeKeep-v0")
    obs, _info = kept.reset(seed=0)
    _obs, _reward, _terminated, _truncated, step = kept.step(expert_action("space_env/LongitudeKeep-v0", obs))
    assert step["success"] and step["error_m"] < 800.0
    kept.close()
    coast = gym.make("space_env/LongitudeKeep-v0")
    coast.reset(seed=0)
    _obs, _reward, _terminated, _truncated, step = coast.step(0)
    assert step["success"] is False and step["error_m"] > 1_500.0
    coast.close()
