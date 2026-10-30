"""Reference geometry becomes usable without silently invoking legacy frames."""
import copy
import json

import numpy as np
import pytest

from space_env.core.constants import C_LIGHT, MU_EARTH
from space_env.core.frames import ecef_to_geodetic
from space_env.core.ground import GroundStation
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.access import ReferenceEphemeris
from space_env.simulation.snapshot import digest
from space_env.validation.force_crosscheck import force_crosscheck


@pytest.fixture(scope="module")
def orbit():
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    return propagate(ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 0, np.sqrt(MU_EARTH/7e6)])),
                     6000, context, sample_sec=30)


def _station(ephemeris, clock):
    fixed = ephemeris.context.matrix(ephemeris.epoch.shifted(clock)) @ ephemeris.state(clock)[:3]
    lat, lon, _height = ecef_to_geodetic(fixed)
    return GroundStation(name="Overflight", lat_deg=lat, lon_deg=lon, min_elevation_deg=5)


def test_hermite_states_match_closed_form_between_nodes(orbit):
    ephemeris = ReferenceEphemeris.from_result(orbit)
    motion, radius = np.sqrt(MU_EARTH/7e6**3), 7e6
    for clock in np.linspace(15, 5985, 30):
        exact = np.array([radius*np.cos(motion*clock), 0, radius*np.sin(motion*clock)])
        assert np.linalg.norm(ephemeris.state(float(clock))[:3]-exact) < .1
    assert ephemeris.state(0) == pytest.approx(orbit["states_si"][0])
    assert ephemeris.state(6000) == pytest.approx(orbit["states_si"][-1])


def test_geometric_access_roots_peak_and_range_rate(orbit):
    ephemeris = ReferenceEphemeris.from_result(orbit)
    station = _station(ephemeris, 3000)
    report = ephemeris.access(station, scan_sec=20)
    window = next(row for row in report["windows"] if row["aos_sec"] < 3000 < row["los_sec"])
    assert window["max_elevation_deg"] > 89.9
    for clock in [window["aos_sec"], window["los_sec"]]:
        assert ephemeris.look(station, clock)["elevation_deg"] == pytest.approx(5, abs=.001)
    at = 2900.
    look = ephemeris.look(station, at, frequency_hz=2.2e9)
    derivative = (ephemeris.look(station, at+.1)["range_m"]-ephemeris.look(station, at-.1)["range_m"])/.2
    assert look["range_rate_m_s"] == pytest.approx(derivative, abs=.01)
    assert look["doppler_hz"] == pytest.approx(-look["range_rate_m_s"]/C_LIGHT*2.2e9)
    assert report["limits"]


def test_station_access_start_is_clipped_not_invented_aos(orbit):
    ephemeris = ReferenceEphemeris.from_result(orbit)
    report = ephemeris.access(_station(ephemeris, 0))
    assert report["windows"][0]["aos_sec"] == 0
    assert report["windows"][0]["clipped_start"]


def test_portable_ephemeris_does_not_consult_packaged_eop(orbit, monkeypatch):
    sealed = {**orbit, "sha256": digest(orbit)}

    def fail(*args, **kwargs):
        raise AssertionError("source EOP must not be consulted")

    monkeypatch.setattr(EOPSeries, "packaged", fail)
    restored = ReferenceEphemeris(json.loads(json.dumps(sealed)))
    assert restored.state(345.6).shape == (6,)
    assert restored.look(_station(restored, 3000), 3000)["elevation_deg"] > 89.9


@pytest.mark.parametrize("clock", [-1, 6001, float('nan'), float('inf'), True])
def test_no_extrapolation_or_nonfinite_queries(orbit, clock):
    with pytest.raises(ValueError):
        ReferenceEphemeris.from_result(orbit).state(clock)


@pytest.mark.parametrize("mode", ["corrupt", "frame", "time", "cadence", "backend", "eop-policy"])
def test_bad_ephemerides_are_not_accepted(orbit, mode):
    body = copy.deepcopy(orbit)
    if mode == "frame":
        body["frame"] = "GCRF"
    elif mode == "time":
        body["epoch_utc"] = "2026-07-18T12:00:01Z"
    elif mode == "cadence":
        body["times_sec"][1] = 0
    elif mode == "backend":
        body["manifest"]["erfa_version"] = "different"
    elif mode == "eop-policy":
        body["manifest"]["eop"]["allow_predictions"] = "false"
    sealed = {**body, "sha256": digest(body)}
    if mode == "corrupt":
        sealed["sha256"] = "bad"
    with pytest.raises(ValueError):
        ReferenceEphemeris(sealed)


def test_external_force_comparison_records_shared_inputs_and_backend():
    report = force_crosscheck()
    assert report["passed"]
    assert len(report["forces"]) == 9
    assert report["oracle"]["library"] == "boinor"
    assert report["shared_inputs"]
    assert "independent ephemerides and atmosphere" in report["not_qualified"]


def test_access_roots_match_independently_evaluated_circular_geometry(orbit):
    from scipy.optimize import brentq

    from space_env.core.frames import geodetic_to_ecef

    ephemeris = ReferenceEphemeris.from_result(orbit)
    station = _station(ephemeris, 3000)
    window = next(row for row in ephemeris.access(station)["windows"] if row["aos_sec"] < 3000 < row["los_sec"])
    lat, lon = np.deg2rad([station.lat_deg, station.lon_deg])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    ground = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
    motion = np.sqrt(MU_EARTH/7e6**3)

    def exact_margin(clock):
        r = np.array([7e6*np.cos(motion*clock), 0, 7e6*np.sin(motion*clock)])
        delta = ephemeris.context.matrix(ephemeris.epoch.shifted(clock)) @ r-ground
        return np.dot(delta, up)/np.linalg.norm(delta)-np.sin(np.deg2rad(station.min_elevation_deg))

    for key in ["aos_sec", "los_sec"]:
        expected = brentq(exact_margin, window[key]-3, window[key]+3, xtol=1e-7)
        assert abs(expected-window[key]) < .01


def test_crosscheck_detects_injected_force_error(monkeypatch):
    import importlib

    module = importlib.import_module("space_env.validation.force_crosscheck")
    original = module.acceleration

    def wrong(state, epoch, context, model):
        result = original(state, epoch, context, model)
        return result+np.array([1e-5, 0, 0]) if model.j2 else result

    monkeypatch.setattr(module, "acceleration", wrong)
    report = module.force_crosscheck()
    assert not report["passed"]
    assert all(not row["passed"] for row in report["forces"] if row["name"].startswith("j2"))


def test_cli_reference_access_writes_sealed_ground_network(orbit, tmp_path, capsys):
    from space_env.cli import main

    ephemeris = ReferenceEphemeris.from_result(orbit)
    station = _station(ephemeris, 3000)
    trajectory = tmp_path / "orbit.json"
    trajectory.write_text(json.dumps({**orbit, "sha256": digest(orbit)}))
    stations = tmp_path / "stations.json"
    stations.write_text(json.dumps({"stations": [station.model_dump(mode="json")]}))
    target = tmp_path / "access.json"
    main(["reference-access", str(trajectory), str(stations), str(target)])
    assert json.loads(capsys.readouterr().out)["windows"] > 0
    result = json.loads(target.read_text())
    assert result["sha256"] == digest({k: v for k, v in result.items() if k != "sha256"})
    original = target.read_bytes()
    stations.write_text(json.dumps({"stations": [{**station.model_dump(), "unrecognized": 1}]}))
    with pytest.raises(SystemExit):
        main(["reference-access", str(trajectory), str(stations), str(target)])
    assert target.read_bytes() == original


def test_combined_report_exposes_force_crosscheck():
    from space_env.validation.gcrs_report import reference_physics_report

    report = reference_physics_report(crosscheck=True)
    assert report["passed"]
    assert report["force_crosscheck"]["shared_inputs"]
