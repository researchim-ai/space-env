"""Explicit frame/time contracts and declared reference-orbit applicability."""
import json
from pathlib import Path

import numpy as np
import pytest

from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR
from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    acceleration,
    propagate,
)
from space_env.reference.frames import c2t_matrix


def context():
    return FrameContext(EOPSeries.packaged())


def test_frozen_sofa_erfa_rotation_fixture():
    fixture = Path(__file__).resolve().parents[1] / "src/space_env/reference/fixtures/c2t06a.json"
    data = json.loads(fixture.read_text())
    actual = c2t_matrix(*data["tt_jd"], *data["ut1_jd"], data["xp_rad"], data["yp_rad"])
    assert np.max(np.abs(actual - data["expected_matrix"])) < data["absolute_tolerance"]
    assert np.allclose(actual @ actual.T, np.eye(3), atol=1e-14, rtol=0)
    assert np.linalg.det(actual) == pytest.approx(1, abs=1e-14)


def test_time_uses_si_seconds_through_leap_second_and_preserves_microseconds():
    epoch = Epoch.utc("2016-12-31T23:59:59Z")
    leap = epoch.shifted(1)
    midnight = epoch.shifted(2)
    assert leap.iso() == "2016-12-31T23:59:60.000000000Z"
    assert midnight.iso() == "2017-01-01T00:00:00.000000000Z"
    assert Epoch.utc("2017-01-01T00:00:00Z").seconds_since(epoch) == pytest.approx(2, abs=1e-9)
    epoch = Epoch.utc("2026-07-18T12:00:00.123456Z")
    assert epoch.shifted(1e-6).seconds_since(epoch) == pytest.approx(1e-6, abs=1e-10)
    a, b = epoch.tt_jd
    assert ((a-epoch.tai1)+(b-epoch.tai2))*86400 == pytest.approx(32.184, abs=1e-9)


@pytest.mark.parametrize("text", ["2026-07-18T12:00:00", "2030-01-01T00:00:00Z", "2026-07-18T12:00:60Z"])
def test_unsupported_utc_is_not_silently_reinterpreted(text):
    with pytest.raises((ValueError, Warning)):
        Epoch.utc(text)


def test_eop_never_extrapolates_or_silently_accepts_predictions():
    series = EOPSeries.packaged()
    with pytest.raises(ValueError, match="coverage"):
        series.at(Epoch.utc("2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="predicted"):
        series.at(Epoch.utc("2026-10-20T00:00:00Z"))
    allowed = EOPSeries.packaged(allow_predictions=True)
    assert allowed.at(Epoch.utc("2026-10-20T00:00:00Z")).dut1_s == pytest.approx(-.0399555, abs=1e-13)


def test_eop_interpolates_ut1_tai_across_utc_leap():
    eop = EOPSeries([
        {"utc": "2016-12-31T00:00:00Z", "dut1_s": -.4, "xp_arcsec": .1, "yp_arcsec": .2, "status": "synthetic"},
        {"utc": "2017-01-01T00:00:00Z", "dut1_s": .6, "xp_arcsec": .1, "yp_arcsec": .2, "status": "synthetic"},
        {"utc": "2017-01-02T00:00:00Z", "dut1_s": .6, "xp_arcsec": .1, "yp_arcsec": .2, "status": "synthetic"},
    ], source="synthetic leap continuity fixture")
    assert eop.at(Epoch.utc("2016-12-31T12:00:00Z")).dut1_s == pytest.approx(-.4)
    assert eop.at(Epoch.utc("2017-01-01T00:00:00Z")).dut1_s == pytest.approx(.6)


@pytest.mark.parametrize("utc", ["2026-07-18T12:00:00Z", "2026-08-20T08:15:00Z"])
def test_velocity_covariance_and_ground_fixed_point_transform(utc):
    c = context()
    epoch = Epoch.utc(utc)
    ground = np.array([R_EARTH_EQUATOR, 0, 0, 0, 0, 0])
    inertial = c.transform(ground, epoch, source="ITRS", destination="GCRS")["state_si"]
    assert 460 < np.linalg.norm(inertial[3:]) < 470
    covariance = np.diag([100., 200., 300., .01, .02, .03])
    fixed = c.transform(inertial, epoch, source="GCRS", destination="ITRS", covariance=covariance)
    restored = c.transform(fixed["state_si"], epoch, source="ITRS", destination="GCRS", covariance=fixed["covariance_si"])
    assert np.allclose(fixed["state_si"], ground, atol=1e-8, rtol=0)
    assert np.allclose(restored["state_si"], inertial, atol=1e-8, rtol=0)
    assert np.allclose(restored["covariance_si"], covariance, atol=1e-10, rtol=1e-12)
    assert np.linalg.eigvalsh(fixed["covariance_si"]).min() > 0
    with pytest.raises(ValueError, match="legacy"):
        c.transform(ground, epoch, source="GCRF", destination="ITRS")


@pytest.mark.parametrize("duration", [86400., 259200.])
def test_circular_orbit_long_arc_matches_closed_form(duration):
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    radius = 7e6
    speed = np.sqrt(MU_EARTH/radius)
    initial = ReferenceState(epoch, np.array([radius, 0, 0]), np.array([0, 0, speed]))
    result = propagate(initial, duration, context(), sample_sec=600)
    times, states = np.array(result["times_sec"]), np.array(result["states_si"])
    phase = np.sqrt(MU_EARTH/radius**3)*times
    expected = np.stack([radius*np.cos(phase), phase*0, radius*np.sin(phase),
                         -speed*np.sin(phase), phase*0, speed*np.cos(phase)], axis=1)
    assert np.max(np.linalg.norm(states[:, :3]-expected[:, :3], axis=1)) < 1.
    assert np.max(np.linalg.norm(states[:, 3:]-expected[:, 3:], axis=1)) < 1e-3


def test_j2_force_matches_gradient_of_independently_written_potential():
    c, epoch = context(), Epoch.utc("2026-07-18T12:00:00Z")
    r = np.array([6e6, -3e6, 2e6])
    rotation = c.matrix(epoch)

    def potential(position):
        fixed = rotation @ position
        radius = np.linalg.norm(fixed)
        return MU_EARTH/radius * (1-J2_EARTH*(R_EARTH_EQUATOR/radius)**2*.5*(3*(fixed[2]/radius)**2-1))

    gradient = np.array([(potential(r+axis*10)-potential(r-axis*10))/20 for axis in np.eye(3)])
    actual = acceleration(np.concatenate((r, [0, 7500, 0])), epoch, c, ForceModel(j2=True))
    assert np.linalg.norm(actual-gradient) < 1e-8


def test_drag_opposes_velocity_relative_to_rotating_earth():
    c, epoch = context(), Epoch.utc("2026-07-18T12:00:00Z")
    state = np.array([6.8e6, 0, 0, 0, 7500, 0])
    central = acceleration(state, epoch, c, ForceModel())
    drag = acceleration(state, epoch, c, ForceModel(drag=True))-central
    relative_velocity = c.jacobian(epoch) @ state
    assert np.dot(c.matrix(epoch) @ drag, relative_velocity[3:]) < 0


@pytest.mark.parametrize("model", [ForceModel(srp=True), ForceModel(third_body=True), ForceModel(j2=True, drag=True, srp=True, third_body=True)])
def test_configured_perturbations_are_finite_and_propagate(model):
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    initial = ReferenceState(epoch, np.array([6.8e6, 0, 0]), np.array([0, 7600, 0]))
    result = propagate(initial, 60, context(), model=model)
    assert np.isfinite(result["states_si"]).all()
    assert result["times_sec"][-1] == 60
    assert result["manifest"]["forces"]["srp"] == model.srp


def test_profile_rejects_bad_inputs_and_unbounded_output():
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    with pytest.raises(ValueError):
        ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 7500, 0]), frame="GCRF")
    initial = ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 7500, 0]))
    for kwargs in ({"sample_sec": 1e-9}, {"rtol": .1}, {"max_step_sec": float('nan')}):
        with pytest.raises(ValueError):
            propagate(initial, 10, context(), **kwargs)
    with pytest.raises(ValueError):
        ForceModel(atmosphere="implicit-fallback")
    with pytest.raises(ValueError):
        ForceModel(mass_kg=-1)


def test_legacy_snapshot_explicitly_rejects_reference_profile():
    from space_env.simulation.snapshot import digest, world_from_payload, world_payload
    from tests.test_execution_integrity import world

    w = world()
    assert w.engines["a"].physics_profile == "legacy-training/1"
    payload = world_payload(w)
    assert payload["definitions"]["a"]["physics_profile"] == "legacy-training/1"
    payload["definitions"]["a"]["physics_profile"] = "gcrs-reference/1"
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    with pytest.raises(ValueError, match="separate"):
        world_from_payload(payload)


def test_reference_cli_produces_sealed_profile_and_preserves_bad_output(tmp_path, capsys):
    from space_env.cli import main
    from space_env.simulation.snapshot import digest

    spec = tmp_path / "state.json"
    data = {"epoch_utc": "2026-07-18T12:00:00Z", "frame": "GCRS", "r_m": [7e6, 0, 0],
            "v_m_s": [0, 7500, 0], "duration_sec": 600, "forces": {"j2": True}}
    spec.write_text(json.dumps(data))
    output = tmp_path / "orbit.json"
    main(["reference-propagate", str(spec), str(output)])
    assert json.loads(capsys.readouterr().out)["frame"] == "GCRS"
    result = json.loads(output.read_text())
    assert result["sha256"] == digest({k: v for k, v in result.items() if k != "sha256"})
    assert result["manifest"]["forces"]["j2"] is True
    before = output.read_bytes()
    data["frame"] = "GCRF"
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        main(["reference-propagate", str(spec), str(output)])
    assert output.read_bytes() == before


def test_eccentric_inclined_24h_arc_matches_kepler_equation():
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    a, eccentricity, inclination = 7.4e6, .05, .9
    rotation = np.array([[1, 0, 0], [0, np.cos(inclination), -np.sin(inclination)],
                         [0, np.sin(inclination), np.cos(inclination)]])
    initial = ReferenceState(epoch, rotation @ np.array([a*(1-eccentricity), 0, 0]),
                             rotation @ np.array([0, np.sqrt(MU_EARTH/a*(1+eccentricity)/(1-eccentricity)), 0]))
    result = propagate(initial, 86400, context(), sample_sec=900)
    motion = np.sqrt(MU_EARTH/a**3)
    errors = []
    for clock, state in zip(result["times_sec"], result["states_si"]):
        mean = (motion*clock) % (2*np.pi)
        eccentric = mean
        for _ in range(12):
            eccentric -= (eccentric-eccentricity*np.sin(eccentric)-mean)/(1-eccentricity*np.cos(eccentric))
        position = rotation @ np.array([a*(np.cos(eccentric)-eccentricity), a*np.sqrt(1-eccentricity**2)*np.sin(eccentric), 0])
        errors.append(np.linalg.norm(position-np.array(state[:3])))
    assert max(errors) < 1.


def test_24h_j2_numerical_convergence_does_not_claim_independent_force_truth():
    initial = ReferenceState(Epoch.utc("2026-07-18T12:00:00Z"), np.array([7e6, 0, 0]), np.array([0, 5000, 5700]))
    left = propagate(initial, 86400, context(), model=ForceModel(j2=True), max_step_sec=120, sample_sec=86400)
    right = propagate(initial, 86400, context(), model=ForceModel(j2=True), max_step_sec=60, sample_sec=86400)
    assert np.linalg.norm(np.array(left["states_si"][-1][:3])-np.array(right["states_si"][-1][:3])) < .1


def test_six_dimensional_transform_agrees_with_astropy_on_identical_eop(monkeypatch):
    from astropy import units as u
    from astropy.coordinates import (
        GCRS,
        ITRS,
        CartesianDifferential,
        CartesianRepresentation,
    )
    from astropy.table import QTable
    from astropy.time import Time as AstroTime
    from astropy.utils import iers

    monkeypatch.setattr(iers.conf, "auto_download", False)
    c, epoch = context(), Epoch.utc("2026-07-18T12:00:00Z")
    nodes = c.eop.manifest()["rows"]
    table = QTable({"MJD": [AstroTime(row["utc"]).mjd for row in nodes] * u.d,
                    "UT1_UTC": [row["dut1_s"] for row in nodes] * u.s,
                    "PM_x": [row["xp_arcsec"] for row in nodes] * u.arcsec,
                    "PM_y": [row["yp_arcsec"] for row in nodes] * u.arcsec})
    moment = AstroTime(epoch.iso(), scale="utc")
    moment.delta_ut1_utc = c.eop.at(epoch).dut1_s
    state = np.array([6e6, -3e6, 2e6, 1000, 6500, -2500])
    with iers.earth_orientation_table.set(iers.IERS(table)):
        coordinate = GCRS(CartesianRepresentation(state[:3]*u.m,
                          differentials=CartesianDifferential(state[3:]*u.m/u.s)), obstime=moment)
        result = coordinate.transform_to(ITRS(obstime=moment))
    expected = np.concatenate((result.cartesian.xyz.to_value(u.m),
                               result.cartesian.differentials['s'].d_xyz.to_value(u.m/u.s)))
    actual = c.transform(state, epoch, source="GCRS", destination="ITRS")["state_si"]
    assert np.linalg.norm(actual[:3]-expected[:3]) < 1e-5
    assert np.linalg.norm(actual[3:]-expected[3:]) < 1e-4


def test_reference_report_declares_what_is_not_qualified():
    from space_env.validation.gcrs_report import reference_physics_report

    report = reference_physics_report(long_arcs=True)
    assert report["passed"]
    assert len(report["checks"]) == 7
    assert "perturbed 24/72h against independent external force models" in report["unqualified"]


@pytest.mark.parametrize("bad", [np.eye(3), np.diag([1, 1, 1, -1, 1, 1]), np.full((6, 6), float('nan'))])
def test_bad_covariance_is_not_propagated(bad):
    with pytest.raises(ValueError):
        context().transform([7e6, 0, 0, 0, 7500, 0], Epoch.utc("2026-07-18T12:00:00Z"),
                            source="GCRS", destination="ITRS", covariance=bad)


def test_eop_sparse_data_and_time_outside_applicability_fail_early():
    with pytest.raises(ValueError, match="gap"):
        EOPSeries([
            {"utc": "2026-07-18T00:00:00Z", "dut1_s": 0, "xp_arcsec": 0, "yp_arcsec": 0, "status": "synthetic"},
            {"utc": "2026-07-20T00:00:00Z", "dut1_s": 0, "xp_arcsec": 0, "yp_arcsec": 0, "status": "synthetic"},
        ], source="sparse invalid fixture")
    with pytest.raises(ValueError, match="applicability"):
        Epoch.utc("2026-12-31T23:59:59Z").shifted(10)


def test_mixed_profile_world_restore_fails_before_any_actor_is_changed():
    from space_env.simulation.snapshot import world_from_payload, world_payload
    from space_env.simulation.world import MissionWorld
    from tests.test_execution_integrity import world

    a = world()
    b = world_from_payload(world_payload(a))
    b.engines["a"].spacecraft.config.name = "b"
    w = MissionWorld({"a": a.engines["a"], "b": b.engines["a"]}, epoch=a.epoch, horizon_sec=10)
    snapshot = w.snapshot()
    before = world_payload(w)
    snapshot["engines"]["a"]["time_sec"] = 1
    snapshot["engines"]["b"]["physics_profile"] = "gcrs-reference/1"
    with pytest.raises(ValueError, match="profile"):
        w.restore(snapshot)
    assert world_payload(w) == before
