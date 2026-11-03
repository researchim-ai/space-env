"""Independent input generation, exact dual certificates and native witnesses."""

from __future__ import annotations

import copy
import json
import random
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from examples.scaling.independent import configuration
from space_env.planning import PlanningRun, make_input
from space_env.scaling.benchmark import Workload, run_scale
from space_env.scaling.contact_capacity import (
    contact_capacity,
    dual_upper_bound,
    verify_capacity_certificate,
)
from space_env.scaling.independent import (
    IndependentFleetSpec,
    OrbitPlane,
    assess_geometry,
    circular_state,
    geometry_delivery_evidence,
    independent_bundle,
    independent_workload,
    verify_independent_bundle,
)
from space_env.simulation.snapshot import digest


def config():
    c = configuration()
    c.update(horizon_sec=60.0, decision_sec=30.0, min_delivered=0)
    for p in c["planes"]:
        p["phases_deg"] = p["phases_deg"][:1]
    return c


def toy(*, terminals=1, same_actor=False, blocked=False):
    products = [
        {"id": "p", "actor": "a", "destination": "s", "size_bits": 10.0},
        {
            "id": "q",
            "actor": "a" if same_actor else "b",
            "destination": "s",
            "size_bits": 10.0,
        },
    ]
    vehicles = {a: {"rate_bps": 1.0} for a in ["a", "b"]}
    sites = {"s": {"terminals": terminals}}
    windows = {(a, "s"): [(0.0, 10.0)] for a in vehicles}
    return contact_capacity(
        products,
        vehicles,
        sites,
        windows,
        10.0,
        [{"station": "s", "start_sec": 0.0, "end_sec": 10.0}] if blocked else [],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "same_plane",
        "same_plane_opposite",
        "duplicate_id",
        "duplicate_site",
        "bad_phase",
        "bool_phase",
        "too_many",
        "gate",
        "decision",
    ],
)
def test_invalid_independent_contracts(mutation):
    c = config()
    if mutation == "same_plane":
        c["planes"][1].update(inclination_deg=55.0, raan_deg=360.0)
    if mutation == "same_plane_opposite":
        c["planes"][1].update(inclination_deg=125.0, raan_deg=180.0)
    if mutation == "duplicate_id":
        c["planes"][1]["id"] = c["planes"][0]["id"]
    if mutation == "duplicate_site":
        c["stations"][1]["name"] = c["stations"][0]["name"]
    if mutation == "bad_phase":
        c["planes"][0]["phases_deg"] = [360.0]
    if mutation == "bool_phase":
        c["planes"][0]["phases_deg"] = [True]
    if mutation == "too_many":
        c["planes"] *= 5
    if mutation == "gate":
        c["min_delivered"] = 64
    if mutation == "decision":
        c["decision_sec"] = 120.0
    with pytest.raises(ValueError):
        IndependentFleetSpec.model_validate(c)


def test_circular_state_has_correct_invariants_and_plane_normal():
    p = OrbitPlane(id="p", inclination_deg=55.0, raan_deg=120.0, phases_deg=[0.0])
    for phase in [0.0, 90.0, 180.0, 270.0]:
        r, v = circular_state(p, phase)
        assert np.linalg.norm(r) == pytest.approx(p.radius_m)
        assert float(r @ v) == pytest.approx(0.0, abs=1e-5)
        assert np.cross(r, v)[2] / np.linalg.norm(np.cross(r, v)) == pytest.approx(
            np.cos(np.deg2rad(55))
        )


def test_station_input_change_does_not_move_orbits_or_filter_products():
    c = config()
    before = copy.deepcopy(c)
    a = independent_workload(c)
    c["stations"][0].update(lat_deg=80.0, lon_deg=-160.0)
    b = independent_workload(c)
    assert (
        a["scenario"]["definition"]["ephemerides"]
        == b["scenario"]["definition"]["ephemerides"]
    )
    assert len(a["scenario"]["definition"]["products"]) == 6
    assert before["stations"][0]["lat_deg"] == 0.0
    assert (
        a["geometry_gates"]["native_spec_sha256"]
        != b["geometry_gates"]["native_spec_sha256"]
    )
    assert independent_workload(before) == a


def test_zero_contact_pairs_and_products_are_retained():
    c = config()
    c["stations"] = [
        {"name": "pole", "lat_deg": 90.0, "lon_deg": 0.0, "min_elevation_deg": 89.0}
    ]
    w = independent_workload(c)
    a = assess_geometry(w["scenario"])
    assert a["pairs_with_contacts"] == 0 and a["pairs_without_contacts"] == 2
    assert len(w["scenario"]["definition"]["products"]) == 2
    assert a["capacity"]["certificate"]["integer_upper_binary_lp_floor"] == 0


def test_capacity_station_and_actor_constraints_are_distinct():
    one = toy()
    two = toy(terminals=2)
    same = toy(terminals=2, same_actor=True)
    assert one["certificate"]["integer_upper_binary_lp_floor"] == 1
    assert two["certificate"]["integer_upper_binary_lp_floor"] == 2
    assert same["certificate"]["integer_upper_binary_lp_floor"] == 1
    for r in [one, two, same]:
        assert verify_capacity_certificate(r)["verified"]
    assert toy(blocked=True)["certificate"]["integer_upper_conservative_ceiling"] == 0


def test_arbitrary_duals_still_bound_known_feasible_solution():
    p = toy()["problem"]
    rng = random.Random(73)
    for _ in range(20):
        y = [rng.uniform(-50, 50) for _ in p["capacities_sec"]]
        z = [rng.uniform(-50, 50) for _ in p["products"]]
        r = dual_upper_bound(p, y, z)
        assert (
            Fraction(
                int(r["fractional_upper_numerator"]),
                int(r["fractional_upper_denominator"]),
            )
            >= 1
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_capacity",
        "negative_work",
        "flow_index",
        "bool_index",
        "negative_flow",
        "nan_dual",
        "shape",
    ],
)
def test_invalid_certificate_inputs_rejected(mutation):
    r = toy()
    p = r["problem"]
    y = r["inequality_dual"]
    z = r["equality_dual"]
    if mutation == "negative_capacity":
        p["capacities_sec"][0] = -1.0
    if mutation == "negative_work":
        p["products"][0]["required_tx_sec"] = -1.0
    if mutation == "flow_index":
        p["flows"][0]["product"] = 100
    if mutation == "bool_index":
        p["flows"][0]["product"] = True
    if mutation == "negative_flow":
        p["flows"][0]["upper_sec"] = -1.0
    if mutation == "nan_dual":
        z[0] = float("nan")
    if mutation == "shape":
        y = []
    with pytest.raises(ValueError):
        dual_upper_bound(p, y, z)


def test_resealed_fake_certificate_not_accepted():
    r = toy()
    r["certificate"]["integer_upper_binary_lp_floor"] = 0
    r.pop("sha256")
    r["sha256"] = digest(r)
    with pytest.raises(ValueError, match="exact dual"):
        verify_capacity_certificate(r)


def test_solver_failure_retains_safe_trivial_bound_without_success(monkeypatch):
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "linprog",
        lambda *a, **kw: SimpleNamespace(
            success=False, status=1, message="intentional time limit"
        ),
    )
    r = toy()
    assert not r["solver_success"] and r["primal_fractional_products_estimate"] is None
    assert r["certificate"]["integer_upper_binary_lp_floor"] == 2
    assert verify_capacity_certificate(r)["verified"]


def test_bundle_reproduction_and_tampering():
    r = independent_bundle(config())
    assert verify_independent_bundle(r)["verified"]
    r["input_spec"]["stations"][0]["lon_deg"] = 150.0
    r.pop("sha256")
    r["sha256"] = digest(r)
    with pytest.raises(ValueError, match="geometry"):
        verify_independent_bundle(r)


def test_resealed_count_and_readiness_metadata_rejected():
    r = independent_bundle(config())
    r["assessment"]["pairs_without_contacts"] += 1
    r["assessment"].pop("sha256")
    r["assessment"]["sha256"] = digest(r["assessment"])
    r.pop("sha256")
    r["sha256"] = digest(r)
    with pytest.raises(ValueError, match="metadata"):
        verify_independent_bundle(r)
    r = independent_bundle(config())
    r["industrial_readiness"] = True
    r.pop("sha256")
    r["sha256"] = digest(r)
    with pytest.raises(ValueError, match="envelope"):
        verify_independent_bundle(r)


def test_diagnostic_certificate_is_not_inserted_in_actor_input():
    w = independent_workload(config())
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    public = json.dumps(make_input(run.session).view())
    for secret in [
        "capacity_certificate",
        "inequality_dual",
        "geometry_gates",
        "fractional_upper",
    ]:
        assert secret not in public


def test_native_witness_rejects_report_identity_and_gives_explicit_bounds():
    w = independent_workload(config())
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    run.run()
    report = run.session._runtime._backend.report()
    g = Workload.model_validate(w).geometry_gates
    r = geometry_delivery_evidence(w["scenario"], report, g)
    assert r["passed"] and r["total_products"] == 6
    bad = copy.deepcopy(report)
    bad["products"].pop()
    with pytest.raises(ValueError, match="identity"):
        geometry_delivery_evidence(w["scenario"], bad, g)


def test_legacy_workload_does_not_serialize_optional_geometry_gate():
    w = independent_workload(config())
    w.pop("geometry_gates")
    assert "geometry_gates" not in Workload.model_validate(w).model_dump(mode="json")


def test_independent_cli_alias_and_bounded_input(tmp_path, capsys):
    from space_env.cli import main

    path = tmp_path / "input.json"
    path.write_text(json.dumps(config()))
    with pytest.raises(SystemExit) as exc:
        main(["independent-fleet", str(path), str(path)])
    assert exc.value.code == 2
    assert (
        json.loads(capsys.readouterr().out)["error"]["code"]
        == "independent_contract_error"
    )
    target = tmp_path / "bundle.json"
    main(["independent-fleet", str(path), str(target)])
    assert verify_independent_bundle(json.loads(target.read_text()))["verified"]


def test_real_native_independent_fixed_site_workload_and_process_restore(tmp_path):
    w = independent_workload(config())
    w.update(comparison_group="restored")
    restored = copy.deepcopy(w)
    restored.update(id="restored", resume_after_decisions=1, resume_in_new_process=True)
    r = run_scale(
        {"id": "geometry-short", "workloads": [w, restored]}, tmp_path / "native"
    )
    assert r["ok"], r["rows"]
    assert r["semantic_comparisons"]["restored"]["passed"]
    assert all(row["gates"]["geometry_delivery"] for row in r["rows"])
