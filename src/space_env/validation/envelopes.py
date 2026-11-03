"""Frozen analytical evidence, with fixed budgets and deliberately narrow scope.

This verifies equations at enumerated points, not calibration or every value in
an input interval. No oracle is imported from the offline vector generator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from space_env.simulation.snapshot import digest

FIXTURE = (
    Path(__file__).resolve().parents[1] / "reference/fixtures/analytic_vectors_v1.json"
)
INDEPENDENCE = "independently-coded-analytic-equations/shared-parameters"
# Engineering numerical comparison budgets, declared before first execution.
BUDGETS = {
    "orbit": {
        "position_m": 2.0,
        "velocity_m_s": 0.003,
        "interpolated_position_m": 20.0,
        "interpolated_velocity_m_s": 0.5,
    },
    "conduction": {"temperature_k": 1e-6, "balance_j": 1e-6},
    "radiation": {"temperature_k": 1e-5, "balance_j": 0.001},
    "rigid": {"quaternion_chord": 1e-9, "omega_rad_s": 1e-11, "wheel_nms": 1e-11},
    "rf": {"fspl_db": 1e-10, "eirp_dbw": 1e-10, "cn0_db_hz": 1e-10, "margin_db": 1e-10},
}
# Pin the complete case input/time domain. Re-signing a fixture cannot expand it,
# lower integration accuracy, replace constants, or inject a larger error budget.
CASE_DOMAINS = {
    "elliptic-leo": "a05e24e76c185563b2961d5d97f4812b2e51f16615d893c0bad3065fc73f00c2",
    "elliptic-high-e": "ede3282ee8302c8b8f125428eb1833a054c85f22f63f5fe6fd7d823f1713b4eb",
    "conduction-0w": "90174d3f3fedde48fa5c87696d5d280fc320a251eb3a823640166bcd0a27d147",
    "conduction-4w": "d46eaa76181b8bb325a03c390d3aaa17004f055b39d764085f31d9523dc4aaa8",
    "vacuum-cooling": "fb5efbf8dc95cdcc83f455968548234134bfaacadfb9225ade47dcc61fba3deb",
    "rigid-coast": "f34d13a3607051783fd99f6398139d96dc7455f12ccc0a808ba529d8e020bf72",
    "rigid-wheel": "a4c6c64bb0ab61a57333bc32058623e877b8be714b669faf384af1e3774519ee",
    "rigid-external": "a32c6ad62f1f2d3cdc373f080b17115860ef68745d3e7f79e036d907b8ddefef",
    "rf-s-band": "a8a33c4ea64b04989d0c1411511f113fd876156381aa66bac614308b3f964304",
    "rf-x-band": "64bb7c7826557fc122e8884b814f518047cc899a6909c5a1d096938df19508f3",
}
SCOPES = {
    "orbit": "two eccentric inclined central-only GCRS orbits; shared mu/initial state; 60s sampling",
    "conduction": "two equal-capacity synthetic nodes, constant sources, no bath/radiation/heater",
    "radiation": "one synthetic node emitting into 0K vacuum; constant capacity/emissivity",
    "rigid": "principal-axis coast/unsaturated wheel/external torque; ideal three-wheel standalone kernel",
    "rf": "two free-space scalar link budgets; fixed gains/noise and declared scalar losses",
}
GAPS = {
    "orbit": [
        "flight ephemeris/calibrated force uncertainty",
        "joint drag/SRP/attitude truth",
    ],
    "attitude": [
        "off-axis coupled pointing/tracking truth",
        "wheel hardware/friction/calibration",
    ],
    "power": [
        "external battery/PV/PMAD calibration and uncertainty under coupled loads"
    ],
    "thermal": [
        "thermal-vacuum measurements",
        "time-varying Earth/surface optical properties",
    ],
    "payload": ["instrument/scene-calibrated image/SAR quality and pointing error"],
    "rf": ["measured channel/weather/antenna/modem performance"],
    "dtn": ["external network/protocol traces and operational interoperability"],
    "coupled_mission": [
        "real tasking/slew/multiplane/multistation workload qualification",
        "external flight/operations validation",
        "long-run resource enforcement",
    ],
}


def _finite_tree(value, depth=0):
    if depth > 8:
        raise ValueError("analytic fixture nesting exceeds budget")
    if isinstance(value, dict):
        if len(value) > 32 or any(not isinstance(k, str) for k in value):
            raise ValueError("invalid analytic object")
        for item in value.values():
            _finite_tree(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 128:
            raise ValueError("analytic vector count exceeds budget")
        for item in value:
            _finite_tree(item, depth + 1)
    elif isinstance(value, str):
        if len(value) > 1024:
            raise ValueError("analytic text exceeds budget")
    elif (
        type(value) not in (int, float)
        or abs(value) > 1e100
        or not math.isfinite(value)
    ):
        raise ValueError("analytic fixture requires finite non-boolean scalars")


def validate_analytic_fixture(payload):
    """Validate all domains before any solver work. Expected values are not trusted."""
    _finite_tree(payload)
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "provenance",
        "cases",
        "sha256",
    }:
        raise ValueError("invalid analytic fixture fields")
    body = {k: v for k, v in payload.items() if k != "sha256"}
    if (
        payload["schema"] != "space-env.analytic-vectors/1"
        or digest(body) != payload["sha256"]
    ):
        raise ValueError("unsupported or corrupt analytic fixture")
    p = payload["provenance"]
    fields = {
        "generator",
        "generator_sha256",
        "independence",
        "backend",
        "shared_inputs",
        "scope",
    }
    if not isinstance(p, dict) or set(p) != fields or any(not p[k] for k in fields):
        raise ValueError("incomplete analytic provenance")
    if (
        p["independence"] != INDEPENDENCE
        or p["generator"] != "scripts/generate_analytic_vectors.py"
    ):
        raise ValueError("unsupported analytic independence or generator")
    if not isinstance(p["generator_sha256"], str) or len(p["generator_sha256"]) != 64:
        raise ValueError("invalid generator digest")
    try:
        if len(bytes.fromhex(p["generator_sha256"])) != 32:
            raise ValueError("invalid generator digest")
    except ValueError as exc:
        raise ValueError("invalid generator digest") from exc
    if p["shared_inputs"] != [
        "SI constants",
        "initial states",
        "synthetic model parameters",
        "binary64 arithmetic",
    ]:
        raise ValueError("analytic shared inputs must be explicit")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASE_DOMAINS):
        raise ValueError("analytic suite must contain every frozen case")
    seen = set()
    widths = {"orbit": 6, "conduction": 2, "radiation": 1, "rigid": 10, "rf": 4}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "kind",
            "inputs",
            "times_sec",
            "expected",
        }:
            raise ValueError("invalid analytic case fields")
        name, kind = case["id"], case["kind"]
        if not isinstance(name, str) or name not in CASE_DOMAINS or name in seen:
            raise ValueError("unknown or duplicate analytic case")
        seen.add(name)
        domain = {k: case[k] for k in ("id", "kind", "inputs", "times_sec")}
        if digest(domain) != CASE_DOMAINS[name]:
            raise ValueError("analytic case outside frozen input/time domain")
        expected = case["expected"]
        if not isinstance(expected, list) or len(expected) != len(case["times_sec"]):
            raise ValueError("analytic expected rows differ from time domain")
        for row in expected:
            if (
                not isinstance(row, list)
                or len(row) != widths[kind]
                or any(type(v) not in (int, float) for v in row)
            ):
                raise ValueError("invalid analytic expected vector")
    return payload


def _check(case, metric, measured, *, time_sec=None, solver=None):
    limit = BUDGETS[case["kind"]][metric]
    if not math.isfinite(measured):
        raise ValueError("nonfinite analytic measurement")
    return {
        "case": case["id"],
        "kind": case["kind"],
        "metric": metric,
        "measured": float(measured),
        "tolerance": limit,
        "passed": measured <= limit,
        **({"time_sec": time_sec} if time_sec is not None else {}),
        **({"solver": solver} if solver else {}),
    }


def _orbit(case, long_arcs):
    from space_env.core.constants import MU_EARTH
    from space_env.reference import (
        EOPSeries,
        Epoch,
        FrameContext,
        ReferenceState,
        propagate,
    )
    from space_env.reference.access import ReferenceEphemeris

    p = case["inputs"]
    if p["mu_m3_s2"] != MU_EARTH:
        raise ValueError("analytic orbit and native constants differ")
    pairs = [
        (t, row)
        for t, row in zip(case["times_sec"], case["expected"])
        if long_arcs or t <= 600
    ]
    initial = p["initial_si"]
    result = propagate(
        ReferenceState(
            Epoch.utc(p["epoch_utc"]), np.array(initial[:3]), np.array(initial[3:])
        ),
        max(t for t, _ in pairs),
        FrameContext(EOPSeries.packaged()),
        sample_sec=p["sample_sec"],
    )
    ephemeris = ReferenceEphemeris.from_result(result)
    rows = []
    for t, exact in pairs:
        sampled = t % p["sample_sec"] == 0
        if sampled:
            state = np.array(result["states_si"][result["times_sec"].index(t)])
        else:
            state = ephemeris.state(t)
        prefix = "" if sampled else "interpolated_"
        for metric, section in (
            ("position_m", slice(0, 3)),
            ("velocity_m_s", slice(3, 6)),
        ):
            error = np.linalg.norm(state[section] - np.asarray(exact)[section])
            rows.append(
                _check(
                    case,
                    prefix + metric,
                    float(error),
                    time_sec=t,
                    solver="DOP853" if sampled else "Hermite/60s",
                )
            )
    return rows


def _thermal(case):
    from space_env.reference.thermal_network import (
        HEAT_SOURCES,
        ThermalNetworkSettings,
        advance_network,
    )

    p = case["inputs"]
    radiative = case["kind"] == "radiation"
    names = ["a"] if radiative else ["a", "b"]
    initial = [p["initial_k"]] if radiative else p["initial_k"]
    nodes = {
        name: {
            "initial_k": temp,
            "capacity_j_per_k": p["capacity_j_per_k"],
            "trip_k": 1000.0,
            "resume_k": 900.0,
            **(
                {"emissivity": p["emissivity"], "emission_area_m2": p["area_m2"]}
                if radiative
                else {}
            ),
        }
        for name, temp in zip(names, initial)
    }
    if radiative:
        from space_env.reference.thermal_network import SIGMA_SB

        if p["sigma_w_m2_k4"] != SIGMA_SB:
            raise ValueError("analytic radiation and native constants differ")
    links = (
        []
        if radiative
        else [{"a": "a", "b": "b", "conductance_w_per_k": p["conductance_w_per_k"]}]
    )
    source = [0.0] if radiative else p["internal_w"]
    rows = []
    for solver in ("DOP853", "Radau"):
        config = ThermalNetworkSettings(
            model_id="analytic-envelope/1",
            parameter_status="synthetic",
            source="frozen synthetic analytic inputs; not calibrated hardware",
            nodes=nodes,
            links=links,
            routing={k: {"a": 1.0} for k in HEAT_SOURCES},
            earth_environment=False,
            max_evaluations=50000,
            solver=solver,
        )
        for t, exact in zip(case["times_sec"], case["expected"]):
            result = advance_network(
                config,
                dict(zip(names, initial)),
                t,
                internal_w=dict(zip(names, source)),
                external_w=dict.fromkeys(names, 0.0),
            )
            error = max(
                abs(result["temperatures_k"][name] - temp)
                for name, temp in zip(names, exact)
            )
            rows.append(_check(case, "temperature_k", error, time_sec=t, solver=solver))
            rows.append(
                _check(
                    case,
                    "balance_j",
                    max(abs(v) for v in result["node_balance_error_j"].values()),
                    time_sec=t,
                    solver=solver,
                )
            )
    return rows


def _rigid(case):
    from space_env.reference.rigid_attitude import (
        RigidAttitudeModel,
        RigidAttitudeState,
        advance_rigid_attitude,
    )

    p = case["inputs"]
    model = RigidAttitudeModel(
        model_id="analytic-envelope/1",
        source="synthetic principal-axis analytic fixture",
        inertia_kg_m2=tuple(
            tuple(p["inertia_kg_m2"][i] if i == j else 0.0 for j in range(3))
            for i in range(3)
        ),
        wheel_inertia_kg_m2=(p["wheel_inertia_kg_m2"],) * 3,
        max_torque_nm=(0.01,) * 3,
        max_momentum_nms=(1.0,) * 3,
        max_step_sec=p["max_step_sec"],
    )
    initial = RigidAttitudeState(omega_body_rad_s=(0.0, 0.0, p["omega_z_rad_s"]))
    rows = []
    for t, exact in zip(case["times_sec"], case["expected"]):
        result, _ = advance_rigid_attitude(
            model,
            initial,
            t,
            motor_torque_nm=(0.0, 0.0, p["motor_z_nm"]),
            external_torque_body_nm=(0.0, 0.0, p["external_z_nm"]),
        )
        q, wanted = np.array(result.quaternion_wxyz), np.array(exact[:4])
        # q and -q represent the same orientation.
        rows.append(
            _check(
                case,
                "quaternion_chord",
                float(min(np.linalg.norm(q - wanted), np.linalg.norm(q + wanted))),
                time_sec=t,
                solver="RK4",
            )
        )
        rows.append(
            _check(
                case,
                "omega_rad_s",
                float(np.linalg.norm(np.array(result.omega_body_rad_s) - exact[4:7])),
                time_sec=t,
                solver="RK4",
            )
        )
        rows.append(
            _check(
                case,
                "wheel_nms",
                float(
                    np.linalg.norm(np.array(result.wheel_momentum_nms) - exact[7:10])
                ),
                time_sec=t,
                solver="RK4",
            )
        )
    return rows


def _rf(case):
    from space_env.reference.radio import (
        BOLTZMANN_J_K,
        C_M_S,
        RFReceiver,
        RFTransmitter,
        rf_budget,
    )

    p = case["inputs"]
    if p["c_m_s"] != C_M_S or p["boltzmann_j_k"] != BOLTZMANN_J_K:
        raise ValueError("analytic RF and native constants differ")
    tx = RFTransmitter(
        frequency_hz=p["frequency_hz"],
        rf_power_w=p["rf_power_w"],
        gain_dbi=p["gain_dbi"],
        feeder_loss_db=p["feeder_loss_db"],
        modes=(
            {
                "id": "test",
                "information_rate_bps": p["information_rate_bps"],
                "bits_per_symbol": 2,
                "code_rate": 0.5,
                "required_eb_n0_db": p["required_eb_n0_db"],
                "implementation_loss_db": p["implementation_loss_db"],
            },
        ),
    )
    rx = RFReceiver(
        gt_db_k=p["gt_db_k"],
        min_frequency_hz=1e9,
        max_frequency_hz=10e9,
        bandwidth_hz=20e6,
        polarization_loss_db=p["polarization_loss_db"],
        other_loss_db=p["other_loss_db"],
    )
    result = rf_budget(tx, rx, p["range_m"], fade_db=p["fade_db"])
    actual = [
        result["fspl_db"],
        result["eirp_dbw"],
        result["cn0_db_hz"],
        result["modes"][0]["margin_db"],
    ]
    return [
        _check(case, metric, abs(value - exact))
        for metric, value, exact in zip(
            ("fspl_db", "eirp_dbw", "cn0_db_hz", "margin_db"),
            actual,
            case["expected"][0],
        )
    ]


def verify_analytic_fixture(payload, *, long_arcs=False):
    """Run every case; numeric mismatches return failed evidence, never disappear."""
    if type(long_arcs) is not bool:
        raise ValueError("long_arcs must be boolean")
    validate_analytic_fixture(payload)
    rows, outcomes, errors = [], {}, []
    for case in payload["cases"]:
        kind = case["kind"]
        try:
            if kind == "orbit":
                checks = _orbit(case, long_arcs)
            elif kind in {"radiation", "conduction"}:
                checks = _thermal(case)
            elif kind == "rigid":
                checks = _rigid(case)
            else:
                checks = _rf(case)
            if not checks:
                raise ValueError("analytic case produced no measurements")
            rows.extend(checks)
            outcomes[case["id"]] = (
                "passed" if all(r["passed"] for r in checks) else "failed"
            )
        except (ValueError, RuntimeError, OverflowError) as exc:
            outcomes[case["id"]] = "execution_error"
            errors.append(
                {"case": case["id"], "type": type(exc).__name__, "message": str(exc)}
            )
    return copy.deepcopy(
        {
            "fixture_sha256": payload["sha256"],
            "provenance": payload["provenance"],
            "passed": not errors and all(row["passed"] for row in rows),
            "checks": rows,
            "execution_errors": errors,
            "evaluated_cases": [
                {
                    "id": c["id"],
                    "kind": c["kind"],
                    "scope": SCOPES[c["kind"]],
                    "status": outcomes[c["id"]],
                    "inputs": c["inputs"],
                    "evaluated_times_sec": [
                        t
                        for t in c["times_sec"]
                        if outcomes[c["id"]] != "execution_error"
                        and (long_arcs or c["kind"] != "orbit" or t <= 600)
                    ],
                    "deferred_times_sec": [
                        t
                        for t in c["times_sec"]
                        if not long_arcs and c["kind"] == "orbit" and t > 600
                    ],
                }
                for c in payload["cases"]
            ],
        }
    )


def analytic_envelope_report(*, long_arcs=False, external=False):
    """A passing numerical suite leaves hardware/mission applicability unresolved."""
    from space_env import __version__

    if type(external) is not bool:
        raise ValueError("external must be boolean")
    analytic = verify_analytic_fixture(
        json.loads(FIXTURE.read_text()), long_arcs=long_arcs
    )
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"})
    }
    try:
        distribution_version = version("space-env")
    except PackageNotFoundError:
        distribution_version = None
    result = {
        "schema": "space-env.physics-envelopes/1",
        "long_arcs": long_arcs,
        "passed": analytic["passed"],
        "qualification_status": "needs_external_evidence",
        "industrial_readiness": False,
        "analytic": analytic,
        "unqualified": copy.deepcopy(GAPS),
        "budgets": copy.deepcopy(BUDGETS),
        "solver_controls": {
            "thermal": {
                "max_step_sec": 0.05,
                "max_evaluations_per_call": 50000,
                "rtol": 1e-9,
                "atol_k": 1e-7,
            },
            "orbit": {
                "max_step_sec": 60.0,
                "rtol": 1e-11,
                "position_atol_m": 1e-4,
                "velocity_atol_m_s": 1e-7,
            },
        },
        "budget_basis": "predeclared engineering comparison thresholds; not physical uncertainty",
        "mission_evidence_policy": "historical campaign receipts bind their original build; this report grants no mission qualification",
        "build": {
            "package_version": __version__,
            "installed_distribution_version": distribution_version,
            "distribution_matches_source_version": distribution_version == __version__,
            "source_data_sha256": digest(hashes),
            "files_sha256": hashes,
            "inventory_policy": "all package .py/.json/.csv/.yaml/.yml files; not a wheel signature or native mission build ID",
            "dependencies": {
                name: version(name) for name in ("numpy", "scipy", "pyerfa", "pydantic")
            },
        },
    }
    if external:
        from space_env.validation.gcrs_report import reference_physics_report
        from space_env.validation.orbit_fixtures import verify_packaged_orbit_fixtures

        result["external_formula_shared_inputs"] = verify_packaged_orbit_fixtures()
        result["backend_regression_and_circular"] = reference_physics_report(
            long_arcs=long_arcs
        )
        result["passed"] = bool(
            result["passed"]
            and result["external_formula_shared_inputs"]["passed"]
            and result["backend_regression_and_circular"]["passed"]
        )
    result["evidence_registry"] = [
        {
            "id": "analytic-vectors/1",
            "independence": INDEPENDENCE,
            "status": "passed" if analytic["passed"] else "failed",
            "scope": "only enumerated input/time points; no continuous validated parameter interval",
            "fixture_sha256": analytic["fixture_sha256"],
        },
        {
            "id": "frozen-boinor-j2",
            "independence": "external-force-formula/shared-inputs",
            "status": (
                "passed"
                if result["external_formula_shared_inputs"]["passed"]
                else "failed"
            )
            if external
            else "not_run",
            "scope": "existing 24/72h J2 and 24h J2+third-body cases; shared ERFA/EOP/constants",
        },
        {
            "id": "erfa-c2t06a",
            "independence": "same-backend-upstream-regression",
            "status": (
                "passed"
                if result["backend_regression_and_circular"]["checks"][0]["passed"]
                else "failed"
            )
            if external
            else "not_run",
            "scope": "one upstream frame vector; not independent frame truth",
        },
        {
            "id": "operational-campaigns",
            "independence": "native-replay-and-resource-ledgers",
            "status": "not_run",
            "scope": "historical B11-Q4 receipt belongs to its original wheel; not transferred",
        },
    ]
    result["sha256"] = digest(result)
    return result
