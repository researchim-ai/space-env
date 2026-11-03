"""Q5-B: frozen off-axis gyrostat, rotating force and fixed-station comparisons.

No oracle equations or offline generator imports belong in this verifier.
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
from space_env.validation.envelopes import _finite_tree

FIXTURE = (
    Path(__file__).resolve().parents[1] / "reference/fixtures/pointing_vectors_v1.json"
)
INDEPENDENCE = (
    "analytic-gyrostat+independent-rk4+independent-contact-equations/shared-erfa-inputs"
)
# Frozen engineering comparison budgets, not calibrated physical uncertainties.
BUDGETS = {
    "attitude": {
        "rotation_frobenius": 2e-8,
        "omega_rad_s": 1e-9,
        "wheel_nms": 1e-10,
        "boresight_chord": 2e-8,
        "pointing_deg": 2e-6,
        "saturation_duration_sec": 1e-10,
    },
    "capacity": {
        "rotation_frobenius": 1e-6,
        "omega_rad_s": 1e-6,
        "wheel_nms": 1e-10,
        "boresight_chord": 1e-6,
        "pointing_deg": 1e-4,
        "saturation_duration_sec": 0.010001,
    },
    "refinement": {"rotation_frobenius": 1e-6, "fine_error_excess": 0.0},
    "burn": {
        "rotation_frobenius": 2e-8,
        "omega_rad_s": 1e-9,
        "wheel_nms": 1e-10,
        "direction_chord": 3e-6,
        "impulse_ns": 1e-5,
        "position_m": 0.002,
        "velocity_m_s": 2e-5,
        "mass_kg": 1e-10,
    },
    "contact": {
        "range_m": 0.1,
        "range_rate_m_s": 0.02,
        "elevation_deg": 1e-5,
        "azimuth_deg": 1e-5,
        "doppler_hz": 0.2,
        "pointing_chord": 1e-8,
        "window_count_error": 0.0,
        "boundary_sec": 0.02,
    },
}
CASE_DOMAINS = {
    "offaxis-coast": "3086f2b3ac75a100fc8acc1952047d0d2b5e445fbc2db2d9d7085910ffbc6a1b",
    "offaxis-wheel": "3cf2186e68ee28e08180d2ba7b754fc66b1645a870b6acab0e45407bce1bab83",
    "offaxis-positive-cap": "2c56403e128b207b3cbd973d75a1583bd4d367dbf8a19e3b9a4b6da7bef5931e",
    "offaxis-negative-cap": "d59c4913e1ac0a9b1b756095ad6643cf135259b433304ce56497e364f900f399",
    "offaxis-reverse-through-zero": "348d7b43cbe4d8274ef9d773acf05edab3948bdf4699f9d0dacb6a6c68ef403e",
    "offaxis-balanced-unloading": "692974d74f2af68a0c6c21fc41b6037e599af539cbcc4efd4f661fcf3545d35d",
    "offaxis-rotating-burn": "e4c33dac8b8d0b88102f0ac018503c85460592e96b1a6a9b90228f39acf0f2b2",
    "offaxis-offset-burn": "0d6799a1141e70ab2869b13496d58fec4bcdfa39c214ae32199b1a04e720fcef",
    "plane-a-equatorial": "b710212f4602123b325e22f74327d489f17baab211dc27a9be97eac99bc55891",
    "plane-a-iberia": "2ee31ba10a0d1bbae1af28818151be991c4ed3b24e8b4ebf802c64285d29d1ee",
    "plane-a-australia": "16a21bb5929689f4066c5d99e276303b2c65e098056ddeb0454e9ae9724615f8",
    "plane-a-arctic": "49115329e012f1f19ee89ed01a94eaca5c90815a3763ec07b0e37a2de260493c",
    "plane-b-equatorial": "942521060b6040535eb74ab3d42bb186e26b1a22760d49463d6c760d647fed61",
    "plane-b-iberia": "e7706d9bc924fb526807237feb3b634312bc8bde94b854c4ac2db1fb25c3d73c",
    "plane-b-australia": "eff08102e1aee465b97122e0367af7e9e9c778e0c6a2848af54eaa81a01bd907",
    "plane-b-arctic": "98dbe5c04fe3a79bdd343ee03eee8f64940be279aa516a75ccea54b2c0bd31e3",
}
SCOPES = {
    "attitude": "symmetric gyrostat with three-axis spin and tilted initial orientation; wheel torque on symmetry axis only",
    "forced_attitude": "one asymmetric body, unsaturated equal external/wheel torques; independent matrix RK4, no RCS fuel/hardware qualification",
    "burn": "centered symmetric and offset asymmetric constant-flow burns; small mass loss, fixed inertia and body-fixed force/moment",
    "contact": "fixed WGS84 sites, two analytic circular planes, constant synthetic EOP, shared ERFA; geometric contact only",
}


def validate_pointing_fixture(payload):
    """Pin every input/time domain and validate all shapes before solver work."""
    _finite_tree(payload)
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "provenance",
        "cases",
        "sha256",
    }:
        raise ValueError("invalid pointing fixture fields")
    if payload["schema"] != "space-env.pointing-vectors/1" or payload[
        "sha256"
    ] != digest({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("unsupported or corrupt pointing fixture")
    p = payload["provenance"]
    fields = {
        "generator",
        "generator_sha256",
        "independence",
        "erfa_version",
        "shared_inputs",
        "scope",
    }
    if not isinstance(p, dict) or set(p) != fields or any(not p[k] for k in fields):
        raise ValueError("incomplete pointing provenance")
    if any(
        not isinstance(p[k], str) or not p[k].strip()
        for k in fields - {"shared_inputs"}
    ):
        raise ValueError("pointing provenance text must be nonblank strings")
    if (
        p["independence"] != INDEPENDENCE
        or p["generator"] != "scripts/generate_pointing_vectors.py"
    ):
        raise ValueError("unsupported pointing independence or generator")
    sha = p["generator_sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError("invalid pointing generator digest")
    if p["shared_inputs"] != [
        "SI constants",
        "initial state and synthetic parameters",
        "ERFA c2t06a and leap table",
        "constant synthetic EOP",
    ]:
        raise ValueError("shared pointing inputs must be explicit")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASE_DOMAINS):
        raise ValueError("pointing suite requires every frozen case")
    seen = set()

    def matrix(value, count, width):
        if (
            not isinstance(value, list)
            or len(value) != count
            or any(
                not isinstance(row, list)
                or len(row) != width
                or any(type(v) not in (int, float) for v in row)
                for row in value
            )
        ):
            raise ValueError("invalid pointing expected matrix")

    for c in cases:
        if not isinstance(c, dict) or set(c) != {
            "id",
            "kind",
            "inputs",
            "times_sec",
            "expected",
        }:
            raise ValueError("invalid pointing case fields")
        name = c["id"]
        if not isinstance(name, str) or name not in CASE_DOMAINS or name in seen:
            raise ValueError("unknown or duplicate pointing case")
        seen.add(name)
        if (
            digest({k: c[k] for k in ("id", "kind", "inputs", "times_sec")})
            != CASE_DOMAINS[name]
        ):
            raise ValueError("pointing case outside frozen input/time domain")
        expected, count = c["expected"], len(c["times_sec"])
        if c["kind"] in {"attitude", "forced_attitude"}:
            matrix(expected, count, 17)
        elif c["kind"] == "burn":
            if not isinstance(expected, dict) or set(expected) != {
                "states_si_mass",
                "attitude_samples",
                "impulse_ns",
                "directions",
            }:
                raise ValueError("invalid rotating burn expected fields")
            matrix(expected["states_si_mass"], count, 7)
            matrix(expected["attitude_samples"], count, 15)
            matrix([expected["impulse_ns"]], 1, 3)
            matrix(expected["directions"], len(c["inputs"]["direction_times_sec"]), 3)
        else:
            if not isinstance(expected, dict) or set(expected) != {"looks", "windows"}:
                raise ValueError("invalid contact expected fields")
            matrix(expected["looks"], count, 8)
            spans = expected["windows"]
            if not isinstance(spans, list) or len(spans) > 32:
                raise ValueError("contact interval budget exceeded")
            matrix(spans, len(spans), 2)
            last = 0.0
            for begin, end in spans:
                if not last <= begin < end <= c["inputs"]["duration_sec"]:
                    raise ValueError(
                        "contact expected intervals overlap or exceed coverage"
                    )
                last = end
    return payload


def _check(case, metric, error, *, group=None, time_sec=None, **extra):
    if not math.isfinite(error):
        raise ValueError("nonfinite pointing comparison")
    budget = group or case["kind"]
    tolerance = BUDGETS[budget][metric]
    return {
        "case": case["id"],
        "kind": case["kind"],
        "metric": metric,
        "measured": float(error),
        "tolerance": tolerance,
        "budget_group": budget,
        "passed": bool(error <= tolerance),
        **({"time_sec": time_sec} if time_sec is not None else {}),
        **extra,
    }


def _rigid_config(p, *, step=None):
    from space_env.reference.rigid_attitude import (
        RigidAttitudeModel,
        RigidAttitudeState,
    )

    model = RigidAttitudeModel(
        model_id="pointing-vectors/1",
        source="frozen synthetic symmetric gyrostat",
        inertia_kg_m2=tuple(
            tuple(p["inertia_kg_m2"][i] if i == j else 0.0 for j in range(3))
            for i in range(3)
        ),
        wheel_inertia_kg_m2=(p["wheel_inertia_kg_m2"],) * 3,
        max_torque_nm=(p["torque_limit_nm"],) * 3,
        max_momentum_nms=(p["momentum_limit_nms"],) * 3,
        max_step_sec=p["step_sec"] if step is None else step,
    )
    state = RigidAttitudeState(
        quaternion_wxyz=tuple(p["initial_quaternion_wxyz"]),
        omega_body_rad_s=tuple(p["omega_body_rad_s"]),
        wheel_momentum_nms=(0.0, 0.0, p["h_z_nms"]),
    )
    return model, state


def _rotation(q):
    from space_env.reference.spacecraft_bus import rotate_body

    return np.column_stack([rotate_body(q, axis) for axis in np.eye(3)])


def attitude_checks(case):
    from space_env.reference.rigid_attitude import advance_rigid_attitude
    from space_env.reference.spacecraft_bus import angle_deg, rotate_body

    p, rows = case["inputs"], []
    model, initial = _rigid_config(p)
    coarse_final = None
    for t, expected in zip(case["times_sec"], case["expected"]):
        result, diagnostics = advance_rigid_attitude(
            model,
            initial,
            t,
            motor_torque_nm=(0.0, 0.0, p["motor_z_nm"]),
            external_torque_body_nm=tuple(p.get("external_body_nm", (0.0, 0.0, 0.0))),
        )
        reference = np.array(expected[:9]).reshape(3, 3)
        actual = _rotation(result.quaternion_wxyz)
        bore = rotate_body(result.quaternion_wxyz, p["boresight_body"])
        errors = {
            "rotation_frobenius": np.linalg.norm(actual - reference),
            "omega_rad_s": np.linalg.norm(
                np.array(result.omega_body_rad_s) - expected[9:12]
            ),
            "wheel_nms": np.linalg.norm(
                np.array(result.wheel_momentum_nms) - expected[12:15]
            ),
            "boresight_chord": np.linalg.norm(bore - reference @ p["boresight_body"]),
            "pointing_deg": abs(angle_deg(bore, p["target_gcrs"]) - expected[15]),
            "saturation_duration_sec": abs(
                diagnostics["capacity_limited_sec"][2] - expected[16]
            ),
        }
        group = "capacity" if expected[16] > 0 else "attitude"
        rows.extend(
            _check(case, m, float(v), group=group, time_sec=t)
            for m, v in errors.items()
        )
        coarse_final = errors["rotation_frobenius"]
    # Saturation is clipped over an RK4 substep, not an exact event split. Test
    # both its fixed error bound and refinement; do not call it exact-time truth.
    if case["expected"][-1][16] > 0:
        finer, state = _rigid_config(p, step=p["step_sec"] / 2)
        t = case["times_sec"][-1]
        result, _ = advance_rigid_attitude(
            finer, state, t, motor_torque_nm=(0.0, 0.0, p["motor_z_nm"])
        )
        exact = np.array(case["expected"][-1][:9]).reshape(3, 3)
        fine_error = float(np.linalg.norm(_rotation(result.quaternion_wxyz) - exact))
        rows.append(
            _check(
                case,
                "rotation_frobenius",
                fine_error,
                group="refinement",
                time_sec=t,
                step_sec=p["step_sec"] / 2,
            )
        )
        rows.append(
            _check(
                case,
                "fine_error_excess",
                fine_error - 0.6 * coarse_final - 1e-10,
                group="refinement",
                time_sec=t,
                coarse_error=float(coarse_final),
                fine_error=fine_error,
            )
        )
    return rows


def _context(p):
    from space_env.reference import EOPSeries, FrameContext

    return FrameContext(
        EOPSeries(
            [
                {
                    "utc": f"2026-07-{day}T00:00:00Z",
                    "dut1_s": p.get("dut1_s", 0.1),
                    "xp_arcsec": p.get("xp_arcsec", 0.05),
                    "yp_arcsec": p.get("yp_arcsec", 0.25),
                    "status": "synthetic",
                }
                for day in (18, 19, 20)
            ],
            source="Q5-B explicit constant synthetic EOP; not IERS observations",
        )
    )


def burn_checks(case):
    from space_env.core.constants import MU_EARTH
    from space_env.core.propulsion import G0_M_S2
    from space_env.reference import Epoch, ForceModel, ReferenceState, propagate
    from space_env.reference.rigid_attitude import (
        advance_rigid_attitude,
        thrust_torque_nm,
    )
    from space_env.reference.spacecraft_bus import rotate_body
    from space_env.reference.thrust import RotatingThrust

    p, expected, rows = case["inputs"], case["expected"], []
    if p["mu_m3_s2"] != MU_EARTH or p["g0_m_s2"] != G0_M_S2:
        raise ValueError("burn comparison constants differ")
    cfg, state = _rigid_config(p["attitude"])
    body = p["attitude"]["boresight_body"]
    external = thrust_torque_nm(p["lever_arm_body_m"], np.asarray(body) * p["thrust_n"])
    _, path = advance_rigid_attitude(
        cfg,
        state,
        p["duration_sec"],
        external_torque_body_nm=external,
        return_path=True,
    )
    for t, wanted in zip(case["times_sec"], expected["attitude_samples"]):
        actual, _ = advance_rigid_attitude(
            cfg, state, t, external_torque_body_nm=external
        )
        rows.extend(
            [
                _check(
                    case,
                    "rotation_frobenius",
                    float(
                        np.linalg.norm(
                            _rotation(actual.quaternion_wxyz)
                            - np.asarray(wanted[:9]).reshape(3, 3)
                        )
                    ),
                    time_sec=t,
                ),
                _check(
                    case,
                    "omega_rad_s",
                    float(
                        np.linalg.norm(
                            np.asarray(actual.omega_body_rad_s) - wanted[9:12]
                        )
                    ),
                    time_sec=t,
                ),
                _check(
                    case,
                    "wheel_nms",
                    float(
                        np.linalg.norm(
                            np.asarray(actual.wheel_momentum_nms) - wanted[12:15]
                        )
                    ),
                    time_sec=t,
                ),
            ]
        )
    law = RotatingThrust(
        p["thrust_n"],
        p["isp_s"],
        p["dry_mass_kg"],
        tuple(rotate_body(state.quaternion_wxyz, body)),
        tuple(body),
        tuple(path["attitude_times_sec"]),
        tuple(tuple(q) for q in path["attitude_quaternions_wxyz"]),
    )
    for t, wanted in zip(p["direction_times_sec"], expected["directions"]):
        rows.append(
            _check(
                case,
                "direction_chord",
                float(np.linalg.norm(law.direction_at(t) - wanted)),
                time_sec=t,
            )
        )
    rows.append(
        _check(
            case,
            "impulse_ns",
            float(np.linalg.norm(law.impulse_vector() - expected["impulse_ns"])),
        )
    )
    initial = p["initial_si"]
    result = propagate(
        ReferenceState(
            Epoch.utc(p["epoch_utc"]), np.array(initial[:3]), np.array(initial[3:])
        ),
        p["duration_sec"],
        _context(p),
        model=ForceModel(mass_kg=p["mass_kg"]),
        sample_sec=1.0,
        max_step_sec=0.05,
        thrust=law,
    )
    for t, wanted in zip(case["times_sec"], expected["states_si_mass"]):
        i = result["times_sec"].index(t)
        delta = np.array(result["states_si"][i]) - wanted[:6]
        rows.extend(
            [
                _check(
                    case, "position_m", float(np.linalg.norm(delta[:3])), time_sec=t
                ),
                _check(
                    case, "velocity_m_s", float(np.linalg.norm(delta[3:])), time_sec=t
                ),
                _check(
                    case, "mass_kg", abs(result["masses_kg"][i] - wanted[6]), time_sec=t
                ),
            ]
        )
    return rows


def contact_checks(case, *, provider_cache=None):
    from space_env.core.constants import (
        C_LIGHT,
        MU_EARTH,
        R_EARTH_EQUATOR,
        R_EARTH_POLAR,
    )
    from space_env.core.ground import GroundStation
    from space_env.reference import Epoch, ReferenceState, propagate
    from space_env.reference.access import ReferenceEphemeris
    from space_env.reference.acquisition import AcquisitionTarget
    from space_env.reference.spacecraft_bus import pointing_quaternion, rotate_body

    p, expected, rows = case["inputs"], case["expected"], []
    if (p["mu_m3_s2"], p["earth_equator_m"], p["earth_polar_m"], p["c_m_s"]) != (
        MU_EARTH,
        R_EARTH_EQUATOR,
        R_EARTH_POLAR,
        C_LIGHT,
    ):
        raise ValueError("contact comparison constants differ")
    key = digest({k: v for k, v in p.items() if k not in {"station", "frequency_hz"}})
    cache = {} if provider_cache is None else provider_cache
    if key not in cache:
        initial = p["initial_si"]
        result = propagate(
            ReferenceState(
                Epoch.utc(p["epoch_utc"]), np.array(initial[:3]), np.array(initial[3:])
            ),
            p["duration_sec"],
            _context(p),
            sample_sec=p["sample_sec"],
        )
        cache[key] = ReferenceEphemeris.from_result(result)
    provider = cache[key]
    station = GroundStation(**p["station"])
    target = AcquisitionTarget(
        id=station.name,
        lat_deg=station.lat_deg,
        lon_deg=station.lon_deg,
        alt_m=station.alt_m,
    )
    for t, wanted in zip(case["times_sec"], expected["looks"]):
        look = provider.look(station, t, frequency_hz=p["frequency_hz"])
        for metric, index in (
            ("range_m", 0),
            ("range_rate_m_s", 1),
            ("elevation_deg", 2),
            ("azimuth_deg", 3),
            ("doppler_hz", 4),
        ):
            delta = look[metric] - wanted[index]
            if metric == "azimuth_deg":
                delta = (delta + 180) % 360 - 180
            rows.append(_check(case, metric, abs(delta), time_sec=t))
        q = pointing_quaternion(provider, target, t)
        rows.append(
            _check(
                case,
                "pointing_chord",
                float(np.linalg.norm(rotate_body(q, [0, 0, 1]) - wanted[5:])),
                time_sec=t,
            )
        )
    actual = provider.access(station, scan_sec=20.0, root_tolerance_sec=0.0001)
    pairs = [[r["aos_sec"], r["los_sec"]] for r in actual["windows"]]
    count_error = abs(len(pairs) - len(expected["windows"]))
    boundary_error = (
        max(
            (
                abs(a - b)
                for row, ref in zip(pairs, expected["windows"])
                for a, b in zip(row, ref)
            ),
            default=0.0,
        )
        if not count_error
        else p["duration_sec"]
    )
    rows.append(
        _check(
            case,
            "window_count_error",
            float(count_error),
            actual_windows=pairs,
            expected_windows=expected["windows"],
        )
    )
    rows.append(_check(case, "boundary_sec", boundary_error))
    return rows


def verify_pointing_fixture(payload):
    validate_pointing_fixture(payload)
    checks, errors, cases, cache = [], [], [], {}
    for c in payload["cases"]:
        try:
            if c["kind"] in {"attitude", "forced_attitude"}:
                rows = attitude_checks(c)
            elif c["kind"] == "burn":
                rows = burn_checks(c)
            else:
                rows = contact_checks(c, provider_cache=cache)
            if not rows:
                raise ValueError("pointing case produced no measurements")
            checks.extend(rows)
            status = "passed" if all(r["passed"] for r in rows) else "failed"
        except (ValueError, RuntimeError, OverflowError) as exc:
            errors.append(
                {"case": c["id"], "type": type(exc).__name__, "message": str(exc)}
            )
            status = "execution_error"
        cases.append(
            {
                "id": c["id"],
                "kind": c["kind"],
                "status": status,
                "inputs": c["inputs"],
                "evaluated_times_sec": c["times_sec"]
                if status != "execution_error"
                else [],
                "scope": SCOPES[c["kind"]],
            }
        )
    return copy.deepcopy(
        {
            "passed": not errors and all(r["passed"] for r in checks),
            "fixture_sha256": payload["sha256"],
            "provenance": payload["provenance"],
            "checks": checks,
            "execution_errors": errors,
            "cases": cases,
        }
    )


def pointing_validation_report():
    from space_env import __version__

    result = verify_pointing_fixture(json.loads(FIXTURE.read_text()))
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"}
    }
    try:
        distribution = version("space-env")
    except PackageNotFoundError:
        distribution = None
    result.update(
        schema="space-env.pointing-validation/1",
        qualification_status="needs_external_evidence",
        industrial_readiness=False,
        budgets=copy.deepcopy(BUDGETS),
        budget_basis="predeclared engineering comparison thresholds; not physical uncertainty",
        build={
            "package_version": __version__,
            "installed_distribution_version": distribution,
            "distribution_matches_source_version": distribution == __version__,
            "files_sha256": hashes,
            "source_data_sha256": digest(hashes),
            "inventory_policy": "all package .py/.json/.csv/.yaml/.yml; not a wheel signature or native mission build ID",
            "dependencies": {
                n: version(n) for n in ("numpy", "scipy", "pyerfa", "pydantic")
            },
        },
        evidence_levels={
            "attitude": "closed-form symmetric gyrostat, independent rotation-matrix composition",
            "forced_attitude": "independent rotation-matrix RK4 versus production quaternion RK4",
            "burn": "independent Cartesian/matrix fixed-step RK4 versus DOP853/quaternion RK4+SLERP",
            "contact": "independent analytic circular orbit/WGS84/ENU/bisection; shared ERFA/EOP",
        },
        unqualified=[
            "arbitrary asymmetric tensor forced solution and hardware calibration",
            "exact wheel saturation event times; native partial-step clipping is approximate",
            "flight-calibrated thrust/lever arms, variable inertia and general off-center mission qualification",
            "closed-loop mission pointing/tracking performance and payload quality",
            "independent real EOP/frames/ephemerides/antenna/weather truth",
            "continuous event completeness for sub-grid/grazing contacts",
            "multiplane/multistation resource contention and full mission service",
        ],
        mission_evidence_policy="no historical mission qualification transfer; these are enumerated synthetic equation checks",
    )
    result["sha256"] = digest(result)
    return result
