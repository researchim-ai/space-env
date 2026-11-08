"""Independent analytic oracles for selected core and reference rigid physics, without network access.

Production routines are used only on the measured side. Constants are shared
model parameters, not independently validated geophysical measurements.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from space_env.core import gravity, link, propulsion, thermal
from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR, SIGMA_SB
from space_env.core.propagate import propagate_state
from space_env.reference import rigid_attitude
from space_env.simulation.snapshot import digest


def source_manifest() -> dict:
    """Hash on-disk Python sources plus available repository suite artifacts.

    Physics/runtime-directory coverage conservatively binds local model code. This
    is provenance of local bytes, not attestation of loaded code or authenticity.
    """
    package = Path(__file__).resolve().parents[1]
    files = {
        "src/space_env/" + p.relative_to(package).as_posix(): p
        for directory in ("core", "reference", "runtime", "simulation")
        for p in sorted((package / directory).rglob("*.py"))
    }
    for name in (
        "__init__.py",
        "validation/__init__.py",
        "validation/profiles.py",
        "validation/report.py",
        "validation/ai_reference_suite.py",
    ):
        files["src/space_env/" + name] = package / name
    repository = package.parents[1]
    companions = (
        "tests/test_ai_reference_suite.py",
        "examples/validation/ai_reference_suite.py",
        "docs/AI_PHYSICS_REFERENCE.md",
    )
    missing = []
    for name in companions:
        path = repository / name
        if path.is_file():
            files[name] = path
        else:
            missing.append(name)
    return {
        "algorithm": "sha256",
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
        "unavailable_repository_artifacts": missing,
        "scope": "All core/reference/runtime/simulation Python sources, package/validation entry imports, and suite artifacts; excludes other import-only package areas, third-party sources and data.",
    }


def seal_report(body: dict) -> dict:
    """Use the native audit canonical digest, excluding the seal itself."""
    body = {key: value for key, value in body.items() if key != "sha256"}
    return body | {"sha256": digest(body)}


def verify_report(report: dict, *, check_current_sources: bool = True) -> bool:
    """Verify integrity, and optionally compare provenance with this checkout.

    A valid seal says nothing about whether physics checks passed. Missing/malformed
    seal fields fail verification. A hash is not a signature or trusted timestamp.
    """
    try:
        body = {key: value for key, value in report.items() if key != "sha256"}
        return (
            report["schema"] == "space-env.ai-reference-suite/2"
            and report["sha256"] == digest(body)
            and report["sources_sha256"] == digest(report["sources"])
            and (not check_current_sources or report["sources"] == source_manifest())
        )
    except (KeyError, TypeError, ValueError):
        return False


def _runtime():
    # Versions are observed in this process, not a claimed compatibility matrix.
    package_map = metadata.packages_distributions()
    loaded = {name.split(".")[0] for name in sys.modules}
    distributions = sorted(
        {distribution for name in loaded for distribution in package_map.get(name, [])}
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "loaded_distributions": {
            name: metadata.version(name) for name in distributions
        },
        "scope": "Installed versions mapped from modules loaded in this process, including import-time dependencies; not all are numerical engines used by checks.",
    }


def _rigid_checks():
    """Closed-form principal-axis dynamics, independent of production invariants."""
    model = rigid_attitude.RigidAttitudeModel(
        model_id="ai-analytic-principal-axis/1",
        source="synthetic analytic fixture",
        inertia_kg_m2=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        wheel_inertia_kg_m2=(0.01, 0.01, 0.01),
        max_torque_nm=(0.1, 0.1, 0.1),
        max_momentum_nms=(1.0, 1.0, 1.0),
        efficiency=0.8,
        idle_power_w=2.0,
        torque_loss_w_per_nm2=1.5,
        max_step_sec=0.01,
    )
    rows = []
    for sign in (-1.0, 1.0):
        duration, speed = 2.0, sign * 0.4
        result, _ = rigid_attitude.advance_rigid_attitude(
            model,
            rigid_attitude.RigidAttitudeState(omega_body_rad_s=(0.0, 0.0, speed)),
            duration,
        )
        # q and -q describe the same attitude; compare the nearer representative.
        expected = np.array(
            [math.cos(speed * duration / 2), 0.0, 0.0, math.sin(speed * duration / 2)]
        )
        actual = np.array(result.quaternion_wxyz)
        if np.dot(actual, expected) < 0:
            actual = -actual
        scope = "Reference rigid kernel used by optional mission rigid_bus; free principal-z rotation, Izz=4 kg m^2, omega=+/-0.4 rad/s, 2 s, step<=0.01 s; no controller or mission integration."
        rows.append(
            _check(
                f"reference_free_rotation_{sign}",
                actual,
                expected,
                1e-11,
                "1",
                scope,
                "Exact body-to-inertial quaternion [cos(omega*t/2),0,0,sin(omega*t/2)].",
            )
        )
        rows.append(
            _check(
                f"reference_free_rate_{sign}",
                result.omega_body_rad_s,
                [0.0, 0.0, speed],
                1e-12,
                "rad/s",
                scope,
                "Torque-free rotation about a principal axis has constant angular rate.",
            )
        )
        torque, inertia, wheel_inertia = sign * 0.05, 2.0, 0.01
        driven, _ = rigid_attitude.advance_rigid_attitude(
            model,
            rigid_attitude.RigidAttitudeState(),
            duration,
            motor_torque_nm=(torque, 0.0, 0.0),
        )
        angle = -torque * duration**2 / (2 * inertia)
        expected = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        actual = np.array(driven.quaternion_wxyz)
        if np.dot(actual, expected) < 0:
            actual = -actual
        scope = "Reference rigid kernel; rest to constant +/-0.05 Nm wheel torque on principal x, Ixx=2 kg m^2, Jwheel=0.01 kg m^2, 2 s, no clipping/saturation/external torque; not coupled mission bus."
        rows.append(
            _check(
                f"reference_motor_rotation_{sign}",
                actual,
                expected,
                1e-11,
                "1",
                scope,
                "Constant body angular acceleration -u/I; angle=-u*t^2/(2I).",
            )
        )
        rows.append(
            _check(
                f"reference_motor_rate_{sign}",
                driven.omega_body_rad_s,
                [-torque * duration / inertia, 0.0, 0.0],
                1e-12,
                "rad/s",
                scope,
                "Integrated angular impulse on the body, omega=-u*t/I.",
            )
        )
        rows.append(
            _check(
                f"reference_wheel_momentum_{sign}",
                driven.wheel_momentum_nms,
                [torque * duration, 0.0, 0.0],
                1e-12,
                "N m s",
                scope,
                "Integrated wheel impulse h=u*t; opposite body angular momentum.",
            )
        )
        # Start from rest: mechanical power is nonnegative and linear in time.
        work = 0.5 * (torque * duration) ** 2 * (1 / inertia + 1 / wheel_inertia)
        dc = work / 0.8 + (2.0 + 1.5 * torque**2) * duration
        for quantity, measured, expected_energy in (
            ("motor_work", driven.motor_work_j, work),
            ("dc_energy", driven.dc_energy_j, dc),
            ("heat", driven.heat_j, dc - work),
        ):
            rows.append(
                _check(
                    f"reference_{quantity}_{sign}",
                    measured,
                    expected_energy,
                    1e-10,
                    "J",
                    scope,
                    "Exact kinetic work W=(u*t)^2*(1/I+1/J)/2; E_DC=W/eta+(P_idle+k*u^2)*t; heat=E_DC-W. Synthetic constitutive electrical model, no hardware calibration.",
                )
            )
    return rows


def _check(name, actual, expected, tolerance, units, applicability, reference):
    """Maximum absolute component error; nonfinite measurements always fail."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: actual/reference shapes differ")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    error = float(np.max(np.abs(actual - expected))) if finite else None
    return {
        "name": name,
        "actual": actual.tolist() if np.isfinite(actual).all() else None,
        "expected": expected.tolist() if np.isfinite(expected).all() else None,
        "finite": finite,
        "max_absolute_error": error,
        "tolerance": tolerance,
        "units": units,
        "passed": finite and error <= tolerance,
        "applicability": applicability,
        "reference": reference,
    }


def _kepler_state(seconds, a, e, rotation):
    # Monotonic E-e sin(E)=M is solved by bracketing, not the production orbit code.
    mean = (math.sqrt(MU_EARTH / a**3) * seconds) % math.tau
    lo, hi = 0.0, math.tau
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid - e * math.sin(mid) < mean:
            lo = mid
        else:
            hi = mid
    anomaly = (lo + hi) / 2
    c, s = math.cos(anomaly), math.sin(anomaly)
    rate = math.sqrt(MU_EARTH / a**3) / (1 - e * c)
    return (
        rotation @ np.array([a * (c - e), a * math.sqrt(1 - e * e) * s, 0.0]),
        rotation @ np.array([-a * s * rate, a * math.sqrt(1 - e * e) * c * rate, 0.0]),
    )


def run_suite() -> dict:
    """Run fixed fixtures and return a strict-JSON-compatible evidence report.

    Exceptions propagate: a broken execution cannot be mistaken for a pass.
    Tolerances are fixed here and never fitted from the measured output.
    """
    sources = source_manifest()
    checks = []
    for eccentricity in (0.0, 0.2):
        a, inclination = 9e6, 0.7
        c, s = math.cos(inclination), math.sin(inclination)
        rotation = np.array([[0.0, -c, s], [1.0, 0.0, 0.0], [0.0, s, c]])
        r, v = _kepler_state(0.0, a, eccentricity, rotation)
        duration = 2 * math.tau * math.sqrt(a**3 / MU_EARTH) + 123.0
        _, _, samples = propagate_state(
            r,
            v,
            duration,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_j2=False,
            include_drag=False,
            substep_sec=10.0,
        )
        expected = [_kepler_state(t, a, eccentricity, rotation) for t, _, _ in samples]
        scope = (
            f"Core inertial two-body only; a=9000000 m, e={eccentricity}, "
            "inclination=0.7 rad, two periods +123 s, RK4 step <=10 s; all samples."
        )
        for index, (quantity, tolerance, unit) in enumerate(
            (("position", 1.0, "m"), ("velocity", 0.001, "m/s"))
        ):
            checks.append(
                _check(
                    f"kepler_e{eccentricity}_{quantity}",
                    [sample[index + 1] for sample in samples],
                    [state[index] for state in expected],
                    tolerance,
                    unit,
                    scope,
                    "Elliptic Kepler equation solved by 60 bisections; analytic derivative.",
                )
            )
        # Keep reports compact: retain maximum errors and fixture metadata, not thousands of states.
        for row in checks[-2:]:
            row["sample_count"] = len(samples)
            row.pop("actual")
            row.pop("expected")

    def potential(r):
        radius = float(np.linalg.norm(r))
        return (
            MU_EARTH
            / radius
            * (
                1
                - J2_EARTH
                * (R_EARTH_EQUATOR / radius) ** 2
                * (3 * (r[2] / radius) ** 2 - 1)
                / 2
            )
        )

    for index, position in enumerate(([7e6, 0, 0], [0, 0, 8e6], [6e6, -4e6, 3e6])):
        r = np.array(position, dtype=float)
        # Five-point derivative reduces cancellation/truncation error without copying force algebra.
        h = 100.0
        gradient = [
            (
                potential(r - 2 * h * u)
                - 8 * potential(r - h * u)
                + 8 * potential(r + h * u)
                - potential(r + 2 * h * u)
            )
            / (12 * h)
            for u in np.eye(3)
        ]
        actual = gravity.acceleration_m_s2(
            r,
            np.zeros(3),
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_j2=True,
            include_drag=False,
        )
        checks.append(
            _check(
                f"j2_potential_{index}",
                actual,
                gradient,
                2e-9,
                "m/s^2",
                "Core z-axis J2 plus central gravity; fixed equatorial, polar and oblique points; no other forces.",
                "Gradient of positive zonal gravitational potential; five-point stencil h=100 m.",
            )
        )

    ranges = [1e3, 1e6, 4e7]
    frequencies = [1e8, 2.2e9, 3e10]
    checks.append(
        _check(
            "friis_loss",
            [link.free_space_path_loss_db(d, f) for d in ranges for f in frequencies],
            [
                10 * math.log10((4 * math.pi * d * f / 299792458.0) ** 2)
                for d in ranges
                for f in frequencies
            ],
            0.01,
            "dB",
            "Vacuum far-field isotropic spreading; 3 ranges x 3 frequencies; excludes gains, noise and fading.",
            "Friis power ratio (4*pi*d/lambda)^2 with exact SI c; allows rounded production 32.44 dB constant.",
        )
    )

    for duration in (100.0, 10000.0):
        initial, dry, thrust, isp = 120.0, 100.0, 20.0, 250.0
        exhaust = isp * 9.80665
        final = max(dry, initial - thrust * duration / exhaust)
        mass, dv = propulsion.mass_after_thrust(initial, duration, thrust, isp, dry)
        scope = f"Constant thrust=20 N, Isp=250 s, initial/dry=120/100 kg, duration={duration} s; isolated mass/impulse helper, not finite-burn trajectory."
        checks.append(
            _check(
                f"rocket_mass_{duration}",
                mass,
                final,
                1e-12,
                "kg",
                scope,
                "Integrated constant mass flow, capped at dry mass.",
            )
        )
        checks.append(
            _check(
                f"rocket_dv_{duration}",
                dv,
                exhaust * math.log(initial / final),
                1e-10,
                "m/s",
                scope,
                "Tsiolkovsky ideal rocket equation, no external forces.",
            )
        )

    for power in (0.0, 250.0, 1000.0):
        area, epsilon, capacity = 2.0, 0.8, 5000.0
        equilibrium = (power / (epsilon * SIGMA_SB * area) + 3.0**4) ** 0.25 - 273.15
        actual = thermal.step_temperature_c(
            equilibrium,
            30.0,
            q_sun_w=power,
            q_diss_w=0.0,
            q_heater_w=0.0,
            radiating_area_m2=area,
            emissivity=epsilon,
            heat_capacity_j_k=capacity,
        )
        checks.append(
            _check(
                f"thermal_equilibrium_{power}",
                actual,
                equilibrium,
                1e-10,
                "degC",
                "Single gray node, 3 K sink, constant 0/250/1000 W input, 30 s Euler step at equilibrium; no transient accuracy claim.",
                "Stefan-Boltzmann equilibrium T=(Q/(epsilon*sigma*A)+Tspace^4)^(1/4).",
            )
        )
    actual = thermal.step_temperature_c(
        20.0,
        60.0,
        q_sun_w=100.0,
        q_diss_w=20.0,
        q_heater_w=5.0,
        radiating_area_m2=0.0,
        emissivity=0.8,
        heat_capacity_j_k=5000.0,
    )
    checks.append(
        _check(
            "thermal_linear_heating",
            actual,
            21.5,
            1e-12,
            "degC",
            "Zero radiating area, constant 125 W, C=5000 J/K for 60 s.",
            "Exact calorimetry Delta T=Q*dt/C.",
        )
    )
    checks.extend(_rigid_checks())
    if source_manifest() != sources:
        raise RuntimeError("source files changed during validation")
    return seal_report(
        {
            "schema": "space-env.ai-reference-suite/2",
            "passed": all(row["passed"] for row in checks),
            "checks": checks,
            "runtime": _runtime(),
            "sources": sources,
            "sources_sha256": digest(sources),
            "error_metric": "maximum absolute component error, inclusive tolerance; nonfinite fails",
            "shared_inputs": ["MU_EARTH", "J2_EARTH", "R_EARTH_EQUATOR", "SIGMA_SB"],
            "external_engine_comparison": {
                "performed": False,
                "reason": "This suite runs analytic references only; no external engine invoked.",
            },
            "not_qualified": [
                "independent physical constants",
                "external-engine agreement",
                "operational orbit accuracy",
                "atmosphere, drag, SRP, eclipses, ephemerides and frames",
                "coupled finite burns",
                "thermal transient integration accuracy",
                "mission rigid controller, battery coupling and wheel saturation/regeneration",
            ],
        }
    )
