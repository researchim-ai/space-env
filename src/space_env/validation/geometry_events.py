"""Analytic-orbit event comparisons with explicit shared-frame/model limitations."""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from space_env.core.constants import AU, MU_EARTH, R_EARTH_EQUATOR, R_EARTH_POLAR, R_SUN
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
from space_env.reference.acquisition import (
    AcquisitionSensor,
    AcquisitionTarget,
    acquisition_windows,
)
from space_env.reference.geometry_events import eclipse_windows


def _bisect_windows(margin, begin, end, step=20.0):
    """Independent bracketing/bisection; no production search or SciPy root call."""
    grid = np.linspace(begin, end, math.ceil((end - begin) / step) + 1)
    previous = margin(float(grid[0]))
    start = begin if previous > 0 else None
    intervals = []
    for a, b in pairwise(grid):
        after = margin(float(b))
        if (previous > 0) != (after > 0):
            lo, hi, sign = float(a), float(b), previous > 0
            while hi - lo > 1e-7:
                middle = 0.5 * (hi + lo)
                if (margin(middle) > 0) == sign:
                    lo = middle
                else:
                    hi = middle
            root = 0.5 * (hi + lo)
            if after > 0:
                start = root
            elif start is not None:
                intervals.append((start, root))
                start = None
        previous = after
    if start is not None:
        intervals.append((start, end))
    return intervals


def _compare(name, actual, expected, tolerance):
    values = [(r["start_sec"], r["end_sec"]) for r in actual]
    error = (
        max(
            (
                abs(a - b)
                for row, ref in zip(values, expected)
                for a, b in zip(row, ref)
            ),
            default=0.0,
        )
        if len(values) == len(expected)
        else None
    )
    return {
        "name": name,
        "actual_windows": values,
        "expected_windows": expected,
        "max_boundary_error_sec": error,
        "tolerance_sec": tolerance,
        "passed": bool(expected) and error is not None and error <= tolerance,
    }


def geometry_event_report():
    radius, duration = 7e6, 6000.0
    motion = math.sqrt(MU_EARTH / radius**3)
    epoch, context = (
        Epoch.utc("2026-07-18T12:00:00Z"),
        FrameContext(EOPSeries.packaged()),
    )

    def exact_position(t):
        return radius * np.array([math.cos(motion * t), 0.0, math.sin(motion * t)])

    record = propagate(
        ReferenceState(epoch, exact_position(0), [0.0, 0.0, radius * motion]),
        duration,
        context,
        sample_sec=10.0,
    )
    provider = ReferenceEphemeris.from_result(record)
    lat, lon, _ = ecef_to_geodetic(
        context.matrix(epoch.shifted(3000)) @ exact_position(3000)
    )
    station = GroundStation(
        name="AnalyticOverflight", lat_deg=lat, lon_deg=lon, min_elevation_deg=5.0
    )
    phi, lam = np.deg2rad([lat, lon])
    eccentricity2 = 1 - (R_EARTH_POLAR / R_EARTH_EQUATOR) ** 2
    normal_radius = R_EARTH_EQUATOR / math.sqrt(1 - eccentricity2 * math.sin(phi) ** 2)
    # Independent WGS84 position and geodetic normal, not production look/ENU.
    ground = np.array(
        [
            normal_radius * math.cos(phi) * math.cos(lam),
            normal_radius * math.cos(phi) * math.sin(lam),
            normal_radius * (1 - eccentricity2) * math.sin(phi),
        ]
    )
    up = np.array(
        [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)]
    )

    def independent_geometry(t):
        rotation = context.matrix(epoch.shifted(t))
        r = exact_position(t)
        fixed_line = rotation @ r - ground
        distance = np.linalg.norm(fixed_line)
        elevation_sine = float(np.dot(up, fixed_line) / distance)
        to_satellite = r - rotation.T @ ground
        off_nadir_cosine = float(np.dot(to_satellite, r) / (distance * radius))
        return elevation_sine - math.sin(math.radians(5)), off_nadir_cosine - math.cos(
            math.radians(45)
        )

    access = provider.access(station, scan_sec=20.0)
    station_rows = [
        {"start_sec": r["aos_sec"], "end_sec": r["los_sec"]} for r in access["windows"]
    ]
    expected_station = _bisect_windows(
        lambda t: independent_geometry(t)[0], 0.0, duration
    )
    target = AcquisitionTarget(id="analytic-target", lat_deg=lat, lon_deg=lon)
    acquisition = acquisition_windows(
        provider, target, AcquisitionSensor(), settings={"scan_sec": 10.0}
    )
    expected_acquisition = _bisect_windows(
        lambda t: min(independent_geometry(t)), 0.0, duration
    )
    sun = np.array([AU, 0.0, 0.0])
    eclipses = eclipse_windows(
        provider, settings={"scan_sec": 20.0}, sun_gcrs=lambda e: sun
    )

    def angular_margin(t, full):
        r = exact_position(t)
        to_sun = sun - r
        separation = math.acos(
            float(
                np.clip(np.dot(-r, to_sun) / (radius * np.linalg.norm(to_sun)), -1, 1)
            )
        )
        earth_angle = math.asin(R_EARTH_EQUATOR / radius)
        sun_angle = math.asin(R_SUN / np.linalg.norm(to_sun))
        return earth_angle + (-sun_angle if full else sun_angle) - separation

    checks = [
        _compare("station_AOS_LOS", station_rows, expected_station, 0.02),
        _compare(
            "point_acquisition", acquisition["windows"], expected_acquisition, 0.02
        ),
    ]
    for key, full in (("umbra", True), ("partial_or_full", False)):
        checks.append(
            _compare(
                key + "_against_spherical_apparent_discs",
                eclipses[key]["windows"],
                _bisect_windows(
                    lambda t, full=full: angular_margin(t, full), 0.0, duration
                ),
                1.0,
            )
        )
    return {
        "schema": "space-env.geometry-event-comparison/1",
        "passed": all(r["passed"] for r in checks),
        "checks": checks,
        "nominal_manifest": record["manifest"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independence": "analytic circular orbit; separately written target/occultation equations and bisection",
        "shared_inputs": [
            "EOP/ERFA GCRS-ITRS matrices",
            "initial orbit and constants",
            "derived target latitude/longitude",
        ],
        "eclipse_comparison": "approximate conical production model versus spherical apparent-disc contacts, constant Sun; 1s model-comparison budget",
        "not_qualified": [
            "independent EOP/frames/ephemeris truth",
            "ellipsoidal occultation/terrain/atmosphere",
            "real illumination flux, short/grazing-event completeness",
            "acquisition execution or sensor calibration",
        ],
        "qualification_status": "needs_external_evidence",
    }
