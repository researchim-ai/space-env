"""Portable GCRS ephemerides and geometric station access, isolated from RF/task outcomes."""
from __future__ import annotations

import copy
import hashlib
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq, minimize_scalar

from space_env.core.constants import C_LIGHT
from space_env.core.frames import geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.reference.frames import FrameContext
from space_env.reference.time import EOPSeries, Epoch, erfa_backend
from space_env.simulation.snapshot import digest


class ReferenceEphemeris:
    """Read a sealed orbit without its original input file or packaged EOP table.

    Hermite interpolation uses GCRS r/v and TAI seconds. Input cadence is explicitly
    bounded; no extrapolation. The bound is not a universal interpolation accuracy.
    """

    def __init__(self, payload: dict, *, max_interval_sec: float = 60.):
        body = {key: value for key, value in payload.items() if key != "sha256"}
        if payload.get("schema") != "space-env.reference-orbit/1" or payload.get("sha256") != digest(body):
            raise ValueError("unsupported or corrupt reference ephemeris")
        if payload.get("frame") != "GCRS" or payload["manifest"].get("profile") != "gcrs-reference/1":
            raise ValueError("reference ephemeris requires explicit GCRS profile")
        if not math.isfinite(max_interval_sec) or not 0 < max_interval_sec <= 120:
            raise ValueError("ephemeris interval limit must be in (0,120] seconds")
        times = np.asarray(payload["times_sec"], dtype=float)
        states = np.asarray(payload["states_si"], dtype=float)
        if times.ndim != 1 or not 2 <= len(times) <= 100001 or states.shape != (len(times), 6):
            raise ValueError("ephemeris needs bounded Nx6 states and N times")
        if not np.isfinite(times).all() or not np.isfinite(states).all() or times[0] != 0 or times[-1] > 259200:
            raise ValueError("ephemeris times/states violate reference applicability")
        if np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > max_interval_sec + 1e-9:
            raise ValueError("ephemeris cadence is unordered or exceeds declared interpolation limit")
        self.epoch = Epoch(*payload["epoch_tai_jd"])
        if abs(Epoch.utc(payload["epoch_utc"]).seconds_since(self.epoch)) > 1e-8:
            raise ValueError("ephemeris UTC and TAI epochs differ")
        data = payload["manifest"]["eop"]
        if type(data.get("allow_predictions")) is not bool:
            raise ValueError("ephemeris EOP prediction policy must be boolean")
        self.context = FrameContext(EOPSeries(data["rows"], source=data["source"], allow_predictions=data["allow_predictions"]))
        if payload["manifest"].get("erfa_version") != erfa_backend().__version__:
            raise ValueError("ephemeris frame backend differs from installed ERFA")
        if payload["manifest"]["leap_table_sha256"] != self.context.manifest()["leap_table_sha256"]:
            raise ValueError("ephemeris leap-second table differs from installed backend")
        self.duration_sec = float(times[-1])
        self.max_interval_sec = float(np.max(np.diff(times)))
        self.sha256 = payload["sha256"]
        self._spline = CubicHermiteSpline(times, states[:, :3], states[:, 3:], axis=0, extrapolate=False)
        self._velocity = self._spline.derivative()
        self._manifest = copy.deepcopy(payload["manifest"])

    @classmethod
    def from_result(cls, result: dict, **kwargs) -> ReferenceEphemeris:
        body = {key: value for key, value in result.items() if key != "sha256"}
        return cls({**body, "sha256": digest(body)}, **kwargs)

    def state(self, clock: float) -> np.ndarray:
        if isinstance(clock, bool) or not math.isfinite(clock) or not 0 <= clock <= self.duration_sec:
            raise ValueError("ephemeris query outside finite time coverage")
        return np.concatenate((self._spline(clock), self._velocity(clock)))

    def look(self, station: GroundStation, clock: float, *, frequency_hz: float | None = None) -> dict:
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError("carrier frequency must be positive and finite")
        if not np.isfinite([station.lat_deg, station.lon_deg, station.alt_m, station.min_elevation_deg]).all():
            raise ValueError("station coordinates and threshold must be finite")
        state = self.state(clock)
        epoch = self.epoch.shifted(clock)
        fixed = self.context.transform(state, epoch, source="GCRS", destination="ITRS")["state_si"]
        ground = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
        delta = fixed[:3] - ground
        distance = float(np.linalg.norm(delta))
        if distance <= 0:
            raise ValueError("station and spacecraft positions coincide")
        lat, lon = np.deg2rad([station.lat_deg, station.lon_deg])
        east = -np.sin(lon)*delta[0] + np.cos(lon)*delta[1]
        north = -np.sin(lat)*np.cos(lon)*delta[0] - np.sin(lat)*np.sin(lon)*delta[1] + np.cos(lat)*delta[2]
        up = np.cos(lat)*np.cos(lon)*delta[0] + np.cos(lat)*np.sin(lon)*delta[1] + np.sin(lat)*delta[2]
        rate = float(np.dot(delta, fixed[3:]) / distance)
        result = {"time_sec": float(clock), "station": station.name, "range_m": distance,
                  "range_rate_m_s": rate, "elevation_deg": float(np.rad2deg(np.arctan2(up, np.hypot(east, north)))),
                  "azimuth_deg": float(np.rad2deg(np.arctan2(east, north)) % 360)}
        if frequency_hz is not None:
            result["doppler_hz"] = -frequency_hz * rate / C_LIGHT
        return result

    def access(self, station: GroundStation, *, scan_sec: float = 30., root_tolerance_sec: float = .001) -> dict:
        if not math.isfinite(scan_sec) or not 0 < scan_sec <= 120:
            raise ValueError("access scan must be in (0,120] seconds")
        if not math.isfinite(root_tolerance_sec) or not 1e-6 <= root_tolerance_sec <= 1:
            raise ValueError("root tolerance must be in [1e-6,1] seconds")
        if math.ceil(self.duration_sec/scan_sec) > 100000:
            raise ValueError("access scan exceeds 100000 intervals")
        # Velocity needs adjacent EOP nodes, even when position itself is covered.
        self.context.jacobian(self.epoch)
        self.context.jacobian(self.epoch.shifted(self.duration_sec))
        grid = np.append(np.arange(0, self.duration_sec, scan_sec), self.duration_sec)

        def margin(clock):
            return self.look(station, float(clock))["elevation_deg"] - station.min_elevation_deg

        values = [margin(clock) for clock in grid]
        start = 0. if values[0] >= 0 else None
        windows = []
        for i, (left, right) in enumerate(pairwise(grid)):
            before, after = values[i], values[i+1]
            if (before >= 0) == (after >= 0):
                continue
            crossing = float(brentq(margin, left, right, xtol=root_tolerance_sec))
            if after >= 0:
                start = crossing
            elif start is not None:
                windows.append((start, crossing))
                start = None
        if start is not None and start < self.duration_sec:
            windows.append((start, self.duration_sec))
        rows = []
        for begin, end in windows:
            if end-begin <= root_tolerance_sec:
                continue
            # Scan for the peak first: bounded minimization alone assumes unimodality.
            candidates = np.append(np.arange(begin, end, min(scan_sec, (end-begin)/4)), end)
            elevations = [self.look(station, float(t))["elevation_deg"] for t in candidates]
            peak = int(np.argmax(elevations))
            lo, hi = candidates[max(0, peak-1)], candidates[min(len(candidates)-1, peak+1)]
            optimization = minimize_scalar(lambda t: -self.look(station, float(t))["elevation_deg"],
                                           bounds=(lo, hi), method="bounded", options={"xatol": root_tolerance_sec})
            choices = [float(candidates[peak]), float(optimization.x)]
            tca = max(choices, key=lambda t: self.look(station, t)["elevation_deg"])
            rows.append({"aos_sec": begin, "los_sec": end, "peak_sec": tca,
                         "max_elevation_deg": self.look(station, tca)["elevation_deg"],
                         "clipped_start": begin == 0, "clipped_end": end == self.duration_sec})
        return {"schema": "space-env.reference-access/1", "station": station.model_dump(mode="json"),
                "ephemeris_sha256": self.sha256, "profile": "gcrs-reference/1", "windows": rows,
                "coverage_sec": [0., self.duration_sec], "scan_sec": scan_sec,
                "root_tolerance_sec": root_tolerance_sec, "interpolation": "cubic Hermite GCRS position/velocity",
                "max_input_interval_sec": self.max_interval_sec,
                "derivation": {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "erfa_version": erfa_backend().__version__, "eop_source": self.context.eop.source},
                "limits": ["grazing or short passes between scan nodes can be missed",
                           "geometric visibility only; no refraction, weather, RF rate or ground bookings",
                           "root tolerance is not physical event accuracy", "no light-time or relativistic Doppler"]}
