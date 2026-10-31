"""Bounded geometric event searches; sampling resolution is not physical accuracy."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq

from space_env.core.constants import R_EARTH_EQUATOR, R_SUN
from space_env.core.environment import shadow_state
from space_env.reference.frames import _vector
from space_env.reference.orbit import _ephemerides


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    scan_sec: float = Field(default=5.0, ge=1e-6, le=120, strict=True)
    root_tolerance_sec: float = Field(default=0.001, ge=1e-6, le=1, strict=True)
    max_evaluations: int = Field(default=200000, ge=2, le=1000000, strict=True)

    @model_validator(mode="after")
    def root_resolution(self):
        if self.root_tolerance_sec > self.scan_sec:
            raise ValueError("root tolerance must not exceed sampling interval")
        return self


def coverage(provider, start_sec=None, end_sec=None):
    begin = 0.0 if start_sec is None else start_sec
    end = provider.duration_sec if end_sec is None else end_sec
    if any(
        isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t)
        for t in (begin, end)
    ):
        raise ValueError("event coverage must be finite elapsed SI seconds")
    if not 0 <= begin < end <= provider.duration_sec:
        raise ValueError("event query outside trajectory coverage")
    return float(begin), float(end)


def positive_windows(margin, start_sec, end_sec, *, settings=None, boundaries=()):
    """Intervals with positive margin. Zero-duration touches are not operations.

    A component narrower than the maximum grid gap can be missed. No continuous
    completeness certificate is claimed without an external variation bound.
    """
    settings = SearchSettings.model_validate({} if settings is None else settings)
    if (
        any(
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            for t in (start_sec, end_sec)
        )
        or not start_sec < end_sec
    ):
        raise ValueError("search requires an ordered finite interval")
    count = math.ceil((end_sec - start_sec) / settings.scan_sec)
    if count > 100000 or count + 1 > settings.max_evaluations:
        raise ValueError("event search grid exceeds budget")
    extra = []
    for boundary in boundaries:
        if isinstance(boundary, bool) or not math.isfinite(boundary):
            raise ValueError("invalid trajectory boundary")
        if start_sec < boundary < end_sec:
            extra.append(float(boundary))
            if len(extra) > 100000:
                raise ValueError("event boundary budget exceeded")
    grid = np.unique(
        np.concatenate((np.linspace(start_sec, end_sec, count + 1), extra))
    )
    if len(grid) > settings.max_evaluations:
        raise ValueError("event search grid exceeds budget")
    cache = {}

    def evaluate(t):
        t = float(t)
        if t not in cache:
            if len(cache) >= settings.max_evaluations:
                raise ValueError("event evaluation budget exhausted")
            value = float(margin(t))
            if not math.isfinite(value):
                raise ValueError("nonfinite event margin")
            cache[t] = value
        return cache[t]

    values = [evaluate(t) for t in grid]
    # Equality at a single node is a measure-zero touch, not a feasible span.
    begin = start_sec if values[0] > 0 else None
    intervals = []
    for index, (left, right) in enumerate(pairwise(grid)):
        before, after = values[index], values[index + 1]
        if (before > 0) == (after > 0):
            continue
        crossing = float(
            brentq(evaluate, left, right, xtol=settings.root_tolerance_sec, rtol=1e-14)
        )
        if after > 0:
            begin = crossing
        elif begin is not None:
            intervals.append((begin, crossing))
            begin = None
    if begin is not None:
        intervals.append((begin, end_sec))
    merged = []
    for a, b in intervals:
        if merged and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    rows = [
        {
            "start_sec": float(a),
            "end_sec": float(b),
            "duration_sec": float(b - a),
            "clipped_start": a == start_sec and values[0] > 0,
            "clipped_end": b == end_sec and values[-1] > 0,
        }
        for a, b in merged
        if b - a > settings.root_tolerance_sec
    ]
    return {
        "windows": rows,
        "evaluations": len(cache),
        "max_grid_gap_sec": float(np.max(np.diff(grid))),
        "search": settings.model_dump(mode="json"),
        "coverage_sec": [start_sec, end_sec],
        "completeness": "sampled brackets only; feasible/infeasible features narrower than grid gap may be missed",
    }


def provider_boundaries(provider):
    return [row[0] for row in getattr(provider, "segments", ())]


def eclipse_windows(
    provider, *, start_sec=None, end_sec=None, settings=None, sun_gcrs=None
):
    """Events of the SAME spherical conical shadow used by mission power and SRP.

    sun_gcrs is a trusted SDK callback used by validation/custom environment code;
    runtime uses ERFA only. Penumbra windows include full umbra.
    """
    begin, end = coverage(provider, start_sec, end_sec)
    settings = SearchSettings.model_validate({} if settings is None else settings)
    cache = {}

    def margins(t):
        if t not in cache:
            if len(cache) >= settings.max_evaluations:
                raise ValueError("eclipse geometry evaluation budget exhausted")
            state = _vector(provider.state(t), 6)
            epoch = provider.epoch.shifted(t)
            sun = _vector(
                _ephemerides(epoch)[0] if sun_gcrs is None else sun_gcrs(epoch), 3
            )
            if (
                np.linalg.norm(state[:3]) <= R_EARTH_EQUATOR
                or np.linalg.norm(sun) <= R_SUN + R_EARTH_EQUATOR
            ):
                raise ValueError("invalid Earth/Sun geometry for eclipse profile")
            if np.linalg.norm(sun) <= np.linalg.norm(state[:3]):
                raise ValueError(
                    "eclipse profile requires an Earth-bound observer closer than the Sun"
                )
            shadow = shadow_state(state[:3], sun)
            behind = -float(np.dot(state[:3], sun / np.linalg.norm(sun)))
            cache[t] = (
                min(behind, shadow.umbra_radius_m - shadow.r_perp_m),
                min(behind, shadow.penumbra_radius_m - shadow.r_perp_m),
            )
        return cache[t]

    result = {}
    for name, index in (("umbra", 0), ("partial_or_full", 1)):
        result[name] = positive_windows(
            lambda t, index=index: margins(t)[index],
            begin,
            end,
            settings=settings,
            boundaries=provider_boundaries(provider),
        )
    return {
        "schema": "space-env.eclipse-forecast/1",
        "model": "core.spherical-conical-linear-penumbra/1",
        "trajectory_sha256": provider.sha256,
        "coverage_sec": [begin, end],
        "sun_source": "ERFA epv00 GCRS"
        if sun_gcrs is None
        else "caller-declared GCRS callback; not independently verified",
        "geometry_evaluations": len(cache),
        **result,
        "limits": [
            "spherical Earth; approximate conical boundaries and linear penumbra flux",
            "same model as mission power/SRP; not an apparent-disc illumination upgrade",
            "no terrain/atmospheric refraction/Moon occultation; no unconditional short-event completeness",
        ],
    }
