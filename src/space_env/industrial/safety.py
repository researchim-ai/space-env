"""Conservative continuous-distance bounds, conditional on a supplied speed bound.

Unlike sampled closest approach, unresolved intervals remain indeterminate.
This is a floating-point bound with a declared numerical position allowance,
not formal interval arithmetic or a certification of the supplied trajectory.
"""

from __future__ import annotations

import math

import numpy as np

from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.core.relative import cw_stm


def interval_separation(
    position,
    duration_sec,
    keepout_m,
    speed_bound_m_s,
    *,
    max_depth=16,
    max_evaluations=4097,
    numerical_allowance_m=1e-7,
):
    for value in (duration_sec, keepout_m, speed_bound_m_s, numerical_allowance_m):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative interval parameters required")
    if (
        type(max_depth) is not int
        or not 0 <= max_depth <= 24
        or type(max_evaluations) is not int
        or not 2 <= max_evaluations <= 65537
    ):
        raise ValueError("invalid interval evaluation budget")
    samples = {}

    def distance(t):
        if t not in samples:
            vector = np.asarray(position(t), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("trajectory must return a finite three-vector")
            samples[t] = float(np.linalg.norm(vector))
        return samples[t]

    left, right = distance(0.0), distance(float(duration_sec))
    pending = [(0.0, float(duration_sec), left, right, 0)]
    unresolved, lowers = 0, []
    while pending:
        a, b, da, db, depth = pending.pop()
        # A nearest endpoint lies at most h/2 away; distance is speed-Lipschitz.
        lower = max(
            0.0, min(da, db) - speed_bound_m_s * (b - a) / 2 - numerical_allowance_m
        )
        if min(da, db) + numerical_allowance_m < keepout_m:
            return {
                "status": "unsafe",
                "evaluations": len(samples),
                "witness_sec": min(samples, key=samples.get),
                "sampled_min_m": min(samples.values()),
                "lower_bound_m": 0.0,
            }
        if lower >= keepout_m:
            lowers.append(lower)
            continue
        if depth == max_depth or len(samples) >= max_evaluations or b == a:
            unresolved += 1
            lowers.append(lower)
            continue
        mid = (a + b) / 2
        dm = distance(mid)
        pending.extend(((a, mid, da, dm, depth + 1), (mid, b, dm, db, depth + 1)))
    return {
        "status": "indeterminate" if unresolved else "safe",
        "evaluations": len(samples),
        "sampled_min_m": min(samples.values()),
        "lower_bound_m": min(lowers),
        "unresolved_intervals": unresolved,
    }


def cw_speed_bound(state, n_rad_s, duration_sec):
    """Analytic absolute-term bound on CW velocity over [0,T]."""
    x = np.asarray(state, dtype=float)
    if (
        x.shape != (6,)
        or not np.isfinite(x).all()
        or not math.isfinite(n_rad_s)
        or n_rad_s < 0
        or not math.isfinite(duration_sec)
        or duration_sec < 0
    ):
        raise ValueError("bounded CW state and nonnegative mean motion/time required")
    if n_rad_s < 1e-12:
        return float(np.linalg.norm(x[3:]))
    # sin/cos in [-1,1], |cos-1|<=2; velocity has no secular t term.
    a, _, c, u, v, w = abs(x)
    return float(
        np.linalg.norm(
            (
                3 * n_rad_s * a + u + 2 * v,
                12 * n_rad_s * a + 2 * u + 7 * v,
                n_rad_s * c + w,
            )
        )
    )


def check_cw_interval(state, n_rad_s, duration_sec, keepout_m, **kwargs):
    x = np.asarray(state, dtype=float)
    bound = cw_speed_bound(x, n_rad_s, duration_sec)
    return interval_separation(
        lambda t: (cw_stm(n_rad_s, t) @ x)[:3], duration_sec, keepout_m, bound, **kwargs
    )


def central_pericenter_m(state):
    """Conic pericenter for an unforced central Earth orbit, shared SI constant."""
    x = np.asarray(state, dtype=float)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError("finite central-orbit state required")
    r = float(np.linalg.norm(x[:3]))
    h = np.cross(x[:3], x[3:])
    if not R_EARTH_EQUATOR < r < 1e9 or np.linalg.norm(h) <= 1:
        raise ValueError("degenerate central orbit")
    eccentricity = float(np.linalg.norm(np.cross(x[3:], h) / MU_EARTH - x[:3] / r))
    return float(h @ h / MU_EARTH / (1 + eccentricity))


def check_central_interval(
    left_state,
    right_state,
    left_position,
    right_position,
    duration_sec,
    keepout_m,
    **kwargs,
):
    """Coast bound from conic pericenters, not CW or sampled minimization.

    |a_rel| <= mu/rp_left² + mu/rp_right²,
    |v_rel(t)| <= |v_rel(0)| + a_bound*T.
    Numerical trajectory error is an explicit allowance, not a proved ODE bound.
    """
    if (
        isinstance(duration_sec, bool)
        or not math.isfinite(duration_sec)
        or not 0 <= duration_sec <= 30
    ):
        raise ValueError("central interval is bounded to thirty seconds")
    left, right = (
        np.asarray(left_state, dtype=float),
        np.asarray(right_state, dtype=float),
    )
    radii = [central_pericenter_m(x) for x in (left, right)]
    lower = [r - max(0.001, r * 1e-12) for r in radii]
    if min(lower) < R_EARTH_EQUATOR + 100000:
        raise ValueError("central coast pericenter is outside the 100-km envelope")
    acceleration = math.fsum(MU_EARTH / r**2 for r in lower)
    speed = float(np.linalg.norm(left[3:] - right[3:])) + acceleration * duration_sec
    result = interval_separation(
        lambda t: np.asarray(left_position(t)) - np.asarray(right_position(t)),
        duration_sec,
        keepout_m,
        speed,
        numerical_allowance_m=kwargs.pop("numerical_allowance_m", 0.01),
        **kwargs,
    )
    return {
        **result,
        "scope": "central-unforced-GCRS-coast/1",
        "speed_bound_m_s": speed,
        "pericenter_lower_bounds_m": lower,
    }
