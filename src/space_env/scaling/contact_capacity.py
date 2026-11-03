"""Fractional delivery relaxation with an exact binary-input dual bound.

Finite box bounds make any finite y<=0,z a valid lower bound for min(-sum f).
Fraction arithmetic verifies that bound without trusting solver feasibility.
This certifies the stated relaxation, not physical contact completeness.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np

from space_env.simulation.snapshot import digest


def dual_upper_bound(problem, inequality_dual, equality_dual):
    if problem.get("schema") != "space-env.contact-capacity-problem/1" or set(
        problem
    ) != {"schema", "products", "capacities_sec", "flows"}:
        raise ValueError("unsupported capacity problem schema")
    n = len(problem["products"])
    if (
        not 1 <= n <= 64
        or len(problem["capacities_sec"]) > 8192
        or len(problem["flows"]) > 20000
    ):
        raise ValueError("capacity certificate dimensions exceed the bounded profile")

    def number(v, positive=False):
        return (
            not isinstance(v, bool)
            and isinstance(v, (int, float))
            and math.isfinite(v)
            and (v > 0 if positive else v >= 0)
        )

    if any(not number(v) for v in problem["capacities_sec"]) or any(
        not number(p["required_tx_sec"], True) for p in problem["products"]
    ):
        raise ValueError("invalid finite capacity coefficients")
    for flow in problem["flows"]:
        if (
            type(flow["product"]) is not int
            or not 0 <= flow["product"] < n
            or any(
                type(flow[k]) is not int
                or not 0 <= flow[k] < len(problem["capacities_sec"])
                for k in ["actor_row", "station_row"]
            )
            or not number(flow["upper_sec"], True)
        ):
            raise ValueError("invalid capacity flow topology/bounds")
    if (
        len(inequality_dual) != len(problem["capacities_sec"])
        or len(equality_dual) != n
    ):
        raise ValueError("dual vector shape mismatch")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in [*inequality_dual, *equality_dual]
    ):
        raise ValueError("dual values must be finite numbers")
    y = [Fraction.from_float(float(min(0.0, v))) for v in inequality_dual]
    z = [Fraction.from_float(float(v)) for v in equality_dual]
    lower = sum(
        Fraction.from_float(float(b)) * a for b, a in zip(problem["capacities_sec"], y)
    )
    for i, p in enumerate(problem["products"]):
        # Equality: -required_tx_sec*f + sum(x) = 0, 0<=f<=1.
        reduced = -1 + Fraction.from_float(float(p["required_tx_sec"])) * z[i]
        lower += min(Fraction(0), reduced)
    for v in problem["flows"]:
        reduced = -y[v["actor_row"]] - y[v["station_row"]] - z[v["product"]]
        lower += min(Fraction(0), reduced * Fraction.from_float(float(v["upper_sec"])))
    upper = min(Fraction(n), -lower)
    integer_upper = min(n, max(0, -(-upper.numerator // upper.denominator)))
    estimate = float(upper)
    if Fraction.from_float(estimate) < upper:
        estimate = math.nextafter(estimate, math.inf)
    return {
        "fractional_upper_numerator": str(upper.numerator),
        "fractional_upper_denominator": str(upper.denominator),
        "fractional_upper_float_rounded_up": estimate,
        "integer_upper_binary_lp_floor": min(
            n, max(0, upper.numerator // upper.denominator)
        ),
        "integer_upper_conservative_ceiling": integer_upper,
        "method": "exact Fraction weak-duality bound on binary64 LP coefficients; conservative integer ceiling",
    }


def contact_capacity(products, vehicles, stations, windows, horizon, outages):
    """All-ready fixed-rate products, fixed destination, one TX/actor and station slots."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    names = sorted(vehicles)
    sites = sorted(stations)
    bounds = {0.0, float(horizon)}
    # A 1ms outward padding relaxes the native root boundaries, not physical truth.
    padded = {
        pair: [(max(0.0, a - 0.001), min(horizon, b + 0.001)) for a, b in spans]
        for pair, spans in windows.items()
    }
    for spans in padded.values():
        bounds.update(t for span in spans for t in span)
    for outage in outages:
        bounds.update([float(outage["start_sec"]), float(outage["end_sec"])])
    times = sorted(bounds)
    if len(times) > 513:
        raise ValueError("capacity interval budget exceeded")
    p = [
        {
            "id": row["id"],
            "actor": row["actor"],
            "station": row["destination"],
            "required_tx_sec": float(
                row["size_bits"] / vehicles[row["actor"]]["rate_bps"]
            ),
        }
        for row in products
    ]
    capacities, flows = [], []
    for left, right in pairwise(times):
        if right <= left:
            continue
        mid = (left + right) / 2
        span = float(right - left)
        rows_a = {a: len(capacities) + i for i, a in enumerate(names)}
        capacities.extend([span] * len(names))
        rows_s = {s: len(capacities) + i for i, s in enumerate(sites)}
        capacities.extend([span * stations[s]["terminals"] for s in sites])
        for i, row in enumerate(p):
            pair = (row["actor"], row["station"])
            if any(a <= mid < b for a, b in padded[pair]) and not any(
                o["station"] == row["station"] and o["start_sec"] <= mid < o["end_sec"]
                for o in outages
            ):
                flows.append(
                    {
                        "product": i,
                        "actor_row": rows_a[row["actor"]],
                        "station_row": rows_s[row["station"]],
                        "upper_sec": span,
                    }
                )
                if len(flows) > 20000:
                    raise ValueError("capacity variable budget exceeded")
    problem = {
        "schema": "space-env.contact-capacity-problem/1",
        "products": p,
        "capacities_sec": capacities,
        "flows": flows,
    }
    n = len(p)
    count = n + len(flows)
    ub_r = []
    ub_c = []
    ub_v = []
    eq_r = list(range(n))
    eq_c = list(range(n))
    eq_v = [-x["required_tx_sec"] for x in p]
    for j, v in enumerate(flows, n):
        ub_r.extend([v["actor_row"], v["station_row"]])
        ub_c.extend([j, j])
        ub_v.extend([1.0, 1.0])
        eq_r.append(v["product"])
        eq_c.append(j)
        eq_v.append(1.0)
    a = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(capacities), count)).tocsr()
    e = coo_matrix((eq_v, (eq_r, eq_c)), shape=(n, count)).tocsr()
    solution = linprog(
        [-1.0] * n + [0.0] * len(flows),
        A_ub=a,
        b_ub=capacities,
        A_eq=e,
        b_eq=np.zeros(n),
        bounds=[(0.0, 1.0)] * n + [(0.0, v["upper_sec"]) for v in flows],
        method="highs",
        options={"time_limit": 10.0, "maxiter": 10000},
    )
    y = (
        [float(min(0.0, v)) for v in solution.ineqlin.marginals]
        if solution.success
        else [0.0] * len(capacities)
    )
    z = [float(v) for v in solution.eqlin.marginals] if solution.success else [0.0] * n
    certificate = dual_upper_bound(problem, y, z)
    result = {
        "schema": "space-env.contact-capacity-bound/1",
        "problem": problem,
        "problem_sha256": digest(problem),
        "inequality_dual": y,
        "equality_dual": z,
        "certificate": certificate,
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "primal_fractional_products_estimate": float(-solution.fun)
        if solution.success
        else None,
        "limits": "upper bound on relaxed fixed-rate service, not a feasible plan, an optimum of the native controller or calibrated RF coverage",
    }
    result["sha256"] = digest(result)
    return result


def verify_capacity_certificate(value):
    if value.get("sha256") != digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ) or value.get("problem_sha256") != digest(value["problem"]):
        raise ValueError("capacity certificate checksum mismatch")
    expected = dual_upper_bound(
        value["problem"], value["inequality_dual"], value["equality_dual"]
    )
    if digest(value["certificate"]) != digest(expected):
        raise ValueError("capacity exact dual bound mismatch")
    return {
        "verified": True,
        "native_reexecuted": False,
        "solver_reexecuted": False,
        "scope": "exact binary-input relaxation bound, not physical contact truth or native optimality",
    }
