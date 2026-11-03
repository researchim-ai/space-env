"""Offline stdlib Decimal oracle. Never imports the production material/solver.

Reproduce only explicitly; normal acceptance checks consume pinned frozen bytes.
Published correlation shared, independent polynomial sum and adaptive Simpson.
"""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path


def generate():
    with localcontext() as context:
        context.prec = 45
        d = Decimal
        coefficients = tuple(
            map(
                d,
                (
                    "46.6467",
                    "-314.292",
                    "866.662",
                    "-1298.3",
                    "1162.27",
                    "-637.795",
                    "210.351",
                    "-38.3094",
                    "2.96344",
                ),
            )
        )

        def cp(t):
            x = t.log10()
            return d(10) ** sum(a * x**i for i, a in enumerate(coefficients))

        def integral(a, b):
            def simpson(a, b, fa, fm, fb):
                return (b - a) * (fa + 4 * fm + fb) / 6

            def refine(a, b, fa, fm, fb, whole, tolerance, depth):
                mid = (a + b) / 2
                fl, fr = cp((a + mid) / 2), cp((mid + b) / 2)
                left, right = simpson(a, mid, fa, fl, fm), simpson(mid, b, fm, fr, fb)
                delta = left + right - whole
                if abs(delta) <= 15 * tolerance:
                    return left + right + delta / 15
                if depth == 0:
                    raise ValueError("Decimal oracle quadrature exhausted")
                return refine(
                    a, mid, fa, fl, fm, left, tolerance / 2, depth - 1
                ) + refine(mid, b, fm, fr, fb, right, tolerance / 2, depth - 1)

            fa, fm, fb = cp(a), cp((a + b) / 2), cp(b)
            return refine(a, b, fa, fm, fb, simpson(a, b, fa, fm, fb), d("1e-8"), 24)

        points = [
            {"temperature_k": t, "cp_j_per_kg_k": float(cp(d(t)))}
            for t in [4, 5, 10, 20, 50, 77, 100, 150, 200, 250, 280, 300]
        ]
        energies = [
            {"start_k": a, "end_k": b, "enthalpy_j_per_kg": float(integral(d(a), d(b)))}
            for a, b in [(4, 10), (10, 77), (77, 150), (150, 250), (250, 300), (4, 300)]
        ]
    result = {
        "schema": "space-env.material-oracle/1",
        "scope": "independent arithmetic/integration of a shared published correlation, not independent measurements",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decimal_precision": 45,
        "integral_absolute_target_j_per_kg": 1e-8,
        "points": points,
        "energies": energies,
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return result


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(
        json.dumps(generate(), ensure_ascii=False, indent=2) + "\n"
    )
