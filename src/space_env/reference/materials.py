"""Published material correlation, not a spacecraft hardware calibration.

Specific heat of 6061-T6 aluminium: NIST cryogenic materials, 4--300 K.
No extrapolation, clipping, or interpretation of fit error as Gaussian noise.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from space_env.runtime.schema import StrictModel

MATERIAL_ID = "nist-al6061-t6-cp/1"
SOURCE_URI = (
    "https://trc.nist.gov/cryogenics/materials/6061%20Aluminum/6061_T6Aluminum_rev.htm"
)
SOURCE_SHA256 = "a2c62b7f4f774db34903bfcaf1479df326ab42de7a7e734a664aa7a0e76c972f"
COEFFICIENTS = (
    46.6467,
    -314.292,
    866.662,
    -1298.3,
    1162.27,
    -637.795,
    210.351,
    -38.3094,
    2.96344,
)
MIN_K, MAX_K = 4.0, 300.0


def _temperature(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not MIN_K <= value <= MAX_K
    ):
        raise ValueError(
            "NIST Al6061-T6 specific heat requires 4 <= T <= 300 K; no extrapolation"
        )
    return float(value)


def specific_heat_j_per_kg_k(temperature_k: float) -> float:
    """Evaluate the published log10 polynomial; this is a correlation, not raw data."""
    x = math.log10(_temperature(temperature_k))
    p = COEFFICIENTS[-1]
    for coefficient in reversed(COEFFICIENTS[:-1]):
        p = p * x + coefficient
    return 10.0**p


def specific_enthalpy_delta_j_per_kg(start_k: float, end_k: float) -> float:
    """Signed integral of cp(T), with a bounded independent quadrature budget."""
    from scipy.integrate import quad

    start, end = _temperature(start_k), _temperature(end_k)
    if start == end:
        return 0.0
    low, high = sorted((start, end))
    value, error, *details = quad(
        specific_heat_j_per_kg_k,
        low,
        high,
        epsabs=1e-8,
        epsrel=1e-10,
        limit=100,
        full_output=True,
    )
    # QUADPACK appends an error message on nonconvergence.
    if (
        len(details) != 1
        or not math.isfinite(value)
        or not math.isfinite(error)
        or error > max(1e-8, abs(value) * 1e-10)
    ):
        raise ValueError("material enthalpy quadrature did not meet its error budget")
    return value if end > start else -value


class MaterialHeatCapacity(StrictModel):
    material_id: Literal["nist-al6061-t6-cp/1"] = MATERIAL_ID
    mass_kg: float = Field(gt=0, le=1e6, strict=True)
    assumption: str = Field(min_length=1, max_length=2000, pattern=r"\S")

    def capacity(self, temperature_k: float) -> float:
        return self.mass_kg * specific_heat_j_per_kg_k(temperature_k)

    def stored(self, start_k: float, end_k: float) -> float:
        return self.mass_kg * specific_enthalpy_delta_j_per_kg(start_k, end_k)


def material_evidence() -> dict:
    """Owned JSON-ready evidence; node mass and geometry are caller assumptions."""
    return {
        "material_id": MATERIAL_ID,
        "source": {
            "uri": SOURCE_URI,
            "content_sha256": SOURCE_SHA256,
            "retrieved_on": "2026-11-03",
            "content_scope": "original HTML bytes",
            "locator": "first property table, Specific Heat column a--i and equation",
            "kind": "published_material_correlation",
        },
        "coefficients_ascending": list(COEFFICIENTS),
        "equation": "cp(T) = 10**sum(a[i] * log10(T)**i)",
        "units": "J/(kg*K)",
        "temperature_range_k": [MIN_K, MAX_K],
        "reported_curve_fit_error_percent_relative_to_data": 5.0,
        "error_interpretation": "source fit error; not measurement sigma, a confidence interval or spacecraft prediction accuracy",
        "hardware_calibration_complete": False,
        "industrial_readiness": False,
    }
