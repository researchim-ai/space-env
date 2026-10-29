"""
Профили достоверности. Числовые допуски заданы здесь и не читаются из остатка прогона.

reference — короткая двухтелая дуга, по которой сверяются метры и скорость.
training — силы учебной миссии. Допуск метров reference-дуги на сутки и 72 часа не переносится.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FidelityProfile:
    """Силы, кадр, шкала, интегратор и допуски одного профиля."""

    name: str
    forces: tuple[str, ...]
    frame: str
    time_scale: str
    integrator: str
    substep_sec: float
    event_tolerance_sec: float
    max_arc_sec: float
    outcome: str
    position_tolerance_m: float | None = None
    velocity_tolerance_m_s: float | None = None
    contact_tolerance_sec: float | None = None
    convergence_ratio_min: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_FRAME = "numerical_cowell_labeled_gcrf_iau1980"
_TIME = "calendar_utc_without_leap_seconds"

REFERENCE_PROFILE = FidelityProfile(
    name="reference",
    forces=("two_body",),
    frame=_FRAME,
    time_scale=_TIME,
    integrator="rk4_stage_epoch",
    substep_sec=5.0,
    event_tolerance_sec=1.0,
    max_arc_sec=600.0,
    outcome="meters_on_this_arc_only",
    position_tolerance_m=50.0,
    velocity_tolerance_m_s=1.0,
    contact_tolerance_sec=1.0,
    convergence_ratio_min=8.0,
)

TRAINING_PROFILE = FidelityProfile(
    name="training",
    forces=("two_body", "j2", "table_drag", "cannonball_srp", "third_body_when_requested"),
    frame=_FRAME,
    time_scale=_TIME,
    integrator="rk4_stage_epoch",
    substep_sec=10.0,
    event_tolerance_sec=1.0,
    max_arc_sec=72.0 * 3600.0,
    outcome="mission_delivery_list",
)


def optional_backends() -> list[dict[str, object]]:
    """Внешняя библиотека не заменяет отчёт и не вводит переход TEME→GCRF."""
    return [
        {
            "name": "boinor",
            "role": "kepler_arc_for_the_reference_profile",
            "replaces_validation": False,
            "teme_to_gcrf": False,
        },
        {
            "name": "orekit",
            "role": "optional_kepler_propagator",
            "replaces_validation": False,
            "teme_to_gcrf": False,
        },
        {
            "name": "sgp4",
            "role": "optional_tle_in_teme",
            "replaces_validation": False,
            "teme_to_gcrf": False,
        },
        {
            "name": "spice",
            "role": "optional_ephemeris",
            "replaces_validation": False,
            "teme_to_gcrf": False,
        },
    ]
