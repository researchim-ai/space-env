"""Эталоны с измеренной ошибкой. Профиль, который не сошёлся, остаётся experimental."""

from space_env.validation.profiles import REFERENCE_PROFILE, TRAINING_PROFILE, optional_backends
from space_env.validation.report import applicability_limits, core_report, reference_report

__all__ = [
    "REFERENCE_PROFILE",
    "TRAINING_PROFILE",
    "applicability_limits",
    "core_report",
    "optional_backends",
    "reference_report",
]
