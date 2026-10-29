"""
Оценка кадра отдельно от факта экспозиции и от сдачи.
Порог облачности 0.8 задан до прогона: это не прогноз погоды и не паспорт сенсора.
"""
from __future__ import annotations

CLOUD_BLOCK = 0.8
CLOUD_PARTIAL = 0.2


def cloud_cover(fraction: float, sigma: float, unit: float) -> float:
    """Доля облака. unit от 0 до 1 сдвигает долю на ±sigma. Нулевой sigma оставляет долю."""
    if float(sigma) <= 0.0:
        return float(fraction)
    shift = float(sigma) * (2.0 * float(unit) - 1.0)
    return min(1.0, max(0.0, float(fraction) + shift))


def frame_grade(
    *,
    off_nadir_deg: float,
    half_fov_deg: float,
    cloud_fraction: float = 0.0,
    images_in_eclipse: bool = False,
    quality_off_nadir_deg: float | None = None,
    slew_rate_deg_s: float = 0.0,
    smear_rate_deg_s: float = 0.0,
) -> str:
    """
    full, partial или none.
    none — кадра нет. partial — экспозиция есть, полное качество заявки она не закрывает.
    Радар не наследует облачность. Нулевой порог смаза скорость разворота не читает.
    """
    if off_nadir_deg > float(half_fov_deg):
        return "none"
    if not images_in_eclipse and cloud_fraction >= CLOUD_BLOCK:
        return "none"
    limit = float(half_fov_deg if quality_off_nadir_deg is None else quality_off_nadir_deg)
    if off_nadir_deg > limit:
        return "partial"
    if not images_in_eclipse and cloud_fraction >= CLOUD_PARTIAL:
        return "partial"
    if smear_rate_deg_s > 0.0 and slew_rate_deg_s > smear_rate_deg_s and not images_in_eclipse:
        return "partial"
    return "full"
