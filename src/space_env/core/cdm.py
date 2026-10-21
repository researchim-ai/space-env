"""
Разбор CCSDS Conjunction Data Message в KVN (ключевые слова, не XML).
Берутся TCA, промах, скорость сближения, площадь столкновения и RTN-ковариации обоих объектов.
Суммарная ковариация — сумма независимых блоков. Pc из сообщения сохраняется, но не подменяет расчёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CdmObject:
    designator: str = ""
    covariance_km2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class ConjunctionMessage:
    tca: str = ""
    miss_distance_m: float = 0.0
    relative_speed_m_s: float = 0.0
    hard_body_radius_m: float = 15.0
    reported_probability: float | None = None
    objects: list[CdmObject] = field(default_factory=list)

    def combined_covariance_m2(self) -> np.ndarray:
        total = np.zeros((3, 3))
        for obj in self.objects:
            total = total + obj.covariance_km2 * 1.0e6
        return total

    def combined_sigma_m(self) -> float:
        """Среднее СКО в плоскости радиус/нормаль, м."""
        cov = self.combined_covariance_m2()
        radial = max(cov[0, 0], 1.0)
        normal = max(cov[2, 2], 1.0)
        return float(np.sqrt(0.5 * (radial + normal)))


def _strip_units(value: str) -> str:
    if "[" in value:
        value = value.split("[", 1)[0]
    return value.strip()


def _float(value: str) -> float:
    return float(_strip_units(value).replace("D", "E"))


def parse_cdm_kvn(text: str) -> ConjunctionMessage:
    """Читает KVN-сообщение. Комментарии с # отбрасываются."""
    message = ConjunctionMessage()
    current: CdmObject | None = None
    cov = np.zeros((3, 3))
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if key == "TCA":
            message.tca = _strip_units(value)
        elif key == "MISS_DISTANCE":
            # В стандарте обычно километры.
            message.miss_distance_m = _float(value) * 1000.0
        elif key == "RELATIVE_SPEED":
            message.relative_speed_m_s = _float(value) * 1000.0
        elif key == "COLLISION_PROBABILITY":
            message.reported_probability = _float(value)
        elif key == "AREA_PC":
            area = _float(value)
            if area > 0.0:
                message.hard_body_radius_m = float(np.sqrt(area / np.pi))
        elif key == "OBJECT":
            if current is not None:
                current.covariance_km2 = cov
                message.objects.append(current)
            current = CdmObject()
            cov = np.zeros((3, 3))
        elif key == "OBJECT_DESIGNATOR" and current is not None:
            current.designator = _strip_units(value)
        elif key == "CR_R":
            cov[0, 0] = _float(value)
        elif key == "CT_T":
            cov[1, 1] = _float(value)
        elif key == "CN_N":
            cov[2, 2] = _float(value)
        elif key == "CT_R":
            cov[0, 1] = cov[1, 0] = _float(value)
        elif key == "CN_R":
            cov[0, 2] = cov[2, 0] = _float(value)
        elif key == "CN_T":
            cov[1, 2] = cov[2, 1] = _float(value)
    if current is not None:
        current.covariance_km2 = cov
        message.objects.append(current)
    if message.miss_distance_m <= 0.0:
        raise ValueError("в CDM нет MISS_DISTANCE")
    return message
