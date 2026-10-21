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
    # 6×6 в метрах и м/с. None — в сообщении нет блока скоростей, переносить такую матрицу нельзя.
    state_covariance_si: np.ndarray | None = None


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

    def combined_state_covariance_si(self) -> np.ndarray | None:
        """Сумма 6×6, если блок скоростей есть у каждого объекта. Иначе None."""
        if not self.objects or any(obj.state_covariance_si is None for obj in self.objects):
            return None
        total = np.zeros((6, 6))
        for obj in self.objects:
            total = total + obj.state_covariance_si
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


# Индексы RTN: R T N Rdot Tdot Ndot. Единицы в KVN — км и км/с.
_POSITION = {"CR_R": (0, 0), "CT_T": (1, 1), "CN_N": (2, 2), "CT_R": (0, 1), "CN_R": (0, 2), "CN_T": (1, 2)}
_VELOCITY = {
    "CRDOT_RDOT": (3, 3),
    "CTDOT_TDOT": (4, 4),
    "CNDOT_NDOT": (5, 5),
    "CTDOT_RDOT": (3, 4),
    "CNDOT_RDOT": (3, 5),
    "CNDOT_TDOT": (4, 5),
}
_CROSS = {
    "CRDOT_R": (0, 3),
    "CRDOT_T": (1, 3),
    "CRDOT_N": (2, 3),
    "CTDOT_R": (0, 4),
    "CTDOT_T": (1, 4),
    "CTDOT_N": (2, 4),
    "CNDOT_R": (0, 5),
    "CNDOT_T": (1, 5),
    "CNDOT_N": (2, 5),
}


def _to_si(raw: np.ndarray) -> np.ndarray:
    si = np.array(raw, dtype=float, copy=True)
    si[:3, :3] *= 1.0e6
    si[:3, 3:] *= 1.0e3
    si[3:, :3] *= 1.0e3
    si[3:, 3:] *= 1.0e6
    return si


def parse_cdm_kvn(text: str) -> ConjunctionMessage:
    """Читает KVN-сообщение. Комментарии с # отбрасываются."""
    message = ConjunctionMessage()
    current: CdmObject | None = None
    cov = np.zeros((3, 3))
    state = np.zeros((6, 6))
    saw_velocity = False

    def _close() -> None:
        nonlocal current, cov, state, saw_velocity
        if current is None:
            return
        current.covariance_km2 = cov
        if saw_velocity:
            current.state_covariance_si = _to_si(state)
        message.objects.append(current)

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
            _close()
            current = CdmObject()
            cov = np.zeros((3, 3))
            state = np.zeros((6, 6))
            saw_velocity = False
        elif key == "OBJECT_DESIGNATOR" and current is not None:
            current.designator = _strip_units(value)
        elif key in _POSITION:
            i, j = _POSITION[key]
            value_f = _float(value)
            cov[i, j] = cov[j, i] = value_f
            state[i, j] = state[j, i] = value_f
        elif key in _VELOCITY or key in _CROSS:
            i, j = (_VELOCITY if key in _VELOCITY else _CROSS)[key]
            value_f = _float(value)
            state[i, j] = state[j, i] = value_f
            saw_velocity = True
    _close()
    if message.miss_distance_m <= 0.0:
        raise ValueError("в CDM нет MISS_DISTANCE")
    return message
