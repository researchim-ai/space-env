"""
Разбор CCSDS Conjunction Data Message в KVN (ключевые слова, не XML).
Берутся TCA, промах, скорость сближения, площадь столкновения и RTN-ковариации обоих объектов.
Единица в скобках читается. Без скобок остаётся договорённость CCSDS: км, км/с и км².
Суммарная ковариация — сумма независимых блоков одной системы и одной эпохи.
Pc из сообщения сохраняется, но не подменяет расчёт и не является эксплуатационной оценкой.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CdmObject:
    designator: str = ""
    frame: str = ""
    epoch: str = ""
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
    diagnostics: tuple[str, ...] = ()

    def combined_covariance_m2(self) -> np.ndarray:
        total = np.zeros((3, 3))
        for obj in self.objects:
            total = total + obj.covariance_km2 * 1.0e6
        return total

    def combined_state_covariance_si(self) -> np.ndarray | None:
        """Сумма 6×6, если блок скоростей есть у каждого объекта. Иначе None."""
        if self.diagnostics or not self.objects or any(obj.state_covariance_si is None for obj in self.objects):
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


_UNIT = re.compile(r"\[([^\]]+)\]")


def _strip_units(value: str) -> str:
    if "[" in value:
        value = value.split("[", 1)[0]
    return value.strip()


def _number_and_unit(value: str) -> tuple[float, str | None]:
    match = _UNIT.search(value)
    number = _strip_units(value).replace("D", "E")
    unit = None if match is None else match.group(1).strip().lower().replace(" ", "").replace("**", "").replace("^", "")
    return float(number), unit


def _scale(unit: str | None, default: str, accepted: dict[str, float], key: str) -> float:
    chosen = default if unit is None else unit
    if chosen not in accepted:
        known = ", ".join(sorted(accepted))
        raise ValueError(f"единица [{chosen}] у {key} не поддерживается, допустимы {known}")
    return accepted[chosen]


_LENGTH = {"m": 1.0, "km": 1000.0}
_SPEED = {"m/s": 1.0, "km/s": 1000.0}
_AREA = {"m2": 1.0, "km2": 1.0e6}
_POS2 = {"m2": 1.0, "km2": 1.0e6}
_VEL2 = {"m2/s2": 1.0, "km2/s2": 1.0e6}
_CROSS_SCALE = {"m2/s": 1.0, "km2/s": 1.0e6}


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


def _write_symmetric(matrix: np.ndarray, i: int, j: int, value: float) -> None:
    matrix[i, j] = matrix[j, i] = value


def _psd(matrix: np.ndarray) -> bool:
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(1.0, float(np.max(np.abs(symmetric))))
    return bool(np.all(np.linalg.eigvalsh(symmetric) >= -1.0e-8 * scale))


def parse_cdm_kvn(text: str) -> ConjunctionMessage:
    """Читает KVN-сообщение. Комментарии с # отбрасываются."""
    message = ConjunctionMessage()
    current: CdmObject | None = None
    cov = np.zeros((3, 3))
    state = np.zeros((6, 6))
    saw_velocity = False
    notes: list[str] = []

    def _close() -> None:
        nonlocal current, cov, state, saw_velocity
        if current is None:
            return
        current.covariance_km2 = cov
        if saw_velocity:
            current.state_covariance_si = state.copy()
            if not _psd(state):
                notes.append(f"ковариация {current.designator or 'объекта'} не положительно полуопределена")
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
            number, unit = _number_and_unit(value)
            message.miss_distance_m = number * _scale(unit, "km", _LENGTH, key)
        elif key == "RELATIVE_SPEED":
            number, unit = _number_and_unit(value)
            message.relative_speed_m_s = number * _scale(unit, "km/s", _SPEED, key)
        elif key == "COLLISION_PROBABILITY":
            number, _unit = _number_and_unit(value)
            message.reported_probability = number
        elif key == "AREA_PC":
            number, unit = _number_and_unit(value)
            area = number * _scale(unit, "m2", _AREA, key)
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
        elif key == "COV_REF_FRAME" and current is not None:
            current.frame = _strip_units(value)
        elif key == "REF_FRAME_EPOCH" and current is not None:
            current.epoch = _strip_units(value)
        elif key in _POSITION and current is not None:
            number, unit = _number_and_unit(value)
            metres2 = number * _scale(unit, "km2", _POS2, key)
            i, j = _POSITION[key]
            _write_symmetric(cov, i, j, metres2 / 1.0e6)
            _write_symmetric(state, i, j, metres2)
        elif key in _VELOCITY and current is not None:
            number, unit = _number_and_unit(value)
            i, j = _VELOCITY[key]
            _write_symmetric(state, i, j, number * _scale(unit, "km2/s2", _VEL2, key))
            saw_velocity = True
        elif key in _CROSS and current is not None:
            number, unit = _number_and_unit(value)
            i, j = _CROSS[key]
            _write_symmetric(state, i, j, number * _scale(unit, "km2/s", _CROSS_SCALE, key))
            saw_velocity = True
    _close()
    frames = {item.frame for item in message.objects if item.frame}
    if len(frames) > 1:
        raise ValueError("ковариации объектов в разных системах не складываются")
    epochs = {item.epoch for item in message.objects if item.epoch}
    if len(epochs) > 1:
        notes.append("эпохи ковариаций не совпадают")
    message.diagnostics = tuple(notes)
    if message.miss_distance_m <= 0.0:
        raise ValueError("в CDM нет MISS_DISTANCE")
    return message
