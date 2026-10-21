"""Память и сброс по опубликованным скоростям. Порог линии сюда не входит."""
from __future__ import annotations


def store_bits(
    buffer_bits: float,
    dt_sec: float,
    *,
    rate_bps: float,
    memory_bits: float,
    imaging_sec: float,
    imaging_cap_sec: float,
) -> tuple[float, float, float]:
    """
    Кладёт сжатый поток в память, пока не кончились витковый лимит съёмки или объём.
    Возвращает новый буфер, накопленные секунды съёмки и добавленные биты.
    """
    if dt_sec <= 0.0 or rate_bps <= 0.0 or memory_bits <= 0.0:
        return buffer_bits, imaging_sec, 0.0
    window = min(dt_sec, max(0.0, imaging_cap_sec - imaging_sec))
    room = max(0.0, memory_bits - buffer_bits)
    added = min(room, rate_bps * window)
    used = added / rate_bps
    return buffer_bits + added, imaging_sec + used, added


def dump_bits(buffer_bits: float, dt_sec: float, rate_bps: float) -> tuple[float, float]:
    """Снимает из буфера не больше, чем модем успевает за шаг. Возвращает остаток и сброшенное."""
    if dt_sec <= 0.0 or rate_bps <= 0.0 or buffer_bits <= 0.0:
        return buffer_bits, 0.0
    moved = min(buffer_bits, rate_bps * dt_sec)
    return buffer_bits - moved, moved
