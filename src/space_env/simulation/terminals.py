"""Интервалы терминала внутри шага. Один слот не отдаёт двум аппаратам одно и то же время."""
from __future__ import annotations


def _merge(pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in pieces if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1e-9:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def assign_terminal_intervals(
    claims: list[tuple[str, str, float, float]],
    slots_by_station: dict[str, int],
) -> dict[str, list[tuple[float, float]]]:
    """
    claims: станция, аппарат, начало, конец.
    На каждом пересечении остаются первые slots имён в алфавитном порядке.
    Свободный хвост другого аппарата сохраняется.
    """
    by_station: dict[str, list[tuple[str, float, float]]] = {}
    vehicles: set[str] = set()
    for station, vehicle, start, end in claims:
        if float(end) <= float(start):
            continue
        by_station.setdefault(str(station), []).append((str(vehicle), float(start), float(end)))
        vehicles.add(str(vehicle))
    allowed: dict[str, list[tuple[float, float]]] = {name: [] for name in vehicles}
    for station, rows in by_station.items():
        slots = max(1, int(slots_by_station.get(station, 1)))
        marks = sorted({moment for _vehicle, start, end in rows for moment in (start, end)})
        for left, right in zip(marks, marks[1:]):
            if right <= left + 1e-12:
                continue
            active = sorted(
                vehicle
                for vehicle, start, end in rows
                if start <= left + 1e-9 and end >= right - 1e-9
            )
            for vehicle in active[:slots]:
                allowed[vehicle].append((left, right))
    return {name: _merge(pieces) for name, pieces in allowed.items()}
