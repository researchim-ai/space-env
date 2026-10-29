"""
Книга наземных окон. Один терминал держит одно окно.
Более ранний приоритет проходит раньше. Пустая книга ничего не бронирует.
"""
from __future__ import annotations


def book_passes(requests: list[dict], terminals: dict[str, int]) -> dict:
    """Жадная бронь. Пересечение — полуинтервал [start, end)."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    order = sorted(
        requests,
        key=lambda item: (-int(item.get("priority", 0)), float(item["start_sec"]), str(item["vehicle"])),
    )
    held: dict[str, list[tuple[float, float]]] = {}
    for item in order:
        station = str(item["station"])
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        slots = max(1, int(terminals.get(station, 1)))
        if end <= start:
            rejected.append({**item, "reason": "пустое окно"})
            continue
        overlaps = [
            window
            for window in held.get(station, [])
            if start < window[1] and end > window[0]
        ]
        if len(overlaps) >= slots:
            rejected.append({**item, "reason": "терминал занят"})
            continue
        kept = {
            "vehicle": str(item["vehicle"]),
            "station": station,
            "start_sec": start,
            "end_sec": end,
            "priority": int(item.get("priority", 0)),
        }
        accepted.append(kept)
        held.setdefault(station, []).append((start, end))
    return {"accepted": accepted, "rejected": rejected}
