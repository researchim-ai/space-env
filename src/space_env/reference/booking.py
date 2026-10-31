"""One-terminal reservations with half-open intervals and single-radio actors."""
from __future__ import annotations

from pydantic import Field

from space_env.reference.operations import _Config


class TerminalBooking(_Config):
    id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    station: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)


def validate_bookings(bookings, cancelled, stations):
    """Sweep effective historical/future intervals; end precedes start on ties."""
    changes = []
    for b in bookings.values():
        end = min(b.end_sec, cancelled.get(b.id, b.end_sec))
        if end > b.start_sec:
            changes.extend([(b.start_sec, 1, b), (end, -1, b)])
    occupied, actors = dict.fromkeys(stations, 0), set()
    for _time, sign, b in sorted(changes, key=lambda x: (x[0], x[1], x[2].id)):
        if sign < 0:
            occupied[b.station] -= 1
            actors.remove(b.actor)
        else:
            if b.actor in actors:
                raise ValueError("actor has overlapping single-radio bookings")
            actors.add(b.actor)
            occupied[b.station] += 1
            if occupied[b.station] > stations[b.station].terminals:
                raise ValueError("station terminal reservation conflict")
