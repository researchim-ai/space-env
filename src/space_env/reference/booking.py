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


class ExternalReservation(_Config):
    """Provider revision with optimistic concurrency; one record occupies one terminal."""
    provider: str = Field(min_length=1, max_length=128)
    external_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1, strict=True)
    expected_previous_revision: int = Field(ge=0, strict=True)
    status: str
    actor: str = Field(min_length=1)
    station: str = Field(min_length=1)
    start_sec: float = Field(ge=0, strict=True)
    end_sec: float = Field(gt=0, strict=True)
    source: str = Field(min_length=1)


def import_reservations(mission, payload, bookings, cancelled, records, boundaries):
    """Stage imports through the same booking sweep as local operator events.

    A revision truncates the old allocation at now; history is never overwritten.
    Records are local evidence, not an API call or an external provider guarantee.
    """
    from space_env.simulation.snapshot import digest

    if set(payload) != {'records'} or not isinstance(payload['records'], list) or not 1 <= len(payload['records']) <= 1000:
        raise ValueError('reservation import requires 1-1000 records')
    for value in payload['records']:
        row = ExternalReservation.model_validate(value)
        if row.status not in {'confirmed', 'cancelled'} or row.start_sec >= row.end_sec:
            raise ValueError('invalid external reservation status/interval')
        if row.actor not in mission.vehicles or row.station not in mission.stations or row.end_sec > mission.horizon_sec:
            raise ValueError('external reservation references unknown resource or exceeds horizon')
        # Length-delimited components prevent provider/ID separator collisions.
        key = f'{len(row.provider)}:{row.provider}:{row.external_id}'
        old = records.get(key)
        canonical = row.model_dump(mode='json')
        fingerprint = digest(canonical)
        if old is not None and row.revision == old['revision'] and fingerprint == old['sha256']:
            continue
        previous = 0 if old is None else old['revision']
        if row.expected_previous_revision != previous or row.revision <= previous:
            raise ValueError('stale external reservation revision')
        if old is not None and old['booking_id'] is not None:
            bid = old['booking_id']
            if min(bookings[bid].end_sec,cancelled.get(bid,bookings[bid].end_sec)) <= mission.clock:
                raise ValueError('external reservation already ended or cancelled')
            cancelled[bid] = mission.clock
        elif row.status == 'cancelled':
            raise ValueError('cannot cancel absent external allocation')
        bid = None
        if row.status == 'confirmed':
            if row.start_sec < mission.clock:
                raise ValueError('import cannot backdate an allocation')
            bid = '@external/'+digest({'key':key,'revision':row.revision})
            if bid in bookings or len(bookings) >= 10000:
                raise ValueError('external booking ID conflict or booking budget exhausted')
            bookings[bid] = TerminalBooking(id=bid,actor=row.actor,station=row.station,
                                           start_sec=row.start_sec,end_sec=row.end_sec)
            boundaries.update((row.start_sec,row.end_sec))
        records[key] = {'revision':row.revision,'sha256':fingerprint,'booking_id':bid,'record':canonical}
    validate_bookings(bookings,cancelled,mission.stations)
