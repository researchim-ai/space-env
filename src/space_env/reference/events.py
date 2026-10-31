"""Causal operator events for fixed-ephemeris operations, applied only at now."""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

from space_env.reference.booking import TerminalBooking, validate_bookings
from space_env.reference.dependencies import dependency_children, dependency_order
from space_env.reference.operations import OnboardProduct, StationOutage
from space_env.simulation.snapshot import canonical_json

if TYPE_CHECKING:
    from space_env.reference.operations import ReferenceDownlinkMission


def validate_event_envelope(event: dict) -> None:
    if not isinstance(event, dict) or set(event) != {"id", "type", "time_sec", "source", "reason", "payload"}:
        raise ValueError("event requires exactly id,type,time_sec,source,reason,payload")
    for field in ("id", "source", "reason", "type"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"nonempty event {field} is required")
    time = event["time_sec"]
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
        raise ValueError("event time must be finite")
    if not isinstance(event["payload"], dict):
        raise TypeError("event payload must be an object")
    if event["type"] not in {"onboard_product_added", "product_priority_changed", "station_outage_added", "station_recovered",
                             "product_cancelled", "product_deadline_changed", "station_rate_changed", "transmitter_rate_changed",
                             "terminal_booking_added", "terminal_booking_cancelled", "station_reservation_policy_changed"}:
        raise ValueError("unsupported reference event type")


def apply_events(mission: ReferenceDownlinkMission, events: list[dict]) -> dict:
    """Atomically apply an ordered batch; IDs cannot be reused or backdated.

    onboard_product_added represents already acquired bytes becoming available,
    not an imaging command or a physical acquisition model.
    """
    if not isinstance(events, list) or not events or len(events) > 1000:
        raise ValueError("events must be a nonempty list of at most 1000 events")
    if len(events)+len(mission.events) > 10000:
        raise ValueError("mission event limit reached")
    canonical_json(events)
    if mission.clock >= mission.horizon_sec-1e-9:
        raise ValueError("cannot inject events at or after mission horizon")
    staged_products = dict(mission.products)
    staged_remaining = dict(mission.remaining)
    staged_outages = list(mission.outages)
    cancelled, discarded = dict(mission.cancelled), dict(mission.discarded)
    station_rates, transmitter_rates = dict(mission.station_rates), dict(mission.transmitter_rates)
    bookings, booking_cancelled = dict(mission.bookings), dict(mission.booking_cancelled)
    policies = dict(mission.reservation_required)
    processing_remaining, processing_completed = dict(mission.processing_remaining), dict(mission.processing_completed)
    boundaries = set(mission._boundaries)
    ids = {e["id"] for e in mission.events}
    accepted = []
    for event in events:
        validate_event_envelope(event)
        time = event["time_sec"]
        if time != mission.clock:
            raise ValueError("event time must equal current mission time")
        if event["id"] in ids:
            raise ValueError("duplicate event ID")
        ids.add(event["id"])
        payload = event["payload"]
        if event["type"] == "onboard_product_added":
            p = OnboardProduct.model_validate(payload)
            if (p.id in staged_products or p.actor not in mission.vehicles
                    or p.ready_sec < mission.clock or p.ready_sec >= p.deadline_sec
                    or p.deadline_sec > mission.horizon_sec
                    or (p.destination and p.destination not in mission.stations)):
                raise ValueError("invalid new onboard product")
            if len(staged_products) >= 10000:
                raise ValueError("product limit reached")
            occupied = sum(staged_remaining[p.id] for p in staged_products.values() if p.actor == payload["actor"])
            if occupied+p.size_bits > mission.vehicles[p.actor].memory_bits:
                raise ValueError("new product exceeds available onboard memory")
            if p.workspace_bits > mission.vehicles[p.actor].memory_bits:
                raise ValueError("processing workspace exceeds memory")
            staged_products[p.id], staged_remaining[p.id] = p, p.size_bits
            discarded[p.id] = 0.
            processing_remaining[p.id], processing_completed[p.id] = p.processing_sec, None
            boundaries.update((p.ready_sec, p.deadline_sec))
        elif event["type"] == "product_priority_changed":
            if set(payload) != {"product_id", "priority"} or payload.get("product_id") not in staged_products:
                raise ValueError("priority event requires a known product and priority")
            p = staged_products[payload["product_id"]]
            if staged_remaining[p.id] == 0:
                raise ValueError("cannot reprioritize a delivered product")
            staged_products[p.id] = OnboardProduct.model_validate({**p.model_dump(), "priority": payload["priority"]})
        elif event["type"] in {"product_cancelled", "product_deadline_changed"}:
            fields = {"product_id"} if event["type"] == "product_cancelled" else {"product_id", "deadline_sec"}
            if set(payload) != fields or payload.get("product_id") not in staged_products:
                raise ValueError("invalid product lifecycle event")
            p = staged_products[payload["product_id"]]
            if staged_remaining[p.id] == 0:
                raise ValueError("product is already resolved")
            if event["type"] == "product_cancelled":
                discarded[p.id] += staged_remaining[p.id]
                staged_remaining[p.id] = 0.
                cancelled[p.id] = mission.clock
                processing_remaining[p.id] = 0.
            else:
                updated = OnboardProduct.model_validate({**p.model_dump(), "deadline_sec": payload["deadline_sec"]})
                if not max(mission.clock, updated.ready_sec) < updated.deadline_sec <= mission.horizon_sec:
                    raise ValueError("updated deadline must follow now/readiness and fit horizon")
                staged_products[p.id] = updated
                boundaries.add(updated.deadline_sec)
        elif event["type"] in {"station_rate_changed", "transmitter_rate_changed"}:
            key = "station" if event["type"] == "station_rate_changed" else "actor"
            if set(payload) != {key, "rate_bps"}:
                raise ValueError("rate event requires target and rate_bps")
            targets = station_rates if key == "station" else transmitter_rates
            rate = payload["rate_bps"]
            if (payload[key] not in targets or isinstance(rate, bool) or not isinstance(rate, (int, float))
                    or not math.isfinite(rate) or rate < 0):
                raise ValueError("rate must be finite, nonnegative and target known")
            if key == "actor" and rate > mission.vehicles[payload[key]].rate_bps:
                raise ValueError("transmitter rate cannot exceed nominal hardware rate")
            targets[payload[key]] = float(rate)
        elif event["type"] == "terminal_booking_added":
            booking = TerminalBooking.model_validate(payload)
            if (booking.id in bookings or booking.actor not in mission.vehicles or booking.station not in mission.stations
                    or booking.start_sec < mission.clock or not booking.start_sec < booking.end_sec <= mission.horizon_sec):
                raise ValueError("invalid terminal booking")
            if len(bookings) >= 10000:
                raise ValueError("booking limit reached")
            bookings[booking.id] = booking
            validate_bookings(bookings, booking_cancelled, mission.stations)
            boundaries.update((booking.start_sec, booking.end_sec))
        elif event["type"] == "terminal_booking_cancelled":
            if set(payload) != {"booking_id"} or payload.get("booking_id") not in bookings:
                raise ValueError("booking cancellation requires known booking_id")
            bid = payload["booking_id"]
            if bid in booking_cancelled or bookings[bid].end_sec <= mission.clock:
                raise ValueError("booking already cancelled or ended")
            booking_cancelled[bid] = mission.clock
            boundaries.add(mission.clock)
        elif event["type"] == "station_reservation_policy_changed":
            if (set(payload) != {"station", "required"} or payload.get("station") not in policies
                    or not isinstance(payload["required"], bool)):
                raise ValueError("station policy requires known station and boolean required")
            policies[payload["station"]] = payload["required"]
        elif event["type"] == "station_outage_added":
            outage = StationOutage.model_validate(payload)
            if (outage.station not in mission.stations or outage.start_sec < mission.clock
                    or outage.end_sec <= outage.start_sec or outage.end_sec > mission.horizon_sec):
                raise ValueError("outage must be known station and now/future interval within horizon")
            staged_outages.append(outage)
            boundaries.update((outage.start_sec, outage.end_sec))
        elif event["type"] == "station_recovered":
            if set(payload) != {"station"} or payload.get("station") not in mission.stations:
                raise ValueError("recovery requires a known station")
            station = payload["station"]
            if not any(o.station == station and o.start_sec <= mission.clock < o.end_sec for o in staged_outages):
                raise ValueError("station has no active outage to recover")
            staged_outages = [
                StationOutage(station=o.station, start_sec=o.start_sec, end_sec=mission.clock)
                if o.station == station and o.start_sec < mission.clock < o.end_sec else o
                for o in staged_outages if not (o.station == station and o.start_sec == mission.clock)]
            boundaries.add(mission.clock)
        else:
            raise ValueError("unsupported reference event type")
        accepted.append(copy.deepcopy(event))
    # All validation and allocation precede the commit; no geometry is rebuilt.
    staged = copy.copy(mission)
    staged.products, staged.remaining = staged_products, staged_remaining
    staged.outages, staged._boundaries = staged_outages, sorted(boundaries)
    staged.events = mission.events+accepted
    staged.cancelled, staged.discarded = cancelled, discarded
    staged.station_rates, staged.transmitter_rates = station_rates, transmitter_rates
    staged.bookings, staged.booking_cancelled, staged.reservation_required = bookings, booking_cancelled, policies
    staged.processing_remaining, staged.processing_completed = processing_remaining, processing_completed
    staged._dependency_order = dependency_order(staged_products)
    staged._dependency_children = dependency_children(staged_products)
    for name in staged_products.keys() - mission.products.keys():
        for parent in staged_products[name].depends_on:
            if parent in cancelled or staged_remaining[parent] != staged_products[parent].size_bits:
                raise ValueError("new dependency requires intact retained input bytes")
    result = staged.report()
    mission.products, mission.remaining = staged.products, staged.remaining
    mission.outages, mission._boundaries, mission.events = staged.outages, staged._boundaries, staged.events
    mission.cancelled, mission.discarded = staged.cancelled, staged.discarded
    mission.station_rates, mission.transmitter_rates = staged.station_rates, staged.transmitter_rates
    mission.bookings, mission.booking_cancelled, mission.reservation_required = staged.bookings, staged.booking_cancelled, staged.reservation_required
    mission.processing_remaining, mission.processing_completed = staged.processing_remaining, staged.processing_completed
    mission._dependency_order = staged._dependency_order
    mission._dependency_children = staged._dependency_children
    return result
