"""Blocking one-hop relay: data becomes available only after world time advances.

A conservative sampled link profile, not a continuous RF/energy qualification.
Legacy Fleet.relay retains its instantaneous accounting semantics.
"""
from __future__ import annotations

import math

import numpy as np

from space_env.core.link import earth_occults, isl_rate_bps
from space_env.simulation.world import MissionWorld


def _idle_participant(world: MissionWorld, actor: str, start: float, end: float) -> None:
    engine = world.engines[actor]
    busy = world.commands.running_actors()
    busy.update(world.commands.get(ident)["actor_id"] for ident in world._batch)
    if actor in busy or engine._downlink_armed or engine._imaging_requested or engine._burn is not None:
        raise ValueError("relay participant has an active operation")
    events = [item.scheduled_at_sec for item in engine.scenario.planned_events]
    events.extend(item.at_sec for item in engine.scenario.anomaly_injects)
    events.extend(item.at_sec for item in engine.scenario.conjunction_injects)
    events.extend(item.at_sec for item in engine._restores if not item.done)
    if any(start < clock <= end for clock in events):
        raise ValueError("split relay at participant event boundaries first")
    if any(lo < end and start < hi for lo, hi in [*world.relay_used.get(actor, []), *engine._terminal_used]):
        raise ValueError("relay terminal already reserved")


def relay_step(world: MissionWorld, source: str, destination: str, dt_sec: float) -> dict:
    """Advance the whole world, then commit a bounded subset of ready inventory.

    No newly created/processed data can borrow the preceding interval. Failed or
    closed links still consume the declared mission time. Invalid inputs do not.
    Participants must be idle; unrelated actors may continue running commands.
    """
    if source not in world.engines or destination not in world.engines:
        raise KeyError("unknown relay participant")
    if source == destination:
        raise ValueError("relay participants must differ")
    if not math.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("relay duration must be positive and finite")
    start = world.now_sec()
    end = start + dt_sec
    if end > world.horizon_sec + 1e-9:
        raise ValueError("relay exceeds mission horizon")
    for actor in (source, destination):
        _idle_participant(world, actor, start, end)
    sender, receiver = world.engines[source], world.engines[destination]
    eligible = {}
    ttl = sender.spacecraft.config.comms.ttl_sec
    receiver_ttl = receiver.spacecraft.config.comms.ttl_sec
    for product in sender.capture()["products"]:
        ready = float(product.get("available_time_sec", product.get("generated_time_sec", 0)))
        born = float(product.get("generated_time_sec", 0))
        if product.get("custodian") != sender.spacecraft.config.name or product.get("delivered_to"):
            continue
        if ready > start + 1e-9 or any(limit > 0 and born + limit <= end + 1e-9 for limit in (ttl, receiver_ttl)):
            continue
        key = (str(product.get("product_id", "")), float(product.get("offset_bits", 0)))
        eligible[key] = eligible.get(key, 0.) + float(product.get("onboard_bits", 0))
    left, right = sender.preview_positions(dt_sec), receiver.preview_positions(dt_sec)
    if len(left) != len(right) or any(abs(a[0] - b[0]) > 1e-9 for a, b in zip(left, right)):
        raise ValueError("relay trajectory samples must share a time grid")
    rate = min((0. if earth_occults(a[1], b[1]) else isl_rate_bps(float(np.linalg.norm(a[1] - b[1]))))
               for a, b in zip(left, right))
    ready_radio = sender.radio_ready() and receiver.radio_ready()
    checkpoint = world.snapshot(include_history=False)
    try:
        world.step(dt_sec, {source: "wait", destination: "wait"})
        world.settle()
        # A maneuver/inject-free profile is required above; also screen the actual
        # endpoint, since preview is not a certificate of continuous visibility.
        if earth_occults(sender.orbit.r, receiver.orbit.r):
            rate = 0.
        rate = min(rate, isl_rate_bps(float(np.linalg.norm(sender.orbit.r - receiver.orbit.r))))
        if not ready_radio or not sender.radio_ready() or not receiver.radio_ready():
            rate = 0.
        offered = min(rate * dt_sec, sum(eligible.values()), receiver._room_bits())
        records = sender.release_bits(offered, destination=destination, eligible=eligible) if offered > 0 else []
        ack = float(sender.spacecraft.config.comms.ack_sec)
        for record in records:
            for key in ("ack_due_sec", "ack_from", "acked"):
                record.pop(key, None)
            record["available_time_sec"] = max(float(record.get("available_time_sec", 0)), end)
            if ack > 0:
                record.update(ack_due_sec=end + ack, ack_from=source, acked=False)
        received = receiver.accept_custody(records)
        for actor in (source, destination):
            world.relay_used.setdefault(actor, []).append((start, end))
        world.note_truth()
        return {"source": source, "destination": destination, "start_sec": start,
                "end_sec": end, "bits": received, "status": "transferred" if received > 0 else "no_transfer",
                "profile": "serial-sampled-relay/1"}
    except Exception:
        world.restore(checkpoint)
        raise
