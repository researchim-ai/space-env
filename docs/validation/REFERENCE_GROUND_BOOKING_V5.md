# Ground resource booking: twelve-task package

Implemented 2026-10-31 for reference-fixed-downlink/1. This is an in-process
mission resource model, not a live ground-network booking integration, a flight
qualified system, or a large-scale performance qualification.

## Completed tasks

1. **Reservation model.** TerminalBooking records id, actor, station, start_sec,
   end_sec. One booking reserves one terminal. IDs remain allocated forever.
   Known actor/station and now <= start < end <= horizon are required. All
   intervals are half-open. No geometry or throughput guarantee is inferred.
2. **Station capacity conflicts.** A sweep validates reservation overlap against
   each station's terminal count. End events precede starts at equal timestamps,
   so adjacent bookings are valid. Conflicting additions reject the entire event
   batch, including earlier otherwise-valid events in that batch.
3. **Single-radio conflicts.** An actor cannot hold overlapping bookings, even
   across different stations. This is the current one-transmitter profile, not
   a multi-radio spacecraft model. Nonoverlapping handovers are supported.
4. **Reserved versus open allocation.** Active reservations withhold capacity
   even when their owner is idle, unready, has no data or cannot transmit. Open
   actors compete only for remaining terminals. Execution and planning maintain
   separate open-slot counts so idle reservations cannot be silently stolen.
   Alphabetical arbitration remains for competing open actors in direct steps.
5. **Reservation-only station policy.** station_reservation_policy_changed has
   payload station and boolean required. Default false permits open access to
   free terminals; true permits only reservation owners. This is current service
   policy, independent of geometric visibility and outages.
6. **Active cancellation.** terminal_booking_cancelled payload booking_id ends
   the effective reservation at now, preserving the original interval and
   historical usage. Capacity can be reassigned at exactly that timestamp in
   the same ordered atomic batch. Data and energy do not change on cancellation.
7. **Future cancellation and identity retention.** A future booking can be
   cancelled before it starts, releasing its entire capacity. Ended or already
   cancelled bookings cannot be cancelled again. Booking IDs cannot be reused;
   event IDs and booking IDs are separate namespaces.
8. **Boundary-correct execution.** Start/end/cancellation times participate in
   the resource timeline. One external step can cross multiple reservations;
   ownership changes at the exact known boundary. A booking never overrides an
   outage, geometry, destination, zero throughput or battery constraint.
9. **Planner integration.** The greedy planner filters unauthorized pairs,
   reserves open capacity separately and reevaluates at booking boundaries.
   Copy-based replanning preserves bookings and history. A granted reservation
   is not proof of task feasibility or planner optimality.
10. **RL/replay/CLI integration.** Events enter the existing /2 journal, state
    fingerprint, restoration and reference-events/reference-adapt commands.
    Both Gym interfaces expose station_permitted; the event environment also
    exposes reserved, reservation_required and open_terminals with fixed shapes.
    Masks describe service permission, not geometry, resource feasibility or a
    guarantee when several open actors request the same terminal. Future event
    tape contents remain hidden; already submitted future bookings are known.
11. **Service-window diagnostics.** mission.service_windows(actor, station)
    intersects future geometric contacts, known outages, reservations and current
    policy from now to the horizon. Windows carry mode=reserved/open and merge
    adjacent equal-mode intervals. Current rate limits are held constant; zero
    throughput returns no windows. No battery, traffic, competing open actions,
    or unobserved future event prediction is included. The method is read-only.
12. **Reservation utilization.** Report booking_utilization measures elapsed
    reserved seconds, actual receipt-overlap transmission seconds, idle seconds
    and utilization. Future time is not charged as elapsed. A cancelled future
    booking has zero elapsed and utilization=null. Idle includes outages or lack
    of geometry, not just operator inefficiency. These are simulated duration
    metrics, not invoices, spectrum utilization or physical antenna duty cycles.

## Events and scenario

```json
{
  "events": [
    {"id": "reserve-1", "type": "terminal_booking_added", "time_sec": 1,
     "source": "ground-operator", "reason": "confirmed mission pass",
     "payload": {"id": "pass-b", "actor": "b", "station": "Ground", "start_sec": 1, "end_sec": 7}}
  ]
}
```

As for all reference events, time_sec must equal the current mission clock.
At most 10000 bookings are retained. Conflicts are checked after each addition;
release a conflicting old booking before adding its replacement. Capacity
validation includes effective historical intervals. There is no implicit
preemption or optimizer-generated reservation admission.

```bash
python examples/reference_operations/booking_demo.py /tmp/space-env-bookings
space-env reference-adapt /tmp/space-env-bookings/prefix.json \
  /tmp/space-env-bookings/events.json /tmp/space-env-bookings/cli-run.json \
  --report /tmp/space-env-bookings/cli-schedule.json
space-env reference-replay /tmp/space-env-bookings/cli-run.json
```

The two-satellite example uses a reservation-only station, sequential owners and
an existing outage. All 5 Mbit arrive on time. The first booking uses 3 of 6
elapsed reserved seconds; the second uses 1 of 1. Portable replay and a two-step
RL rollout are checked. This is a deterministic regression scenario, not an
industrial fleet workload benchmark.

## Remaining gates

The event-based admission API and masks use the separate reference backend;
legacy station booking APIs are not silently treated as equivalent. Not included:
external network synchronization/authentication, contractual priority/preemption,
prices/SLA settlement, aggregate RF sharing, antenna slew/acquisition and setup
costs, multi-radio spacecraft, uncertainty-aware scheduling, independent optimum
comparisons, or load qualification. Reservation scans and utilization reports
currently scale with retained history; the 10000 bound is not an endurance claim.
Runtime/source matching for replay remains strict; no old journal migration is
implied. The event environment's observation dictionary gained fields, so saved
policies must declare their observation contract or be retrained/adapted.
