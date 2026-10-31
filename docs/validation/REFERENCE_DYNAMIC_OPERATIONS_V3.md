# Dynamic reference mission operations

Update: [Service lifecycle v4](REFERENCE_SERVICE_LIFECYCLE_V4.md) adds cancellation,
discard accounting, deadline changes and throughput limits. Statements below
about the absence of deletion/cancellation describe the earlier v3 boundary.

Implemented 2026-10-31. This package connects causal mission changes, audit/replay,
replanning, and a fixed-capacity RL episode. It extends the separate
`reference-fixed-downlink/1` resource model. It does not migrate MissionWorld or
qualify a flight simulator, acquisition model or optimal industrial planner.

## Operator events

`mission.apply_events(events)` and `recorder.inject(events)` accept an ordered
atomic batch of 1–1000 events, with a limit of 10000 applied events per mission.
Every event requires exactly these fields:

```json
{
  "id": "kiruna-fault-1",
  "type": "station_outage_added",
  "time_sec": 12,
  "source": "ground-network-controller",
  "reason": "antenna maintenance",
  "payload": {"station": "Ground", "start_sec": 12, "end_sec": 20}
}
```

`id`, `source` and `reason` must be nonempty; IDs cannot be reused. Event time
must equal the current mission clock exactly. No backdating, implicit clock
advance, or future submission as if already observed is permitted. Operations
must be inside the mission horizon. Events in one batch are applied in order
on staged dictionaries; validation/report failures leave the mission unchanged.
The source/reason are user-supplied provenance, not authenticated signatures.

Supported types:

- `onboard_product_added`: payload is an OnboardProduct definition. Unique ID,
  known owner/destination, now <= ready < deadline <= horizon and currently free
  memory are required. This models externally supplied, already acquired bytes
  arriving in inventory, **not** an imaging command, sensor result or validated
  physical acquisition. Memory becomes occupied immediately, even if ready is
  later. Delivered products are retained in history and IDs are never recycled.
- `product_priority_changed`: payload product_id and nonnegative integer priority.
  Product must be known and not fully delivered. Size, delivery history and
  deadline are not rewritten.
- `station_outage_added`: payload station/start_sec/end_sec. Interval must start
  now or later and end within the horizon. Overlapping outages form a union for
  availability. Contact/outage boundaries split physical transmission steps.
- `station_recovered`: payload station. Ends all active outages at now, preserving
  past downtime and future scheduled outages. Requires at least one active outage.
  A just-started interval may be removed entirely; recovery does not rewrite the
  immutable base spec or the earlier event provenance.

Reports include the applied event history and an inventory ledger per actor:
initial, injected, delivered and remaining bits, plus balance error. This ledger
makes exogenous data injection explicit, alongside the existing Wh energy ledger.
There is no deletion/cancellation or spontaneous generation of inventory.

## CLI and replay

```bash
# events.json is {"events": [event, ...]} at prefix's current clock.
space-env reference-events prefix.json events.json updated-prefix.json
space-env reference-replan updated-prefix.json remaining.json --report schedule.json
space-env reference-resume updated-prefix.json remaining.json completed.json
space-env reference-replay completed.json

# One command: verify prefix, inject, replan, execute, save result and schedule.
space-env reference-adapt prefix.json events.json completed.json --report schedule.json
```

Both commands preserve the source file. All input/output paths must differ.
Invalid events, runtime mismatch or planning budget exhaustion occur before
output writes. Each output is atomic; schedule/run publication is not a single
filesystem transaction. `--decision-sec` and `--max-steps` bound adaptive planning.

Run schema /2 interleaves normal transition rows with kind=events rows, each
containing the pre-event observation and post-event result. Replay checks the
checksum, runtime, every observation/result, all event validation and final
state. `from_record` reconstructs the full state by replay before allowing
continuation. Runs without event rows retain schema /1. As before, source/runtime
changes require the matching runtime or an explicit future migration; checksum
integrity is not cryptographic proof of an authentic operator or expert.

Events change the state fingerprint. Earlier state-bound plans are rejected
by CLI execution even if the clock has not changed. The adaptive SDK entry point
is `space_env.reference.adaptation.adapt_reference_run(path, events, ...)`, which
returns a new recorder and verified schedule without writing files.

## RL with new products during an episode

`ReferenceEventDownlinkEnv(spec, event_tape, product_slots=16, dt_sec=30)` has fixed
actors/stations and a fixed product slot capacity **per actor**. Observation
shapes and the MultiDiscrete action space remain unchanged when products arrive.
No slot reuse: capacity must cover the full episode, including delivered products.
Maximum configurable slot capacity is 1000; these are bounds, not performance
qualification. Future tape contents/IDs are absent from observations and info;
only applied events and currently known inventory are exposed. The environment
still provides full current resource state, not a realistic telemetry estimator.

Each actor action: 0 wait; then one block of automatic selection plus N slots
for each station. For one station and N=3: 1 automatic, 2 slot1, 3 slot2, 4 slot3.
Empty slots or destination-incompatible selections wait and appear in
`invalid_actors`; they never choose another product silently. Slot IDs/order are
returned in info. Occupied slots stay active after delivery with remaining=0.

A step returns early at the next event timestamp, applies the whole event group,
and lets the agent react. Time-zero events apply on reset. `elapsed_sec` records
the actual step duration: this is a variable-duration control process, so agents
using continuous-time discounting must account for it. Nominal Gym gamma alone
does not model a fixed discount per second. There is no early termination while
future events remain. Reward is newly completed on-time Mbit; truncation occurs
at the horizon if products remain incomplete. If a contextual event constraint
fails (e.g. memory), the entire step, including preceding transmission, rolls back.

The tape envelope, ordering, IDs and slot capacity are checked upfront; contextual
constraints are checked at application time. No future event is applied early to
validate it against an incorrect inventory. This environment does not write the
operator journal automatically; use the recorder/adaptive API for portable
mission replay, and the existing rollout writer for RL transition JSONL.

## Runnable example and open gates

```bash
python examples/reference_operations/dynamic_demo.py /tmp/space-env-dynamic-demo
```

Two actual GCRS ephemerides, a shared station, initial outage, new onboard product
and another station fault produce a 6 Mbit on-time outcome. The demo verifies
portable replay, zero inventory-balance residual and a successful two-step RL
episode. This is a regression fixture, not an industrial workload benchmark.

Still open: calibrated acquisition/processing pipelines, task cancellation and
retention policy, changing constellation membership, external station bookings,
RF/antenna/slew constraints, look-ahead energy, uncertainty-aware planning,
independent optimum/regret benchmarks and measured large-fleet endurance.
All known fixed-ephemeris and simplified resource model limitations remain.
