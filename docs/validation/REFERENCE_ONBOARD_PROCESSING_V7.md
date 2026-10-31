# Reference onboard processing v7

Implemented 2026-10-31. This extends the separate fixed-ephemeris operational
profile with an explicit in-place processing queue. It does not simulate image
formation, scientific data quality, compression or actual onboard software.

## Configuration and meaning

VehicleResources gains processing_power_w (default 0) and min_processing_soc
(default 0.1). OnboardProduct gains processing_sec and workspace_bits (both default
0). Existing products therefore remain immediately process-ready. Positive work
must be paired with meaningful user-supplied power/duration/memory parameters;
zero power is supported for synthetic fixtures, not inferred hardware efficiency.

```json
{
  "id": "raw-scene", "actor": "a", "size_bits": 2000000,
  "ready_sec": 1, "deadline_sec": 20, "priority": 3,
  "processing_sec": 3, "workspace_bits": 1000000
}
```

The product's bytes already occupy onboard memory. ready_sec releases the input
for processing; it is not a predicted processing completion time. The same
product ID and size become eligible for downlink after processing completes.
No extra data are injected by processing and no input/output compression ratio
is implied. Exogenous arrivals still require onboard_product_added provenance.

## Execution contract

There is one processor per actor. Among ready, unresolved, memory/energy-feasible
jobs it selects descending priority, deadline, then ID. Work is preemptible at
resource/event/action boundaries without setup or checkpoint overhead. Priority
changes or new higher-priority jobs can interrupt work; accumulated progress is
retained. No explicit per-job CPU selection command is provided yet.

Persistent inventory plus the running job's workspace must fit vehicle memory.
Scratch memory is temporary and released when not computing, including on pause
or preemption; preserving partially completed work requires no additional memory
in this surrogate. There is no persistent scratch-data content or swap cost.
Workspace alone cannot exceed memory. A currently blocked job can become feasible
when another processed product is downlinked and frees inventory space.

For a given actor, CPU and transmitter are mutually exclusive. Automatic CPU
work takes precedence; other actors may compute/transmit concurrently. If no CPU
job is feasible, ready data may transmit. This deliberately serial hardware
profile avoids assuming independent compute/radio buses; it is not a general
multi-core or concurrent CPU/RF resource allocation model.

Processing consumes constant configured additional power while running, on top
of the existing base load. Work stops at min_processing_soc, with exact resource
splitting at the floor and modeled recharge thresholds. No computational work is
credited while power-inhibited. Solar generation and existing electrical ledger
remain shared; CPU power is not double charged. CPU heating is not coupled to
the optional one-node transmitter thermal model, which cools while TX is idle.

Steps split at input readiness, processing completion and resource boundaries.
Raw data cannot transmit early. Completion is tracked by remaining processing
seconds, with the same four-endpoint-ulp numerical snapping used by transfer
completion. Extreme sub-nanosecond work sizes are not qualified. Late processing
may complete, but subsequent late delivery earns no on-time reward.

Cancellation discards undelivered inventory and remaining work, preserving CPU
seconds, energy already consumed, history and any valid earlier completion.
Failed steps roll back progress, completion times, CPU logs, energy and data.
Event batches stage processing state together with the rest of the mission.

## Commands, replay and planning

`mission.step(dt, actions, products=..., processing={actor: bool})` accepts a
per-step CPU enable map. Missing actors default to enabled. The recorder and
reference-run/reference-resume accept the same optional processing mapping:

```json
{"steps": [
  {"dt_sec": 1, "actions": {"a": "Ground"}, "processing": {"a": false}},
  {"dt_sec": 5, "actions": {"a": "Ground"}, "processing": {"a": true}}
]}
```

Pausing CPU can let already processed data transmit; it cannot make raw data
ready. Controls are recorded and replayed alongside resource outcomes. New
processing fields are additive to the run /1 steps; event-containing runs still
use /2. Runtime matching remains strict, without journal migration guarantees.

Reports add per-product processing_remaining_sec, processing_completed_sec and
downlink_ready (requires undelivered bytes as well as completed work/readiness).
The processing section records per-actor CPU seconds and attributable Wh, plus
intervals with job, start/end, workspace, inventory and total memory. CPU seconds
include work later cancelled. These are deterministic simulation costs, not
measured processor performance, FLOPs or scientific product correctness.

The planner waits for CPU completion, omits unprocessed products from downlink
candidates and avoids assigning a computing actor to a station. It then resumes
resource-constrained downlink planning using actual outcomes. CPU admission at
planning time uses conservative zero-generation eligibility; modeled solar
execution can differ, so execution verification remains binding. CPU jobs are
automatic unless a caller supplies pause controls; no global joint compute-radio
optimization or independent optimum proof is claimed.

## RL contract

Both Gym facades expose processing_remaining_sec, workspace_bits and per-actor
processing_seconds. The event environment pads job fields into stable actor/slot
arrays and supports dynamically arriving raw products.

With cpu_control=False (default), previous station/product action sizes are
retained and CPU runs automatically. With cpu_control=True, each actor gets two
blocks: the original block enables CPU; the second block performs the same radio
action with CPU paused. In ReferenceDownlinkEnv, reset info.action_choices maps
all choices; missing processing in the first block means true. In the event
environment info.cpu_pause_offset gives the block width. Choice 0 waits on radio
but computes; choice equal to that offset waits with CPU paused. Invalid/empty
product slots still wait on radio without silently changing the CPU enable bit.

New observation fields require adapting/retraining saved policies. Rewards remain
newly completed on-time product Mbit. No reward is earned merely for CPU activity,
partial work, workspace reservation, readiness or cancelled products.

## Runnable scenario and qualification

```bash
python examples/reference_operations/processing_demo.py /tmp/space-env-processing
space-env reference-replan /tmp/space-env-processing/prefix.json \
  /tmp/space-env-processing/remaining.json --report /tmp/space-env-processing/replanned.json
space-env reference-resume /tmp/space-env-processing/prefix.json \
  /tmp/space-env-processing/remaining.json /tmp/space-env-processing/cli-run.json
space-env reference-replay /tmp/space-env-processing/cli-run.json
```

The example combines two GCRS actors, a shared station/outage, an initial CPU
pause and a newly arrived high-priority processing job. It verifies 6 Mbit on-time
delivery, CPU work of 3 seconds per actor, replay and a successful ten-step RL
rollout. Configured CPU energy is 0.01 Wh and 0.0125 Wh. Inventory accounting stays
unchanged by in-place computation. This is a synthetic regression scenario.

Tests cover readiness, memory blocking/release, priority preemption, pause,
energy-floor stopping, cancellation, partition agreement, rollback, dynamic
arrival, CLI recording/replay and both Gym checkers. Still open: calibrated
acquisition/compression algorithms, output-size changes, job DAGs, persistent
scratch/checkpoint costs, CPU thermal and contention, multi-core/concurrent RF,
uncertainty/optimality benchmarks and large-fleet endurance. General MissionWorld
is not migrated by this package; full repository pytest is a separate gate.
