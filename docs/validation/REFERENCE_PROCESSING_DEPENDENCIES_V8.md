# Processing prerequisites and input retention v8

Implemented 2026-10-31 in the separate reference operations backend. This adds
same-actor prerequisites to the in-place processing resource model. It does not
create scientific output bytes, execute algorithms, or implement a cross-satellite
dataflow/compression pipeline. Every product still owns its independently loaded
inventory; dependencies express compute ordering and input retention only.

## Model

OnboardProduct gains depends_on: a tuple/list of at most 128 product IDs, empty by
default. Dependent products require positive processing_sec. Input references must
exist and belong to the same actor. Unknown, duplicate, self and cyclic references
are rejected. Validation uses a stable heap-based topological sweep, not recursive
DFS; a 2000-node chain is tested independently of mission physics. The 10000-product
limit remains an admission bound, not a performance qualification.

```json
{"id": "analysis-input", "actor": "a", "size_bits": 1000000,
 "ready_sec": 1, "deadline_sec": 15, "processing_sec": 2,
 "workspace_bits": 500000, "priority": 9, "depends_on": ["raw-scene"]}
```

This example's 1 Mbit already exists in memory; no conversion of raw-scene into
analysis-input is implied. All parents must finish their own processing and reach
input readiness before a child can run, regardless of child priority. Zero-work
roots are considered processed once ready. Positive-work children complete only
through actual CPU execution. Join and shared-input dependency patterns are tested.

## Retention and cancellation

A parent's undelivered inventory is retained while any unresolved direct child
still has processing work. Both automatic and explicitly selected downlink honor
this retention. Pausing CPU does not bypass retention or make raw bytes ready.
After the last child completes or is cancelled, the parent's pin is released.
Retention can increase storage pressure; no automatic eviction or deadlock
resolution is implied when workspace cannot fit.

Cancelling a required input before its waiting/partial child finishes blocks that
child. Completed work and energy costs remain, but no fabricated progress or
automatic recovery occurs. Operators may explicitly cancel the blocked child;
there is no implicit cascade or dependency rewrite. A child that finished before
its input was cancelled remains a valid processed product: completed computation
is not retroactively undone, and its own downstream children can still run.

New dynamic dependencies require intact, uncancelled input inventory at admission.
References to already partially or fully downlinked inputs are rejected; there is
no hidden retrieval from Earth or another actor. An atomic addition batch can
contain forward references, because its final graph is validated before commit.
Graph indexes, product state, events and reports are committed together; a cycle
or invalid new input leaves all staged changes uncommitted. For replacements,
new nodes/IDs are required; dependency editing on existing products is unsupported.

## APIs, planning and RL

Existing initial specs and onboard_product_added events accept depends_on.
Existing reference-events/reference-adapt and portable journal /1 or /2 replay
support the model without a separate command language. Runtime matching remains
strict. Same-actor constraints are intentional; physical relay of prerequisite
bytes is a future model, not an assumption implicit in an ID reference.

CPU scheduling respects prerequisite completion before priority. Planner downlink
candidates respect retained inputs, while execution remains the final feasibility
check. All former workspace, power, quota, thermal and station booking limits
still apply. No independent optimal DAG scheduler or feasibility proof is claimed.

Product reports expose depends_on, dependency_blockers and retained_for. The
retained_for list describes unresolved consumers, not a reference count on a real
filesystem. Gym facades expose dependency_edges and retained with stable shapes:
rows are children and columns are parents. ReferenceDownlinkEnv uses global
product order from reset info. The event environment uses per-actor product slots;
future tape nodes/edges remain absent until those products arrive. New observations
require saved policy contract adaptation. Padded adjacency uses quadratic slot
space; large-graph RL throughput/memory has not been qualified.

## Reproduce

```bash
python examples/reference_operations/dependency_demo.py /tmp/space-env-dependencies
space-env reference-replan /tmp/space-env-dependencies/prefix.json \
  /tmp/space-env-dependencies/remaining.json --report /tmp/space-env-dependencies/report.json
space-env reference-resume /tmp/space-env-dependencies/prefix.json \
  /tmp/space-env-dependencies/remaining.json /tmp/space-env-dependencies/cli-run.json
space-env reference-replay /tmp/space-env-dependencies/cli-run.json
```

The synthetic two-GCRS-actor scenario includes a shared station/outage and a new
dependent input arriving at t=0.5. Its parent finishes processing at t=1; child
computation starts at t=1. All 6 Mbit are delivered on time. Portable replay and
a successful nine-step RL rollout are verified. This is a resource-control
regression scenario, not scientific validation of a payload algorithm.

Open gates: data materialization and changing output sizes, calibrated algorithms,
persistent scratch/checkpoint overhead, job/input lifetimes and recovery policy,
physical cross-actor dependencies, independent optimum/regret and measured
large-fleet workloads. Full repository pytest remains a separate gate.
