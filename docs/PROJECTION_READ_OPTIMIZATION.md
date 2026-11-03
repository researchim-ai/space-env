# B11-Q2 — scoped observation reads

This package removes repeated telemetry materialization. It does not modify
physical integration, RF/thermal/geometry cadence, payload quality, tolerances,
resource arbitration or the declared B11-Q1 workload gates.

## Implementation and ownership

`ReferenceDownlinkMission.observation(actor=None)` preserves the aggregate API.
An explicitly selected actor materializes only that actor. `ReferenceBackend`
uses this path for local reads, avoiding full-formation construction followed by
discarding the other actors.

`TaskSession._refresh` creates one private `_ProjectionReads` per synchronous
refresh. It memoizes owned public actor observations, inventories, station
telemetry, current command rows and static capabilities. Central and local
projections select their own actor scopes from these reads, maintain independent
entity registries and produce their own numeric graphs, catalogues and bindings.
Only the refresh-local object holds these cached values; it is not retained on
TaskSession, stored in checkpoints or exposed to agents.

The projector and mask builder treat these intermediate values as read-only.
Public `TaskSession.observe`, runtime observations and inventory still return
owned data. No native mutable histories are shared between forks to obtain this
optimization. Backend capture/restore and audit evidence contracts are unchanged.

Crucially, `TaskSession.step` still uses the real candidate runtime for joint
admission after cancellation. A projection snapshot is never used as a native
admission oracle. Every new refresh reads the new runtime state, even when time
has not advanced. Projection failure still discards the trial transaction.

## Verification

`tests/test_projection_reads.py` checks exact cached/uncached projection equality,
separate role registries, one underlying read per key per refresh, no retained
cross-refresh cache, single-role read scope, unchanged native reports after
reads, nested public mutation isolation and independent fork state. A native
single-actor read is instrumented to ensure it does not build other actors' RF
observations. Existing task tests exercise cancel-and-resubmit, rejected joint
cancellation and projection-capacity rollback.

`examples/campaign/projection_parity.py` offers a reproducible same-build A/B
control. Its diagnostic-only temporary patches restore the previous full-actor
materialization and uncached projection path, without editing package files or
changing physics. Four small missions compare every task-transition hash, full
controller trace and complete native backend report against optimized execution.

```bash
python examples/campaign/projection_parity.py /tmp/projection-parity.json
```

This is two actors and one request over 20 physical seconds per case, not a large
formation qualification. Its wall times are single samples under shared-host
load and are not a stable speedup claim. Profiles are separate instrumented
runs; cumulative profiler rows overlap and must not be summed.

Source edits change trajectory provenance and derived science IDs. Therefore an
unmodified older-build backend report is not expected to have the same digest.
Same-build diagnostic controls and within-build native recovery/replay comparisons
avoid erasing provenance fields to manufacture equality.

## Qualification and limits

The new large run uses the **identical** frozen B11-Q1 manifest: 12 actors,
40 requests, 120 physical seconds, two straight repeats and one prefix-recovered
repeat, three workers, timeout 2400 seconds per worker, execution gate 1200
seconds. Other memory, latency, checkpoint, quality and replay gates are unchanged.
All failed rows remain in the denominator. The previous three timeouts remain
in the historical B11-Q1 receipt.

Final measured outcomes and environment caveats belong in
[the acceptance receipt](acceptance/B11_PROJECTION_READS.json). Do not infer large
readiness from fewer read calls or a small passing mission. Repeated full native
evidence copies, transaction capture, trial forks and long-horizon history growth
remain separate optimization targets in [the core plan](COUPLED_PERFORMANCE_NEXT.md).
