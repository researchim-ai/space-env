# B09: scale, compact replay and streaming contract

## Guarantees

- Runtime `audit_mode="compact"` retains request, revision, idempotency key,
  response and SHA-256 of the full operation evidence. Final native report,
  original/effective scenario and source/dependency/seed manifest stay in the
  checkpoint. Runtime schema `/2` cold-reexecutes each operation and verifies its
  full evidence digest. Default runtime audit remains `full`; factory defaults
  to compact. Full `/1` checkpoints remain supported within matching builds.
- Task/planning `checkpoint(compact=True)` uses schema `/2` and transition
  digests. Cold restore recomputes physical transitions and policy decisions,
  including masks, ACK rewards, feedback and RNG. It is not unchecked native
  state deserialization. Source/dependency incompatibility still fails closed.
- Runtime forks share committed, internally append-only audit/cache entries;
  public readers return deep copies. Cache dictionaries and audit lists themselves
  are distinct. Mutable physical state, commands, histories and RNG remain
  isolated through backend capture/restore; rollback follows the same path.
  Mutating private journal objects is outside the supported API.
- Frame cache keys contain all six exact ERFA arguments, without time rounding
  or altered physics/safety grids. A bounded 4096-entry LRU stores immutable
  matrix tuples; every caller receives a fresh NumPy array. An uncached path
  enables exact comparison. Cache metrics never enter physical manifests.
- `iter_dataset` fully verifies an episode first, then yields its owned public
  rows. Atomic JSONL writers pull one row at a time with byte/row/file limits;
  late producer failures preserve the previous destination. Array digests are
  compatible with canonical JSON arrays, without building the entire array.
- Factory row buffers and duplicate index use local SQLite with a 2 MiB page
  cache and declared disk/row budgets. Per-episode savepoints roll back exported
  rows and dedup entries together. Resource-budget/IO failure aborts unpublished
  builds. Semantic verification streams expected/file hashes. Total bundle-byte
  budget now includes root receipt bytes as well as members.
- Benchmark workers are spawned, one episode per process, at most 8 outstanding
  jobs. Manifest order is stable; independent seeds stay native; worker errors
  remain in the evaluation denominator. No silent retry. Verification recomputes
  semantics serially and ignores measurements/process IDs, not mission outcomes.
- Local corpus registry verifies ready bundles through native replay before a
  transaction. Re-registering identical content is idempotent. New corpus labels
  do not bypass actual trajectory duplicate/cross-split checks. Registry is
  explicit, local and separate from immutable bundles; foreign databases are
  rejected without replacing their contents.

## Measured acceptance

See [B09 receipt](acceptance/B09_SCALING.json) for exact checks and observed
workloads, and [published probes](../examples/scaling/README.md). The
`scale-benchmark` command freezes a versioned typed manifest before execution;
errors/timeouts remain in the receipt and cause exit 1. Measurements include
hardware/thread settings, complete-step p50/p95, checkpoint/fork/replay/export
cost and peak RSS in a fresh process per workload. Outcome/trace hashes allow
full/compact comparisons without treating timing as replay facts.

## Limits and remaining qualification

This removes corpus-wide RL/SFT row accumulation and repeated copying of audit
payloads. It does **not** provide constant-memory physics: a single native
mission, task transitions, original scenario, current mutable backend history,
checkpoint JSON and bounded case/evaluation metadata still occupy memory. Full
before/after backend reports are currently constructed transiently to hash
compact operation evidence. Cold replay deliberately remains linear in history;
compact checkpoints are not fast binary snapshots or random-access recovery.

Probes with cloned fixed ephemerides do not establish distinct-orbit fleet
qualification. Short coupled EO/CPU/DTN/fault probes do not prove 72-hour fleet
performance, all subsystem combinations or mission reliability. Broader
multi-plane geometry, long coupled missions, scheduling/forecast services,
learned-policy benchmarks, calibrated hardware/flight data and B12 independent
industrial pilot evidence remain acceptance gates. No assertion of superiority
to independently qualified industrial simulators follows from these tests.

Parallel `workers>1` execution requires Python 3.11+ (fresh-process recycling);
Python 3.10 supports `workers=1` and receives an explicit validation error for
parallel requests. Actual Python 3.10 worker execution is not claimed.

## B09-Q1: distinct-orbit workloads and mask cost

`space_env.scaling.workloads.FleetWorkloadSpec` bounds actors (1–64), products
(1–1023), planes, resource/horizon/decision budgets and repeated executions before
propagating distinct J2/DOP853 GCRS arcs. It generates individual downlink action
choices, grouped controller roles and ground stations. Products are already
onboard; these resource workloads do not add imaging/processing, calibrated RF,
coupled bus dynamics or collision avoidance. Native profile limits are retained.

A read-only stack sample on 64/1000 identified repeated native outcome reports in
`admissible`, solely to query horizon. `MissionRuntime.horizon_sec` and
`command_status()` now expose those existing values without recomputing outcome
metrics. Command lists remain owned copies under the same lock. Projection,
planning input and admission paths use these readers; a test forbids full report
construction there. Physical masks, command rechecks after cancellation, time
sampling and final outcome accounting remain unchanged. Paired 16/100 native
reports/traces match before and after this optimization.

Workers have explicit numerical-library thread variables and preserve parent
isolation/user-site flags. Affinity, CPU model and cgroup CPU/memory limits are
reported where available. Latency/execution limits are frozen in the manifest.
Aggregate success requires all declared repeats and identical native/trace hashes
among successful repeats; failed/infeasible rows remain in the denominator.
Two repetitions verify repeatability on this host, not statistical reliability,
independent hardware qualification or a fleet SLA. See
[B09-Q1 receipt](acceptance/B09_Q_FLEET.json) for exact stages and retained failed
profiles. The interrupted pre-optimization probe is recorded, not relabelled as
successful. Frozen manifests are preserved as gzip-compressed JSON alongside
receipts; decompress before passing them to `scale-benchmark`.

Remaining gates include long *coupled* missions, sustained heterogeneous arrivals,
256 actors/10000 tasks (currently outside native/task limits), durable per-phase
recovery, mutable-history memory, a dependency-clean CI matrix and independent
industrial calibration. The measured profiles do not close these gates.
