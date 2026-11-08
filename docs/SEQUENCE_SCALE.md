# Bounded actual sequence planner qualification

Run from the checkout with the installed environment (no extra dependencies):

```sh
python examples/network/sequence_scale.py --output /tmp/sequence-small --select tiny pair reference backlog boundary --workers 2 --timeout 120
python examples/network/sequence_scale.py --audit /tmp/sequence-small
python -m pytest tests/test_sequence_scale.py -q
```

The immutable workload ladder is `tiny` (1 actor, 1 request, 60 s), `pair`
(2, 3, 150 s), `reference` (3, 3, 180 s), `backlog` (4, 8, 600 s), and
`boundary` (4, 16, 1200 s). All use 5 s decisions, one ground terminal,
50,000 search nodes, 12 order trials and a directed adjacent-actor relay chain.
For 600/1200 s rows, release times are `round_to_5s((horizon-180)*i/(requests-1))`;
deadlines are release plus the original per-request deadline, capped at the horizon.
Earlier rows release at zero and scale original deadlines by `min(1,horizon/180)`.
Long rows offset target longitude by `0.058 * release_sec` degrees, wrapped to
[-180,180), as a declared along-track target grid. This rule is independent of
planner outcomes; it does not certify contact or capture feasibility.
They derive from the first public mission-suite relay case, with explicit
actor/request/horizon edits before any execution. Repeated target locations
are separate identified requests. This spans the currently supported 1–4 actor,
1–16 request and 60–1200 s network limits, but is not exhaustive, Cartesian,
random, blind, calibrated, or evidence for thousands of spacecraft. Larger
rows are opt-in and need not finish or fulfill all requests. Default selection
is only tiny/reference. A fresh output directory is required; no resume/retry.

The real `sequence_lanes` planner compiles the network action catalogue, and
the privileged scripted reference policy executes it on the native runtime.
No external LLM is measured. Existing scaling environment/RSS helpers and
native episode replay are reused; the generic PlanningRun scaling worker does
not expose this network compilation path, so this module owns a separate worker.

Each selected cell launches a fresh Python interpreter, with numerical thread
environment variables set to one. `--workers` accepts integers 1–4 (default 1).
A bounded parent ThreadPoolExecutor supervises up to that many isolated worker
processes concurrently; no native mission state is shared between threads.
Results always follow the frozen selection order, regardless of completion order.
The deadline starts at launch, excluding queue time. Parent orchestration wall
time and per-cell start/finish offsets are recorded; audit checks recorded
concurrency against the frozen worker limit. The plan/report schema is version 2;
version 1 receipts must be rerun rather than silently upgraded. The parent enforces a per-cell wall
deadline (including imports, compile, execution, replay and artifact output),
maximum 300 s, and polls Linux `/proc` RSS every roughly 20 ms against a 1 GB
default **per-worker** budget (`run(max_rss_bytes=...)`). The frozen plan records
`aggregate_potential_rss_bytes = workers * max_rss_bytes`: with workers=2 and
1 GB per worker this is 2 GB, even if fewer cells are selected. This is a potential
worker RSS budget, not measured simultaneous RSS, not a cgroup limit, and excludes
parent memory. Shared pages may be counted in multiple processes. Queued cells
consume no worker budget until launched; active workers are at most
`min(workers, selected_count)`. It kills the process group on timeout/RSS excess. RSS polling
can miss brief spikes and is not an OS memory reservation or strict allocator
cap; recorded process high-water RSS supplies a second qualification gate.
The suite is bounded to five process launches and at most four concurrent workers. Errors/timeouts remain in the selected
denominator. A draining thread preserves the first 64 KiB of stderr, discards
excess output and binds the saved UTF-8 diagnostic to the report. Worker runtime
exceptions also produce a sealed error type/message artifact; no zero metrics
are fabricated. A failed import may only leave stderr and the exit code.
A parent crash can leave an incomplete report; there is no crash recovery.

`compile.wall_sec` measures MissionWorkspace.network setup, forecast, geometry,
action compilation and initial native session, minus the separately timed real
scheduler call. `planning.wall_sec` covers that scheduler only. Native execution
includes reference-policy selection, Python protocol/record assembly and native
steps. Validation and cold replay have separate timings. RSS at each phase end
is **process cumulative high-water RSS**, not incremental memory or an independent
phase peak. Import memory is included; import time is outside phase times and
inside parent worker wall time. Single observations are not throughput, latency
percentiles or statistical confidence estimates. Timing receipts are measurements,
not cryptographically attested clocks, and cannot be regenerated by replay.

On Linux the shared `planning.runner.process_peak_rss_bytes` helper reads
`/proc/self/status` **VmHWM**, the current executable's resident high-water mark.
It deliberately does not use `getrusage(RUSAGE_SELF).ru_maxrss`: Linux can carry
that watermark across fork/exec from a much larger parent, falsely charging
training-process memory to an isolated simulation worker. Missing/invalid Linux
proc evidence is unknown rather than a fabricated zero or parent measurement.
Other supported resource platforms retain their native ru_maxrss convention.
`tests/test_peak_rss.py` includes a real allocated-parent/exec-child regression.
Historical measurements using the previous helper must be rerun on this source
revision; existing source binding prevents treating them as current qualification.

Before the first worker, the sealed plan freezes all configurations, selection,
limits and hashes of package Python sources. Documentation/test-only edits do not
invalidate package evidence; workload configuration changes do. Workers check hashes before/after and the
parent records `sourcehashfinal` after all cells. Audit requires the same current
sources and regenerated workload ladder, rejects duplicate/missing/unsafe
selections and replays every completed native episode. It binds metrics, planner,
configuration, schedule admission, search counters, PID and phase schema to the
sealed receipts. Source edits, including adding package Python modules, invalidate
prior evidence: finish source changes first, then rerun in a new output directory.
Transient edits restored between checks and malicious resealing of plausible
timing fields are outside the provenance model. Installed dependency versions
are not hashed; worker hardware/thread/cgroup context is recorded.

`valid` means evidence validation succeeded, including failure rows; `qualified`
requires every selected cell to replay, stay within measured RSS budget, deliver
exactly its planned count and have no failed calls. It does not require all
requested products to be planned. Full goal achievement is separately visible in
native metrics. `full_matrix` means the entire frozen ladder was selected in
canonical order; partial evidence never qualifies unselected cells. No global
sequence optimality, physical coverage completeness or mission-wide scalability
is established. Auditing reexecutes native episodes in the auditing process;
worker runtime bounds apply to the measurement run, not to audit.


The targeted tests execute tiny/reference serially and with workers=2, verify
concurrent intervals and distinct fresh PIDs, and compare full native episode
signatures and metrics. Parallel timeouts and launch errors retain every selected
cell in order. This establishes deterministic parity on those two regression
workloads; it does not measure parallel speedup or qualify the larger cells.
The five-cell workers=2, 120 s/cell command above is the final ladder invocation
to run only after all package sources are frozen. In the worst case worker
execution occupies up to `ceil(selected_count/workers) * timeout` plus bounded
launch/poll/log overhead; runtime contention can change timings and timeout rates.
