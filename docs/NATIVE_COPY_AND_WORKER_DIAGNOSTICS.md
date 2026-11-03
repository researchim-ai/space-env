# B11-Q3 — owned native copies and worker diagnostics

This package changes data materialization and measurement observability. Native
forces, integration tolerances, geometry/RF/thermal sampling, resource admission,
payload quality, physical-time reward and declared B11-Q1 gates remain intact.

## Native data ownership

`simulation.owned.owned_copy` specializes exact built-in scalar/dict/list trees.
A shared memo preserves repeated references and cycles. Tuples, NumPy arrays,
container subclasses, custom reduction and deepcopy hooks use Python's standard
copy machinery. It does not serialize through JSON, truncate history or share
mutable science records with a parent or caller.

Dynamic native observations now have a private borrowed construction path used
only inside locked `MissionRuntime.observe`, immediately followed by the final
owned copy. Public/default backend and lifecycle artifact observations retain
owned science data. No borrowed references are kept across reads or revisions,
returned through task tools, stored in checkpoints or fed to a policy.

Controlled native capture still clones a complete mutable state graph. It builds
an owned lifecycle dictionary in the shared deepcopy memo, retaining the alias
between lifecycle transport and mission DTN. The rest of mission/bus state uses
standard deepcopy. Unknown lifecycle subclasses retain the original capture path.
There is no append-only-history sharing or skipped nested transaction snapshot.

## Scalar trainer accounting

A truth-guarded internal accounting path obtains exactly the completion, priority,
acknowledged delivered bits, energy, grouped discarded bits and bus-safe values
used by task reward. Product outcome calculation is shared with the native report;
actor/product summation order and grouping remain unchanged. Full science reports
are not constructed merely to compute these scalars. Other backends retain the
previous full-report fallback. This adds no agent observation or forecast input.

## Audit and transaction publication

Compact audit reuses the exact canonical JSON bytes already validated inside the
native transaction to compute its evidence SHA-256. Fields are not removed from
the hashed evidence. Full/compact event formats and replay rules remain the same.
Owned event and response preparation now completes before publication. A failure
while preparing the event copy restores backend state and command revision;
the idempotency key remains reusable. This is covered by forced-copy-failure tests.

## Bounded worker stage receipts

Every scale-benchmark repeat has an assigned numeric directory, independent of
user workload names: `workers/000000/progress.json`, etc. The CLI passes that path
to the fresh interpreter; the frozen workload inputs/gates are unchanged.

Each file is replaced atomically and contains the workload digest, PID, monotonic
elapsed/stage times, decision count, physical time, process CPU and observed peak
RSS. It reports initialization, prefix execution/checkpoint, resume, execution,
fork, checkpoint, evidence, replay and export. Decision updates are throttled to
one per second; phase changes and terminal worker status are written immediately.
Blocking fork/checkpoint/replay/export calls do not have a background heartbeat;
resource/time fields are the last sample, not a continuously live gauge. Evidence
phase bookkeeping also includes releasing the live run and garbage collection.
There is at most one current file per repeat, at most 32 completed stage entries
and at most 32 KiB per file. The existing 128-repeat bound therefore limits retained
diagnostic file payload to 4 MiB, apart from bounded atomic replacement overlap.

`completed` means that the worker wrote a result; it does **not** mean that the
qualification gates passed. A killed/timed-out worker can leave a last `running`
snapshot. An exception records its current phase and a bounded error description.
The supervisor retains these files and a validated last snapshot even when the
temporary input/result directory disappears. Malformed, oversized, torn or wrongly
bound progress files are ignored as diagnostics, never accepted as a native result.

Per-worker files can be read while the manifest-ordered supervisor is waiting for
an earlier repeat. Aggregate `progress.json` still lists completely collected rows.
Neither format is an automatic resume queue, a checkpoint, native replay evidence
or a policy input. A completed progress file without a valid worker result fails.
Diagnostics are included in measured worker overhead; peak RSS remains a process
high-water observation, not an enforced OS quota or episode allocation delta.

## Verification and qualification

`tests/test_native_copy.py` exercises ownership, cycles, shared references, mutable
tuple children, NumPy, container subclasses, memo/hooks, capture aliases, scalar
reward parity and rollback on a forced event-copy failure. Existing cancellation,
joint-admission, projection-overflow and corrupt-replay negatives remain active.

`tests/test_worker_progress.py` checks atomic publication, throttling, workload
binding, invalid/truncated/oversized input, disk failure, retained timeout phase,
failed-repeat denominators, stable parallel result order, missing native results
and unchanged native/controller semantics with diagnostics enabled.

```bash
python examples/campaign/native_copy_parity.py /tmp/native-copy-parity.json
```

The diagnostic temporarily restores standard copies, default owned backend reads,
full scalar accounting and original capture in the same source build. Five small
cases compare every task transition hash, controller trace and complete native
report. Single shared-host timing samples are not a stable speedup claim. Build
changes naturally alter science provenance IDs; old receipts are retained intact.

[Acceptance](acceptance/B11_NATIVE_COPY.json) records final executed tests, installed
wheel checks, the frozen four-actor fault matrix and the unchanged 12-actor,
40-request, 120-second large formation profile. The large profile still has three
fresh workers, 2400-second timeouts and the original 1200-second execution, latency,
RSS/checkpoint, quality, replay, completion and semantic-parity gates.

This remains synthetic nearby-formation geometry and an engineered baseline
policy. Passing bounded cases does not qualify arbitrary multi-plane, multi-day,
flight-calibrated missions, expert teachers or a multi-tenant hosted service.

The final large run completed in all three processes. All technical, native
replay/accounting and export gates passed; overall qualification failed solely on
the mission objective (28/40). All 40 products were acquired, processed and handed
to relays. The conservative controller's 4-second ground windows explain its
28-window bound under this profile. Raw semantics matched across all repeats,
while the formal comparison group correctly stayed failed. See the
[next operational package](GROUND_DELIVERY_NEXT.md); the old threshold is unchanged.
