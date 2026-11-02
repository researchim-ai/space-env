# B09-M1: owned evidence and replay-once export

```python
from space_env.tasks import VerifiedEpisode

verified = VerifiedEpisode(task_checkpoint)  # complete native cold replay first
verified.export_bundle('/tmp/new-training-bundle')  # RL + centralized SFT
verified.export('/tmp/local-rl.jsonl', role='control-0')
```

`VerifiedEpisode.from_planning_checkpoint(record)` independently cold-replays the
planning policy, requests, RNG, transitions and final native state, then checks
existing conservation/ACK/public-view/mask gates. Its `verification` property is
an owned copy of the replay/quality receipt. No public constructor accepts an
unchecked live session or a caller-supplied `verified=True` flag. This proves
execution consistency, not expert-teacher quality or mission/flight qualification.

Every public row is copied before yielding. Mutating input records, previously
yielded rows, observations, returned verification receipts or idempotent responses
cannot change internal evidence or future rows. Private implementation attributes
are not a mutation API. Concurrent consumers can export from the same immutable
verified evidence to separate destinations; they must not mutate private state.

`export_bundle` supports RL/SFT for the central role or RL-only for selected local
roles. Format/role combinations are validated before writing. File paths use
numeric role slots, not caller-supplied path fragments. A new sibling staging
directory holds streamed JSONL and a digest-bound receipt with roles, formats,
row counts, canonical array hashes and file hashes. Total byte budget includes
the root receipt. Publication is a directory rename; late IO, producer or budget
errors remove staging and leave no partial target. Existing targets are rejected.
This is atomic visibility on the local filesystem, not a distributed transaction
or a guarantee of power-loss durability. There is no automatic resume/overwrite.
The receipt binds exports to the source episode; independent native verification
still requires its original compatible-build checkpoint.

`export_dataset` and `iter_dataset` remain compatible single-export APIs, each
verifying the episode on entry. Consumers needing several outputs should retain
one `VerifiedEpisode`. Factory quality/collection now uses a single verified
planning episode per case/policy for all role/format exports; scale workers use
one independent planning/native replay followed by RL/SFT bundle export.

Committed task observations and transitions are internally append-only. Current
views, historical adjacent views, central/local references and the idempotency
cache share owned evidence. Public accessors return copies. Canonical JSON values,
digests, transition/reward semantics, native sampling and cold-replay requirements
are preserved. Compact/full checkpoint schemas remain `/2` and `/1`; matching
source/dependency fingerprints are still required. Checkpoint serialization can
expand shared Python references back into repeated JSON values.

The scale worker releases its original run before allocating the independently
replayed run, and collects unreachable cycles at that boundary. Its measured
peak RSS includes interpreter, execution, checkpoint, replay and export costs;
it is not a delta allocation or steady-state training-only memory figure.

See [acceptance](acceptance/B09_M_MEMORY_EXPORT.json) for measured 64/1000 evidence,
build stages, exact regression scope and retained limitations. Mutable native
history, per-episode transitions, full checkpoint serialization, linear cold
replay, crash recovery and long heterogeneous coupled missions remain open.
