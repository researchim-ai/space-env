# B10-C1 — scoped event cursors and coherent local recovery

Engineering implementation for the bounded B10 operator profile, building on
[B10-B](OPERATOR_DURABILITY_CONTRACT.md). This package adds incremental history
reads and online SQLite backups. It does not raise actor/decision/history limits,
implement streamed artifacts or introduce another spacecraft physics model.

## Cursor API

POST `/api/operator/sessions/<id>/events` with the usual bearer capability and JSON:

```json
{"channel":"journal","limit":64,"cursor":null}
```

- `channel`: `journal` (default) or `navigation`.
- `limit`: strict integer 1–128, default 64. Unknown fields are rejected.
- `cursor`: omitted/null for a fresh read; otherwise the prior returned cursor.
- Response schema `space-env.operator-events/1`: `entries`, `cursor`, `has_more`,
  `visible_count`, native task `revision` and `control_epoch`.
- Each entry has a zero-based **role-projected** `sequence`, `kind`, and `event`
  (`decision`/`control`) or `frame` (`navigation`). It is not a global native
  sequence that reveals how many hidden decisions another role executed.
- Use the returned cursor even for an empty page: after the next mutation, polling
  it yields the new visible suffix. `has_more=false` describes this response,
  not the end of a still-live mission. This is polling, not SSE or a long poll.

Decision visibility and actor filtering match existing state projections. All
roles currently see the same control events/reasons, as in B10-B state. Do not
encode secrets in public role names, action IDs or control reasons. Navigation
contains past observed frames only, not future ephemerides or private truth.

Journal order is physical decision revision, then its subsequent control events
in control-epoch order. A handover at revision n is after decision n and before
decision n+1. Import/recovery control events follow the same rule. Returned entries
are independent copies. History reads do not advance time, change revision,
append storage commits, sample physical RNG or populate a session cursor cache.

Cursors are opaque versioned payloads signed with HMAC-SHA256 using the presented
bearer. They bind session ID, channel, projected prefix length and prefix hash.
They contain no bearer or database capability hash. Cross-capability/channel/
session reuse or malformed signatures are invalid requests (400). Revoked bearers
remain unauthorized (401). Missing/changed consumed prefixes fail with explicit
`cursor_conflict` (409), rather than skipping or replaying rows silently.
A capability holder can construct cursors for their own authorized projection;
HMAC is not a substitute for projection/authorization. There is no untrusted-code
sandbox or independent authenticated human identity in this mechanism.

Compatible-build SQLite restart preserves cursor continuity because there is no
process-local cursor key. Credential rotation revokes previous cursors. Portable
archive import creates a new session/capabilities, so use a fresh cursor there.
The signature is bounded to 2048 characters. Prefix hashing is O(retained history),
not an asymptotically constant-time database event index. Existing 256-decision,
257-frame and 1024-control-event bounds remain; latency is not qualified for
unbounded histories. Legacy full state responses and the browser UI are unchanged.

## Local backup and restore

SDK:

```python
from space_env.api.operator_maintenance import (
    backup_operator_store, verify_operator_store,
)

receipt = backup_operator_store(
    ".space-env/operator.sqlite3", "/private/backups/mission-001.sqlite3",
    timeout_sec=30.0,
)
verification = verify_operator_store("/private/backups/mission-001.sqlite3")
```

The output parent must already exist. Output must be a new file outside package
source/assets. Existing files, directories and dangling symlinks are rejected.
Do not copy only the live SQLite main file with `cp`: accepted commits may still
reside in WAL. Use SQLite's online backup API, which makes a coherent snapshot
while other source connections can continue committing. Snapshot time is determined
by SQLite copying; it is not a caller-selected application revision.

Copying uses private same-filesystem staging and a finite page/copy deadline. The
copy is validated before publication: SQLite integrity, all stored snapshot/journal
heads and cold native replay of every **active** session, including its history.
Closed tombstones receive journal/snapshot-integrity checks, not another full
closed-episode physics replay. File SHA-256 and byte count are computed afterward.
A reader never receives an acknowledged partial file. Exclusive hard-link
publication prevents a racing writer from overwriting an existing destination;
file and parent directory are fsynced where supported. The published file is 0600.
This requires a local filesystem that supports hard links. Failure is reported
rather than falling back to an unsafe overwrite.

`timeout_sec` (0.01–600 finite seconds) bounds SQLite copying through its progress
callback, **not total backup wall time**. Subsequent native verification uses
native numerical/budget guards and can take longer. A pre-publication failure
removes private staging and leaves no output. A failure after publication, such
as directory-fsync failure, can leave a complete but unacknowledged file; inspect
that file with `verify`, do not assume the missing acknowledgment means absence.

The receipt is returned by SDK or printed to stdout by CLI. It is not a second
atomically published receipt sidecar. Integrity hashes detect accidental mismatch;
they are not a signed anti-tamper ledger, an encryption layer or an externally
trusted calibration certificate.

CLI (local OS owner, no HTTP maintenance routes):

```bash
python -m space_env.api.operator_maintenance backup .space-env/operator.sqlite3 /private/backups/mission-001.sqlite3
python -m space_env.api.operator_maintenance verify /private/backups/mission-001.sqlite3
python -m space_env.api.operator_maintenance restore /private/backups/mission-001.sqlite3 /private/recovery/operator.sqlite3
```

`restore` makes a new verified copy; it never replaces the running database.
Stop/select the service and explicitly configure `SPACE_ENV_OPERATOR_DB` to the
restored file. IDs, capability hashes, control epochs and compatible cursors are
retained. Old plaintext connection files still authorize their backed-up sessions;
a restore is **not** credential rotation. Protect backups and use B10-B local
recovery when rotation is required. Tokens themselves are not printed or restored
from hashes. No automatic server restart, retention deletion or backup scheduler
is performed.

`verify` also first makes a private online snapshot. It cannot combine a native
replay from an old live head with metadata from a concurrently committed new head.
Temporary copies are removed after verification. Source/dependency guards remain
strict: backups do not migrate checkpoints to incompatible builds, even when the
package version string is unchanged.

## Acceptance and remaining gates

`examples/operator/maintenance_smoke.py` exercises native step → cursor → online
backup → independent restore → continuation. Reference CI includes
`tests/test_operator_maintenance.py`; installed CI includes the smoke outside the
checkout. [Receipt](acceptance/B10_MAINTENANCE.json) records the actual local runs.

Remaining B10-C/B11 gates: upload/export streaming, durable indexed event storage
and retention policy, backup scheduling/restore drills at scale, measured long-run
latency and disk growth, worker quotas/cancellation/metrics and multi-user SSO/TLS.
No flight/hardware qualification or distributed/network-filesystem guarantee is
established by these tests. The website is not regenerated by this package.
