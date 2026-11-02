# B10-B: durable local Mission Control and controller ownership

The default `python -m space_env.api.server` stores sessions in
`.space-env/operator.sqlite3`. Set `SPACE_ENV_OPERATOR_DB` to an absolute path
outside the installed package to select another local database. SDK callers use
`OperatorStore(path)`; `OperatorStore()` deliberately remains ephemeral.

## Commit and recovery

Successful mutations are acknowledged only after a SQLite WAL transaction commits
both the current snapshot and its chained commit receipt. Generation compare and
swap rejects competing writers. Native revision and control epoch fence commands;
a handover changes ownership without advancing simulation time. Lost commit
acknowledgments are reconciled against the persisted snapshot. Uncertain IO fails
closed; restart and replay are required before using that handle again.

Startup verifies snapshot/journal hashes, native checkpoint replay, historical
navigation and controller history. Incompatible source/dependency fingerprints or
corruption fail startup rather than silently dropping sessions. Closed sessions
remain tombstoned. Direct writes to the internal TaskSession are unsupported and
rejected by the adapter. This is local SQLite durability, not a distributed
scheduler, network-filesystem deployment or signed tamper-proof ledger.

## Capabilities and reconnection

Save a connection file with the explicit button in `/operator`. It contains bearer
secrets: protect it like a password. The database contains capability hashes, not
plaintext tokens. Reloading the page does not close the mission or create another
mission when active sessions already exist. Import the connection file to attach.
No tokens are saved in browser localStorage.

If credentials are lost, the OS owner can list and recover sessions locally:

```bash
python -m space_env.api.operator_recovery .space-env/operator.sqlite3 --list
python -m space_env.api.operator_recovery .space-env/operator.sqlite3 SESSION_ID /tmp/operator-connection.json
```

Recovery rotates all capabilities and increments the control epoch. Output is
created exclusively with mode 0600; existing files are never overwritten. The
secret file and database are not one atomic transaction: if file writing fails
after rotation, repeat recovery to another output path. There is no unauthenticated
HTTP recovery route. Capability roles do not prove a person's identity.

## Handover and portable archives

POST `.../handover` requires the admin capability and `controller` (`operator` or
`agent`), `expected_revision`, `expected_control_epoch`, `idempotency_key`, `reason`.
Steps carry `control_epoch`; omission is compatible only at initial epoch zero.
Only the current controller's channel can step. Local roles retain actor scope.
Native idempotency keys are separated by controller epoch and role.

GET `.../archive` returns a secret-free portable archive with native checkpoint,
observed navigation, journal and controller history. POST `/api/operator/restore`
with `{"archive": ...}` validates it and issues a new ID and fresh capabilities.
Raw external task checkpoints remain supported. Scripted/random checkpoints need
explicit `{"record": ..., "takeover": true}`: original native execution is
verified before rebuilding the external-controlled continuation. Original policy
origin and checkpoint hash are retained; imported commands are not relabelled as
human demonstrations or expert actions.

Training ZIPs include `operator-provenance.json` alongside the existing verified
RL/SFT files. Controller history is a sidecar; native policy origin stays external.
Neither capability names nor replay establish expert quality.

## Bounds and remaining gates

Limits remain eight live sessions, 16 actors, 256 decisions/actions, 257 navigation
frames and 8 MiB HTTP input. Stored snapshots are capped at 32 MiB; control history
at 1024 events; SQLite page budget is approximately 4 GiB. Full-history snapshots
and cold replay are intentionally bounded and have not been throughput-qualified.
Streaming artifacts/event cursors, long missions, retention/backup tooling, worker
quotas, SSO/TLS/account isolation, deployment hardening and independently calibrated
flight models remain separate gates. Existing legacy HTTP routes are not upgraded
to enterprise authentication by this package.

## Reproducible acceptance

`python examples/operator/durable_smoke.py` executes the native relay mission,
restarts the store, transfers ownership twice and verifies replay/training ZIP
and archive cloning. With optional Playwright and Chromium installed, run
`python examples/operator/browser_smoke.py` for real HTTP/browser restart,
reconnection, owner permissions and portable archive import on desktop/mobile.
The script leaves synthetic screenshots, receipt and private test connection
files in a new temporary directory.

Archive/checkpoint downloads preserve the original HTTP bytes; imports embed
the original JSON text in the request. Parsing and serializing those artifacts
through JavaScript would change number representations and invalidate hashes.
