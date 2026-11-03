# B10-C2 — bounded disk artifacts and chunk transport

This engineering package adds a process-local transfer service for native archives,
checkpoints and verified RL/SFT ZIP. It retains existing replay, provenance and
interactive scenario limits. It does not implement an object-storage service,
durable transfer queue or unbounded-memory native JSON parser.

## Authorization and routes

An existing session's **admin** capability is required. Transfers bind the source
session ID and exact capability hash. Operator/agent/viewer capabilities cannot
create one. Different admin capabilities cannot use another transfer ID. Source
closure or credential rotation rejects subsequent transfer operations. Authorization
is checked at an operation boundary; it does not revoke bytes already admitted
into an in-flight response. Transfer IDs and bearer tokens are not URL query keys.

POST `/api/operator/sessions/<id>/artifacts`, JSON:

```json
{"direction":"download","format":"archive"}
```

Supported download formats: `archive`, `checkpoint`, `training`. Upload:

```json
{"direction":"upload","format":"archive","bytes":12345,"sha256":"<64 lowercase hex characters>"}
```

Upload accepts only archive/checkpoint JSON. A scripted/random checkpoint requires
explicit boolean `takeover:true`; native archive imports retain their own recorded
provenance. There is no ZIP decompression/import route and no caller-supplied path
or filename. Unknown fields, coerced byte counts and non-boolean takeover are rejected.

The `space-env.operator-transfer/1` response includes `transfer_id`, state, expected
and received byte counts, SHA-256, busy status and maximum chunk size. Subsequent
routes use the same admin bearer:

- GET `/api/operator/transfers/<transfer_id>/status` returns public transfer metadata,
  never newly issued connection secrets.
- PUT `.../chunks` appends `application/octet-stream` bytes. Exactly one decimal
  `Content-Length` and `X-Upload-Offset` are required. No chunked transfer encoding.
- POST `.../commit`, `{}`, verifies the uploaded digest and runs the existing native
  restore/takeover path. Successful imports receive fresh session IDs/capabilities.
- GET `.../content` streams a ready download with Content-Length, fixed safe
  Content-Disposition, `X-Artifact-SHA256` and no-store. No Range/resume GET contract
  is provided: retry the same immutable artifact from the beginning.
- POST `.../cancel`, `{}`, deletes an unleased transfer and releases its reservation.

Loopback Host, same-origin and existing bearer controls apply to these routes.
Existing JSON requests retain the 8 MiB body limit. Chunks are 1–1,048,576 bytes;
HTTP reads use at most 64 KiB and a 15-second whole-chunk deadline with `read1`,
rather than resetting an unlimited drip-feed timeout. Rejected requests close the
connection because their body may be unread. Downloads write at most 64 KiB per
iteration with a socket timeout; total download wall time is not qualified.

## Upload and import semantics

A chunk must append at the acknowledged received offset or repeat a byte range
already fully accepted. The latter is compared byte-for-byte and does not advance
the offset. Gaps, overlapping unacknowledged bytes and different retry bytes fail
409. Short input, timeout and write failures truncate the unacknowledged suffix
before retry. If rollback cannot repair the file, the transfer is discarded.
Final-byte digest mismatch returns 422 and deletes the failed upload/reservation.
No native operation is executed by transferring bytes.

Upload socket reads hold only the transfer lease, not the global mission lock;
other native mission decisions can proceed. Each transfer has at most one active
operation. Busy cancellation returns 409: it does not pretend to interrupt native
replay/restore midway through a commit.

Commit states are `uploading` → `ready` → `committing` → `consumed` or `failed`.
A successful result is cached until cancellation/expiry, including its private
connection response. Repeating commit returns that same result rather than
creating another session. A failed restore attempt is terminal for this transfer:
it is never blindly retried, because persistent import may have committed before
an IO/readback acknowledgment failed. Status includes a resulting session ID if
one was registered in this process. On uncertain outcomes inspect local sessions
and use OS-owner recovery; do not assume absence of a response means absence of
a mission. This is not durable, cross-process exactly-once import submission.

Successful consumption removes input bytes. If that cleanup fails, the cached
result remains usable and its disk reservation remains charged until later cleanup.
Invalid native JSON/replay never receives a success response. Missing source or
incompatible dependency/source fingerprints remain errors; transfer does not
migrate checkpoint builds or weaken native validation.

## Downloads and verified ZIP

JSON is encoded directly into a bounded disk file with original Python number
representations. The content path does not reserialize it in JavaScript. Training
uses `OperatorSession.export_to(...)`: VerifiedEpisode checks native replay, writes
the bounded bundle, adds operator-provenance.json, then compresses files through
64 KiB reads into a bounded ZIP file. No whole ZIP or whole member is assembled in
BytesIO. Existing bytes-returning `export()` remains compatible at its 16 MiB limit.
New disk/native payload and ZIP bounds are 32 MiB; native rows remain external-origin
when appropriate, not silently labelled expert demonstrations.

The operator UI uses the new transfer routes for download and authorized native
imports, validates SHA-256, uploads `File.slice()` blocks and cancels its temporary
transfer after use. Browser Blob hashing and JSON classification can read a bounded
whole file; native checkpoint/replay and final JSON decoding also materialize their
objects. **Transport streaming is not a constant-RSS guarantee for the simulator,
JSON parser or browser.** Old small unauthenticated local-owner import is retained
for compatibility when no admin connection is attached; large import requires an
existing admin session. This remains a local console, not enterprise account isolation.

## Resource and lifecycle bounds

- Maximum eight transfer records per process, 32 MiB per artifact, 128 MiB reserved
  spool budget. A training export reserves 64 MiB to cover bundle plus ZIP peak;
  ready downloads release excess reservation to actual file size.
- Upload reserves its declared size before accepting bytes. Completed receipt-only
  records count toward job capacity until cancelled/expired, even after disk bytes
  are released. Quota exhaustion returns 429.
- Spool root and job directories are private (0700); payloads are 0600. Paths are
  generated locally, outside package source/assets. API does not return disk paths.
- Unleased idle records expire after 600 seconds by default. Cleanup is cooperative
  on subsequent operations or explicit SDK `cleanup()`, not a background scheduler.
  Status polling does not extend a record's idle lifetime.
  Leased files are never deleted under a reader/writer. Cleanup failure keeps
  accounting and record identity for repair/retry.
- Normal server shutdown closes the spool after request threads finish. Transfer
  state is not restarted from disk. An abrupt kill can leave a private temporary
  directory; OS temporary-file policy/manual OS-owner cleanup is still required.
- SDK callers should use their own `ArtifactStore` and close it after leased
  FileBody responses are closed. Unclosed SDK readers retain leases by design.

These are conservative engineering caps, not measured claims of large-fleet or
long-mission throughput. Native interactive limits remain unchanged. Indexed durable
artifact metadata, resumable process restart, deployment quotas, hard cancellation
of computation, retention scheduling and external storage remain later work.

## Acceptance

`tests/test_operator_artifacts.py` covers format/role scope, exact retry, EOF/timeout
rollback, SHA mismatch, busy/expiry, quota, cleanup failures and unacknowledged import.
`examples/operator/artifact_smoke.py` checks the real HTTP path, negative headers,
training ZIP and a >8 MiB wire fixture (native JSON plus whitespace padding). That
fixture demonstrates transport behavior, not realistic long-mission size or RSS.
Browser acceptance covers native restart/roles plus transfer download/import.
[Receipt](acceptance/B10_ARTIFACTS.json) records actual executed checks and limits.
