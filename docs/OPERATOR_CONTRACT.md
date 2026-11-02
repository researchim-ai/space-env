# B10-A: native Mission Control

Historical B10-A scope below. Current persistence, ownership and takeover semantics
are defined by [B10-B](OPERATOR_DURABILITY_CONTRACT.md), which supersedes the
in-memory and no-agent-handover limitations described here.

Run with the reference dependencies installed:

```bash
python -m space_env.api.server
```

Open **http://127.0.0.1:8766/operator**. The previous Studio remains at `/studio`
with a link to the native operator. Both are packaged in the wheel. There is no
browser-side physics executor in Mission Control: it sends declared B06 task
action IDs through `TaskTools`/`TaskSession` and renders returned state.

The fleet preset has four distinct J2 arcs, twelve preloaded products and a
600-second resource horizon. The relay preset is the published synthetic B07/B08
three-actor EO→CPU→ISL custody→ground ACK mission, with coupled resources and a
20-second horizon. It is frozen in `configs/operator_relay.json` and labelled
synthetic, not calibrated flight data. The bounded route-aware suggestion uses
only the same causal B07 public planning input; it is a heuristic, not an optimal
plan. Suggestions alone never advance the mission. The browser's automatic run
sends ordinary revision-bound steps and can stop between accepted steps.

## Operations and visibility

- POST `/api/operator/sessions`: `{"preset":"fleet-demo"}` or
  `{"preset":"relay-demo"}`, or explicit `{"scenario":...,"task":...}`.
  The session issues random bearer capabilities: `operator`, read-only `viewer`,
  artifact-only `admin`, and `role:<task role>` local controllers. Creation and
  restore are unauthenticated **local owner operations**, not a multi-user login.
- GET `/api/operator/sessions/<id>`: scoped observation, current/past navigation,
  journal and static horizon/decision/capacity information. Use
  `Authorization: Bearer <token>`. Query-string tokens are not accepted.
- POST `.../step`: `action_ids`, strict integer `expected_revision`, nonempty
  `idempotency_key`, optional physical `advance_sec`. Same body/key returns the
  original transition without a second physical step or duplicate journal entry.
  New stale requests return 409. A schema-invalid or unauthorized request does
  not advance time. A valid declared action rejected by B06 consumes its native
  decision interval and reports rejection; the UI makes that cost visible.
- GET `.../suggest`: central-visible bounded route-aware suggestion. Local-role
  capabilities cannot inspect the joint suggestion. Choosing a local role on
  the owner page also hides other actor navigation and observations server-side.
- GET `.../checkpoint`: admin-only compact native task checkpoint, including
  control-plane inputs. POST `.../verify` recomputes native action execution from
  the beginning; this is action/physics replay, not replay of an external human's
  reasoning or a browser policy implementation.
- POST `.../export`: admin-only replay-verified RL/SFT ZIP using B09-M
  `VerifiedEpisode`. The payload is bounded to 16 MiB before compression. Entries
  have fixed ZIP metadata and safe relative filenames. Operator data is marked
  `policy_origin=external`, never silently promoted to expert/scripted teacher.
- POST `/api/operator/restore`: `{"record": <operator task checkpoint>}`.
  Source/dependency/native replay must match. Only external-origin task checkpoints
  are accepted; scripted PlanningRun/agent handover needs a separate provenance
  contract. Restore creates a new session and rotates credentials. The operation
  journal is reconstructed from verified transitions; navigation starts with the
  current restored frame, without inventing earlier tracks.
- POST `.../close` revokes the session and old object handles. Verify, export and
  close accept an empty JSON object. Tokens are not included in native checkpoints
  or training rows and are not stored in browser localStorage.

The owner page holds all issued capabilities in memory; changing its displayed
role does not erase the owner's separate admin capability. Delegating a single
viewer/local bearer through the API gives only that capability, not the owner's
other tokens. Roles identify task permissions, not verified human identities.

## Geometry and UI semantics

Navigation is an additional ideal operator projection, not a new agent sensor
model. The adapter reads **current** native state providers and their **sealed
EOP contexts**, uses exact reference GCRS→ITRS conversion and WGS-84 geodetic
coordinates, then filters actors by role. It never exposes ephemeris samples,
future states, private seeds or native configuration in the observation route.
It does not substitute a packaged EOP table for a mission's own context. Current
provider reads do not create operations or advance the simulator clock.

The 2D map and 3D globe are on the same page with command masks/reasons, actor
resources, products/jobs and the operation journal. Both use the same returned
ITRS positions. Tracks join only already observed frames; they are display
segments, not integrated forecasts. Frame history is bounded to 257 snapshots.
Models are enlarged, orientation/lighting illustrative, and the globe is a visual
projection rather than a calibrated attitude/illumination instrument. Ground-map
labels avoid the dateline chord and overlap; 3D supports focusing a visible actor.
WebGL failure leaves controls, telemetry and the 2D map available.

## Bounded local deployment

The interactive profile accepts at most 16 actors, 256 templates/decisions,
4096 entities and 16384 relations per task, with at most eight sessions. These
are operator-profile limits, not increases to native simulator limits. HTTP
requests are capped at 8 MiB; JSON/content length/origin checks apply. New
operator routes require loopback hostnames, same-origin browser requests, JSON
POST bodies and bearer capabilities on existing sessions. They do not enable
CORS. State/artifact responses are `no-store`; an unavailable optional reference
backend produces a structured 503. Existing legacy routes are not upgraded to
enterprise authentication by this package.

This is a local development/engineering console. It is not a public service with
SSO, account isolation, TLS termination, durable sessions, rate-limited worker
queues or independently qualified operator procedures. Session storage is process
memory; download a checkpoint before closing/restarting. Expensive verify/export
holds a session lock for consistency. Larger fleets, estimation/forecast display,
policy provenance across handover, persistent audit and pinned-hardware latency
qualification remain separate work. No flight/industrial qualification follows
from an internal browser smoke.

[Acceptance and exact scope](acceptance/B10_OPERATOR.json).
The installed smoke is `python examples/operator/smoke.py`; browser checks also
cover role switching, native manual steps, an automatic relay mission, downloads,
replay and narrow mobile layout.

During B10-A the adapter is the sole writer of each live operator session.
Programmatic agents and the human page share that session through the same
revision-bound adapter/API. Service implementation objects and their contained
TaskSession/runtime handles are internal; direct SDK writes bypass the adapter
journal and are outside this ownership contract. Coordinated attachment/handover
of an independently running SDK session requires the next ownership/provenance
contract. Matching simulator classes alone do not establish such coordination.

Large downloaded checkpoints can exceed the 8 MiB HTTP/import limit even when
initial inputs were smaller. Use the compatible-build SDK for such artifacts;
streamed/compressed artifact transfer and durable storage remain B10-B work.
