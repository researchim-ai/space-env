# B11-Q4 — operational ground windows and observed-ACK reclamation

The historical four-second case and its failed 28/40 receipt remain unchanged.
This package introduces a separately frozen task configuration, not a relaxed
completion threshold, longer horizon, faster radio or different physics kernel.

## Versioned task configuration

`CampaignSpec.ground_control_sec` is a strict finite duration in [2, 8] seconds,
with the historical 4-second default. `ack_release_actions` is an optional strict
boolean (default false). The default generator retains the family/1 metadata and
legacy serialized policy configuration. A non-default option uses family/2 and
records both operational settings. Its geometry, force, power, thermal, RF, cloud,
CPU, product-size, terminal-capacity and custody parameters are unchanged.

The dedicated `examples/campaign/ground.py` freezes 8 cameras + 4 relays, 5 waves
of 8 requests at 0/20/40/60/80s, a 120s horizon, one terminal, two-second ground
controls, two straight repeats and one cold prefix restore after decision 7.
The original 2400s timeout, 1200s execution, 60s step-p95, 512 MiB checkpoint,
6 GiB peak-RSS, completion=1.0, native witnesses/replay/quality/export gates remain.
The main large case does not enable ACK cancellation: the window change is isolated.

```bash
python examples/campaign/ground.py /tmp/ground-v2.json --run /tmp/ground-v2
```

The general generator also accepts `--ground-control-sec` and
`--ack-release-actions`. A new manifest/output path is required. Historical
manifests are not overwritten, silently reinterpreted or relabelled as passing.

## Causal observed-ACK cancellation

`PolicyConfig.reclaim_acked_downlinks` is opt-in. Its disabled default is omitted
from canonical serialization, preserving existing frozen policy inputs. The
runner then asks the trusted public-input bridge to include accepted controls'
product references and static claims of currently visible conditional commands.
Unreleased command parameters, scenario, seed, future faults/rollouts and native
admission/forecast oracles are not supplied to a policy.

The policy can release an actor only when an enabled, owned cancellation action
exists and **all** of that actor's active controls are downlinks for locally owned
products observed as delivered with zero remaining bits. An active incoming relay
or any other active owned work prevents this optimization. TX completion,
awaiting ACK, custody completion and job completion are insufficient.

Conditional replacement additionally requires a visible, owned, locally ready
product with positive remaining bits. Delivered, awaiting-ACK, failed, cancelled,
expired and still-processing products are excluded. This explicit check prevents
reselecting the old delivered product when the single-action mask initially says
only `actor_busy`. Native probes caught and verified this failure before acceptance.

Candidate search respects role slots, joint limits, ownership, product conflicts,
shared-ground capacity, existing bookings and incoming half-duplex reservations.
A conditional command requires its selected ACK cancellation. Only the cancelled
actor's active claims/station use are removed; booking reservations stay intact.
Stalled-work cancellation keeps its previous conservative behavior. Cooldowns and
bounded search remain active; no optimality guarantee is claimed.

TaskSession still forks a trial, cancels, performs real native re-admission, then
advances physics. A rejected replacement discards cancellation as well. No mask
cache or predicted future success substitutes for this transaction. The single-
action mask may remain zero for an accepted conditional pair; the existing native
quality verifier explicitly checks this post-cancel case.

## Conditional preflight, not physical feasibility certification

`preflight.json` binds diagnostics to the frozen manifest and recognizes a narrow
2s acquisition / 2s CPU / 4s relay chain with unique products, no preload/injection/
cancellation, a two-second decision grid and one terminal. Uniform ground windows
are rounded to the decision grid for this conservative controller's reservation
cadence. Unsupported topology, chains, grids or early-release policies return
`applicable=false`; assumptions remain explicit.

Under the stated no-early-termination/full-window assumptions, the old profile
permits at most 28 windows after the initial pipeline. The two-second profile has
56 potential window slots, capped at 40 request completions. An upper bound of 40
is **not** a proof that late arrivals, RF/setup/ACK, visibility and resources make
40 feasible. Conversely, the old conditional bound is not a physical impossibility
proof for arbitrary controllers/actions. Diagnostics never skip workers, change
inputs or weaken a gate; actual native execution/replay remains required.

## Separate reporting without qualification laundering

`result.json.qualification_domains` separately reports technical runtime,
native integrity/witnesses, mission expectations and any additional gates. Missing
or timed-out observations stay `not_observed` in the full expected denominator.
Mission expectations include deliberately negative cases; passing them is not
successful customer service.

Raw native/trace signature equality is observable even if a mission target fails.
It is labelled separately from the formal comparison group. Overall `ok`, failed
repeat counts, successful-only timing summaries and formal group qualification
retain their original semantics. No failed goal is converted into overall success.

## Evidence and limits

The native tests exercise short windows, a different station geometry, slow rate,
setup longer than one window, delayed ACK both inside an active long window and
after a short control, ownership/role/slot/cooldown checks, late tasks, hidden loss
seed/future outage, cancel-and-resubmit, rejection rollback and prefix replay.
Slow RF can legitimately fail the mission. Windows may compose over several
controls; ACK is never synthesized or forced to fit a two-second interval.

The installed smoke includes paired short-window native replay/export and a
three-actor recovery mission with real observed-ACK cancellation followed by
verified RL/SFT export. The CI definition runs it outside the checkout. Actual
local test, wheel and frozen qualification outcomes are recorded in
[acceptance](acceptance/B11_GROUND.json); clean remote CI is not inferred from a
local system-site-packages venv.

Geometry remains a favorable synthetic nearby formation, not arbitrary multi-plane
or multi-day flight simulation. Teachers remain engineered baselines. Independent
mission/physical calibration, wider horizons/distributions, enforced hosted-worker
quotas and immutable history storage remain separate qualification work.

The final new large profile passed all gates in three fresh processes: 40/40
requests each, 40 acquisitions/CPU/custody completions, 32 later-released requests
fulfilled, all four ground senders, one exact native/trace signature pair across
straight repeats and prefix recovery. Maximum execution 1164.7s, step p95 26.47s,
peak RSS 1.76 GiB and checkpoint 72.2 MiB met the original gates. The old four-second
case remains failed at 28/40. This is a bounded synthetic profile qualification,
not arbitrary mission, flight or hosted-service certification. Next work is in
[the industrial roadmap](INDUSTRIAL_NEXT_AFTER_GROUND.md).
