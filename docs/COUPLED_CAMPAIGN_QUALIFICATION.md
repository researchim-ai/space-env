# B11-Q1 — coupled formation campaign qualification

This package tests the existing native mission engine with a new, predeclared
coupled workload family. It does not add a second simulator, claim flight accuracy
or certify arbitrary constellations. The new input family and the observed results
must be distinguished from earlier fixed-ephemeris/preloaded-product B09-Q probes.

## What the campaign actually executes

`space_env.scaling.campaign.CampaignSpec` / `campaign_workload` build a synthetic,
closely spaced Earth-observation formation with distinct initial orbit states:

- 1–12 camera actors, 1–4 relay actors, no more than 16 total;
- J2 coasts through the native controlled-trajectory backend (no burns commanded),
  finite memory/battery, coupled bus and multi-node
  thermal networks, sampled RF, DTN custody/fragments and ground ACK;
- separate camera and relay roles, two declared action slots per actor, one shared
  ground station/terminal and its aggregate bandwidth/rate pool;
- individual acquisition → onboard CPU transformation → selected relay → ground
  actions for each request; no preloaded products or canned success transitions;
- 1–16 bounded release waves, explicit request deadlines, eligible actors and
  generated raw/final product/job identities;
- nominal, temporary ground outage/recovery, 25% ACK loss, complete ground outage
  and deliberately infeasible relay TTL profiles. Seeds are declared, not searched
  or resampled until a good outcome appears.

Native actor, action, entity, relation, terminal and numerical budgets are not
raised. Configuration combinations that exceed them fail before execution.

This is **formation-local**, not a globally separated multi-plane constellation.
The horizon is bounded to 20–300 physical seconds. It is not a 24/72-hour mission,
continuous global coverage claim, maneuver campaign or qualification of rigid
wheel/RCS ADCS. Longer and spatially broader coupled regimes remain gates.

## Synthetic geometry and quality assumptions

The generator retains fixed initial camera boresights and declares synthetic
5-degree half-angle optical footprints. Before execution, it places each AOI
around the nominal boresight/ellipsoid intersection at the intended acquisition
midpoint. This is a constructive, deliberately favorable target-placement rule,
not a representative distribution of real customer requests or a globally optimal
pointing plan. Bodies use distinct nearby J2 initial states; targets and planned
operations still pass the native geometry/coverage/quality engine.

AOIs, raw/processed sizes, CPU work/switching/scratch, memory and camera parameters
are explicit. Processing is a declared synthetic product/resource transformation,
not a real image-feature extraction pipeline. Optical fulfilment requires native
coverage, GSD/sun thresholds and explicitly supplied synthetic clear-cloud cells.
Clear masks are assumptions, not observed/predicted weather data. The packaged and
sealed fixture EOP manifests were checked equal for this release, with exact frame
matrix parity over the exercised interval.

Policies receive the existing causal PlanningInput only. Future release geometry,
fault schedule, scenario metadata and checkpoint truth are not fed into a policy.
Public action intent names remain public; unreleased request entities are hidden.
The initial boresight/target construction is an offline scenario-generation rule,
not an online forecast capability added to the baseline.

## Frozen gates and native witnesses

`Workload` adds optional `campaign_gates`, `max_completion_fraction`,
`resume_after_decisions` and `comparison_group`. Existing inputs remain valid.
The manifest is validated and frozen before any repeat runs. Completion lower and
upper bounds cannot contradict each other. A negative mission must satisfy its
predeclared zero-completion upper bound, not merely a permissive zero lower bound.

Evaluator-only witnesses read final **native** reports: acquisitions, completed
CPU jobs and custody transfers, completed later-released requests, actual ground
senders, expired transfer states and lost-ACK/re-serialized fragment events.
They do not equate an ACK, transmitted bits or product arrival with a quality-
fulfilled customer request. Shared resource configuration is explicit; the witness
of multiple actual relay senders is not a proof of globally optimal scheduling
or every possible contention interleaving.

The usual native conservation, cold replay, latency, checkpoint/RSS and verified
RL/SFT export gates remain. Outcomes that fail any declared gate are retained,
including worker exceptions and timeouts. There are no hidden retries or easier
replacement scenarios. A passing negative case means the engine represented the
expected failure correctly; it is not counted as successful customer service.

## Paired recovery and repeatability

`resume_after_decisions=N` executes a real prefix, creates a compact planning/native
checkpoint and cold-restores both controller and physics before continuation.
A point reaching termination/budget is rejected as a failed workload. Resume time
and prefix bytes are reported separately. `execution_sec` includes prefix
checkpoint/recovery overhead when configured; this is not pure integrator time.

A `comparison_group` must contain at least two workloads with **identical** scenario,
task and policy inputs. Only top-level execution/measurement settings, including
resume location and repeat count, may differ. All expected repeats must pass;
then native backend and decision-trace hashes must match across the whole group.
This establishes exact semantics within this build, not identical timings.
A performance-failed row remains in raw results; it cannot be removed to obtain
"repeatability". An unqualified group is not evidence that all raw hashes differ.

`ScaleSpec.workers` bounds concurrency to 1–4. A small thread supervisor launches
one fresh interpreter per repeat, with at most `workers` pending/running jobs and
native BLAS/thread variables explicitly set. Results retain frozen manifest order,
even if processes finish out of order. Worker errors remain in denominators.
Each child's PID/thread/affinity/CPU/cgroup evidence is recorded where available.

`progress.json` atomically records finished rows; it is a partial observation,
not final readiness, a durable execution queue or an automatic resume contract.
`result.json` is the final hashed receipt. Concurrent engineering runs affect wall
timings; these measurements do not constitute pinned-hardware tail-latency or
real-time/training-throughput qualification. Peak RSS is process lifetime high
water, including interpreter, execution, recovery and export, not allocation delta.

## Run from a checkout

For a small installation/semantic smoke use `python examples/campaign/installed_smoke.py`.
The full command below is an intentionally substantial qualification suite, not
a quick server health check; it can run for many minutes and consume significant RAM.

```bash
python examples/campaign/qualification.py /tmp/campaign.json \
  --cameras 4 --relays 2 --waves 3 --horizon 90 --workers 3 \
  --run /tmp/campaign-results
```

Defaults include all five modes and paired nominal/recovery variants. Choose
`--mode nominal`, `--mode ttl`, etc. to freeze only selected modes. The output
manifest path must be new; benchmark output must be empty. A frozen manifest can
be executed separately through the existing `space-env scale-benchmark` CLI.
For a larger declared formation:

```bash
python examples/campaign/qualification.py /tmp/formation.json \
  --cameras 8 --relays 4 --waves 5 --wave-gap 20 --horizon 120 \
  --mode nominal --workers 3 --repeats 2
space-env scale-benchmark /tmp/formation.json /tmp/formation-results
```

Published acceptance uses the exact manifests linked by its receipt; they can
have different predeclared repeat/budget settings from these illustrative commands.
`examples/campaign/installed_smoke.py` deliberately uses a small paired coupled
case for installed-wheel correctness, not a claim to repeat the large workload.

## Preserved diagnostic failures

The first four-actor/four-request probe omitted clear-cloud declarations. All
four products reached ground, but zero requests fulfilled the optical-quality
criteria. Native replay/accounting passed and the qualification correctly failed.
Its result and frozen input are retained as
`acceptance/B11_CAMPAIGN_INITIAL_UNKNOWN_CLOUD*`. The fix explicitly added synthetic
cloud truth; it did not weaken the requirement or relabel ACK as fulfilment.

An initial detached launch of the large run left a frozen manifest but no final
result/exit receipt and was no longer running. It is not counted as a success;
the cause was not established. Subsequent qualification uses managed retained
process sessions. The launch diagnostic is preserved in final acceptance.

## Remaining scope

[Acceptance](acceptance/B11_CAMPAIGN.json) records the actual executed sizes,
physical horizons, budgets, hardware, measured results and qualifications. It must
not be generalized into arbitrary long-horizon, global-coverage, image/sensor,
rigid-attitude, flight-calibrated or learned-policy benchmark claims.

Next gates: measured multi-hour/global coupled cases, actual customer geometry and
weather distributions, heterogeneous calibrated buses, maneuver/attitude campaigns,
learned agents and OOD baseline comparisons, and mutable-history/recovery/export
cost improvements under unchanged physical/safety sampling. The bounded local
operator's 32 MiB artifact and 256-decision profile is a separate limit; a large
native benchmark checkpoint need not fit that UI profile.
