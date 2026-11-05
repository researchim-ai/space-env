# B12: equal-information release qualification

## Fixed scope

The bundled `configs/release_scope_v1.json` states supported interfaces, physical
units and separate profile limits. This release benchmark covers kinematic serial
capture → metadata processing → ideal geometric delivery. It does not qualify RF,
rigid attitude, scientific image processing, measured instruments or arbitrary
mission DAGs. Incoming-request/rebind correctness is separately exercised by B11;
these B12 benchmark policies start from a static request graph.

The complete generator emits 32 fixed cases: eight families across one training,
one development and two test orbit groups. Geometry, seeds, sites, resource
changes and service gates are declared before outcomes. Variants of one geometry
group cannot cross splits. The validator rejects duplicate physical definitions,
misdeclared geometry hashes and unsupported profiles. Families are nominal,
all-station outage,20-second deadlines below the conservative planner reservation, energy scarcity, slow processing,
station contention, expanded four-actor catalogue and 1800-second missions.
The five-case smoke subset is explicitly incomplete. The deadline case is a
negative control for these conservative planners; it is not an independent proof
that no possible native controller could meet that deadline. Identity checks bind
the declared numeric/ID graph, not all possible equivalences under renaming or
physical symmetry.

## Comparison contract

One sealed forecast is collected per case. Three policies receive exactly that
value: `priority_greedy`, `deadline_rule`, `bounded_order_search`. All use the same
serial feasibility packer, edge guards, native resources, maximum search-node
allowance and worker timeout. Search uses up to eight deterministic request orders
and accounts for nodes across all calls, including an exhausted final call. This
is a bounded ordering heuristic, not MILP or a global optimum. Actual nodes and
timing are reported; equal upper limits do not imply equal actual computation.

Policies do not receive native outcomes during selection. Future pointings are
materialized afterward through the same authorized capability for every policy.
Native execution and cold replay determine the actual score. The evaluator
retains the full request denominator, priority served, deadline misses, per-actor
completion, Jain fairness, command rejections, bit balance and base-energy deficit.
Native bundle publication verifies execution before exporting RL and conditional
SFT. A verified feasible trace is not an expert demonstration.

## Promotion and uncertainty

Every case/policy row is mandatory, including timeout/error rows with the complete
demand denominator. Negative controls must retain zero deliveries and failed
mission goals. Positive held-out cases must meet service, replay and resource
gates. Comparisons aggregate within geometry groups before computing paired
uncertainty. The reported Hoeffding bound is conditional on independent draws:
frozen deterministic geometry does not establish population sampling. It does
not prove real-world generalization or global statistical superiority.

The small default test-group count yields deliberately conservative intervals;
passing a few mechanisms cannot establish superiority. `expert_teacher_qualified`
remains false. Native per-case SFT candidates and suite-level teacher promotion
are separate gates; a dataset file alone is not an approved expert training set.

## Resource evidence

Each worker starts a new interpreter with BLAS threads fixed to one. Forecast,
planning, execution, checkpoint, cold replay and verified export have separate
measurements; step p50/p95, checkpoint bytes, exported rows/sec and peak RSS are
recorded. Timeout terminates the child. RSS is an observed post-run gate, not an
OS memory boundary. A predeclared total artifact allowance stops later workers
while retaining their failure rows. Journals/checkpoints restore by audited
replay, not constant-time RAM restoration.

The measured envelope groups observed cells by actors, requests and horizon. It
separates native/resource success from service success. Multiple policies and
variants are not independent performance repeats. It gives no interpolation,
production fleet SLA or support claim for unmeasured sizes.

## Commands (installed package, outside checkout)

```bash
python -I -m space_env.planning.release_benchmark freeze /tmp/release-suite.json
python -I -m space_env.planning.release_benchmark run /tmp/release-suite.json /tmp/release-results
python -I -m space_env.planning.release_benchmark verify /tmp/release-results --native
```

Add `--smoke` to `freeze` for the five-case subset. Output directories must not
exist. `progress.json`, worker packets/logs, forecasts, selections, pointings,
verified datasets, measurements and final report are retained. Offline verification
checks exact inventory, policy recomputation, plan correspondence, scoring and
gates. Native mode additionally recomputes forecasts/pointings and replays bundles.
Hashes detect inconsistency; they do not authenticate an untrusted producer.

## Packaging gate

`requirements/release-py311-py312-linux.txt` freezes the currently selected runtime
and test dependency closure; it is not a hash-locked cross-platform resolution.
The installed editable distribution is not used as root dependency truth:
`scripts/freeze_release_constraints.py` reads current `pyproject.toml`, traverses
installed transitive metadata and inventories licenses. Python 3.10 retains its
separate local fast test profile; this lock contains Astropy requiring 3.11+.

Packaging helpers, pinned dependency snapshots and advisory tools are optional
local utilities. CI workflows have been removed at the user's request. They are
not prerequisites for installation, development or release. Use local tests and
manually run examples; see [local setup](../LOCAL_RUN.md). Historical benchmark
receipts retain the policy and measurements applicable when they were recorded.
License metadata inventory is not a completed legal review.

## Full group orchestration and storage capacity probes

`scripts/run_release_qualification.py` freezes and runs the complete suite by
geometry group, verifies every shard and recomputes one global teacher report.
Each shard retains its own inputs, progress, logs, dataset bundles and audit.
Groups can run concurrently; individual policies still share the same per-case
forecast and maximum budgets. Timings then describe a contended shared host.

```bash
python -I scripts/run_release_qualification.py freeze /tmp/release4g.json --artifact-gib 4 --result-mb 128
python -I scripts/run_release_qualification.py run /tmp/release4g.json /tmp/release4g --workers 4
python -I scripts/run_release_qualification.py verify /tmp/release4g
```

Use the absolute script path when running outside checkout. The1GiB and4GiB
profiles differ only in aggregate artifact capacity and suite ID. The second is
a declared capacity experiment after observing evidence storage pressure, not a
new blind holdout or a service-floor reduction. Preserve both original reports;
never relabel earlier rows under the new profile hash. The parent records payload
artifact bytes excluding its final report; hashes/inventory cover every file.


The first4GiB experiment retained a64 MB per-result gate. Three long-mission
exports exceeded it (about64.36 MB), even though native execution and replay
passed. A separately frozen128 MB capacity probe repeats those same three
policies without changing physical inputs or service thresholds and checks exact
episode equality. It is not a new held-out case. The optional full local configuration
uses4GiB aggregate capacity and128 MB per result; local mixed-profile evidence is
not mislabeled as a completed homogeneous full128 MB run.
