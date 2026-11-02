# B08 — bounded synthetic mission and training-data factory

The factory reuses B06 TaskSession and B07 PlanningRun/benchmark. It does not contain
another spacecraft simulator. Schema IDs: `space-env.factory/1`,
`space-env.factory-catalogue/1`, `space-env.factory-bundle/1`,
`space-env.factory-quality/1`, `space-env.factory-dataset/1`.
Acceptance evidence is recorded separately in `acceptance/B08_FACTORY.json`.

## Templates, distributions and provenance

`FactorySpec` contains inline native reference scenarios/tasks, template sources,
family/kind tags, explicit splits, sample counts, policy configurations, export roles,
teacher gates and corpus quality/budget limits. Inputs are copied before validation
and execution. All generated variations are labelled **synthetic**. Neither a source
string nor random sampling establishes hardware calibration or expert policy quality.

Kinds cover ground delivery, EO/processing, relay, recovery, shared terminals and
finite-resource workloads. Structural capability checks verify the required native
components/actions; a family tag is not proof that every instance is feasible or hard.
Templates remain responsible for orbit, sensor, RF, bus and task consistency.

Supported distributions have typed units and targets: product-bit scaling, memory,
battery, transmitter-rate cap, deadline, processing/acquisition work, terminal count,
outage duration and declared relay TTL. Choice and uniform distributions use local
PCG64 generators derived from the explicit factory seed, geometry group, sample
index and parameter/target. They never use global RNG state or retry until success.
Terminal counts use integer choices; lifecycle bit counts use Python round-to-nearest
integer. Runtime validation rejects invalid results rather than silently clamping.

Scaling changes the documented native field, not an arbitrary JSON path. RF mode
curves, optical parameters, controller windows and event payloads are not silently
retuned to make a sampled mission succeed. Acquisition/processing scaling can make
an unchanged control window insufficient. Initial-storage overflow is a configuration
error; an otherwise valid but unreachable deadline/outage is a mission outcome.

Structural template errors that prevent identifying geometry are input-contract
errors before sampling. Every requested case of a structurally usable template stays in the catalogue with its sampled parameters and error,
if any. Invalid draws are not replaced by easier ones. Configuration checks construct
the native task at time zero; they do not consult a future-success oracle. The factory
seed policy is explicit: `derive-network` or `preserve`, with B06's exact meaning.

## Splits, aliases and duplicates

The geometry group ignores experiment IDs, lineage labels, seeds, resource/workload
knobs and actor-order permutations. It retains trajectory/epoch, station geometry,
targets and AOIs. All variants of that underlying group must share one split.
An explicit conflicting assignment raises before any training execution. OOD families
cannot also appear in train/holdout. These split/duplicate guarantees apply within one frozen manifest/bundle; combining
independent corpora requires a cross-bundle group/trajectory index.
Lineage derives from the group fingerprint,
not a caller-provided name; adding seeds does not create independent teacher support.

Typed identity graphs distinguish actor/product/station/request/job/role namespaces
and preserve reference relationships while ignoring supported identity spellings.
Fingerprints use bounded graph refinement, not a complete graph-isomorphism solver
or a universal proof of physical equivalence. They can conservatively overgroup
cases; unknown model-specific identity conventions need separate qualification.

Dataset duplicate checks additionally compare actual public trajectories, action
semantics, masks/catalogues, feedback, physical intervals, discount and reward.
Case/seed labels and hidden mission metadata do not distinguish otherwise identical
training trajectories. Same-split duplicates are excluded with an audit reference;
cross-split duplicates fail quality, even when geometric scenarios differ. These
conservative checks may remove label/role-name augmentations. They are not a proof
that all statistically similar training examples have been detected.

## Execution and quality gates

The factory freezes a portable catalogue before B07 runs all case/policy pairs.
Failures and infeasible cases remain in benchmark denominators and audit records.
Each successful episode is cold-replayed, bound to the expected scenario, task,
policy, seed and trace, then checked against native inventory/lifecycle/DTN/battery/
thermal ledgers. Absolute tolerances are 1e-6 bits, 1e-9 Wh and 1e-6 J, matching the
declared engineering fixtures. Large/unqualified parameter ranges can fail these
strict gates; this is not evidence of calibrated accuracy at those ranges.

Completion and Mbit reward components must match native ground ACK outcomes.
Small generated cases can additionally be checked with B07's trainer-only
`tiny_oracle(TaskSession(case['scenario'], case['task'], seed=case['episode_seed']).checkpoint(), ...)`.
Its proof applies only to its finite-depth masked action tree. It is never a policy
input, does not run in compilation, and is not a universal feasibility label.
Policy views must be finite and exclude privileged fields. Accepted actions must
respect causal masks; B06's explicit cancel-then-recheck exception is recorded.
Role exports use B06's role filtering; SFT requires central control. Local RL
views are projections of a central rollout, not qualified decentralized teachers;
a selected local action may depend on the central controller's broader current view. Late role/export
failure rolls back all training rows from that episode, including duplicate state.

RL retains valid failed/rejected mission transitions. SFT defaults require full
completion, zero rejected decisions, a selected SFT policy and passed declared
held-out gates for **that candidate**. Another policy cannot borrow its qualification.
`require_teacher_gates=False` is an explicit engineering-data opt-out; it never sets
`expert_qualified=True`. B07 thresholds still prove only declared synthetic tests.

Corpus gates cover error budget, minimum RL/SFT rows and unique valid missions.
`ready` means those configured gates passed, not all missions succeeded, all possible
leakage was ruled out, or data is qualified for flight operations. An error allowance
is explicit. Missing teacher evidence with the strict defaults usually quarantines
SFT instead of manufacturing expert demonstrations.

## Bundle layout and learning inputs

`build_dataset(spec, directory)` requires a new destination. It builds in an owned
temporary sibling, writes all files and checks limits before renaming into place.
Completed negative quality results are published locally as `quarantined` audit
bundles. Infrastructure/budget exceptions leave no partially published destination.
Existing outputs are not overwritten. Concurrent independent writers are unsupported.

Only `data/train`, `data/holdout`, `data/ood` contain policy-facing RL/SFT JSONL.
Rows retain B06 observation/action semantics, physical dt/discount, role, provenance,
teacher status and synthetic quality labels. The factory split is authoritative;
`legacy_lineage_split` records the old B06 hash assignment as audit metadata.
Action-map IDs/slots are explicit; SFT rows also carry the matching structured tool
schema. For RL select observation/action/next_observation and training targets;
for SFT select messages/tools. Provenance, group IDs, seed labels and outcome metadata
are not observation features or LLM prompt content.

`manifest.json`, `benchmark.json`, `episodes/`, `report.json` and `bundle.json` are
privileged trainer artifacts: they include scenarios, seeds and native truth. Never
pass the entire bundle to a policy. A bundle is JSON/JSONL, not an executable archive.

All members have byte hashes/sizes and an exact inventory; symlinks and unsafe paths
are rejected. `verify_bundle` defaults to catalogue regeneration, B07 semantic
evaluation, native episode replay and reconstruction of the exact training rows.
`replay=False` / CLI `--integrity-only` checks files/catalogue bindings but does not
verify training semantics, and reports `replay_verified=False`. Checksums provide
integrity/reproducibility, not authenticity against an attacker rewriting all files.

The stable `content_sha256` covers the catalogue, training rows and semantic audit.
The raw bundle hash also covers measured B07 wall time/RSS and can differ between
otherwise reproducible builds. Source/runtime fingerprints retain B06/B07's strict
version/dependency matching; there is no cross-version checkpoint migration here.

## Bounds and next gates

Limits: 64 generated cases, at most five policies (320 episodes), explicit row/byte
budgets and at most 100,000 fingerprint graph nodes. As of B09 the default total
bundle budget is 256 MiB including the root receipt. Catalogue byte limits count
generated cases, not the original template/spec envelope. B09 adds streaming
JSONL and a bounded SQLite corpus/duplicate spool, compact replay checkpoints,
spawned workers and explicit cross-bundle registration; see
[the scaling contract](SCALING_CONTRACT.md). One native episode and its task
transitions still occupy memory. Short synthetic throughput measurements do not
qualify massive fleets, long missions, learned teachers, estimation realism or
independent hardware calibration; B09-Q and B12 remain open.

CLI: `factory-schema output`, `factory-compile spec output`, `factory-build spec dir`,
`factory-verify dir [--integrity-only]`, `factory-register dir registry.sqlite`. Invalid contracts or quarantine return exit 1;
completed quarantine keeps its audit/report for diagnosis. See `examples/factory/`.
