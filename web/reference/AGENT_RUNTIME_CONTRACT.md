# B06 — runtime-backed tasks for AI training

Status: bounded reference engineering profile; acceptance results are recorded in
`acceptance/B06_AGENTS.json`. This layer reuses `MissionRuntime`; it does not
introduce a second spacecraft physics implementation. Legacy Gym environments
remain separate. Schema IDs are versioned independently of the package version.

## Public interfaces

`space_env.tasks.TaskSession(scenario, task, seed=...)` accepts a reference runtime
scenario and a `space-env.task/1` specification. The task declares actor/station
roles, a finite catalogue of command templates, action slots, physical decision
durations, capacities, episode budget, seed policy and objective.

`observe(role='@central')` returns `space-env.task-observation/1`: sparse entities,
relations, causal action catalogue and masks. `step(choices,
expected_revision=..., idempotency_key=..., advance_sec=...)` executes one
synchronized decision. `save`, `load` and `from_checkpoint` are trusted trainer
operations, not policy tools.

`MissionTaskEnv` exposes the same session through Gymnasium. Observations are
padded numeric arrays with explicit entity, relation and per-feature presence
masks. Missing values are zero-filled and marked unknown. `action_masks()` returns
Gym-compatible sampling masks. `variable_duration=True` adds a discrete physical
duration choice. No global Gym registration is required; instantiate with an
explicit scenario and task. Action indices refer to that task's fixed catalogue;
for multi-task training, the trainer must align catalogue semantics across tasks
or supply its own action embeddings. Numeric observations do not yet embed arbitrary
command-template parameters.

`space_env.tasks.parallel.MissionParallelEnv` requires the optional `[marl]` extra.
Every role controls its declared actors/stations. Each joint step advances one
clock, with one shared team reward (do not sum it across agents). With variable
durations, the minimum proposed duration becomes the common next decision time.
Agents disappear on termination/truncation; stepping thereafter is unsupported.

`TaskTools(session, role)` supplies a role-bound structured observation and step
tool. Only the central controller may use `joint_schema()` / `joint_step()`.
Python object privacy is not a security sandbox: untrusted agents must receive
only these tool results, not arbitrary Python access to the trainer process.

## Causality and authority

Local projections contain current locally available telemetry, released requests,
known jobs/products, own commands and transfer progress. Operator roles see their
stations. Central control sees the union. Each role has an independent append-only
entity registry: hidden remote entities cannot shift local slots. Overflow raises;
it never silently truncates an entity graph. Slots are not recycled within an episode.

Orbit state is current GCRS and attitude is the actual modeled quaternion where
available. RF/weather rows carry measurement age. This is an **ideal current local
telemetry profile**, not an estimator, delayed TT&C model or belief-state filter.
Scenario seeds, unreleased request details, future fault schedules and privileged
truth/checkpoints are excluded. Observation bindings hash visible content only.
Action IDs and role/target assignments are public task intent: do not put secrets
in their names. Command payloads referencing unreleased entities stay hidden;
privileged forecast bindings are forbidden in templates.

Masks use observable single-action conditions and static resource claims. They do
not call backend admission or simulate a future to predict success. Joint choices
can still conflict or fail admission. Receiver resources outside a role are not an
oracle. Masked sampling is useful, but does not constitute a feasibility guarantee.

## Transaction and clock contract

One decision forks the runtime, applies cancellations first, rechecks commands,
applies declared operator events, submits a batch, then advances physics once.
Only a successful advance, reward calculation and all-role projection commit.
Projection/physics exceptions leave the original session unchanged. A rejected
choice rolls back all zero-time edits and consumes one real decision interval on
the original state. There is no free retry or fictitious completion. Syntax/stale
revision/idempotency errors are control errors and do not advance time.

Task and native runtime revisions differ intentionally. Repeating the same
idempotency key/request returns the cached response; changing its body fails.
The native runtime must not be mutated behind a live task session. Horizon clips
the final interval and terminates; the decision budget truncates. Discount is
`discount_per_sec ** elapsed_sec`, zero on true termination, retained on truncation.

## Rewards and data

Five fixed objectives are exposed by `objective_registry()`: delivery, throughput,
SLA, efficient and robust. Delivery counts lifecycle-fulfilled requests (or on-time
Mbits in fixed-product reference profiles). Throughput counts only completely
acknowledged native ground products, including late ones. Custody transfer and
partial transmission are not ground delivery. Energy uses native battery-consumed
Wh converted to joules; safe-mode entry penalties are sampled at decision edges,
so transient enter/exit within an interval is not counted. Components are reported
for trainer diagnostics, not included in policy observation features.

`seed_policy='preserve'` keeps configured physics seeds. `derive-network` derives
network/weather seeds from the explicit episode seed and original configuration.
It does not claim to randomize every physical parameter. Gym reset options are
metadata only and cannot silently override a scenario.

`space-env.task-episode/1` is a **control-plane** artifact: it intentionally contains
scenario, seeds, transitions and native checkpoint. Loading replays every decision
from a cold start and verifies transitions and final runtime. It is not a policy
observation or a prompt. Export runs this verification before atomically writing
`space-env.task-dataset/1` JSONL. RL supports role-filtered transitions; SFT currently
supports central joint control only and excludes rejected decisions by default.
SFT tool names match `TaskTools.joint_schema()`. Lineage determines grouped splits;
scenario variants in one lineage cannot leak across train/test through seed changes.
These records prove execution, **not expert-policy quality** or dataset suitability.

## CLI and acceptance

Commands: `task-schema`, `task-run`, `task-resume`, `task-replay`, `task-export`.
Use `--help` for positional arguments. `examples/agent_task/demo.py` executes a
three-actor EO → CPU → ISL → ground mission in nominal/reroute/loss/TTL variants.
It checks conservation, checkpoint replay during fragment transmission and in
flight, and identical native checkpoints across adapters.

## Remaining product gates

This release provides bounded declared action templates, not arbitrary continuous
control synthesis or universal legacy-environment adaptation. It does not provide
an estimator, decentralized delayed communication, qualified expert teacher,
trained-policy benchmark, massive-fleet throughput claim, hardware calibration,
mission certification or flight operations authorization. B07 planning baselines,
B08 data factory, B09 fleet scale and B12 external evidence remain separate gates.
