# B07 — causal baseline planning

This engineering profile adds `space_env.planning` to the B06 task executor.
It is a controller/evaluation layer, not another physical simulator or a claim of
globally optimal mission planning. Final evidence is in `acceptance/B07_PLANNING.json`.

## Knowledge boundary

`make_input(session)` is a trusted trainer bridge. `PlanningInput` stores a canonical
JSON value, is frozen, and returns independent copies. It contains the central B06
observation, declared roles/slots, static resource claims, current active controls,
current station telemetry/reservations and the previous sanitized task feedback.
Command parameters come only from the causal B06 catalogue; unreleased parameters,
future faults/weather/loss, scenario seeds, truth reports and checkpoints are not
passed to policies. Current imported provider revisions are public operator evidence;
unrealized future events and future inactive bookings are excluded.

`BaselinePolicy.decide(input)` accepts this value, not a session/runtime. Baselines
never call backend admission, a forecast service or a future rollout. Python access
to the trainer remains trusted: this is not a hostile-code sandbox. No forecast
capability is implemented in B07; missing future contact/rate/receiver knowledge
stays conditional. Current GCRS, SOC, memory, safe state and measured telemetry retain
the accuracy/age limitations of B06.

## Policies and search

Five policies: `wait`, `random_masked`, `priority`, `earliest_deadline`,
`route_aware_rolling`. Configuration has strict finite bounds for seed, decision
interval, candidate/search budgets, urgency horizon, cooldown and stall thresholds.
The policy RNG is independent of physical RNG and is checkpointed explicitly.
Same-revision retry returns the same decision without drawing again; changed input
at that revision and nonconsecutive policy revisions fail.

Priority and EDF rank available actions by visible priority/deadline. Stable action
IDs break ties. Random-masked samples a subset/permutation with its own generator.
All policies use conservative joint resource/role/product/terminal constraints.
An active reservation and its transmitting owner occupy one terminal, not two.
Receiver half-duplex is conservatively modeled as an exclusive receiving actor,
which may exclude feasible CPU concurrency. Backend admission is still authoritative.

Route-aware rolling scores currently visible pipeline stages: ground ACK opportunity,
relay, processing and acquisition; it considers visible urgency, receiver memory,
contacts and low SOC. It enumerates **current-window compatible subsets**, with a
deterministic node budget and a feasible greedy incumbent. It replans each decision
epoch. `horizon_sec` is an urgency scale, not a propagated trajectory horizon.
Search exhaustion and candidate pruning are reported explicitly. A complete subset
search does not prove physical feasibility or global mission optimality. There is
no internal substitute simulator, continuous control solver or hidden nominal rate.

Native rejection consumes the B06 physical interval and is recorded once. Bounded
cooldown avoids a free admission retry loop. Observed transfer stalls can trigger
a declared actor cancellation; subsequent recovery/routing occurs in later epochs.
The stalled next hop receives a bounded route cooldown, allowing another declared
route to be considered after cancellation.
Cancellation never resets spin, SOC, spent resources or native product TTL. Radio
success and delivery credit come only from the native engine, not planner scores.

## Runner and operator

`PlanningRun(scenario, task, policy, episode_seed=...)` owns the session and policy.
`step()` uses `TaskSession.step` exactly once; `run(decisions=None)` runs to horizon
or budget, or a caller-requested prefix. Physics exceptions restore policy/RNG state;
task transaction rollback remains B06's responsibility. Out-of-band session changes
are rejected. The runner records input hash, decision/reasons/assumptions, feedback,
transition digest and policy-state digest.

`allow_operator=True` enables `step(action_ids=[...])` overrides using the same
declared task catalogue and executor. This is a trainer capability, not an expanded
policy permission. Overrides are explicit in the trace, advance policy observation
state, and survive cold replay. Such runs are labelled `policy_origin='external'`;
pure baselines are scripted, and random-masked is random. For an override,
`estimated_utility` and `baseline_action_ids` describe the original baseline proposal,
not a utility prediction for the operator action. No arbitrary event or raw
backend mutation is admitted through this interface.

`space-env.planning-run/1` contains the privileged B06 episode, full policy state
and audit trace. Loading reruns every policy/operator decision and compares exact
transitions and final native state. Timing/RSS samples are shape/range checked,
stored as original measurements and **not** claimed reproducible by physics replay.
RSS is the process-lifetime high-water mark; it is not an episode allocation delta.
RL/SFT export uses the existing B06 exporter on `run.session.checkpoint()`.

## Benchmark and teacher evidence

`BenchmarkSpec` / `run_benchmark` take a portable, predeclared manifest containing
inline scenarios/tasks, families, lineage-grouped splits, physical episode seeds,
independent policy seeds, budgets and optional teacher gates. Episode seeds are
requested trainer seeds: with B06 `seed_policy=preserve` the configured physics
seeds remain unchanged; distinct row seed labels alone do not establish diversity. The manifest is hashed
before execution and, with `artifacts=...`, written before any episode starts.
The artifacts directory must be empty. OOD families cannot also appear in train or
holdout; variants of one lineage cannot cross splits. Relabelled variants or extra
seeds do not establish independent family support for teacher qualification.

Every case/seed/policy remains in the results, including exceptions and infeasible
missions. The all-episode completion mean assigns zero contribution to errors and
zero-request episodes, and reports the latter explicitly. Native metrics include
request fulfillment, fully ACKed ground Mbits, lateness/latency, battery-consumed
joules, discarded/wasted bits, sampled safe entries, rejections and search effort.
Decision wall time/RSS are measured separately from physical mission seconds.

Teacher gates apply to held-out/OOD rows, per-family independent lineages, paired
priority/EDF baselines and declared thresholds. `passed_declared_gates` proves only
those declared tests. `expert_qualified` remains false: neither execution replay nor
a toy test establishes expert demonstration quality, hardware calibration or global
optimality. Gates that lack held-out evidence fail; failed baselines do not silently
disappear from paired comparisons.

`planning.oracle.tiny_oracle` is trainer/test-only, never imported into a policy.
It exhaustively executes a bounded, finite-depth current-mask action tree on native
TaskSessions. Candidate overflow raises, node exhaustion removes any optimality
claim. Even a fully enumerated tree optimum applies only to that declared tree, depth,
fixed interval and objective, not arbitrary actions or a continuous full mission.

## CLI and remaining gates

Commands: `planning-run scenario task policy output [--seed N] [--decisions N]`,
`planning-resume record output [--decisions N]`, `planning-replay record`,
`planning-evaluate manifest output [--artifacts DIR]` and
`planning-verify-evaluation report`. Verification recomputes the manifest and
compares semantic results, native episode and decision-trace hashes, excluding
wall time/RSS. Input overwrite is refused; evaluation output must be outside the
artifacts directory to protect its frozen manifest and run records. Failed execution
or failed declared teacher gates produce CLI exit 1 while preserving the report.
The example in `examples/planning/` exercises the EO/CPU/relay/ground chain without
a scripted action sequence. B08 data diversity/quality, learned-policy evaluation,
B09 streaming/fleet throughput and B12 independent calibration remain open.
