# Actor-only mission operations suite

This is a small **public regression suite**, not a blind holdout or a measured
LLM evaluation. Five fixed inputs exercise three existing operational families.
The supplied policies are scripted. No new physics controller is implemented.

- **Resource conflict:** `resource_lanes/demo.py`, both literal `wide` and
  `constrained` modes. Camera acquisition, transformation and downlink overlap
  on one actor. The existing memory-bandwidth allocator governs actual work.
  The native goal is all three qualified requests acknowledged at Ground.
  Evidence includes resource-lane intervals and native request outcomes.
- **Comms recovery:** `agent_task/demo.py`, literal `reroute` mode. Transfer
  toward b is cancelled and the second product is redirected through c.
  Native DTN transfer states, custody and ground acknowledgements determine
  outcomes. The goal is both requests fulfilled. **The outage is predeclared;
  this is not dynamic discovery.** The supplied privileged schedule knows timing.
- **Momentum unloading:** `momentum_unloading/demo.py`, literal `nominal` and
  `limited` modes, only its initial three-second unload segment. Actual rigid
  wheel/RCS momentum evolution consumes shared propellant and bus energy.
  The native goal requires a completed unload operation and remaining momentum
  within its declared tolerance. Limited torque is retained even when it fails.
  Subsequent EO operations are deliberately excluded from this task.

The latter is a separate physical control task, rather than a delivery workload
renamed as attitude. The suite does not claim a power-recovery task: energy is
accounted in unloading, but there is no separate power-control evaluation.

## Run and inspect

With the package and reference/coverage dependencies installed:

```bash
python -m space_env.llm.operations_suite --compile /tmp/operations-inputs.json
python -m space_env.llm.operations_suite --output /tmp/operations-run
python -m space_env.llm.operations_suite --audit /tmp/operations-run
python -m pytest tests/test_operations_suite.py -q
```

Outputs must use a new directory. `suite.json` freezes complete scenario/task
inputs, command templates, time partitions, ordered cases and builder provenance
hashes before any policy is run. No seed search, successful-case admission,
resampling, or success filtering occurs. All five cases and both policies are
attempted; errors stay as error rows with null metrics in `progress.json` and
`report.json`. No error is replaced by a scripted fallback.

## APIs and boundaries

`space_env.llm.operations_suite` exposes `compile_suite()`,
`validate_suite(suite)`, `run_suite(output, *, suite=None)`, `audit(directory)`,
`native_evidence(session)`, `operational_goal(family, evidence)` and
`export_qualification(record, family)`.

Each case contains `scenario`, `task`, `steps`, `id` and `family`. Construct
`TaskSession(case['scenario'], case['task'])`, then `LLMEpisode` with the task
text from `TASKS`, to connect an external provider using the existing
`run_agent(episode, policy)` interface. A policy accepts only messages and tool
schemas and returns a tool-call name plus JSON argument string. No model is
called automatically here. Custom-provider receipts require their own harness;
this suite's audit intentionally verifies only its declared two baselines.

Execution uses existing `space_env_actor_joint_step` tools and local actor
projections. There are no central-role or operator injection calls. The adapter
translates only native submit/advance plans, preserving command batches and
physical time; it rejects other runtime operations rather than inventing a
conversion. The comms case reuses its existing actor cancellation catalogue.

`wait` is a public no-action baseline. `reference_schedule` is explicitly
`scripted_privileged`: it receives the existing demonstration timing and action
sequence, never an expert or LLM score. Both run from identical initial native
inputs over identical per-case physical time partitions. Decisions still flow
through native admissibility and rejection handling.

`operational_goal_met` uses native ledgers, never expected case labels. Request
completion uses qualified fulfillment/ACK counts. Unloading uses native status,
target, residual and tolerance. Raw evaluator evidence, including propellant,
energy, transfer and resource ledgers, stays outside actor messages. Raw
`LLMEpisode.metrics` are preserved: its `success` is specifically a terminated
or explicitly finished request-delivery episode. These bounded operation
segments may end with `caller_stopped`, so that field is **not** the operation
goal metric, especially for unloading. Each result also carries `export_qualification`: delivery SFT eligibility is
reported separately, operation expert qualification is false, the split is test,
and training export is false. A successful unload never becomes a delivery-SFT
example. No training export is performed.

Audit verifies seals, frozen inputs, a complete ordered comparison matrix,
policy provenance, every completed dialogue's cold replay and exact native
result binding. Errors remain visible and do not count as successes. There is
no cross-family scalar leaderboard or unsupported superiority claim.

The frozen inputs are packaged in `space_env/bundled_data/llm_operations_v1.json`
and loaded with `importlib.resources`; no checkout or example imports are needed.
The fixture digest is pinned in the module. Its provenance records only the
example builders used during one-time derivation; those files are not loaded or
hashed at run time. Native Python source hashes bind receipts to the installed
package snapshot. Source changes during a run invalidate final reporting; finish concurrent writers and rerun/audit for
final source-bound receipts. The suite does not modify runtime, CI or git.
