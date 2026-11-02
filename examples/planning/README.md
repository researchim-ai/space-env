# A baseline plans the mission

```bash
python examples/planning/demo.py /tmp/planning nominal
python examples/planning/demo.py /tmp/planning-reroute reroute
python examples/planning/demo.py /tmp/planning-loss loss
python examples/planning/demo.py /tmp/planning-ttl ttl
```

The policy selects actions from current observations. There is no teacher action
sequence. The demo saves a policy/runtime prefix, resumes it, verifies a cold final
replay. Add `route_aware_rolling --export` after the mode to export RL/SFT through
B06 (each export independently verifies the episode). Inventory, wire, battery and thermal balances
are checked. TTL mode gives all second-product relay alternatives a short lifetime,
so selecting another template cannot bypass the negative case.

The same runner is available from the CLI:

```bash
space-env planning-run /tmp/planning/scenario.json /tmp/planning/task.json /tmp/planning/policy.json /tmp/planning/cli-run.json
space-env planning-replay /tmp/planning/cli-run.json
```

Other baselines are `wait`, `random_masked`, `priority`, `earliest_deadline`; pass
the name as the third demo argument. Results are not required to match the route
heuristic. Inspect `run.json` only in trusted trainer code: it includes the full
scenario, seeds and native audit state. Policies receive only `PlanningInput`.

For evaluation, write a `space-env.planning-benchmark/1` manifest with inline
`cases` (`id`, `family`, `scenario`, `task`, `episode_seeds`, `split`), `policies`
(PolicyConfig objects) and optional `teacher_gates`; then use
`space-env planning-evaluate manifest.json result.json --artifacts /tmp/bench-runs`.
The artifacts directory must be empty. See [the contract](../../docs/PLANNING_CONTRACT.md)
for split rules, metrics, bounded search, operator overrides and teacher limits.

A small portable benchmark includes one feasible and one closed-station case:

```bash
python examples/planning/benchmark_demo.py /tmp/planning-benchmark
space-env planning-verify-evaluation /tmp/planning-benchmark/evaluation.json
```

All ten case/policy episodes are retained. Execution/replay should pass, while
teacher gates should fail on the unreachable mission. This is intentional:
`planning-evaluate` exits 1 for failed teacher gates and still writes the report.
