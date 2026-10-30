# Reference downlink scheduling baseline

Implemented 2026-10-31 over `reference-fixed-downlink/1`; no claim of industrial
qualification or optimal scheduling. This does not migrate the legacy MissionWorld.

## Execute

```bash
space-env reference-plan spec.json plan.json --report schedule.json
space-env reference-run spec.json plan.json run.json --lineage my-scenario-family
space-env reference-replay run.json
```

SDK: `space_env.reference.scheduling.plan_downlinks(spec, decision_sec=5., max_steps=10000)`.
The input is the existing `space-env.reference-downlink-spec/1`. The returned
`plan` has exactly the format consumed by reference-run. The schedule report
contains a spec checksum, algorithm identity, verified execution outcome, and
unfulfilled products. The portable run embeds the scenario and runtime metadata.

## Contract

The baseline starts at time zero. It knows all prescribed geometry, station
outages, initial inventory, resource constants, priorities and deadlines. It is
a privileged offline baseline, not a partial-observation agent or qualified
expert demonstration. Neither optimality nor infeasibility is certified.

At each decision, eligible actor/station queue heads are ranked by descending
priority, ascending remaining deadline slack, deadline, actor and station ID.
Each actor uses at most one station; station terminal counts are enforced.
Candidates must currently be visible, available, have enough nominal time before
the deadline, and be allowed to transmit by the battery threshold. Destination
restrictions and the simulator's onboard priority queue remain binding.

The scheduler does not reserve an entire future pass or certify sufficient
future energy for completion. It reconsiders decisions at a maximum decision
interval, geometric/contact/outage/readiness/deadline boundaries and predicted
selected-product completion. The executor additionally splits battery and
resource boundaries. Negative-slack products are skipped; an impossible queue
head can block other products on that actor/station. This is reported as an
unfulfilled baseline result, never as proof that the scenario is impossible.

The plan is executed through ReferenceDownlinkMission, then reexecuted from a
fresh mission. Full outcome equality is required before verified=true is
returned. This detects execution inconsistencies, not errors shared by both
runs or physical model inaccuracies. All existing reference resource-model
limits apply: fixed trajectories, configured constant bitrate, simplified
battery and ideal independent Sun tracking; no ACK, imaging, thermal or slews.

Invalid configuration or exhaustion of the decision budget raises an error;
no incomplete plan is returned. CLI computes and verifies before replacing
output files. Each file is written atomically; the plan/report pair is not a
single filesystem transaction. Input, plan and report paths must differ.
Default budget is 10000 decisions; larger fleets and long missions require
performance measurement. No endurance or production scale claim is made.

## Next development gates

1. Explicit product selection commands and cancellation/expiry policy, with
   causal queue semantics shared by planner, Gym and operator interfaces.
2. Rolling replanning from saved mission states and measured observations,
   preserving previously delivered inventory and station reservations.
3. Objective definitions and independent small-instance optimum comparisons;
   quantify greedy regret across held-out scenario families.
4. Look-ahead energy and link-budget uncertainty, setup/slew/antenna costs,
   calibrated contact margins and operational outage handling.
5. Benchmark planner time, memory and dataset size versus actors, products,
   windows and mission duration; replace cumulative reports where needed.
