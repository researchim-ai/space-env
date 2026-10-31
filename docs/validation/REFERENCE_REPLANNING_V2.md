# Reference operations: product selection and rolling continuation

Update: [Dynamic operations v3](REFERENCE_DYNAMIC_OPERATIONS_V3.md) implements
causal onboard-product, priority and station outage/recovery events, event journals
and a fixed-capacity event-driven RL environment. The remaining-work statements
below describe the earlier v2 boundary; v3 does not add acquisition physics.

Implemented 2026-10-31. Applies to the separate fixed-ephemeris downlink profile,
not the legacy MissionWorld, flight control or a qualified industrial planner.

## End-to-end CLI

```bash
# Optional explicit selection while generating an initial plan:
space-env reference-plan spec.json plan.json --report schedule.json --select-products
space-env reference-run spec.json prefix-plan.json prefix.json --lineage my-family
# prefix-plan.json can be any valid partial plan ending before the horizon.
space-env reference-replan prefix.json remaining.json --report remaining-report.json
space-env reference-resume prefix.json remaining.json completed.json
space-env reference-replay completed.json
```

A partial plan may specify products explicitly:

```json
{"steps": [{"dt_sec": 1, "actions": {"a": "Ground"}, "products": {"a": "scene-a"}}]}
```

`products` is optional. Without it, the existing priority/deadline queue is used.
With an explicit product, only that product is transmitted on that actor for the
step. Finishing it does not implicitly start another product. Transmission waits
for its readiness, contact availability, terminal allocation and battery limits.
Selection is stateless across steps, supports preemption/resumption of partial
products and never deletes inventory. A selected completed product simply waits.
Unknown products, wrong ownership, missing station action and incompatible
destination are rejected before any physical state changes.

Station contention is still the mission's alphabetical actor arbitration when
a caller oversubscribes a terminal. The planner avoids oversubscription itself.
Neither product priority nor explicit selection changes the arbitration rule.

## SDK and RL

```python
from space_env.reference.scheduling import plan_remaining_downlinks
from space_env.data_generators.reference_operations import ReferenceOperationsRecorder

recorder = ReferenceOperationsRecorder.from_record("prefix.json")
schedule = plan_remaining_downlinks(recorder.mission)
for step in schedule["plan"]["steps"]:
    recorder.step(step["dt_sec"], step["actions"], products=step.get("products"))
recorder.save("completed.json")
```

Planning uses two private copies of the supplied mission: one for planning and
one for deterministic reexecution. It does not mutate the source. Existing
clock, partial inventory, receipts, energy and all energy ledgers survive.
Restoration reexecutes and checks the entire prefix before making a recorder
available; checksum, runtime, observations, per-step results and final state
must agree. The continued record contains the complete prefix plus continuation
and preserves scenario lineage. It is self-contained but grows with history;
restoration is linear in prefix execution and cumulative reports can be costly.

For RL use `ReferenceDownlinkEnv(spec, product_selection=True)`. The MultiDiscrete
joint action retains 0=wait and 1..N=automatic station selection, then adds
actor-owned product/station combinations compatible with destinations.
`reset()` info `action_choices[actor]` maps each integer to station/product IDs.
All sampled action-space combinations are structurally valid, including when
the selected product is not ready or already complete. Default mode retains
the previous station-only action space. Reward remains newly completed on-time
product Mbit; Gym checker covers both modes. Existing JSON trajectory recording
supports the same MultiDiscrete action representation.

## State binding and compatibility

Schedule report schema is `space-env.reference-downlink-schedule/2`. New plans
include `initial_state_sha256` and `start_sec`, alongside `steps`. The fingerprint
covers the scenario, resource/delivery report and execution intervals. Both
fresh and continuation CLI execution reject a mismatched start state before
executing steps or writing output. This is integrity checking, not signing or
an authorization boundary. SDK callers applying individual steps themselves
are responsible for checking the start fingerprint if the mission may change
between planning and execution.

Old steps-only plans remain executable. Run schema remains
`space-env.reference-operations-run/1` with an additive optional per-step
`products` mapping. Existing no-selection steps retain their execution meaning.
Exact runtime matching remains mandatory; journals made against another source
revision require that runtime or a future explicit migration mechanism.

Source, plan and output paths must differ. Budget exhaustion or validation
failure produces no new output. Individual output files are atomic; two-file
plan/report publishing is not a transactional pair.

## Planning quality and remaining work

The new explicit planner considers ready products with nonnegative nominal
slack, ranked by priority, slack, deadline and IDs. It can bypass an impossible
queue head. The regression example recovers 1 Mbit previously blocked by that
head (3 Mbit versus 2 Mbit total); this is one counterexample, not a performance
benchmark. The old implicit planner remains the default for `reference-plan`;
`--select-products` and `reference-replan` use explicit selection.

Known contacts/outages and full resources are available to this privileged
baseline. No optimum proof, held-out regret evaluation, RF calibration,
independent physical validation, station setup/slew costs or scale qualification
is implied. There is no external booking service or reservation import. Rolling
continuation recomputes assignments from the current state; it does not insert
new jobs or mutate the scenario's outages. Those require versioned causal event
injection and replay contracts before they can safely be added.

Next gates: dynamic task/outage events with provenance, setup/antenna constraints,
look-ahead energy feasibility, small-instance optimality/regret comparisons,
load measurements, and integration with the general mission backend.
