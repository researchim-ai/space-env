# Bounded calendar optimality and portable benchmark v9

Implemented 2026-10-31. This package measures scheduling quality on a deliberately
restricted class instead of declaring the general reference simulator optimal.
The oracle is independent scheduling code over calendars supplied by the existing
reference geometry backend; it is not an independent orbit/physics truth source.

## Oracle scope and objective

`solve_small_downlinks(spec, max_states=65536)` accepts a fresh reference spec with:

- One station with exactly one terminal and a horizon of at most 3600 seconds.
- At most 16 products; integer bit sizes with combined total <= 2^53.
- Fixed per-actor rates, release times, destinations and deadlines, known geometric
  station contacts and initial outages. Each product must fit wholly inside one
  available service interval and finish by its deadline. Transmission is not
  preempted and does not span an outage/contact gap.
- Zero base/TX electrical power, no solar, thermal, TX quota, processing or DAG
  requirements. Initial battery state must permit transmission. There are no
  runtime events or imported bookings in this fresh-spec API.

The objective is completed_on_time_bits, matching the throughput reward. Priority
is a heuristic input, not an objective weight. An optimum may intentionally omit
a high-priority small product or leave some work incomplete. This is not proof
that all mission requirements were satisfied, nor an SLA/priority-weighted optimum.

The subset dynamic program retains the earliest feasible finish for each completed
job subset. With no setup state, resources or dependencies in this class, earlier
finish dominates later finish of the same subset: any remaining job can wait until
the later schedule's next start. Every reachable subset is examined in increasing
mask order; ties use stable product-ID order. Integer bit objectives are exact.
Timing arithmetic uses rational representations of the supplied floating calendar
endpoints, rates and release/deadline values, not accumulated binary-float sums.

This proves the optimum for that represented calendar and class. Geometry/root
accuracy, real station behavior and alternative physical models are outside the
proof. The selected order is independently executed by ReferenceDownlinkMission;
its on-time bit value must equal the integer oracle value exactly. Unsupported
sub-nanosecond waits or executor inconsistencies raise an error instead of returning
a certificate. All existing numerical applicability limits still apply.

At most 65536 reachable subset states are allowed. Exceeding an explicit smaller
budget raises an error; no partial run is labeled optimal. Runtime includes normal
reference spec/geometry construction and comparison-policy execution, so the state
budget is not a wall-clock or total-memory guarantee. Result schema is
space-env.reference-calendar-oracle/1 and optimality_proven_in_scope is explicitly
separate from general_resource_optimality=false.

## Fair baseline comparisons

The report includes a matching-class nonpreemptive priority/slack greedy baseline,
its executable plan, outcome, regret_bits and relative_regret. Its chosen job is
scheduled at the earliest feasible interval; reserving a future job can idle the
station even when some other job could run. This baseline is explicitly identified
and is not silently equated with the full project controller.

The actual project's priority-slack-explicit/1 policy is also executed with its
normal defaults. It can preempt or produce partial transfers. Its outcome is
compared to the oracle only when every transmitted product completes on time in
one contiguous transmission interval (resource-grid fragments may be contiguous).
Otherwise comparable=false and regret_bits=null; an example where the preemptive
policy delivers 4 Mbit while the nonpreemptive optimum is only 2 Mbit is tested.
Such an outcome does not contradict the class-limited optimum.

A reproducible counterexample under the throughput objective gives optimum=4 Mbit
versus priority-greedy=3 Mbit. This demonstrates objective-sensitive heuristic
regret; it does not prove that a priority-oriented policy is universally worse for
other industrial objectives. The benchmark does not tune or replace the general
controller and does not claim an independent optimum for energy, DAG, booking,
thermal, multi-station or stochastic problems.

## Portable benchmark and split discipline

`evaluate_calendar_suite(cases, max_states=65536, total_states=262144)` accepts
1–32 cases. Each requires id, lineage, split (train/test), and the complete spec.
Case IDs must be unique. An identical spec hash or declared lineage cannot occur
across train/test splits. These checks enforce declared metadata only; they cannot
establish that different user-declared families are physically independent or
free of near-duplicates, and they do not turn two synthetic variants into a
scientifically qualified held-out benchmark.

The suite applies per-case and combined state budgets (combined maximum 2097152).
It returns per-split optimum, baseline and regret totals, source/runtime metadata,
embedded scenarios and results, and a checksum. It labels expert_demonstrations
false: a bounded optimum is not automatically a qualified expert for all tasks.
No natural-language SFT reasoning or teacher credentials are invented.

`verify_calendar_benchmark(path)` checks integrity/runtime and reruns all oracle,
baseline and physical execution comparisons. Resigning a falsified summary with
a new checksum is insufficient to pass verification. Original suite/spec files
are unnecessary because the report embeds them. Runtime changes require the
matching runtime or a future explicit migration; checksum integrity is not an
authenticated signature. Each output file is atomic, but a plan/report pair is
not a single publication transaction.

## CLI and reproducible example

```bash
python examples/reference_operations/optimal_demo.py /tmp/space-env-calendar
space-env reference-optimal /tmp/space-env-calendar/spec.json \
  /tmp/space-env-calendar/cli-plan.json --report /tmp/space-env-calendar/cli-report.json
space-env reference-run /tmp/space-env-calendar/spec.json \
  /tmp/space-env-calendar/cli-plan.json /tmp/space-env-calendar/cli-run.json --lineage calendar-demo
space-env reference-replay /tmp/space-env-calendar/cli-run.json
space-env reference-verify-benchmark /tmp/space-env-calendar/benchmark.json
# reference-benchmark suite.json report.json also accepts --max-states/--total-states.
```

The demo keeps both correlated deadline variants in the same test split/family;
it makes no held-out generalization claim. They total 9 Mbit optimal versus
6 Mbit matching-class baseline. Reports include the exact objective, scope,
state count, execution evidence and the actual project policy comparison.

Tests compare DP values with an independently written exhaustive permutation
search on eight deterministic five-job instances; verify release/outage/deadline
constraints, scope rejection, nonzero regret, preemption incomparability, budgets,
CLI execution/replay, split checks, deleted-source portability and recomputation
of tampered results. This is bounded scheduler qualification, not a complete
industrial simulator qualification.

The complete repository profile was also executed: 665 passed, one HTTP health
test failed before server startup because the execution sandbox denied socket
creation. The unchanged same test passed separately with loopback socket access.
This is explicitly not one uninterrupted green full run. Day/three-day campaigns
and the sparse 16-actor/72-hour tasking regression passed. The installed wheel's
console entry point space-env, benchmark recomputation and resource/replay smoke
were checked from /tmp; all 168 package Python modules match the source checkout.
The 16-product/65536-state oracle boundary was additionally executed; its measured
runtime is one local observation, not a performance guarantee. Detailed evidence:
[verification record](../acceptance/REFERENCE_CALENDAR_QUALIFICATION_V9_VERIFICATION.json).

Next gates: larger independently sourced scenario families and objectives,
energy/DAG/station-resource optimum bounds, uncertainty and robustness, calibrated
physical validation, data lineage/correlation audits and large-mission endurance.
