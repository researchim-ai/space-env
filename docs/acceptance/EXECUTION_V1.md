# Execution-integrity implementation

Date: 30 October 2026. Base: `1e5a9f6`, package 0.134.0; uncommitted development.
This is an implementation increment, not completion of the industrial roadmap.

## Implemented

- Timed commands split an external world step at operation-completion boundaries.
  Completion turns off the transmitter/imaging before the remaining interval.
  An exception after a boundary restores the whole external step and command state.
- World snapshots include truth history, task registry, pending batch, rejection state
  and registered sensor channel RNG/cache. Internal rollback stores history lengths
  instead of copying the growing tape on every simulation tick.
- Engine snapshots now also restore the complete subsystem state, telemetry counters,
  cached passes and numerical encounter (including its propagation context).
- Command-book rollback uses copy-on-write records and a shallow internal checkpoint;
  public exports remain detached. A regression covers all lifecycle mutations.
- Running operations and timed windows are indexed independently of completed
  history. Indexes rebuild on checkpoint/JSON restore; JSON bodies are normalized
  so repeating an ID after loading remains idempotent.
- Session failure restores the complete world checkpoint instead of only the command
  book. Cancelling through MissionTools now stops the underlying running operation.
- ParallelFleet restores its initial world on reset, seeds action/simulation RNG,
  validates actions before mutation, clips the final step and clears agents on done.
- RL JSON preserves nested observations/actions and includes schema/seed fields.
  Random policies seed the action space. JSONL writing rejects NaN/Infinity.
  Schema2 preserves array dimensions; consumers that assumed flattened matrices
  must explicitly adapt. Existing one-dimensional Box observations stay lists.
- CI full profile discovers every test file, rather than naming two smoke modules.
  Contract tests run in fast CI; full regression also runs on schedule/manual trigger.
- Scheduled shift executors now charge during long idle periods using ordinary
  `point_sun` commands and reserve time for the return to nadir. This applies to
  DayShift, DishAward and OperatorShift; slew and battery dynamics remain active.
  The tasking `priority` policy is separate and is not silently replaced.

## Regression evidence

`tests/test_execution_integrity.py` adds operation-boundary/dt, rollback, observation
cache/RNG, parallel lifecycle, Dict JSON and invalid-duration cases. Before changes:
9 failed, 1 passed. After changes, the combined integrity/continuity/finale/awareness/
data-generator selection passed29 tests; the broader contract selection passed44.
The selections overlap; these are not73 distinct tests.

Ruff passes on changed modules/tests excluding existing legacy engine/campaign findings;
the engine has seven pre-existing findings (the original file had those plus import
ordering), campaign has four class-metadata findings. No new findings were introduced
there. Compileall passes.
HTTP health test passed separately
outside the socket-restricted sandbox (1 passed,1.28s). Regression and wheel
verification evidence is recorded below.

## Day-mission rebaseline: explicit model change, not repaired planner

The legacy day assertion required at least30/32 deliveries. With the already shipped
sampled nadir-panel cosine, the priority policy depletes batteries and delivers18/32.
The implementation does not change solar generation or alter the scenario.

Controlled diagnostics on the same scenario,120-second policy step:

- Actual sampled cosine:18 deliveries. For sentinel-a, SOC is0 near23910,29480,
  40040 and77380 seconds; other vehicles also encounter depleted periods.
- Counterfactual forcing panel cosine to1 only in an isolated diagnostic process:
  31 deliveries, only req-15 unmet. This is not a valid new spacecraft model and is
  not a product change; it isolates the effect of the generation assumption.

The regression now freezes the exact18 delivered request IDs and all32 terminal
outcomes. It does not claim that18 is optimal, that all14 failures have one cause,
or that the policy meets an industrial service target. A planner that schedules
charging/pointing and correct destination delivery remains a separate requirement.

## Remaining gates

- Unified boundaries for all request/outage/processing/fault events and causally
  settled relay remain broader work; this increment guarantees timed-command ends.
- Portable snapshots still depend on the scenario path; complete configuration,
  journal continuation and cross-process KPI parity need qualification. Encounter
  and pass-cache state are now saved, but not independently qualified for mission use.
- ParallelFleet is not yet certified against the official PettingZoo test suite.
- Frame migration and long-arc fidelity remain open; see
  [FRAME_MIGRATION.md](../validation/FRAME_MIGRATION.md).
- No competitor benchmark, real customer pilot, hosted-service hardening or flight
  qualification is implied. No commit or push is performed.

## Legacy shift regression investigation

The full regression exposed failures in DayShift/DishAward/OperatorShift. DayShift
and DishAward were independently reproduced on the unmodified base checkout
(`1e5a9f6`,2 failed in84.60s); these were not introduced by snapshot changes.
The shift idle executor previously coasted in nadir indefinitely. It now performs
explicit sun-pointing and reserves a worst-case180-degree return interval before
the job. No battery refill, solar-power multiplier or deadline change is used.
Short gaps keep nadir. `tests/test_power_coast.py` checks real charge improvement,
return settling and unchanged job time. Tasking's energy-limited18/32 baseline
remains separate from the scheduled-shift controller.

## Final verification

The expanded integrity/contract selection passed61 tests (9.97s); the final index/
JSON/timed-interface and charging regressions passed23 tests (1.91s). Selections
overlap and are not added as independent coverage totals. Scheduled-shift regression
passed7 tests (7 deselected,250.80s); console/API regression passed8 (1 deselected,
181.15s). The unchanged base DayShift/DishAward comparison failed both tests.

Fresh wheel installed without system site packages in `/tmp/space-env-execution-clean`.
Verification ran from `/tmp` with Python `-I`; imports resolve to this environment's
site-packages. Passed: Dict rollout, all four smoke cases, Studio handler, JSON
snapshot with stations and CLI validate. This is not browser interaction or the full
test suite against the installed wheel.

Resolved versions: numpy2.5.4, boinor0.20.0, gymnasium1.4.0, astropy8.0.2,
pydantic2.14.2. Artifact details and regression status are recorded in
`EXECUTION_V1_VERIFICATION.json`. The initial broad run completed347 of370 tests
(341 passed,6 failed) before interruption to apply and verify fixes. All six failed
nodes subsequently passed in targeted reruns, including the socket test outside
the restricted sandbox. Current collection contains381 tests. The remaining
long-horizon selection passed18 tests in477.57s, including16 spacecraft over72h
(460.51s). Vehicle capabilities passed separately (5 tests). All381 collected
node IDs were exercised across these overlapping runs and successful reruns;
the initial broad run preceded final fixes. This is not a claim of one
uninterrupted green full-suite run on final source. All145 Python modules in the final wheel
match the working-tree source byte for byte.
