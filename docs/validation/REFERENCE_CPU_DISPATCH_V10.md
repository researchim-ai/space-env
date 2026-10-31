# Explicit onboard CPU dispatch v10

Implemented 2026-10-31 for the separate reference operational backend. This adds
agent/operator control over which in-place processing job runs. It does not
implement scientific processing algorithms, a learned CPU scheduler, multi-core
execution, or a globally optimal joint compute/radio controller.

## Mission command

`mission.step(dt, actions, products=..., processing=..., processing_jobs={actor: product_id})`
accepts an optional per-actor CPU selection. Omitted actors retain the automatic
priority/deadline/ID queue. Selection is per external step, not persistent across
calls. It may preempt a partially processed job without resetting its progress;
the existing zero-overhead preemption/scratch-memory model remains in force.

A selected product must exist and belong to the given actor. Combining a selected
job with processing=false for the same actor is rejected before state changes.
Unknown ownership/IDs and non-string selections are also rejected before time,
energy or work advances. A structurally valid selection may be temporarily
ineligible: it waits for readiness, prerequisites, workspace and battery. It does
not silently run a different job. A completed/cancelled selected job also waits
on CPU; no automatic fallback occurs after finishing within the external step.

CPU waiting does not reserve the radio: eligible processed data can transmit if
the requested radio action permits it. A running processor still excludes that
actor's transmitter under the existing serial hardware model. Other actors may
work or transmit concurrently. All thermal/quota/booking/retention rules remain.

`processing_status(product_id, generation_w=0)` returns eligible, reason and the
explicit generation_sample_w used for assessment. Reasons are eligible, cancelled,
completed, not_ready, dependency, memory or power. This is current CPU admission,
not a prediction of deadline feasibility. The first applicable reason is reported.
Generation must be finite/nonnegative; below the configured SoC floor, even a
nonnegative net power sample does not allow immediate work until charge recovers.
Product reports include processing_admission using a conservative zero-generation
sample. The executor still uses its actual configured solar-grid sample: report
admission can therefore differ from physical execution at a solar-supported floor.

## CLI, recording and replay

```json
{"steps": [
  {"dt_sec": 2, "actions": {"a": "Ground"}, "processing_jobs": {"a": "short"}},
  {"dt_sec": 11, "actions": {"a": "Ground"}, "processing_jobs": {"a": "long"}}
]}
```

reference-run and reference-resume accept the new key. ReferenceOperationsRecorder
records it and replay validates both requested selection and actual intervals,
progress, completion, energy and receipts. The optional key is additive to the
current journal transition schema. Existing no-selection plans retain automatic
CPU behavior. Runtime/source matching remains strict; old journals require the
matching runtime or explicit future migration. Default planner behavior remains
automatic CPU scheduling; this is a new controller interface, not an optimum proof.
State-bound plans from an older report contract may also require regeneration.

## Gym action contract

Both ReferenceDownlinkEnv and ReferenceEventDownlinkEnv accept
cpu_job_selection=True. In this mode action_space is Dict with radio and cpu
MultiDiscrete components, preserving the existing radio encoding. Per actor:

- CPU 0: automatic queue.
- CPU 1: pause processing.
- CPU 2+j: select owned product j (base environment) or local slot j (event env).

Base reset info.cpu_product_ids provides the owner-specific order. Event info's
existing product_ids map supplies current slot IDs. Both expose cpu_job_selection
and cpu_commands labels. No future event-tape job is exposed early. An empty
CPU slot in the event environment pauses that actor's CPU for the step and lists
it in invalid_cpu_actors, without changing the requested radio action. Once a
product arrives in that slot, the same selector can address it.

cpu_control=True and cpu_job_selection=True are mutually exclusive because the
older combined radio/pause action blocks would otherwise conflict with the CPU
component. With both options false, the original MultiDiscrete action shape and
automatic behavior are unchanged. Saved policies must opt into the correct action
contract. Both modes pass the Gym checker; existing trajectory recording supports
the Dict actions without flattening away their radio/CPU structure. A scripted
rollout is not evidence of a trained policy or an expert SFT teacher.

## Reproduce

```bash
python examples/reference_operations/cpu_selection_demo.py /tmp/space-env-cpu-choice
space-env reference-replay /tmp/space-env-cpu-choice/run.json
```

The synthetic case has one long high-priority compute job and one short job with
a tight deadline. Automatic CPU priority serves 1 Mbit on time; explicit selection
of the short job followed by the long job serves 2 Mbit, without relabeling
priorities or changing hardware. The example records actual CPU selections,
verifies portable replay, and generates a successful Dict-action RL trajectory.
This is an objective-specific counterexample, not a general scheduler benchmark.

Tests cover ownership/conflicting controls and atomic rejection, no fallthrough,
readiness/dependency waiting, admission diagnostics, dynamic slots, both Gym
checkers, Dict trajectory JSON, and CLI recording/restored continuation. Changes
are checked by reference/fast profiles and installed-wheel smoke. The previous
v9 full regression remains separately documented; a new full run is not implied.

Remaining gates include job-specific processing algorithms, persistent scratch,
setup/switching overhead, multi-core/concurrent hardware, uncertainty, and optimal
joint CPU/radio scheduling. The new dispatch interface makes those policies
expressible but does not qualify them by itself.
