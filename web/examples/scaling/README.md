# B09: bounded, measurable scale

Generate a frozen small workload manifest and run it through installed SDK/CLI:

```bash
python examples/scaling/demo.py /tmp/scale.json
space-env scale-benchmark /tmp/scale.json /tmp/scale-result
```

`--full` adds fixed-ephemeris fanout at 1/4/16/64 actors, paired full/compact
three-actor EO→CPU→DTN→ground-ACK and loss/TTL witnesses. These are synthetic,
short engineering probes. Identical ephemerides in fanout deliberately exercise
resource/action/public-telemetry overhead; they do not represent separated orbital
planes or qualify a 64-spacecraft operational mission. Loss/TTL retain incomplete
missions with declared completion thresholds, rather than resampling successes.

`manifest.json` is frozen before execution; one fresh interpreter runs each
workload/repeat. Results include all failures, hardware/thread environment,
p50/p95 full task-step latency, fork tracemalloc peak, checkpoint bytes, cold
replay, native conservation/ACK gates, verified RL+SFT export throughput and
fresh-process lifetime peak RSS. RSS includes replay/export and interpreter
startup; it is not episode allocation delta. Budgets are predeclared. Performance
numbers depend on hardware, concurrent work, build and declared workload. Hashes
of native reports/traces allow semantic comparison independent of timing.

`FactorySpec.workers` uses spawned one-episode processes with bounded pending
work, deterministic output order and no hidden retries. Keep Python callers
under `if __name__ == '__main__':`. For independently generated ready bundles:

```bash
space-env factory-register /tmp/corpus /tmp/corpus-registry.sqlite
```

This verifies the bundle through replay and atomically registers geometry groups
and actual exported trajectories. Same-content registration is idempotent;
cross-split geometry/trajectory contamination and repeated trajectories across
new bundles are rejected. Registry is local and explicit, outside the immutable
bundle. It is not a distributed dataset service or an automatic merge command.

Parallel `workers>1` execution requires Python 3.11+ (fresh-process recycling);
Python 3.10 supports `workers=1` and receives an explicit validation error for
parallel requests. Actual Python 3.10 worker execution is not claimed.

## Distinct-orbit fleet loads (B09-Q)

```bash
python examples/scaling/fleet.py /tmp/fleet-16-100.json
space-env scale-benchmark /tmp/fleet-16-100.json /tmp/fleet-16-100
python examples/scaling/fleet.py /tmp/fleet-64-1000.json --actors 64 --products 1000
space-env scale-benchmark /tmp/fleet-64-1000.json /tmp/fleet-64-1000
```

`FleetWorkloadSpec` generates distinct initial states with different altitudes and
phases in several orbital planes. Each arc is separately integrated with J2 and
DOP853 in GCRS. Ground stations are initially aligned to the orbital planes;
products already reside onboard and use explicit actor-specific downlink actions.
This is a synthetic fixed-ephemeris operations workload. It does not exercise
controlled maneuvers, imaging, CPU, DTN, RF or coupled attitude/thermal physics.
Groups of up to eight actors share one controller role with eight individual
command slots, inside existing task limits; commands are not replaced by canned
batch plans. A 64/1000 run still contains 1000 independent product commands.

Default two repeats execute in separate processes. Child thread variables are
set explicitly (`worker_threads`, default 1), and isolation/user-site flags are
preserved. Workers report affinity, cgroup CPU/memory limits when available and
CPU model. `max_step_p95_sec` and `max_execution_sec` are optional predeclared
gates. Summaries retain failed repeats in their denominators; timing quantiles
explicitly cover successful repeats only. Unequal native/trace hashes across
successful identical repeats fail aggregate readiness. Two samples are only a
repeatability smoke, not a statistically reliable tail-latency estimate.

`--outage` blocks all stations and retains the zero-delivery negative case with
an explicit completion threshold of zero. Native replay, accounting, latency,
checkpoint and memory gates still apply. Do not mix this execution acceptance
with nominal mission fulfillment. The generator rejects >64 actors and >1023
products: scaling to 256/10000 needs new qualified contracts, not a silent limit
increase. Long arcs may leave ground visibility; defaults do not imply global
continuous coverage or feasibility for arbitrary horizons/product sizes.

A longer fixed-ephemeris resource arc can be declared explicitly:

```bash
python examples/scaling/fleet.py /tmp/fleet-hour.json --actors 4 --products 4 --planes 2 --horizon-sec 3600 --decision-sec 300
space-env scale-benchmark /tmp/fleet-hour.json /tmp/fleet-hour
```

Here all four onboard products are available at the start. This tests a one-hour
orbit/resource arc and replay, not sustained arrival/scheduling or a long coupled
EO/CPU/DTN mission. Native resource and physical sampling grids are unchanged.
The generated priority policy explicitly admits up to 64 candidates for 64 actors;
its former default of 32 left 544/1000 tasks incomplete in 34 seconds. That failed
profile is preserved separately from the corrected, predeclared policy budget.
