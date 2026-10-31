# Transmitter resource health v6

Implemented 2026-10-31 in the separate reference-fixed-downlink/1 backend.
Both controls are opt-in. Existing scenarios without them retain unrestricted
thermal/lifetime operation, but gain telemetry and cumulative TX accounting.

## Model and configuration

```json
{
  "memory_bits": 5000000, "battery_wh": 5, "tx_power_w": 20,
  "rate_bps": 500000, "max_tx_sec": 5,
  "thermal": {
    "ambient_k": 280, "initial_k": 280,
    "capacity_j_per_k": 2, "conductance_w_per_k": 0.2,
    "heat_tx_w": 20, "trip_k": 300, "resume_k": 290
  }
}
```

These fields belong to each vehicle resource configuration. max_tx_sec defaults
to null (no lifetime quota). Zero disables transmission. Thermal defaults to
null (disabled). Temperature/capacity/conductance must be positive, heating
nonnegative, ambient < resume < trip, and heat_tx_w <= electrical tx_power_w.
Derived time constant and heated equilibrium must remain finite. This is a
user-parameterized one-node surrogate, not calibrated spacecraft thermal truth.

The model is C dT/dt = Q - G(T - ambient), with Q=heat_tx_w during actual
transmission and zero otherwise. Each constant-mode interval is advanced with
the analytic exponential solution, using expm1 for small intervals. No passive
solar/albedo/Earth IR heating, radiative T^4 coupling, temperature-dependent
properties, avionics/base heat, antenna setup or multi-node spacecraft geometry
is represented. The ambient reservoir can supply heat when initial T is below
ambient; rejected_heat_j is a signed exchange, not necessarily positive.

At T >= trip, transmission is inhibited until cooling reaches resume. Different
trip/resume temperatures prevent switching chatter. Initially hot transmitters
start locked. Idle cooling continues during outages, lack of data, waiting,
zero throughput, quota exhaustion or service denial. The thermal model does
not consume additional electrical energy: its heat input is a bounded part of
the already charged TX electrical power, not a second power drain.

## Execution, planning and accounting

Actual transmitting duration accumulates per actor, independently of product,
station, calls or episodes' external step partition. max_tx_sec is a cumulative
per-mission budget, not a rolling duty-cycle limit, fatigue law or remaining
useful lifetime estimator. It never resets on cooling or rate changes.

Steps split at product, contact, outage, booking, battery, thermal trip/resume
and quota boundaries. Cooling actors do not consume transmitting terminal
capacity; explicit ground reservations still withhold the owner's capacity
while idle, as required by the booking contract. Only base electrical power is
charged while thermal/quota blocking prevents transmission.

Reports expose TX seconds, remaining quota, nullable temperature/lock/readiness,
heat input, signed rejected heat and the residual:

heat_input - rejected_heat - C * (T - initial_T).

Disabled thermal channels have temperature=null in reports and temperature=0
with thermal_enabled=0 in Gym observations. Locked state uses a 1e-9 K numerical
comparison tolerance. Existing time-step/boundary numerical limits apply; extreme
microsecond thermal constants or extreme throughput are not qualified.

Planning reevaluates at trip/cooling boundaries, hands an open station to other
actors during cooling and checks remaining quota against the candidate's minimum
transmission duration. Explicit product mode can choose a smaller fitting job;
implicit mode retains queue discipline. Thermal feasibility is still executed
through the simulator; there is no global optimality proof or look-ahead battery
optimizer. Other combinations of contention/cooling can remain suboptimal.

Full resource checkpoints include counters, temperature, hysteresis and heat
ledgers. Internal step errors roll them back with inventory and electrical state.
Portable recording, replay and restored continuation reconstruct every resource
transition. Existing runtime/source matching still applies; no migration of
old journals or trained policy observation contracts is implied.

Both Gym facades expose thermal_enabled, thermal_locked, temperature_k,
tx_seconds and tx_quota_remaining_sec. Observation shapes remain fixed within
an episode. Remaining quota is clamped to the mission horizon in observations;
null quota is represented by that horizon. Exact nullable quota remains in the
report. Disabled temperature channels have equal Box bounds, which Gym checker
reports as a warning; checker validation still passes. Rewards remain completed
on-time product Mbit, not temperature, contact or partial-transfer bonuses.

## Reproducible scenario

```bash
python examples/reference_operations/thermal_demo.py /tmp/space-env-thermal
space-env reference-run /tmp/space-env-thermal/spec.json \
  /tmp/space-env-thermal/plan.json /tmp/space-env-thermal/cli-run.json \
  --lineage thermal-reference-demo
space-env reference-replay /tmp/space-env-thermal/cli-run.json
```

The example contains two GCRS vehicles sharing a station, a thermally constrained
0.5 Mbit/s transmitter, hysteretic cooling and a 5-second quota. It verifies 5 Mbit
on-time delivery, four seconds actual transmission, heat-ledger residual, portable
replay and an 18-step successful one-second RL rollout. These numeric parameters
are intentionally synthetic and are not specifications of a real satellite.

Validation includes analytical trip/cooldown times, split-step agreement,
zero/exhausted quotas, electrical and thermal accounting, initially hot hardware,
invalid parameters/derived constants, rollback, quota-aware selection, station
handover, both Gym checkers and recorded replay. Full repository pytest and
large-fleet endurance are separate qualification gates.
