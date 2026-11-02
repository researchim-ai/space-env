# B04-D2b1 — opt-in mission rigid attitude and wheel control

2026-11-01; package 0.134.0. Engineering contract; B04-D2 and B04 remain in progress.

## Configuration and compatibility

Set `definition.spacecraft_bus[actor].attitude.rigid` to enable the reduced
three-wheel rigid kernel for that actor. Other actors/scenarios retain the legacy
rate-limited quaternion model. A mixed constellation advertises
`physics.rigid_attitude.partial_actors`; `commands.wheel_torque` is then available,
but commands for actors without configured wheels are rejected atomically.
The existing coupled-bus-surrogate fidelity label remains an engineering label.

Controller settings contain the explicit `RigidAttitudeModel`, initial body rates
and wheel momenta, constant electrical `supply_w`, PD gains, rate applicability,
rate-settling and payload body-rate tolerances, control interval and lifetime
committed integration-substep budget. Initial quaternion uses the bus's existing
body-to-GCRS wxyz field. All inertia/momentum conventions and reduced-rotor
assumptions in [the kernel contract](RIGID_ATTITUDE_CONTRACT.md) still apply.
No flight calibration or complete ADCS claim is made.

## Commands, control and persistent mechanics

`wheel_torque` takes `wheel_torque: {frame: BODY, motor_torque_nm: [x,y,z]}`.
This is motor torque; its reaction accelerates the carrier in the opposite
sense. Per-axis hardware limits and momentum saturation constrain actual torque.
A parallel command owns the attitude lane, so it conflicts with slew, capture
and directed radio. Managed CPU/omnidirectional radio may run concurrently.
Serial execution retains actor exclusivity.

Existing `slew` becomes a sampled PD wheel controller for configured actors.
It uses the shortest body-frame rotation error, a capped target angular rate,
and the declared wheel authority. It does not teleport the quaternion or stop
rotation when the target is reached. Completion requires sampled angle and rate
tolerances at both interval endpoints over the configured settling period.
The operation records completion time, pointing error and actual angular rate.
After completion there is **no automatic attitude hold**: residual spin coasts.
The rate cap limits the requested target rate, not the physical rate instantaneously.

During capture the controller tracks the moving target; actual predicted
quaternions feed footprint/coverage/quality evidence. Capture additionally checks
a declared maximum absolute body-rate applicability (not a calibrated image-smear
model). Directed radio checks start/mid/end pointing over the candidate interval;
it does not turn the spacecraft automatically. A conservative interval rejection
can discard a shorter usable prefix. Geometry/pointing tests remain sampled.

Cancellation, command expiry, ADCS/bus faults, safe mode and insufficient power
remove motor authority. Angular velocity and wheel momentum remain physical state;
coasting gyrostat motion continues. Fault clearing/recovery does not desaturate
wheels, restore energy or erase spin. Degradation scales target rate and motor
authority. Unpowered wheels are modeled as ideal frictionless rotors.

## Power, heat and numerical scope

While enabled, the controller reserves **and draws the declared constant supply**.
This is an intentionally conservative electrical model, not a measured motor map.
Unused electrical power is dumped as heat. The minimum supply validates against
`idle + sum(torque_limit*(momentum_limit/Jwheel + applicable_rate))/efficiency`
plus the quadratic torque-loss bound. Kernel stages and final state must remain
within the declared rate applicability; violations reject/roll back the advance.

The common scheduler sees the supply before CPU/radio allocation. Bus peak/SOC
admission and reserve boundaries can disable the controller without removing spin.
Mechanical work changes kinetic energy; controller heat equals supplied energy
minus motor mechanical work, including braking heat. Converter losses remain in
the bus ledger. No double charge or automatic regeneration is introduced.
Fixed supply usage is present even for zero requested torque while a control
operation is enabled. No supply is charged during disabled/coasting intervals.

Control ticks are anchored to elapsed mission time to avoid cumulative floating
addition ending a dwell just short of its command boundary. Runtime event/resource
boundaries may subdivide intervals and resample control. Numerical results can
therefore depend on the declared control/resource/event partition; this is not an
assertion of exact partition-independent closed-loop dynamics. The kernel uses
bounded RK4, normalized quaternions and substep-resolution wheel saturation.
PV and bus thermal integration retain the sampled single-node bus assumptions.
The lifetime substep budget is counted for committed motion; previews do not
spend it. Exhaustion fails the mutation atomically, including resource/geometry
and shared bus aliases. It is a computation budget, not a fleet-scale SLA.

## Replay, observations and acceptance

Angular state, wheel momenta, cumulative DC/heat/work, substeps and saturation are
actor-local bus telemetry. `controller.torque` is the requested, authority-capped
motor torque for the last prepared interval; saturation can reduce the actually
applied torque further. Applied effects are reflected in wheel momenta and
capacity-limited time. Truth reports include all configured actors. Existing
runtime journaling reproduces mid-wheel/CPU and mid-tracking checkpoints, retaining
shared bus references; no new unversioned checkpoint format is introduced.

33 tests in `tests/test_rigid_mission.py` cover admission, motor reaction, shared
energy/thermal ledgers, coast/cancel/fault/safe behavior, saturation,
mid-command replay, actor isolation, conflicts, real slew completion,
invalid schemas, SOC/peak limits, degradation, payload-rate and directed-radio
checks, free asymmetric momentum conservation, budgets, rollback and the control
clock/dwell regression. Reversal is also independently covered by kernel tests.

`python examples/rigid_mission/demo.py OUTPUT tuned|weak` executes capture,
concurrent CPU and wheel torque, a mid-operation checkpoint, cancellation,
coast, tracking/capture checkpoint and delivery. Both variants keep the same
request/AOI criteria and payload-rate applicability. Only PD gains differ.
Tuned completes 2/2 requests. Weak delivers the second raw product but fulfills
only 1/2 because AOI coverage is insufficient; delivery alone earns no fulfillment.
These are synthetic controller examples, not a statistically qualified benchmark.

## Explicit next gates

- B04-D2b2 now supports rotating/off-center thrust through explicit engine
  `rigid_coupling`, with a fixed-COM/inertia small-mass-loss limit. Without that
  declaration rigid burns remain rejected. See [ROTATING_THRUST_CONTRACT.md](ROTATING_THRUST_CONTRACT.md).
- No external unloading actuator/command is exposed in missions. Kernel `r cross F`
  and external torque are not free mission actions. An unloading implementation
  must account for real actuator authority, fuel/power/heat and remaining momentum.
- No gravity-gradient/aerodynamic/magnetic disturbance, sensor/estimator, flexible
  appendages, arbitrary wheel layout or flight-controller qualification is added.
- B04-D2/B04 remain open; later B04-E thermal, B05 RF, B09 scaling and B12 industrial
  calibration/pilot acceptance are not replaced by this integration.

## B04-D2b3 update

Dedicated resource-accounted balanced RCS unloading is now available under an
explicit opt-in contract: [MOMENTUM_UNLOADING_CONTRACT.md](MOMENTUM_UNLOADING_CONTRACT.md).
This supersedes the earlier absence of a mission unloading actuator. Full
hardware qualification, multiple tanks and variable inertia/COM remain open.

## B04-E thermal update

An optional multi-node network with routed component heat, orbital radiation,
PV extraction/curtailment and paid thermostatic heaters is now available; see
[THERMAL_NETWORK_CONTRACT.md](THERMAL_NETWORK_CONTRACT.md). The earlier one-node
assumptions above describe legacy actors. Flight/hardware qualification remains
open and the network does not add pressure/exhaust or variable-inertia physics.
