# B04-D2b3 — wheel/RCS momentum unloading

2026-11-02, package 0.134.0. Engineering implementation complete for the declared
ideal profile. B04-D2 engineering packages a/b1/b2/b3 are delivered; independent
hardware/flight qualification is still `needs_external_evidence`. B04 remains open.

## Explicit hardware model and command

Opt in with `definition.unloading.actors[actor]` on a rigid spacecraft. Declare
model ID, synthetic/published/calibrated parameter status, source, positive
per-axis `max_torque_nm`, effective `lever_arm_m`, total scalar nozzle-thrust cap,
Isp, startup and valve power, electrical heat fraction, wall heat, minimum SOC,
and constant-inertia cumulative mass-loss applicability. The required
`propellant_pool: shared-lumped` explicitly acknowledges a single shared reservoir
with the main engine; separate propellants/tanks/pressurant are not modeled.
Source/status are caller declarations, not an automatic calibration certificate.

The geometry is six signed BODY torque channels, each an ideal opposed nozzle
pair (twelve ideal nozzles total). Opposite forces cancel, their moments add.
Positive and negative channels represent different nozzle pairs, not reversible
unphysical thrusters. The three effective arms must be in [0.001,100] m and torque
limits in (0,1000] Nm. Total nozzle thrust is additionally bounded. There is no
minimum impulse bit, pulse modulation, line pressure transient or misalignment.

`momentum_unload` requires `momentum_unload: {frame: BODY,
target_momentum_nms: [x,y,z], tolerance_nms: value}`. Target defaults to zero;
its components must fit wheel capacity, and positive tolerance must be smaller
than the smallest capacity. Admission stores the current quaternion as the
hold target; it changes no momentum, mass or resources.

A command owns attitude and propulsion, excluding simultaneous primary burn,
slew, wheel command, capture and directed radio on the same actor. Managed CPU
and omnidirectional radio can run concurrently under common resource admission.
The new command is available only with declared RCS; other actors are rejected.

Command completion means meeting the command's requested setpoint, not an
independent mission goal. Benchmarks must compare actual residuals against an
immutable task target/tolerance; an agent-chosen easier target must not earn
credit. The current built-in `fulfilled_requests` objective remains the EO
objective. The example fixes the unload target at zero and independently checks
the expected operation outcome; limited-case EO success does not erase its
unloading failure.

## Allocation and mechanics

After paid startup, a sampled controller computes body hold torque from quaternion
error, damping and gyrostat feedforward. Holding takes priority in a conservative
per-axis authority allocation. Remaining RCS authority and wheel torque limits
bound the requested motor momentum change. External torque equals allocated hold
torque plus motor torque. If the total nozzle-thrust limit binds, both commands
are scaled together. Degradation scales wheel and RCS authority. This is a
bounded deterministic allocator, not globally optimal thruster allocation.

Actual applied torques enter the same rigid RK4 as the rest of mission attitude:
`I*w_dot = torque_external - torque_motor - w cross (I*w+h)`;
`h_dot = torque_motor`. Limited authority can leave attitude/rate error and target
momentum. No state variable is assigned its desired value. Completion requires
momentum, rate and pointing tolerances at both sampled interval endpoints over
the configured settling dwell. A missed window fails with actual residuals.
No automatic attitude hold persists after completion/cancellation.

For each active axis with arm r, two opposed jets produce `tau=2*r*Fjet`.
The sum of nozzle thrust magnitudes is therefore `sum(abs(tau_i)/r_i)`, and
mass flow is that scalar divided by `Isp*g0`. `BalancedCoupleThrust` advances
this analytic mass flow through the orbit provider with **zero net force**.
There is no artificial inertial delta-v. Mass-dependent drag/SRP, if configured,
still use the updated mass. Manifests explicitly distinguish scalar nozzle thrust
from net force. Fuel-changing intervals update geometry/binding revisions.

The shared provider dry mass and each declared fixed-COM/inertia loss fraction
truncate actual firing. Primary and RCS segment allocation share the global
maneuver-history budget, including simultaneous actors; primary reservations
have deterministic priority. Both subsystem fuel ledgers subtract the other's
recorded consumption when checking the shared reservoir. No double-spent segment
or incorrectly attributed shared-tank mass loss is accepted.

## Resources, health and numerical limits

Startup draws declared RCS startup power without moving wheels or firing jets.
Control draws the existing conservative constant wheel-controller supply. Valve
power is drawn while nozzle thrust is nonzero. RCS electrical and declared wall
heat feed the existing bus node; wheel braking heat includes released mechanical
energy, without automatic electrical regeneration. RCS operation `dc_energy_j`
is the RCS startup/valve cost only. Wheel DC cost is separately accounted by the
rigid controller; for this fixed-supply model it equals `control_sec*supply_w`.
Common bus totals include both exactly once. RCS wall/pressure/exhaust chemistry
is an imposed assumption, not complete propulsion thermodynamics.

Peak power and the stricter RCS/wheel SOC floor gate and truncate execution.
Faults in bus, ADCS or propulsion, safe mode, thermal trip, fuel/scope exhaustion,
segment limits, cancellation and window expiry stop future actuation while
preserving past paid work, fuel, heat and residual spin/momentum. Fault clearing
and recovery do not resume a failed operation or refill resources. Restarting a
cancelled startup pays startup again. The inertia/COM and ideal frictionless rotor
assumptions remain; there is no free mission external-torque action.

The rate-applicability bound times the smaller kernel/control step must be at
most 0.25 rad. Rigid stage/final rate checks and lifetime substep budgets remain.
The bus thermal/PV models and pointing checks remain sampled. Numerical failures
and mutation errors roll back the entire current transaction, including mass,
angular state, resource counters, geometry and shared provider/bus aliases.
Previously committed work is preserved. Pending commands are not forecast as
already executed; beyond executed mass-flow intervals, the provider coasts.

## Verification and scope closure

43 mission tests cover admission, startup, signed/multi-axis analytic unloading,
residual targets, exact ideal pair mass flow and zero-force orbit comparison,
linear/angular/energy ledgers, limited windows, cancellation, fault/safe/recovery,
per-axis and total nozzle caps, degradation, peak/SOC/thermal limits, dry mass and
constant-inertia scope, local/global segment budgets with a primary engine,
bidirectional shared-tank accounting, nominal covariance, concurrent CPU/radio,
actor isolation, strict schema/parameters, undersampling, quaternion normalization,
checkpoint replay and injected rollback. A concurrent-work case intentionally
fails completion when wheel momentum is reduced but carrier rate has not settled.

`python examples/momentum_unloading/demo.py OUTPUT nominal|limited` restores a
checkpoint during unloading, then executes EO capture/CPU/radio and a tracking
checkpoint. Nominal unloading completes. Limited authority deliberately misses
the unload window and retains momentum; this failure is reported even though
both EO requests can still be fulfilled. Demo success means the declared expected
outcome was verified, not that the limited unloading command succeeded.

These checks close the bounded implementation package, not physical qualification.
Open industrial gates include variable COM/inertia, multiple propellant pools,
pulse/nozzle/pressure calibration, disturbance torques, sensor/estimator and
arbitrary actuator layouts. Subsequent delivery is B04-E multi-node thermal, B05
RF and B09 mission-scale qualification; B12 requires actual pilot/calibration data.

## B04-E thermal update

An optional multi-node network with routed component heat, orbital radiation,
PV extraction/curtailment and paid thermostatic heaters is now available; see
[THERMAL_NETWORK_CONTRACT.md](THERMAL_NETWORK_CONTRACT.md). The earlier one-node
assumptions above describe legacy actors. Flight/hardware qualification remains
open and the network does not add pressure/exhaust or variable-inertia physics.
