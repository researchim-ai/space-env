# B04-D2b2 — rotating and off-center thrust in the mission runtime

2026-11-01, version 0.134.0. Engineering scope; parent B04-D2/B04 remain open.

## Explicit opt-in and applicability

On a rigid actor, configure `actuation.engines[actor].rigid_coupling` with
`lever_arm_body_m` (engine position relative to fixed COM),
`max_mass_loss_fraction` (default 0.001, maximum 0.05) and
`max_attitude_step_rad` (default 0.25, maximum 1). The lever arm is a finite BODY
3-vector, bounded to 100 m. A coupling on a non-rigid actor is rejected; a rigid
body burn without explicit coupling is also rejected. Legacy held-body burns are
unchanged. `physics.rotating_body_thrust.partial_actors` advertises availability.

The declared carrier/wheel inertia and COM stay fixed. This is a **small mass-loss
approximation**, not variable-inertia rocket dynamics. Cumulative mission mass
loss is bounded relative to initial engine actor mass. Execution cuts at that
limit and fails subsequent firing with `constant_inertia_mass_limit`; consumed
fuel, orbit and spin are preserved. No extrapolation silently updates inertia.

The applicable rate limit times the smaller of kernel maximum step and control
interval must not exceed the declared angular-step limit. This prevents obvious
angular undersampling/aliasing; it does not prove a global numerical error bound.
Rate limits are also checked at rigid RK stages/final state. Force-knot histories
are finite, ordered and unit-normalized, with bounded count and maximum implied
integration work. Unsupported regimes reject the mutation.

## One causal execution interval

Only actual firing creates force. Gimbal motion, settling, ignition and attitude
wait remain paid causal phases; they do not preapply mass, force or engine torque.
During firing, the current gimbal direction defines BODY force `F_b=T*d_b` and
engine moment `tau_b=lever_arm cross F_b`. The same constant BODY moment drives
rigid RK4 motion, including existing spin and stored wheel momentum.

A burn owns propulsion and attitude. It does **not** enable an implicit wheel
hold controller or allow a concurrent wheel/slew/capture/directed-radio command.
Managed CPU/omnidirectional radio may coexist within declared common resources.
The body rotates freely under the engine moment and gyrostat dynamics; even a
centerline engine has a changing inertial force when the body already spins.

For each committed firing interval, rigid RK4 supplies quaternion knots. The
sealed `RotatingThrust` law interpolates adjacent knots with shortest SLERP and
rotates the BODY force into GCRS **at orbital RHS evaluation time**. Orbit DOP853
steps are limited by the smallest attitude-knot interval. The force is not frozen
at the left endpoint. Constant-flow mass remains analytic `T/(Isp*g0)`.

Linear impulse is integrated with four-point Gauss quadrature per SLERP interval;
`direction_gcrs` in interval telemetry is the initial direction, not an average.
`impulse_vector_gcrs_ns`, final direction, knot count, BODY moment and external
angular impulse distinguish the executed rotating law. A complete constant-rate
revolution can have near-zero net impulse despite positive consumed propellant.
Ideal scalar rocket delta-v is still not net orbital velocity change.

Only executed intervals replace the dynamic orbit arc. Past history remains
unchanged, geometry revisions invalidate old bindings, and future translation is
coast beyond executed intervals. Future pending burns are not predicted. Arc and
maneuver manifests seal the attitude path; nominal covariance validation compares
the complete thrust manifest, including knots, not just the first direction.

## Ledgers and failure behavior

Rigid integration separately integrates inertial external angular impulse and
`tau_body dot omega_body` external mechanical work. Telemetry checks
`H_inertial-H_initial-external_angular_impulse` and kinetic energy minus initial
energy, motor work and external work. Quaternion normalization and force-knot
interpolation introduce numerical errors that must be qualified by refinement.

Engine fuel/DC/heat/jet ledgers and common bus admission remain active for chemical
and electric engines. The external mechanical-work ledger is not an added DC
charge: imposed thrust exchanges momentum/energy with expelled mass. Existing
jet power is an ideal vehicle-relative assumption; this is **not a complete
rotating exhaust/propellant energy or momentum-flux model**. Chemical wall heat
remains declared; COM/inertia changes, gimbal moving mass and reaction torques are
not represented. Constant-inertia and massless-gimbal applicability are explicit.

Cancellation, faults, safe mode, low SOC, dry mass, phase/window expiry and segment
budgets remove further engine force **and moment**. Spent fuel and acquired spin
remain. A lifetime substep or numerical applicability violation rejects/rolls back
the entire current mutation; it does not fabricate a paid partial success.
Shared provider/bus aliases, mass, geometry, angular state and resources participate
in the existing atomic rollback and journal replay.

## Evidence and remaining work

36 new tests cover analytic rotating force/impulse, full-revolution cancellation,
independent central-gravity/analytic-angle orbital RHS and step refinement,
off-center Euler acceleration/work, noncommuting spin/moment conservation,
ignition without free thrust, chemical/electric energy, fuel and SOC exhaustion,
segment budget without ghost torque, faults/safe/cancel, fixed-inertia mass limit,
strict coupling/path inputs, angular alias guard, history preservation, nominal
covariance PSD and rejection of a tampered attitude path, ownership, rollback and
mid-burn replay. The fine-step velocity error must improve by >20x over the coarse
step and stay below the declared engineering threshold; this is not flight truth.

`python examples/rotating_thrust/demo.py OUTPUT chemical|electric` executes two
EO requests with CPU/radio work around a gimballed, off-center burn; checkpoints
restore mid-gimbal and mid-firing. Both synthetic variants fulfill 2/2 requests.
Source and installed wheel SDK/CLI evidence belong in the acceptance JSON.

Next B04-D2b3: explicit resource-accounted momentum unloading and controlled-burn
actuator allocation, with external authority/fuel/DC/heat and residual momentum.
No free mission external torque command is exposed by this change. Also open:
variable COM/inertia and exhaust qualification, disturbances, sensor/estimator,
flexible body, calibrated hardware, B04-E thermal, B05 RF, B09 scale and B12 pilot.

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
