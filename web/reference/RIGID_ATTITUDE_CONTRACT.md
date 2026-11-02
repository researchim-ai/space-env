# B04-D2a — standalone rigid attitude kernel

2026-11-01; package 0.134.0. B04-D2 and B04 remain in progress.

## Scope and API

`space_env.reference.rigid_attitude` provides strict, JSON-serializable
`RigidAttitudeModel`/`RigidAttitudeState`, pure `advance_rigid_attitude`,
`invariants`, and `thrust_torque_nm`. This is a separately usable physics kernel.
The opt-in mission bridge is now described in
[RIGID_MISSION_CONTRACT.md](RIGID_MISSION_CONTRACT.md). The kernel API remains
separately usable; its state JSON serialization alone does not prove mission
checkpoint support. The mission bridge preserves B04-D1 held-body thrust for
legacy actors. Rotating burns now require explicit engine coupling; see
[ROTATING_THRUST_CONTRACT.md](ROTATING_THRUST_CONTRACT.md).

Inputs use SI units. Quaternion wxyz maps body to inertial axes; angular velocity,
external torque and three orthogonal ideal wheel momenta use body axes.
Inertia is symmetric positive definite with principal-moment triangle inequality.
Provenance is required but caller-supplied, not independently verified calibration.

## Equations and numerical contract

For effective carrier inertia I, body rate w, ideal axial rotor momentum h and motor torque u:

- `I w_dot = torque_external - u - w cross (I w + h)`;
- `h_dot = u`;
- `q_dot = 0.5 q ⊗ (0,w)`;
- inertial angular momentum is `R(q)(I w+h)`;
- declared kinetic energy is `(w.I.w + sum(h_i²/Jwheel_i))/2`.

Motor mechanical power is `u dot (h/Jwheel - w)`. DC power equals idle power,
absolute mechanical power / efficiency and a declared quadratic torque loss.
Heat is DC power minus mechanical power; braking does not regenerate the bus.
Here h/Jwheel is the absolute axial rotor speed in this ideal reduced model;
relative motor speed is h/Jwheel minus the carrier rate component. I excludes
the axial rotor spin energy represented separately by h²/(2Jwheel). Supplying
a locked-spacecraft inertia without converting it to this convention would
double-count that contribution. These are ideal rotor bookkeeping assumptions,
not a motor controller or a complete rotating-assembly model.
External work integrates external torque dot body rate. No implicit desaturation
or clearing momentum on serialization is performed. `r cross F` computes a torque
only; a caller must apply translation and account for the actual external actuator
and its fuel/electrical costs. This standalone kernel does not supply them.

Constant body-axis input torques are integrated with bounded fixed-step RK4.
Motor torque clips per axis; available momentum limits its average torque within
one substep. Thus saturation timing is resolved to `max_step_sec`, not an exact
continuous event. This can soften the final partial substep near capacity.
Quaternion is renormalized after each substep. Step-budget rejection happens before
integration, and inputs are not mutated. No guaranteed error bound is asserted for
arbitrarily high rates; callers must use step refinement to qualify their regime.

## Validation

21 tests cover analytic spherical rotation, asymmetric free-body energy and
inertial momentum, motor reaction/work/heat balances, both saturation directions,
reversal, off-center force torque, external work, no free unloading/regeneration,
JSON continuation and malformed/nonfinite/out-of-range inputs. The independent
analytic and conservation checks are engineering evidence, not flight validation.

Run `python examples/rigid_attitude/demo.py`: wheel acceleration, saturation,
serialized continuation and unloading with explicit external torque. Source and
installed-wheel results are checked separately. Tests and demos do not imply that
this kernel is connected to mission operations.

## Original integration gates — B04-D2b

B04-D2b1 now implements controller/state/resource/replay gates within its explicit
contract. Rotating/off-center thrust is implemented in B04-D2b2 under its explicit
small-mass-loss contract. A dedicated unloading actuator remains open.

1. Opt-in mission schema/fidelity and actor-local persistent state.
2. Torque-limited attitude controller; pointing/rate settling based on actual state.
3. Wheel commands, saturation and explicit unloading actuator with resource costs.
4. Bus power/SOC/thermal admission before torque is applied; faults/cancel leave spin.
5. Off-center engine moment and rotating thrust integration on one bounded timeline.
6. Preserve past trajectory and conservative future forecasting; reject ideal hold
   shortcuts and incompatible actuator combinations.
7. Runtime ownership, rollback, local observations and mid-rotation mission replay.
8. Installed SDK/CLI mission acceptance; then external reference/calibration.

No gravity-gradient/magnetic/aerodynamic torque, structural flexibility, rotor
imbalance, wheel geometry allocation, friction, sensor/estimator or flight controller
is included. No complete ADCS or B04-D2 completion claim is made.

## B04-D2b3 update

Dedicated resource-accounted balanced RCS unloading is now available under an
explicit opt-in contract: [MOMENTUM_UNLOADING_CONTRACT.md](MOMENTUM_UNLOADING_CONTRACT.md).
This supersedes the earlier absence of a mission unloading actuator. Full
hardware qualification, multiple tanks and variable inertia/COM remain open.
