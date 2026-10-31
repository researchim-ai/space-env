# B04-D1: body-frame thrust contract

Date: 2026-11-01. Version: 0.134.0. Parent B04 and B04-D remain in progress.

## Configuration and execution

Opt in with `definition.actuation` in the reference-spacecraft coupled bus profile.
`engines` maps known actors to explicit chemical/electric engine settings: model ID,
source/provenance, mount unit vector, hardware-limited thrust, gimbal cone/rate,
settling, ignition, auxiliary electrical power, heat and minimum SOC.
Synthetic provenance is not flight calibration. The scenario factory declares
`coupled-gcrs-thrust`; legacy coupled scenarios retain their coast fidelity tag.

`body_burn` takes BODY unit direction, thrust in N, desired firing duration in s
and a bounded command window. Admission changes neither mass nor trajectory.
Execution pays finite gimbal movement, settling and ignition before firing.
Gimbal uses unit-vector spherical interpolation; phase boundaries round upward to
the runtime's 1 ns clock, with 1e-6 degree angular completion tolerance.
Body attitude is held during the command. Actual quaternion and current gimbal
rotate force into GCRS. Only executed intervals enter the dynamic orbit provider.
The forecast beyond those intervals is coast: pending engine work is not predicted.
Geometry revisions invalidate stale bindings after actual thrust integration.
Legacy GCRS impulse/finite_burn commands remain unsupported in this coupled profile.

## Conservation and shared resources

Mass flow is `T / (Isp*g0)`. Scalar ideal rocket delta-v and GCRS impulse are
reported separately; neither is the net orbital velocity change under gravity.
Electric jet power is `T*Isp*g0/2`, DC power is jet power/efficiency plus auxiliary
power. Waste heat includes conversion loss and the declared auxiliary heat fraction;
controller nonthermal energy closes the ledger. Chemical wall heat is an explicit
assumption, not a complete chemical energy budget.

Engine power/SOC is reserved ahead of CPU/radio work. Propulsion and attitude claims
exclude capture, slew and directed radio; managed CPU and omnidirectional radio may
run concurrently within common resource limits. No controller memory IO is invented.
Fuel exhaustion, low SOC, power rejection, safe mode, propulsion/ADCS faults, segment
budget, cancellation and command expiry preserve already paid work and consumed fuel.
Gimbal cancellation retains its actual axis and residual settling requirement.
Global segment admission cannot spend the same remaining segment twice.

Checkpoints preserve gimbal, phases, fuel, geometry and shared bus/provider aliases.
Injected post-integration errors roll back the complete runtime mutation.
Actor observations do not expose another actor's engine state. Nominal covariance
propagation is preserved; this does not calibrate thrust uncertainty.

## Validation and limits

`tests/test_actuation.py` covers analytic mass/rocket balances, body rotation, finite
gimbal, preparation without fuel use, cancellation, faults, exhaustion, power and
thermal limits, simultaneous resource work, checkpoint replay, rollback, ownership,
malformed schemas, covariance and the CPU/gimbal boundary rounding regression.
`examples/actuation/demo.py OUTPUT chemical|electric` executes two EO requests,
processing and delivery around a burn, restoring mid-gimbal and mid-firing.
These are engineering checks, not independent flight qualification.

No rigid-body inertia/Euler integration, reaction wheels, angular momentum,
off-center thrust torque, flexible appendages, plume, startup propellant,
throttle transients or calibrated efficiency curves are represented. Thermal
limits retain the sampled single-node bus assumptions. Segment budgets bound
computation, not mission-scale throughput. B04-D2 must add rigid attitude and
momentum before claiming full propulsion/ADCS coupling.
