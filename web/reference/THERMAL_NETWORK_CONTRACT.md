# B04-E — multi-node thermal engineering profile

2026-11-02; package 0.134.0. Implementation verified within the declared reduced
model; thermal hardware/flight qualification remains `needs_external_evidence`.

## Configuration and compatibility

Opt in per actor with `spacecraft_bus[actor].thermal_network`. The existing
`thermal` field remains required for compatibility and is used only by legacy
single-node actors. Network actors use their node capacities and limits instead;
no legacy scalar threshold silently overrides a network node threshold.
Capabilities identify partial-actor network physics and autonomous heaters.

Declare model ID, source and synthetic/published/calibrated parameter status;
these are caller provenance, not a validated calibration certificate. Up to 16
nodes have SI heat capacity, initial K, cold/hot trip and recovery limits, optional
linear bath coupling, emitting/solar areas, thermal emissivity, optical
absorptivity and a BODY unit normal. Up to 120 unique undirected conductive links
join distinct known nodes. No unknown/missing heat routes are silently accepted.

Explicit heat-source routing fractions cover base, CPU, payload, radio, ADCS,
propulsion, conversion and PV extraction. Each nonnegative mapping must sum to
one over known nodes. Node/link/route order is canonicalized before integration;
JSON key sorting must not alter numerical state or journal replay. Link signs in
the ledger follow canonical node-name order.

The carrier is a lumped thermal network, not a CAD/FEA mesh. Internal radiative
view-factor links, variable heat capacity, phase change, fluid loops, self-shadowing,
deploying appendages and thermoelastic/sensor response are not represented.

## Equations and numerical contract

For node i:

`C_i*Tdot_i = Qinternal_i + Qexternal_i + sum(G_ij*(T_j-T_i))`
`             - Gbath_i*(T_i-Tbath_i) - emissivity_i*sigma*Aemit_i*T_i^4`.

Symmetric conductive exchange cancels globally. A configured bath is an explicit
infinite-temperature reservoir; its signed exchanged energy is recorded, not
invented electrical heating. Defaults give zero bath conductance. Stefan-Boltzmann
sigma is declared as 5.670374419e-8 SI. Incoming radiation is supplied separately;
the pure kernel does not silently add a background a caller did not prescribe.

SciPy Radau is the default bounded stiff solver; DOP853 is selectable. Both integrate
bath, outgoing radiation and per-link heat-flow integrals alongside temperature.
Node and global residuals compare those independent counters with stored energy.
They are not manufactured by defining rejected energy as input minus storage.
Relative/absolute tolerances, maximum step, per-call/lifetime derivative-evaluation
budgets and event-history limits are explicit. Rejected implicit stages count
against computation budgets. Solver/numerical-budget failure rolls back the current
mission mutation, including heater requests, resources and shared adapter aliases.
Positive finite-temperature applicability is enforced without temperature clamps.

Nonterminal hot/cold crossings are located on accepted-step dense output, so a
recorded excursion remains latched even after cooling. An exactly constant
threshold does not create repeated crossing events. Detection is numerical;
multiple unresolved crossings inside a step are not guaranteed to be complete.

## Sampled orbital radiation

Environment is held at the beginning of each bounded thermal/execution interval,
aligned to elapsed mission time. BODY normals rotate with the actual spacecraft
quaternion. Direct solar absorption uses distance-scaled solar flux, projected
area and the existing eclipse/penumbra model. Body-fixed surfaces do not track the
Sun without actual attitude motion.

Earth IR is prescribed uniform spherical radiance. The projected Earth-disc view
factor has exact fully visible/hidden branches and bounded 12x48 solid-angle
quadrature at the limb. Incoming Earth IR uses the same thermal emissivity as
outgoing IR. Albedo uses prescribed geometric albedo and a Lambert phase law;
the passive-Lambert bound is 2/3. Geographic/diurnal/cloud radiance variation is
not modeled. Sky background is weighted by the remaining view factor, so Earth
and background do not double-cover a hemisphere. The spherical Earth assumption
and partial-disc quadrature are engineering approximations, not ray tracing.

These radiative inputs are external reservoirs; neither an Earth energy-balance
model nor a calibrated spectral/radiometric detector model is asserted.

## Electrical/thermal accounting

Routing uses actual native transmitter/processor service time, managed CPU/camera
work and current propulsion/ADCS plans. SAR amplifier heat excludes emitted RF
and adds declared support-electronics heat. Rigid motor mechanical work remains
separate; braking increases heat without automatic electrical regeneration.
Chemical wall heat and electric conversion loss retain their declared engine
assumptions. Conversion losses are charged to the declared conversion route.

The existing bus generic electrical heat fraction remains explicit. A DC ledger
separates electrical input, heat, wheel work, external wall heat and declared
nonthermal export. That export includes known RF/jet outputs and unresolved
nonthermal fractions of the surrogate; it is not a measurement or full exhaust
thermodynamic closure. Use/calibrate appropriate fractions rather than interpreting
all power consumption as heat by accident.

PV electrical extraction is a negative panel heat source. Its surface must match
the electrical panel normal and have enough absorbed solar capacity; runtime also
checks local sampled flux. Raw extracted power, converter losses and post-converter
curtailment/dump heat are accounted exactly once. In this profile curtailment is
thermalized at the conversion route; an open-circuit controller with panel-local
curtailment needs a different calibrated topology/model. Charge/discharge/dump
share one declared routing map. Tests close combined battery plus thermal energy
for both charging and full-battery curtailment cases.

## Paid heaters and protection

Optional thermostats have on/off hysteresis, node-specific power and deterministic
priority. Desired heat and admitted power are separate telemetry. Power is reserved
as actual baseline DC demand, constrained by bus peak and minimum SOC. Managed
work and native CPU/radio respect the heater SOC floor. Valve/engine/ADCS power
cannot silently spend the same peak headroom. A bus fault disables heaters.
Heaters can warm a safe/cold actor when sufficient power is available; clearing
safe mode remains an explicit recovery action after **all** nodes meet hysteresis.
Temperature, fuel, momentum and battery are never reset by recovery.

Heater power is assumed to become local heat. Converter losses remain separately
routed. Heater DC and per-node delivered heater energy are cumulative counters.
Heater switching is sampled and can overshoot a threshold within one configured
interval; this is not a zero-latency hardware thermostat or damaged-heater model.

Any critical node crossing latches actor-wide thermal safe mode. This profile has
conservative global protection rather than subsystem-specific thermal masks.
Legacy `temperature_k` summarizes the maximum node value; cold protection and
recovery use the full node map, never that summary alone. Safe-mode survival slew
is not allowed while network thermal recovery conditions remain unsatisfied.

Resources/torques already executed in the interval remain paid; hardware shutdown
occurs at the next execution boundary. Capture/CPU materialization is rejected
when a crossing occurred in its finishing interval, preventing false product
credit. Existing reliable fixed-rate radio can finish an interval before feedback;
this is not continuous RF/temperature failure modeling. Heat, excursions and
partial work are preserved for operator analysis/replanning.

## Evidence and remaining gates

54 new tests cover two-node analytic exchange with both solvers, bath RC,
nonlinear cold-space radiation and equilibrium, stiff conduction, transient and
constant-threshold events, kernel budgets/strict topology, Earth-disc geometry,
actual capture/CPU/radio/SAR/engine/RCS heat, braking, explicit nonthermal export,
PV plus battery/curtailment conservation, night IR, body orientation, paid cold
recovery/heater priority/SOC/peak/fault behavior, per-node protection, canonical
ordering, cold-start and mid-operation replay, event/lifetime budgets and rollback.

Run `python examples/thermal_network/demo.py OUTPUT nominal|hot|cold`.
Nominal and explicitly recovered cold-start cases fulfill 2/2 EO requests. Hot
CPU latches protection and cannot materialize/deliver the final product (0/2);
the demo verifies this expected failure, not a successful hot mission. Checkpoints
cover paid warmup, mid-capture and mid-CPU state. Source and installed SDK/CLI
acceptance are recorded separately.

This closes the declared B04-E engineering package. B04 as a whole is still open:
continuous sensor/radiometric completeness, stochastic uncertainty, calibrated
component models and B12 external pilot acceptance remain gates. Next B05 covers
RF/link/ground competition. CAD/contact/material/radiation calibration, detailed
EPS/pressure/thermal loops and independent mission-scale B09 performance evidence
must not be inferred from these tests or synthetic presets.
