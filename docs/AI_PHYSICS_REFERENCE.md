# Independent analytic physics reference suite

Run locally from the repository root using the project's installed dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/validation/ai_reference_suite.py --output /tmp/ai-physics-reference.json
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_ai_reference_suite.py
```

The example prints JSON, optionally writes the same JSON to `--output`, and exits
0 for a pass or 1 for a failed comparison. Unexpected execution errors propagate
and produce a nonzero exit. It requires no network, downloaded data, or external
reference engine. The output path's parent must already exist.

`space_env.validation.ai_reference_suite.run_suite()` returns 32 checks with
fixed tolerances, units, applicability, reference method, and measured maximum
absolute component error. Tolerances are inclusive and are never estimated from
the run. Nonfinite measurements fail, with unavailable numerical fields serialized
as JSON null. Exceptions never become successful or skipped checks. Orbit checks
retain sample counts and maxima rather than full trajectory arrays.

## Independence and scope

Production routines run only on the measured side. The added reference checks
exercise `reference.rigid_attitude.advance_rigid_attitude`, the numerical kernel
called by the optional mission `reference.rigid_bus` controller. They do not
instantiate `MissionRuntime` or validate its battery/thermal/control integration. The core comparisons do not call the
existing validation reports or production Kepler solver as an oracle. Constants
MU_EARTH, J2_EARTH, R_EARTH_EQUATOR and SIGMA_SB are shared model inputs; their
physical accuracy is not independently established. Analytic references establish
agreement with idealized equations, not agreement with measured flight data.

The existing validation package also contains `report.py`, `gcrs_report.py`, and
`force_crosscheck.py`. The latter compares Boinor force formulas using shared
frame, density, and ephemeris inputs. This new suite does not execute that comparison.
No external-engine comparison is performed or claimed here, regardless of which
optional engines happen to be installed. The report explicitly records this fact.

## Fixed checks and tolerance rationale

- **Two-body orbits (4 checks):** circular and eccentricity 0.2, semimajor axis
  9,000 km, inclination 0.7 rad, a rotated orbital plane, two periods plus 123 s.
  Every RK4 sample (step at most 10 s, including the fractional final step) is
  compared with an elliptic Kepler solution using 60 bisection iterations and
  its analytic velocity. Maximum component tolerances are 1 m and 0.001 m/s,
  allowing finite-step global integration error while detecting meter-scale
  trajectory errors. J2, drag, SRP, third bodies and thrust are disabled.
- **J2 force (3 checks):** equatorial, polar and oblique positions outside Earth.
  The oracle differentiates the scalar positive gravitational potential
  `U=mu/r * [1-J2*(Re/r)^2*(3*(z/r)^2-1)/2]` with a five-point stencil of
  100 m. Acceleration is `grad(U)`. The tolerance is 2e-9 m/s² per component,
  allowing finite-difference cancellation and truncation, well below the J2
  contribution. This validates the core fixed-z-axis J2 convention, not GCRS
  orientation, EOP, or higher harmonics.
- **Radio loss (1 check, 9 points):** Friis power spreading
  `(4*pi*d*f/c)^2`, using exact SI c=299792458 m/s, at distances 1 km,
  1,000 km, 40,000 km and frequencies 100 MHz, 2.2 GHz, 30 GHz. The 0.01 dB
  tolerance accommodates production's rounded 32.44 dB unit-conversion
  constant. Only vacuum far-field isotropic loss is checked; antenna patterns,
  receiver noise, fading and modem capacity are outside scope.
- **Rocket accounting (4 checks):** initial/dry mass 120/100 kg, thrust 20 N,
  Isp 250 s, durations 100 and 10,000 s. Integrated mass flow is capped at dry
  mass; ideal delta-v follows Tsiolkovsky. Absolute tolerances are 1e-12 kg
  and 1e-10 m/s, allowing floating-point roundoff. Both an ordinary burn and
  fuel exhaustion are checked. This covers the mass/impulse helper, not the
  coupled finite-burn trajectory integrator or cutoff timing within a step.
- **Thermal balance (4 checks):** gray-body equilibrium at 0, 250 and 1,000 W,
  area 2 m², emissivity 0.8, capacity 5,000 J/K and a 3 K sink; a 30 s step
  must preserve equilibrium to 1e-10 °C. Exact calorimetry for 125 W over
  60 s with no radiating area must produce a 1.5 °C rise to 1e-12 °C.
  These are roundoff tolerances for stationary and linear solutions. They do
  not qualify Euler integration accuracy for radiative transients or actual
  spacecraft thermal properties.

The tests deliberately corrupt each production model family and require the same
suite to fail. They also check nonfinite rejection, tolerance boundaries, shape
validation, strict JSON serialization, report metadata and runner exit codes.

Passing these fixtures does not qualify atmosphere, eclipses, SRP, ephemerides,
frame transformations, operational mission accuracy, arbitrary orbital regimes,
or an external engine. Broader claims require separate evidence.


## Reference rigid-spacecraft kernel (16 additional checks)

Both torque signs are tested with synthetic diagonal body inertia (2, 3, 4)
kg m², wheel inertia 0.01 kg m² per axis, a 2 s arc and a maximum 0.01 s RK4
step. Torque limit is 0.1 Nm and wheel capacity 1 Nms per axis, so the fixtures
remain below both limits. The reference oracle uses no production invariant,
rotation or quaternion helper.

For free principal-z rotation at +/-0.4 rad/s, the oracle is the exact quaternion
`[cos(omega*t/2), 0, 0, sin(omega*t/2)]` and constant angular rate. Tolerances
are 1e-11 per quaternion component and 1e-12 rad/s. The nearer of q and -q is
used because both encode the same attitude.

For +/-0.05 Nm motor torque about principal x from rest, the analytic results are
`omega_x=-u*t/Ixx`, `h_x=u*t`, and `angle_x=-u*t²/(2*Ixx)`. These check attitude,
body rate and wheel momentum, at 1e-11, 1e-12 rad/s and 1e-12 Nms respectively.
They test reaction-torque sign as well as magnitude. Kinetic work follows
`W=(u*t)²*(1/Ixx+1/Jwheel)/2`, independently evaluated from prescribed inputs.
For the declared synthetic electrical model, efficiency 0.8, idle power 2 W and
loss coefficient 1.5 W/Nm² give `E_DC=W/0.8+(2+1.5*u²)*t` and
`heat=E_DC-W`. Each energy tolerance is 1e-10 J, allowing numerical roundoff
while remaining sensitive to energy-accounting errors. These tolerances apply
only to these smooth principal-axis solutions at the fixed step.

These are actual reference rigid-kernel attitude and energy checks, not an
independent qualification of the full reference-spacecraft mission backend.
Controller behavior, battery coupling, saturation, regeneration, arbitrary
asymmetric tumbling, external disturbances and hardware calibration remain
unqualified. Tests inject quaternion and DC-energy faults and require failure.

## Source binding and runtime evidence

Schema `space-env.ai-reference-suite/2` adds `sources`, `sources_sha256`, a native
canonical `sha256` seal and observed `runtime` information. Source hashes cover
all Python files in `core`, `reference`, `runtime` and `simulation`, plus package
and validation entry imports and the validation harness,
plus the suite's test, example and documentation when available in the checkout.
Missing repository artifacts in an installed package are listed explicitly;
they are never silently represented as hashed. Other import-only package areas (such as LLM/planning), external dependency
source files, data, compiled libraries and interpreter bytes are not sealed.

Sealing reuses `simulation.snapshot.digest` / `canonical_json`, the generic
native snapshot/audit mechanism, rather than importing `llm.episode`. The seal
hashes the complete report except its own `sha256` field, including checks,
tolerances, source manifest and runtime. `sources_sha256` separately identifies
the complete source-manifest object. Source bytes are sampled before validation
and compared again after the checks; a difference aborts the run. The example
verifies the seal against current sources before printing or atomically replacing
an output file using native `simulation.snapshot.atomic_write`.

For an existing JSON report:

```python
import json
from space_env.validation.ai_reference_suite import verify_report

with open('/tmp/ai-physics-reference.json') as stream:
    report = json.load(stream)
assert verify_report(report)  # seal and equality with this checkout's source bytes
assert report['passed']      # separate physics result
```

`verify_report(report, check_current_sources=False)` checks internal integrity
without requiring the current checkout to match. A seal is a reproducibility
checksum, not a signature, trusted timestamp, proof of execution or attestation
of already-loaded Python objects. A source change followed by resealing does not
prove scientific correctness. Tests detect altered checks, changed current-source
hashes, malformed seals and source changes during execution.

`runtime` reports the actual Python version, implementation, executable, platform,
and installed distribution versions mapped from modules loaded in that process.
Import-time dependencies are explicitly included: appearing in that list does
not mean a package was used as a numerical oracle. This implementation imports
the reference package and native snapshot utility, so the runtime must support
those imports (including the reference extra SciPy/pyerfa); it is not a
NumPy-only installation claim. No compatibility matrix beyond the tested runtime
is implied. In-memory monkeypatches are not detectable by on-disk source hashes.

The implementation was exercised locally with CPython 3.12.9 on Linux x86_64
(glibc 2.39), NumPy 2.2.6, Pydantic 2.11.7, SciPy 1.15.2 and pyerfa 2.0.1.5.
The standalone process also loaded Boinor 0.19.0, Astropy 7.2.0,
Gymnasium 1.2.0, pandas 2.3.3 and PyYAML 6.0.3 through imports; none supplied an
external comparison oracle. See the generated JSON for the complete observed
list, including transitive dependencies. Tests used pytest 9.1.1 in this same
Python environment. Other runtimes and minimal dependency installations were not
tested.
