# Frame migration specification

Status: partial implementation in the working tree. Standalone GCRS reference
frames/time/orbit are implemented; see [GCRS_REFERENCE_V1.md](GCRS_REFERENCE_V1.md).
The mission executor remains explicitly legacy-training/1. The complete migration
sequence and qualification gates below are not declared complete.
Base model: space-env 0.134.0, 30 October 2026.

## Existing contract

`NUMERICAL_FRAME = "GCRF"` currently labels a legacy numerical frame. The terrestrial
transform uses GAST and polar motion without the complete celestial-to-terrestrial
rotation. `teme_to_numerical` converts through ITRS into that same legacy frame.
`teme_to_gcrs` is a separate utility and is not the engine's propagation frame.

ERFA ephemerides and the legacy catalogue RAAN convention require an explicit
compatibility review. Enabling a library must not silently change a physical profile.
The measured 46 km difference in one example is a difference of frame conventions,
not a universal propagation error estimate.

## Migration sequence

1. Add a distinct `LEGACY_NUMERICAL` frame/profile identifier. Preserve old serialized
   scenarios via an explicit schema migration; do not relabel their raw vectors GCRS.
2. Define a GCRS reference profile with one UTC/TAI/TT/UT1 conversion service, dated
   leap-second and EOP data, declared epoch ranges and deterministic missing-data rules.
3. Specify position, velocity and covariance transforms. Velocity includes the time
   derivative of the rotation; covariance uses the complete 6x6 state Jacobian.
4. Evaluate Sun/Moon, gravity, drag/atmospheric rotation, illumination, access and
   catalogue local solar time in the profile's declared frame/time convention.
5. Keep legacy mission fixtures immutable. Generate a separate reference fixture
   suite and record its physics/data hashes. Do not tune coordinates to preserve a
   legacy delivery count.
6. Release an explicit profile selector and versioned serialization. A requested
   reference backend without required data/dependencies fails early; no quiet fallback.

## Validation matrix required before promotion

- GCRS/ITRS and TEME/ITRS/GCRS: multiple epochs, inclinations, r/v and 6x6 covariance;
  separate independent fixture generation from implementation under test.
- Geodetic positions, elevation, AOS/LOS, eclipses and local solar time.
- Two-body analytical arc, individual perturbations, finite burns and combined
  24/72-hour reference arcs using matched initial conditions and force settings.
- Stepsize convergence and model/data uncertainty distinct from integration error.
- Mission outcomes and safety on common feasible/infeasible cases across profiles.

For each fixture record source, version/licence, units, frame, epoch, arc length,
numerical tolerance and applicability. Set tolerances from the profile requirements
before measuring. The current 600-second two-body position tolerance is not a
24-hour mission accuracy claim. No new numerical tolerance or operational
qualification is established by this specification.
