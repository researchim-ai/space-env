# Reference pointing: snapshot goals and causal task masks

Дата: 2026-11-04. Development contract, source version 0.134.0.

## Command contract

`RuntimeCommand(kind="slew")` accepts exactly one goal in `slew`:

```python
{"frame": "GCRS", "target_id": "site"}
{"frame": "GCRS", "station_id": "Ground"}
{"frame": "GCRS", "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
```

References must already exist in the acquisition-target or ground-station
registry. Unknown references reject admission before committing the transaction.
The reference direction is evaluated at admission time, using the actor's
current GCRS state and the existing epoch/EOP terrestrial transform:

`line_GCRS = R_GCRS_to_ITRS(t).T @ site_ITRS - position_GCRS(t)`.

The existing `align_z` maps the body +Z axis onto this direction, choosing its
existing deterministic roll convention. This is not a full payload/antenna
mount calibration or a free roll optimisation. The resulting quaternion is
frozen in the operation; it is an inertial snapshot hold, **not continuous
tracking of a terrestrial site**. Reissuing a command resolves a new snapshot.

Admission does not change attitude or charge energy. The existing rate-limited
or rigid wheel controller performs the manoeuvre during advance, retaining
its duration, settling, saturation, power, thermal and failure semantics.
Imaging's existing moving-target tracking remains distinct. A site below the
horizon is still a mathematically defined slew direction; command admission
does not assert visibility or successful acquisition.

Legacy quaternion command serialization omits the new unused fields, preserving
its previous representation. Checkpoint replay resolves each reference at the
original journal time; it does not replace the held goal with a future direction.
Historical acceptance receipts remain attached to their historical source hashes.

## Current read API

```python
runtime.pointing_observation("sat-a", target_id="site")
runtime.pointing_observation("sat-a", station_id="Ground")
```

Returns owned current ideal telemetry: snapshot goal, boresight angular error,
settling, configured payload tolerance, and rigid body rate/limit when present.
For targets with a configured acquisition sensor, also returns current
`acquisition_geometry`, including availability and feasibility. Station geometry
is not yet evaluated by this read (`geometry=None`). No future windows, truth
access, command-admission oracle, predicted dwell or successful outcome is used.
Like other operator runtime reads this is not a role-scoped actor API by itself;
the task projection supplies the role and released-request visibility boundary.

## Opt-in task mask

`TaskSpec(pointing_masks=True)` adds necessary current-state checks:

- imaging requires the request to be publicly released, current feasible
  acquisition geometry, settled attitude, payload-compatible body rate and
  boresight error within the configured tolerance;
- target-reference slew requires a publicly released request for that target
  and currently feasible geometry;
- the task catalogue hides reference command parameters if that actor has no
  publicly visible request for the referenced target;
- reads are cached only within one projection refresh, like existing inventory
  reads, and do not mutate native state.

These are conservative checks, not a prediction of successful future dwell.
Power, Sun keepout, mid-interval motion, interruptions, finite resources and
simultaneous action conflicts remain authoritative in native execution.
Station-reference slews retain ordinary resource/fault checks; no new antenna
visibility/readiness mask is claimed. Opt-in on unsupported bus/geometry
configurations fails explicitly rather than fabricating readiness.

False is omitted from task serialization, preserving legacy task identities.
No new controller, RL teacher, expert planner, calibrated sensor or observation
noise is introduced. These observations remain ideal current telemetry.

## D2b continuation

The snapshot/current-read/mask prerequisite is now used by the bounded
[independent coupled metadata mission](COUPLED_INDEPENDENT_MISSION_V1.md).
Its frozen compiler, native negative cases and process restore have their own
acceptance. Numerical refinement, independent mission/controller holdout and
real sensor/hardware qualification remain open. D2a ready-product receipts are
not relabelled as coupled-mission evidence.

## Предыдущая проверка prerequisite

14 новых tests в `tests/test_reference_pointing.py`: schema/legacy serialization,
unknown-reference atomicity, target/station immutable snapshot, paid advance,
native replay, owned reads, current-state masks, unreleased target suppression,
rigid energy conservation и task checkpoint. Совместно с planning/projection/
runtime-task/bus/rigid regressions: **181 passed**, 2 upstream warnings.
Fast API profile: **110 passed**, 2 upstream warnings. Наборы пересекаются;
полная reference/full suite, installed wheel и independent mission qualification
для этой новой сборки ещё не запускались. Reference CI profile включает новые tests.

Последующая общая проверка новой сборки и installed coupled workloads сохранена
отдельно в [B11_COUPLED](../acceptance/B11_COUPLED.json); исторические численные
результаты предыдущего шага выше не переаттестованы задним числом.
