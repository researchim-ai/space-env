# D2h-B2b-B: incoming demand and explicit policy epochs

## Contract

The simulator starts with the original requests only. New `CustomerRequest` and
`TransformationJob` declarations enter through native `request_added` events on
an aligned decision boundary. They do not create inventory: capture, processing,
and delivery must still execute. The effective planning/goal graph is distinct
from the unchanged simulator specification; the certificate compares its complete
request denominator with actual native state and exact committed event payloads.

`DemandBatch` accepts 1–4 additions per update. The journal is bounded to four
updates and 64 MB. The inherited kinematic chain profile limits actors to 4,
stations to 16, total requests to 64, horizon to 21600 seconds and decision cells
to 1–30 seconds (at most 10000). This planner currently requires a single eligible
actor, an existing target and one metadata processing job per new request.
Repeated requests to the same target use `request_ref`, rather than duplicate
physical targets. Unknown targets, duplicate graph identities and backdated
releases fail validation. An infeasible but well-formed deadline stays in demand;
it does not disappear from service metrics.

The declared policy requires the original service floor plus every incoming
request. This is deliberately explicit; admission control and negotiated SLA
policies are separate future work. A failed mission can produce RL trajectories,
but cannot produce successful-goal SFT candidates.

## Authority, catalogues and replay

`demand_actors` and `demand_stations` grant declaration authority without granting
physical ownership. Physical roles remain disjoint. The `@demand` operator role
is excluded from learner action targets. Empty permission fields and absent
`request_ref` preserve legacy serialized task intent.

Each epoch retains the original simulator definition, all old action templates
and the already executed prefix. Only future suffix actions are replanned. Active
commands are explicitly quiesced through native controls before additions.
A catalogue version hashes generation and actual task specification. Shared
online/offline observations include that version and bind it cryptographically.
`ActorCoordinator.rebind(journal)` native-verifies the successor against the live
source episode, then retires the old controller. Failed rebind leaves it usable;
stale bindings fail before policy state advances. Hashes provide consistency and
provenance, not authentication against a malicious artifact author.

Gym/PettingZoo spaces stay fixed within an epoch. Construct a new environment
from the successor journal to expose the new action vocabulary. A checkpoint
restores through audited cold prefix replay; this is not constant-time RAM restore.
Offline journal checks validate structural ancestry and intent; `native=True`
additionally reconstructs every physical source prefix and recomputes each epoch.

## Run

```bash
python -m examples.planning.demand_mission /tmp/demand-batch --mode batch
python -m examples.planning.demand_mission /tmp/demand-two --mode two_updates
python -m examples.planning.demand_mission /tmp/demand-late --mode tight_deadline
python -m space_env.cli verify-demand-journal /tmp/demand-two/journal.json --native
```

Use `python -m space_env.cli --help` for `demand-journal-create`,
`demand-journal-append` and `actor-controller-rebind`. File workflows refuse to
overwrite existing outputs. The Python example writes input, epoch journals,
pre/post rebind checkpoints, final execution, native audits and role datasets.

## Evidence boundaries

These fixed synthetic mechanism cases exercise event admission, repeated-target
planning, interrupted commands, epoch bindings, complete service denominators,
failed-deadline preservation, lineage and replay. They do not establish forecast
optimality, measured payload fidelity, RF/rigid-body fidelity, overload performance,
expert teacher quality or general industrial readiness. SFT rows are candidates;
`expert_teacher_qualified` stays false. Historical receipts belong to their exact
wheel fingerprints and must not be reclassified under a new source build.
