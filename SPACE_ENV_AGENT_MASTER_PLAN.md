# SPACE-ENV — MASTER IMPLEMENTATION PLAN FOR AUTONOMOUS CODING AGENT

> **Актуализация от 25 октября 2026:** порядок дальнейшей разработки и новые находки аудита — в [docs/STARTUP_DEVELOPMENT_PLAN.md](docs/STARTUP_DEVELOPMENT_PLAN.md) и [аудите](docs/audits/2026-10-25_READINESS_AUDIT.md). Не выполнять push, публикацию или создание PR без отдельного поручения владельца. Не переключать ветку и не перезаписывать существующие изменения. Эти ограничения имеют приоритет над прежними указаниями ниже.

> **Repository:** https://github.com/researchim-ai/space-env/tree/develop  
> **Base branch:** `develop`  
> **Purpose:** Transform the existing Space-Env research codebase into a reproducible, validated, extensible platform for space-mission simulation, RL/ML/LLM-agent training, evaluation, and commercial mission-operations workflows.  
> **Document type:** Executable engineering specification and acceptance checklist.  
> **Important:** No deadlines or estimates. Progress is governed only by verified acceptance criteria.

---

## 0. Instructions to the autonomous agent

You are the lead implementer and technical integrator. **Implement** the work in this document, not merely describe it. Continue through all work packages unless blocked by access, credentials, licensing, outside validation, unavailable infrastructure, or physical evidence that cannot be generated from code. Record blockers precisely and continue with independent packages. Do not mark unverified items as complete. Never claim flight qualification, industrial certification, physically validated accuracy, or successful customer pilots without supporting evidence.

### 0.1 First actions — inspect the real code

1. Check out `develop`, record commit SHA and repository status; create an implementation branch (e.g. `feat/industrial-platform`). Preserve existing work. Do not force-push or delete user data.
2. Inventory ALL existing `src/space_env`, `configs`, `scripts`, `tests`, `docs`, CI and visualization code. Read `README.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/INDUSTRY_LIBRARIES.md`, `pyproject.toml`, `configs/curriculum.yaml`, `configs/rewards.yaml`, and `simulation/engine.py`.
3. Build the project using documented commands, run `pytest`, all environments through Gymnasium checks, available scripts, and a baseline performance benchmark. Store results, stdout/stderr, versions and exit codes under `artifacts/baseline/` (or a gitignored output location with sample manifests checked in).
4. Produce `docs/audits/BASELINE_GAP_ANALYSIS.md`: present/partial/missing/broken status for each requested feature, with file paths and commands. **Do not reimplement existing features**; refactor and extend them.
5. Maintain `docs/IMPLEMENTATION_STATUS.md` with issue IDs, status, test evidence and blockers. Update after each implemented feature.
6. Use small reviewable commits, backward-compatible adapters and migrations. Changes to public contracts need tests and release notes.
7. When a task has dependencies, implement them first; otherwise parallelize independent work through subagents, worktrees or clearly separated branches. Reintegrate and rerun CI.
8. Run the appropriate unit, integration and regression tests after every substantial change. Do not modify tests solely to hide failures. Treat validation failures as findings, not as opportunities to weaken tolerances.

### 0.2 Definition of Done (applies to every work package)

- Actual implementation and API contract exist, with documented invocation and working example.
- Unit tests, a happy-path integration test, failure-path tests, and reproducible test data exist.
- CI covers the feature, or an explicit documented reason explains why an expensive optional job is scheduled separately.
- All units, coordinate frames, timescales, assumptions, numerical tolerances, and domain limitations are stated where relevant.
- Existing registered environments and exports remain functional, or a documented compatibility layer and migration are supplied.
- Deterministic sample outputs, benchmark artifact(s), and a reference to the exact source commit are produced.
- Security, error handling, observability and runtime cost are addressed when applicable.
- Any unverified physics or safety claims are explicitly marked **NOT VALIDATED**.
- `docs/IMPLEMENTATION_STATUS.md` links evidence and records remaining failures; passing a toy smoke test never counts as production readiness.

### 0.3 Delivery structure

Deliver source code, tests, config examples, runnable CLIs, API docs, diagrams, benchmark and validation reports, deployment manifests, sample datasets, example notebooks, and release notes. Add a root `AGENTS.md` summarizing these instructions and the verification commands. Before declaring success, run all acceptance gates in §19 and prepare `docs/RELEASE_READINESS.md`.

### 0.4 Working assumptions and constraints

- Current documented package version when this plan was prepared: `0.45.0`, with Gymnasium environments and physical, mission, and data-generation layers already present. **Verify every detail against the checked-out revision**; do not infer readiness from README claims.
- Existing components include Cowell orbital propagation, J2, drag, SRP, third-body effects, batteries, thermal models, ground passes, communications, CDM/conjunction assessment, constellation operations, scenario configs, REINFORCE trainers, SFT and RL data generators, evaluation scripts, and viewer assets.
- Keep the research model label distinct from any mission-specific validated digital twin. Engineering-grade software quality does not imply flight certification.
- Respect Apache-2.0 license obligations and the licenses of every optional backend. Optional integrations must degrade gracefully.
- Do not submit commands to real spacecraft, authenticated mission operations, or external customer systems without explicit authorization and a separate reviewed integration process. Development defaults to simulation-only.

---

## 1. Product scope and acceptance architecture

Implement six decoupled layers:

1. **Core Simulation** — deterministic clock/event system, world state, orbital and subsystem models, typed actions, snapshots, replay.
2. **Physics Backends & Validation** — fast/engineering/reference modes; independently computed golden cases and declared uncertainty.
3. **Mission Operations** — EO acquisition, resource planning, ground network, conjunction assessment, fleet tasking, optional proximity operations.
4. **Agent Training** — Gymnasium/PettingZoo adapters, classical baselines, RL/MARL, offline imitation, tool-using LLM agents.
5. **Evaluation & Safety** — reproducible benchmark suites, OOD/robustness tests, hard constraints, action verification, audit/evidence reports.
6. **Product Delivery** — typed SDK, CLI/API, Studio UI, telemetry import/export adapters, isolated training workers, deployment and observability.

### 1.1 Architectural rules

- Environments may depend on stable simulation interfaces, not vice versa.
- Training algorithms must not be imported into the physics kernel.
- No hidden use of privileged simulator state inside agent observations; full state access allowed only to explicitly designated oracle baselines and centralized critics during training.
- Rewards and mission-success criteria must be separate from safety constraints; safety cannot be reduced to negative rewards.
- Every backend has explicit units, frames, timescales, version, assumptions and determinism guarantees.
- Hardware/system-specific numerical accuracy must be validated on agreed use cases, not asserted globally.
- Use dependency injection rather than one monolithic `SimulationEngine` accumulating responsibilities.
- Long-running tasks are idempotent and restartable; avoid silent partial writes.

### 1.2 Target repository layout (evolutionary migration, not blind rewrite)

```text
space-env/
  AGENTS.md
  src/space_env/
    core/                  # stable clock, state, actions, event model, units
    physics/               # propagators, attitude, subsystems, adapters
    simulation/            # orchestration, snapshots, replay, fault injection
    mission/               # EO, scheduling, constraints, passes, CDM
    envs/                  # Gymnasium and optional multiagent wrappers
    agents/                # LLM tools, schemas, authorization, agent harness
    rl/                    # algorithm adapters, training, checkpointing
    eval/                  # baselines, suites, reports, robustness
    data_generators/       # SFT, trajectories, dataset manifests
    integrations/          # CCSDS, Orekit, Basilisk etc.
    api/                   # versioned service contracts
  apps/studio/             # browser UI (if frontend stack confirmed)
  scripts/
  configs/
    scenarios/
    rewards/
    curriculum/
    benchmark_suites/
  benchmarks/              # versioned suites and case definitions
  validation/              # reference vectors + physical validation reports
  tests/                   # unit, integration, contract, physics, e2e
  examples/
  docs/
  infra/                   # containers, local stack, opt-in deployment
```

**Migration rule:** retain old import paths and IDs through shims, avoid namespace churn until tests prove parity.

---

## 2. WP-001 — Baseline, repository hygiene, compatibility

### Implement

- Create machine-readable `capabilities.json` or `docs/capabilities.yaml`: module, existing API, owner, maturity, validation status, dependencies, examples, limitations.
- Add `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, clear LICENSE attribution, CHANGELOG and versioning/release policies.
- Add a reproducible environment with lockfile(s); preserve compatibility with supported Python versions per project policy. Verify optional extras install independently.
- Configure Ruff, typing checks (mypy or pyright), pytest/coverage, Bandit or equivalent, dependency audit, secret scanning, build/wheel checks and dependency license report.
- Ensure `python -m build`, package install, imports, Gymnasium entry-point registration and CLI entry points work in clean environments.
- CI job matrix: lint/type/unit/integration/packaging; separate optional physics integrations and expensive benchmarks.
- Regression-test all existing environment IDs, observation and action shapes, `reset`/`step` output, termination/truncation and `info['success']` semantics.

### Acceptance

- Clean install and complete smoke suite succeeds under the declared supported matrix.
- `gymnasium.utils.env_checker.check_env` succeeds for supported Gymnasium environments, except documented tested exceptions.
- A baseline report includes the full environment registry and per-environment deterministic smoke episode.

---

## 3. WP-002 — Deterministic simulation kernel

### Implement

- Design `SimulationConfig`, `WorldState`, `SpacecraftState`, `Action`, `Event`, `SimulationResult` using validated schemas and version fields.
- Standardize absolute epoch plus duration, physical units, frame labels, numerical dtype, source and assumptions.
- Introduce a single event clock, ordered event queue with tie-break rules, deterministic random seed splitting per subsystem and agent, typed commands and typed failure reasons.
- Implement `reset(seed, scenario)`, `step(action)`, `simulate_until(t)`, `snapshot()`, `restore(snapshot)`, `clone()`, `branch(action_sequence)`, `event_log()`.
- Use stable serialization (JSON for manifests; arrays using suitable portable binary format); include schema version, checksums, event sequence numbers, RNG states and model versions.
- Separate state transitions for orbit, attitude, electrical power, thermal, payload, comms, propulsion, faults, task queue and observation generation behind documented protocols.
- Define explicit integration ordering and substep policies. Resolve simultaneous commands deterministically; reject invalid actions with structured errors.
- Explicitly distinguish `terminated`, `truncated`, mission success, unsafe termination, and timeout.

### Test cases

- Same seed/scenario/code/backend/platform produces bitwise-identical or explicitly tolerance-defined trajectories.
- Snapshot→restore→continue equals uninterrupted run.
- Clone branches diverge only after distinct actions.
- Events never reorder across serialization/replay.
- Invalid commands, insufficient fuel, illegal pointing, missing station, impossible windows and NaN/Inf state are handled predictably.
- Existing environment episode outputs are characterized before and after migration.

---

## 4. WP-003 — Physics validation and backend independence

### Implement

Create `PhysicsBackend` protocol with `propagate`, `apply_maneuver`, `transform_frame`, `ground_visibility`, `illumination`, `predict_resources`, `metadata`. Offer modes:

- `fast`: controlled approximations optimized for batched RL rollouts.
- `engineering`: existing and improved detailed simulation.
- `reference`: independent trusted implementations (where installed, licensed and configured).

Integrate or create optional adapters for **Orekit**, **Astropy**, **SGP4**, **SPICE** and **Basilisk**, only when appropriate to their scope. Do not claim Basilisk is a drop-in orbital propagator or that SGP4 is interchangeable with numerical propagation without documenting model/initialization differences.

### Required validation suites

- Two-body Keplerian reference and orbital invariants.
- J2 nodal/perigee drift and sensitivity to epoch, inclination and step size.
- Atmospheric drag using declared atmospheric inputs; perturb density, F10.7 and Ap.
- Solar radiation pressure, Earth eclipse geometry, third-body contributions.
- Impulsive Δv in RTN/inertial frame and maneuver timing consistency.
- TEME/GCRF/ITRF transforms, UTC/TAI/TT/UT1/TDB timescales, leap seconds and Earth orientation data versioning.
- Ground-station visibility, AOS/LOS edge cases, antenna constraints and link budgets.
- Battery conservation, state-of-charge boundaries, thermal balance, heater hysteresis.
- Attitude quaternions, torque and actuator limits (implement where incomplete).
- Relative dynamics, close approach prediction, covariance PSD/symmetry, collision probability and Monte Carlo cross-checks.
- CDM parser conformance with representative positive/negative CCSDS examples.

### Validation artifacts

For each case generate `validation/cases/<case>/` with inputs, reference provenance/version, unit/frame conventions, outputs, comparisons, plots/data and failure criteria. Implement CLI:

```bash
space-env validate physics --suite core --backend engineering --format json
space-env validate physics --suite conjunction --backend reference --format html
```

Use domain-specific error budgets. Determine tolerances from reference quality and application requirements; never set thresholds simply to force a pass. Tag tests by `verified`, `approximate`, `needs-external-validation`.

### Acceptance

- Reference comparisons executable for available optional backends; unavailable extras produce explicit skip with explanation.
- Reports state numerical errors and conditions; no unsupported blanket claim of accuracy.
- Simulation modes are switchable without changing agent-facing contracts.

---

## 5. WP-004 — Scenario DSL and mission catalog

### Implement

- Versioned Pydantic/YAML scenario schema with migrations and JSON Schema export.
- Entities: spacecraft, orbit/ephemeris, bus, sensors, payloads, propulsion, stations, communication links, target requests, contact schedule, conjunction objects, failures, uncertainty, environmental data, mission horizon and decision intervals.
- Strict units and validation; errors reference YAML path and expected constraints.
- Explicit distinction between documented satellite characteristics and synthetic assumptions.
- Import/export scenario bundles with provenance, SHA-256 manifests, license attribution and reproducible source references.
- Scenario generation: parameterized sweeps, correlated perturbations, train/validation/test partitions preventing seed and scenario leakage.
- Example catalogs for existing Sentinel-2, Landsat 9, Iridium, CubeSat and debris use cases; retain synthetic labels.

### Acceptance

- Invalid schemas fail fast; migrations tested; scenarios round-trip without losing material information.
- Examples cover one satellite, fleet, station competition and conjunction with uncertainty.

---

## 6. WP-005 — Mission operations and classical optimization

### Implement

1. **EO acquisition and planning:** target visibility, sun/eclipse, payload duty cycle, pointing and slew, storage, power/thermal impacts, station access, priority, deadline, cloud/weather uncertainty when data support exists, data-delivery SLA.
2. **Ground station management:** antenna occupancy, competing bookings, AOS/LOS, link margins, station outages, handovers, queueing and capacity.
3. **Fleet/constellation planning:** heterogeneous spacecraft, task assignments, intermittent connectivity, inter-satellite link geometry and bandwidth, communications latency and conflicts.
4. **Conjunction assessment:** CDM ingestion, TCA propagation, collision probability, screened maneuver candidates, post-maneuver re-screening, propellant impact and task conflicts. Do not hard-code one Pc threshold as universal policy.
5. **Anomaly operations:** power faults, heater faults, radio latch, missing telemetry, stale data, clock skew and degraded sensors; safe-mode transitions.
6. **Optional advanced use cases:** rendezvous hold, proximity operations, docking and active attitude control **only after** dynamics and verification prerequisites are satisfied.
7. Implement strong non-ML baselines: greedy heuristic, earliest-deadline-first, MILP or CP-SAT (e.g. OR-Tools) for scheduling, MPC/trajectory optimization where physically appropriate. Document solver licensing and optional deps.
8. Canonical typed `MissionPlan`, `PlanStep`, `ConstraintViolation`, `PlanEvaluation`, `ScheduleExplanation` data structures with export formats.

### Acceptance

- Each mission planner has feasible and infeasible fixtures, deterministic benchmarks, constraint reporting, and comparison to a documented baseline.
- Measured metrics: fulfilled priority-weighted requests, delivered bits, energy and storage margins, Δv use, late deliveries, resource utilization, solution runtime and violated constraints.

---

## 7. WP-006 — Gymnasium, multiagent environments and observation integrity

### Implement

- Keep existing `space_env/...-v0` IDs functional. Introduce versioned `-v1` only for meaningful contract changes.
- Validate Gymnasium semantics and observation bounds/dtypes; make action masks and invalid-action policy explicit.
- Standardize `info`: `success`, `unsafe`, `violations`, `mission_metrics`, `episode_id`, `scenario_id`, `physics_backend`, `seed`, `termination_reason`.
- Provide `SpaceEnvWrapper`/factory for horizon, fidelity, safety and scenario selection.
- Add PettingZoo Parallel/AEC or equivalent multiagent interfaces for fleet tasks; document shared rewards, partial observability and centralized critic privileges.
- Provide vectorization wrappers and resets with non-overlapping seeds.
- Implement environment families (without duplicating existing functionality):
  - `AcquisitionScheduling-v1`
  - `GroundNetwork-v1`
  - `EOFleetOps-v1`
  - `FaultRecovery-v1`
  - `AutonomousOperator-v1`
  - `ConjunctionCampaign-v1`
  - `CollisionAvoidance-v1`
  - `ProximityOps-v1` and `DockingControl-v1` (gated advanced scope)
- Prevent hidden target/failure labels, true future sensor values or privileged collision state from leaking into policy observations.
- Add comprehensive observation/action cards, render examples and public registration documentation.

### Acceptance

- All registered environments construct, reset, step and terminate correctly; tests enforce declared spaces.
- Multiagent behavior reproducible and no unauthorized oracle state appears in agent observations.

---

## 8. WP-007 — RL/MARL training infrastructure

### Implement

- Plugin-based trainers using Stable-Baselines3 for PPO/Maskable PPO, SAC, TD3 as action spaces allow; RLlib/CleanRL alternatives when justified. Pin compatible versions.
- Multiagent algorithms: MAPPO/centralized critic and appropriate value decomposition (e.g. QMIX for compatible discrete cooperative settings).
- Include random, scripted heuristic, MPC/optimization and behavior cloning baselines in all relevant experiments.
- Offline learning on recorded decision datasets: behavior cloning first, IQL/CQL where feasible, with proper off-policy datasets and no leakage.
- Constrained RL experiments with explicit cost signals and validated policy shields; report both reward and constraint violations.
- Recurrent/partial-observation training where missing telemetry or delayed decisions matter.
- Curriculum scheduler based on measured performance (not merely a stage list); progressive difficulty, domain randomization and held-out evaluation.
- Multiprocess/vectorized rollout collection, worker seed management, checkpoint/resume, model artifacts, CPU/GPU options and graceful cancellation.
- Config-driven CLI with YAML, reproducible manifests and MLflow (or equivalent) tracking; no embedded credentials.

### Reference commands to implement

```bash
space-env train --env space_env/ResourceManager-v0 --algo ppo --config configs/train/ppo.yaml
space-env train --env space_env/EOFleetOps-v1 --algo mappo --config configs/train/mappo.yaml
space-env evaluate --checkpoint artifacts/models/example --suite benchmarks/eo_core.yaml
space-env compare --suite benchmarks/eo_core.yaml --agents random,heuristic,optimizer,ppo
```

### Acceptance

- Reproducible training and resume demonstration; training curves preserved with independent seeds.
- Train and test seeds are separated; no test-suite driven hyperparameter overfitting.
- Failures and regressions remain visible; a model is not labeled “learned” merely because its training tail beats a random mean once.

---

## 9. WP-008 — Dataset generation, offline RL and LLM-agent training

### Implement

- Single typed episode/trajectory format carrying observations, allowed actions, action masks, tool calls/results, reward components, safety costs, terminal reasons, provenance, policy IDs, physics parameters and source version.
- Data generators for pretraining documents, supervised tool-use traces, behavior-cloning trajectories, preference pairs (with defined labeling protocol) and RL rollout storage.
- JSONL/Parquet export, manifest/checksum, streaming generation, shard merge, schema validation, deduplication and deterministic splits grouped by scenario families and seeds.
- Ensure the agent is not trained on hidden simulator labels unless the task explicitly models privileged supervised targets.
- Implement dataset quality gates: empty/duplicate/contradictory examples, impossible tool calls, invalid units, hallucinated states, policy leakage, deterministic replay of logged transitions.
- Support optional tool-use and multi-turn LLM-agent evaluation harness; model-specific SFT/RL integrations are optional adapters, not a new bespoke foundation model framework.
- Report effective sample count, scenario diversity, train/test overlap, outcome distribution and verification pass rate.

### Acceptance

- Existing `generate_sft_data.py` and `generate_rl_data.py` remain functional or have documented wrappers.
- Representative serialized trajectory round-trips and can be replayed against the same simulator revision.

---

## 10. WP-009 — Safe LLM Mission Agent framework

### Implement

- Typed, versioned tool contracts: `get_spacecraft_state`, `get_ground_station_passes`, `get_power_forecast`, `predict_orbit`, `assess_conjunction`, `simulate_maneuver`, `optimize_acquisition_plan`, `validate_command_plan`, `submit_plan_for_approval`.
- Agent roles with scoped visibility and permissions: planner, diagnosis agent, evaluation agent, safety verifier, human approver.
- Tool calls run inside simulation sandbox with strict argument validation, resource quotas, timeout/error classification, deterministic replay and full audit logs.
- Separate agent reasoning/proposal from *execution*. Execution of sensitive actions defaults to denied unless verifier and approval pass.
- Action plans represented as structured contracts, not ad hoc free text; invalid or ambiguous commands rejected.
- Multi-turn test suite with incomplete telemetry, contradictory alerts, delayed ground contacts, false anomalies, conflicting orders, missing permissions and unknown states.
- Define correct abstention/escalation behavior and calibration metrics.
- Prevent prompt injection from untrusted telemetry, scenario descriptions, log content and imported documents by never treating those as higher-priority instructions.

### Acceptance

- End-to-end simulated agent successfully retrieves observations, calls tools, proposes a plan, obtains verdict, and records the final simulated outcome.
- Unauthorized commands cannot bypass approval gates; tests cover adversarial tool inputs.

---

## 11. WP-010 — Safety constraints and verification framework

### Implement

- Constraint specification engine: battery SOC, temperature, memory, Δv, link geometries, pointing exclusion, forbidden maneuvers and operational state-machine guards.
- Independent `SafetyVerifier` and `ActionShield` that operate outside learned-policy reward optimization.
- Three verdict outcomes at minimum: `allow`, `reject`, `requires_approval`; include reasons, provenance, state/version and validity window.
- Distinguish deterministic hard constraints from uncertain/probabilistic risk assessments. Provide policy-specific configurable thresholds and risk assumptions.
- Safe fallback actions only when actually verified for the current state.
- Fault injection and resilience suite: sensor bias/drift/dropout, comms loss/latency, stale ephemerides, propulsion nonresponse, solar activity variations, clock skew, station outage and corrupted packets.
- OOD testing with scenario perturbations and held-out physical backend; evaluate detection, abstention and graceful degradation.
- Hazard log, preliminary FMEA, evidence traceability, documented misuse cases, human sign-off and approval records.
- Non-bypassable audit events for proposed/approved/rejected/executed commands. This product remains simulation-only unless explicit external approvals and separate mission safety process exist.

### Acceptance

- Tests demonstrate that reward pressure cannot override hard limits.
- Reports include the exact suite/sample sizes and observed failures; no inference of universal safety from zero observed failures.

---

## 12. WP-011 — Benchmark and statistical evaluation suite

### Implement

- Benchmark manifests with versioned scenario list, seed split, physics backend, action limits, policy interface, score definitions and tolerances.
- Suites: `resource_management`, `eo_planning`, `ground_network`, `fault_recovery`, `conjunction`, `fleet_operations`, `transfer`, `domain_shift`, `agent_tool_use`, `physical_validation`.
- Multi-seed learning experiments, bootstrap/confidence intervals where appropriate, paired comparisons and per-scenario failure breakdowns. Publish raw scores, not just aggregate means.
- Benchmarked competitors: random, heuristic, classical solver, imitation, PPO/SAC/MARL as appropriate.
- Metrics: mission success, weighted task completion, delivered data, delay, safety violations, fuel and energy, robustness under shift, wall-clock runtime, memory/CPU/GPU and test coverage.
- Collision risk reports must disclose confidence limitations and rare-event sample sizes; do not use ordinary episode success as a substitute for reliability qualification.
- Generate machine-readable JSON/Parquet and human-readable HTML reports with replay links and exact source/artifact metadata.
- Add benchmark regression gates for changed code and maintain a pinned baseline dataset.

### Acceptance

- One command evaluates the same agent on canonical and holdout suites; outputs reproducible trace-linked comparisons.
- Reports distinguish improvement over random from improvement over a credible domain-specific optimizer.

---

## 13. WP-012 — Performance and scale

### Implement

- Profile actual hotspots; optimize only measured bottlenecks and preserve correctness.
- Batched physics where appropriate, vectorized rewards/observations, multiprocessing or distributed workers, memory-conscious trajectory recording.
- Benchmark fleets at 1, 10, 50, 100+ spacecraft if feasible; measure time per simulated mission duration, rollouts/s, CPU/GPU memory and planning latency.
- Support heterogeneous satellites, dynamic link graphs, partial observability, crosslink capacities and independent agent states.
- Deterministic distributed scheduling where practical; document any non-deterministic accelerator behavior.
- Record scaling curves and trade-offs of fidelity modes.

### Acceptance

- Repeatable profiler results before and after optimization; no meaningful physics/safety regression due to speedups.

---

## 14. WP-013 — SDK, services and integrations

### Implement

- Stable Python SDK with typed public APIs, semantic versioning, examples and migration guide.
- CLI: `inspect`, `scenario validate`, `simulate`, `replay`, `train`, `evaluate`, `compare`, `validate physics`, `dataset generate`, `report`, `serve`.
- Optional FastAPI-based REST endpoints with OpenAPI docs for scenarios, jobs, policies, evaluations, metrics and replay assets. Include pagination, quotas and structured errors.
- External data interfaces: CCSDS Space Packets and CDM parsing, orbital element/TLE ingestion via SGP4, optional SPICE kernels and station data, all with provenance and input validation.
- Integrations must be simulation-only by default; no automatic production uplink.
- Use versioned job/status models, idempotency keys, job cancellation and safe retry. Support local filesystem then optional object storage.
- Secret management, input isolation and HTTP auth (when remote service exposed); never hardcode keys.

### Acceptance

- Minimal SDK and REST examples run against test scenarios; invalid input returns structured errors.
- API contract tests and compatibility tests pass.

---

## 15. WP-014 — Space-Env Studio (developer / customer UI)

### Implement

- Scenario editor with schema-driven form, JSON/YAML import/export and validation messages.
- Simulation controls: run, pause, reset, branch, replay and inspection of events/telemetry.
- Orbital visualization and ground track (reuse current viewer where practical), spacecraft state panels, station contact windows and EO target coverage.
- Training dashboard with reward/success/safety time series, run comparison and model/seed/backend metadata.
- Mission planning dashboard with proposed schedule, priority requests, ground station contention, fuel/energy/storage usage and infeasibility explanations.
- Safety / evaluation page showing constraint violations, uncertainty, reference-validation status, test-suite versions and source links.
- Export reproducible HTML/JSON reports, not screenshot-only results.
- Accessibility, empty/error/loading states, secure rendering of imported user text and UI e2e tests.
- If a full UI cannot be justified by the inspected architecture, implement a minimal working Studio atop existing viewer and stable API; document deferred polish separately.

### Acceptance

- A user can load a bundled EO scenario, run a simulated mission, compare solver/heuristic/policy, inspect constraints and replay a failure without editing code.

---

## 16. WP-015 — Deployment, observability and security

### Implement

- Multi-stage containers, non-root runtime, pinned images and base image vulnerability scans.
- Local `docker compose` developer stack for API, worker, experiment tracking and optional database/object storage.
- Optional self-hosted deployment manifests (Kubernetes/Helm) only after local deployment is verified.
- Structured JSON logs, correlation/episode/job IDs, metrics, tracing hooks, health/readiness checks.
- RBAC for remote API, team workspace isolation, audit retention controls, rate limits and resource quotas.
- Background job queue with retry/restart semantics, checkpointing and cancellation.
- CI supply-chain checks: SBOM, dependency/license scans, secret scans, artifact checksums, signed releases where feasible.
- Backups, restore drill, migrations and operating runbooks for any stateful service.
- Threat model imported data, tool-use injection, arbitrary code execution, deserialization and file path handling.

### Acceptance

- Clean local deployment and test scenario via API/worker/UI succeed; security smoke tests and restart/restore scenario pass.

---

## 17. WP-016 — Commercialization-ready demonstrator

### MVP: Earth Observation Mission Planning and Agent Evaluation

Build an end-to-end demo using a **synthetic/reference fleet** and clearly labeled source assumptions:

1. Configure satellites, ground stations, task requests, priorities, resources and mission horizon.
2. Generate target/pass opportunities and constraints.
3. Produce plans with greedy heuristic, CP-SAT/MILP optimizer and trained agent.
4. Validate every plan with independent resource and safety checks.
5. Simulate and report executed mission outcomes; show failure examples and uncertainty.
6. Compare task completion, delivery latency, constraints, resource usage, planning latency and robustness.
7. Show results in Studio and export shareable reproducible benchmark reports.
8. Demonstrate an LLM agent proposing and explaining a plan using sandbox tools, but require verifier/approval before simulated execution.

### Commercial deliverables

- `docs/product/PRODUCT_REQUIREMENTS.md`: ideal customer, explicit use case, non-goals, differentiation, buyer outcomes.
- `docs/product/INTEGRATION_GUIDE.md`: supported interfaces, deployment assumptions and limitations.
- `docs/product/SECURITY_AND_TRUST.md`: threat model, access and data handling, validation boundaries.
- `docs/product/PILOT_ACCEPTANCE_TEMPLATE.md`: customizable customer KPIs, reference data needs, sign-off conditions.
- `docs/product/LICENSING_REVIEW.md`: current Apache-2.0 terms, dependency licenses and any proposed optional commercial components (do not unilaterally change existing code license).
- `examples/eo_commercial_demo/` with scenario, policy/configs, instructions, seed-fixed sample output and complete demo regression.

**Do not fabricate customers, pricing validation, paid pilots, mission qualification or operational authorizations.** Such work must be marked as requiring external business/engineering confirmation.

---

## 18. Cross-cutting implementation specifications

### 18.1 Suggested Python contracts (adapt after code audit)

```python
from typing import Protocol, Mapping, Any

class PhysicsBackend(Protocol):
    name: str
    version: str
    def propagate(self, state: Any, dt_seconds: float, *, parameters: Mapping[str, Any]) -> Any: ...

class Simulation(Protocol):
    def reset(self, *, seed: int, scenario: Any) -> tuple[Any, dict]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]: ...
    def snapshot(self) -> bytes: ...
    def restore(self, payload: bytes) -> None: ...

class SafetyVerifier(Protocol):
    def verify(self, state: Any, proposed_action: Any, *, horizon_seconds: float) -> Any: ...
```

Make actual types strict and serializable. Avoid `Any` in final public schemas; this skeleton only indicates responsibility boundaries.

### 18.2 Run manifest fields

`run_id`, UTC timestamp, git SHA, package version, Python version, dependency lock hash, operating system, physics backend/version, environment ID, scenario ID/hash, observation/action schema versions, seed(s), policy ID/hash, reward version, constraint version, validation status, results artifact URIs and checksum(s).

### 18.3 Standard acceptance scenario families

- Nominal EO capture and delivery.
- Station contention between heterogeneous satellites.
- Power/thermal survival through eclipse.
- Radio fault during a short pass.
- High-uncertainty conjunction and candidate avoidance maneuver.
- Satellite constellation with intermittent crosslinks and packet routing.
- Delayed/missing/contradictory telemetry for LLM mission agent.
- Domain shift in drag, ephemeris, timekeeping and hardware limits.
- Infeasible task request and safe abstention.

### 18.4 Priority handling and dependency graph

```text
WP-001 Audit & Quality
    ├── WP-002 Simulation Kernel
    │   ├── WP-003 Physics Validation
    │   ├── WP-004 Scenario DSL
    │   └── WP-006 Gym/MARL APIs
    ├── WP-005 Mission Planning (depends on scenarios + physics)
    ├── WP-007 RL Infrastructure (depends on stable Gym API)
    ├── WP-008 Data/LLM Training (depends on trajectories + metadata)
    ├── WP-009 Agent Tool Framework (depends on mission API + safety)
    ├── WP-010 Safety Framework (depends on typed state/commands)
    ├── WP-011 Benchmarks (depends on baseline + validation)
    ├── WP-012 Scale (after correctness and profiling)
    ├── WP-013 SDK/API (after stable domain contracts)
    ├── WP-014 Studio (after API + report contracts)
    ├── WP-015 Deployment (after runnable services)
    └── WP-016 EO Product Demo (integrates validated components)
```

Execute in **dependency order**, not calendar order. Parallelize packages when testable interfaces exist. P0: WP-001/002/003/004/005/006/007/010/011 and EO demo core. P1: WP-008/009/012/013/014/015 and commercial packaging. P2: advanced robotic docking and wider fidelity extensions. Dependencies may shift after audit; record rationales.

---

## 19. Final acceptance gates (must all be reported, not necessarily all passed)

### Gate A — Build and compatibility

- [ ] Clean install + build works.
- [ ] Supported Python matrix passes.
- [ ] Existing examples and registered environments retain public compatibility.
- [ ] Lint, types, unit and integration tests pass or known failures clearly recorded.

### Gate B — Reproducibility

- [ ] Scenarios, seeds, configs and dependencies recorded in run manifest.
- [ ] Snapshot/restore/replay verified with fixtures.
- [ ] Model checkpoint reload reproduces declared evaluation behavior.

### Gate C — Physics and mission correctness

- [ ] Physics reference tests and measured errors published.
- [ ] Unvalidated models clearly labeled and excluded from unsupported assurance claims.
- [ ] EO, link budget, power/thermal, maneuvers and conjunction constraints tested on negative cases.

### Gate D — RL, ML and agent evaluation

- [ ] Classical baselines and RL baselines operate under the same test conditions.
- [ ] Independent training/test scenarios and seed splits enforced.
- [ ] Confidence intervals/failure distributions available.
- [ ] LLM tool agent supports typed calls, replay and safe abstention.

### Gate E — Safety

- [ ] Hard constraints enforced independently of agent reward.
- [ ] No real-command execution path enabled by default.
- [ ] Adversarial/fault/OOD suites generate traceable safety reports.
- [ ] Remaining hazards explicitly recorded.

### Gate F — Product and operations

- [ ] Full EO reference demo runs from fresh install.
- [ ] Studio + API/CLI reports reference the same ground-truth run IDs.
- [ ] Local deployment, authentication and observability tested.
- [ ] Pilot integration, model assumptions, licenses and security docs published.

### Mandatory final report format

At the end generate `docs/RELEASE_READINESS.md` with:

1. Exact checked-out SHA and final implementation SHA.
2. Implemented features with paths, commands and evidence links.
3. Detailed automated test results and skipped jobs.
4. Physics validation results, error budgets and any **NOT VALIDATED** models.
5. RL versus random/heuristic/optimizer numbers for specified benchmark suites (never invent results).
6. Runtime/scale performance data and hardware information.
7. Critical risk register (severity, impact, reproduction steps, mitigation).
8. Remaining blockers requiring humans, customer data, lab tests, qualified domain validation, credentials or external permissions.
9. Step-by-step run instructions for SDK, CLI, Studio, EO demo and local deployment.
10. Explicit readiness verdict by category: research tool, engineering evaluation platform, customer pilot, real mission/flight qualification (do not conflate).

---

## 20. Autonomous execution loop

Repeat this loop until all implementable tasks are verified or blocked with evidence:

```text
1. Inspect latest source + implementation status.
2. Select highest-priority unblocked work package according to dependencies.
3. Write exact contract, acceptance tests and minimal integration fixture.
4. Implement using existing code where possible.
5. Run focused tests + global regression checks.
6. Fix defects without disabling or weakening tests.
7. Run relevant numerical/benchmark/reference validation.
8. Update docs, status, manifests, examples and release notes.
9. Commit cohesive changes, record commit hash and next tasks.
10. Re-evaluate dependencies and continue.
```

If there are external blockers, create a reproducible stub/mock **clearly labeled as synthetic**, document the integration contract, and proceed with the rest of the system. Never silently replace unavailable real data or reference models with synthetic values while continuing to claim equivalence.

### Required summary returned to project owner

- Implemented and tested work packages.
- Test commands and actual observed results.
- Benchmark and physics validation results with provenance.
- What remains unverified and exactly why.
- How to launch the working product demonstrator.
- Exact files to review and proposed next merge steps.

---

## Source-of-truth references

- Project: https://github.com/researchim-ai/space-env/tree/develop
- README: https://raw.githubusercontent.com/researchim-ai/space-env/develop/README.md
- Documented architecture: https://raw.githubusercontent.com/researchim-ai/space-env/develop/docs/IMPLEMENTATION_PLAN.md
- Package configuration: https://raw.githubusercontent.com/researchim-ai/space-env/develop/pyproject.toml

> This specification is an execution target, not evidence that any proposed feature already exists. The autonomous agent must inspect the current repository and use test results—not documentation claims—to determine completion.
