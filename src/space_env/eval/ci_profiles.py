"""Contract, reference and complete repository regression profiles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

PROFILES = {
    "fast": (
        "tests/test_version.py",
        "tests/test_cards.py",
        "tests/test_agent_surface.py",
        "tests/test_execution_integrity.py",
        "tests/test_power_coast.py",
        "tests/test_resource_timeline.py",
        "tests/test_portable_run.py",
        "tests/test_agent_run.py",
        "tests/test_mission_contract.py",
    ),
    "reference": (
        "tests/test_dynamics.py",
        "tests/test_physics_rest.py",
        "tests/test_reference_physics.py",
        "tests/test_reference_access.py",
        "tests/test_reference_operations.py",
        "tests/test_runtime.py",
        "tests/test_reference_dynamic.py",
        "tests/test_reference_covariance.py",
        "tests/test_reference_geometry_events.py",
        "tests/test_reference_lifecycle.py",
        "tests/test_spacecraft_bus.py",
        "tests/test_reference_pointing.py",
        "tests/test_coupled_mission.py",
        "tests/test_mission_study.py",
        "tests/test_controller_clock.py",
        "tests/test_fixed_actuators.py",
        "tests/test_full_orbit.py",
        "tests/test_mission_plan.py",
        "tests/test_forecast_delivery.py",
        "tests/test_chain_planning.py",
        "tests/test_incident_planning.py",
        "tests/test_actor_online.py",
        "tests/test_demand_journal.py",
        "tests/test_release_benchmark.py",
        "tests/test_area_coverage.py",
        "tests/test_payload_quality.py",
        "tests/test_observations.py",
        "tests/test_resource_lanes.py",
        "tests/test_actuation.py",
        "tests/test_rigid_attitude.py",
        "tests/test_rigid_mission.py",
        "tests/test_rotating_thrust.py",
        "tests/test_momentum_unloading.py",
        "tests/test_thermal_network.py",
        "tests/test_material_capacity.py",
        "tests/test_radio_network.py",
        "tests/test_custody_network.py",
        "tests/test_runtime_tasks.py",
        "tests/test_planning.py",
        "tests/test_factory.py",
        "tests/test_scaling.py",
        "tests/test_fleet_scaling.py",
        "tests/test_campaign_qualification.py",
        "tests/test_projection_reads.py",
        "tests/test_native_copy.py",
        "tests/test_worker_progress.py",
        "tests/test_ground_scheduling.py",
        "tests/test_physics_envelopes.py",
        "tests/test_pointing_validation.py",
        "tests/test_calibration.py",
        "tests/test_ensemble_qualification.py",
        "tests/test_process_restart.py",
        "tests/test_independent_geometry.py",
        "tests/test_verified_exports.py",
        "tests/test_operator.py",
        "tests/test_operator_durable.py",
        "tests/test_operator_maintenance.py",
        "tests/test_operator_artifacts.py",
    ),
    "full": tuple(
        str(path.relative_to(_ROOT))
        for path in sorted((_ROOT / "tests").glob("test_*.py"))
    ),
}


def command(profile: str, shard: str | None = None) -> list[str]:
    files = PROFILES[profile]
    extra = []
    if shard is not None:
        if profile != "full" or shard not in {
            "other",
            "campaign",
            "tasking-base",
            "tasking-sixteen",
        }:
            raise ValueError("shard requires full and one declared partition")
        if shard == "other":
            files = tuple(
                f
                for f in files
                if f not in {"tests/test_campaign.py", "tests/test_tasking.py"}
            )
        elif shard == "campaign":
            files = ("tests/test_campaign.py",)
        else:
            files = ("tests/test_tasking.py",)
            extra = [
                "-k",
                "not sixteen_vehicles"
                if shard == "tasking-base"
                else "sixteen_vehicles",
            ]
    if not files:
        raise ValueError("тесты профиля отсутствуют; запускайте из исходного checkout")
    return [sys.executable, "-m", "pytest", "-q", *files, *extra]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if (
        len(args) not in {1, 3}
        or args[0] not in PROFILES
        or (len(args) == 3 and args[1] != "--shard")
    ):
        names = ", ".join(PROFILES)
        print(f"нужен один профиль: {names}", file=sys.stderr)
        return 2
    try:
        invocation = command(args[0], args[2] if len(args) == 3 else None)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return subprocess.call(invocation)


if __name__ == "__main__":
    raise SystemExit(main())
