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
        "tests/test_area_coverage.py",
        "tests/test_payload_quality.py",
        "tests/test_observations.py",
        "tests/test_resource_lanes.py",
        "tests/test_actuation.py",
        "tests/test_rigid_attitude.py",
    ),
    "full": tuple(str(path.relative_to(_ROOT)) for path in sorted((_ROOT / "tests").glob("test_*.py"))),
}


def command(profile: str) -> list[str]:
    files = PROFILES[profile]
    if not files:
        raise ValueError("тесты профиля отсутствуют; запускайте из исходного checkout")
    return [sys.executable, "-m", "pytest", "-q", *files]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in PROFILES:
        names = ", ".join(PROFILES)
        print(f"нужен один профиль: {names}", file=sys.stderr)
        return 2
    return subprocess.call(command(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
