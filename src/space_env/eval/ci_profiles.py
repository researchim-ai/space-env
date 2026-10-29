"""Три профиля проверки. Полный профиль — дымовые миссии, не кампания на 72 часа."""
from __future__ import annotations

import subprocess
import sys

PROFILES = {
    "fast": (
        "tests/test_version.py",
        "tests/test_cards.py",
        "tests/test_agent_surface.py",
    ),
    "reference": (
        "tests/test_dynamics.py",
        "tests/test_physics_rest.py",
    ),
    "full": (
        "tests/test_mission_benchmark.py",
        "tests/test_session_dataset.py",
    ),
}


def command(profile: str) -> list[str]:
    files = PROFILES[profile]
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
