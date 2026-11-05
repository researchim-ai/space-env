"""Isolated installed-wheel probe, with exact tested dependency versions."""

from __future__ import annotations

import argparse
import json
import site
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import space_env
    from space_env.planning.release_benchmark import Suite
    from space_env.planning.release_cases import release_suite
    from space_env.scaling.restart import build_identity

    prefix = args.prefix.resolve()
    assert sys.flags.isolated and sys.prefix != sys.base_prefix
    assert not site.ENABLE_USER_SITE
    assert Path(space_env.__file__).resolve().is_relative_to(prefix)
    pyvenv = (prefix / "pyvenv.cfg").read_text().lower()
    assert "include-system-site-packages = false" in pyvenv
    versions = {}
    for line in args.constraints.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = metadata.version(name)
        assert actual == expected, (name, actual, expected)
        versions[name] = actual
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    s = Suite.model_validate(release_suite())
    assert len(s.cases) == 32
    from space_env.api.server import handle_get

    for path in ("/studio", "/operator", "/viz/app.js"):
        status, body, _ = handle_get(path, {})
        assert status == 200 and body, path
    body = {
        "isolated": True,
        "dependency_clean": True,
        "package_path": str(Path(space_env.__file__).resolve()),
        "python": sys.version,
        "dependencies": versions,
        "frozen_suite_cases": len(s.cases),
        "build": build_identity(),
        "installed_assets_checked": True,
        "scope": "local installed-profile probe; not remote matrix or full CI",
    }
    if args.output.exists():
        raise ValueError("install receipt must not exist")
    args.output.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dependency_clean": True,
                "frozen_suite_cases": 32,
                "pinned_dependencies": len(versions),
            }
        )
    )


if __name__ == "__main__":
    main()
