"""Resolve installed dependency closure, never the whole developer environment.

Produces explicit version constraints for the tested Linux reference/coverage/MARL
profile. Build frontend is provisioned separately. Re-run in a prepared environment
and review differences before replacing the checked-in lock snapshot.
"""

import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import tomllib
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def freeze():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    requirements = list(project["dependencies"])
    for extra in ("reference", "coverage", "marl"):
        requirements += project["optional-dependencies"][extra]
    pending = [(r.name, set(r.extras)) for r in map(Requirement, requirements)] + [
        ("pytest", set()),
        ("build", set()),
    ]
    seen = {}
    versions = {}
    licenses = {
        "space-env": {
            "version": project["version"],
            "requires_python": project["requires-python"],
            "license_expression": project.get("license"),
            "license_files": project.get("license-files", []),
            "scope": "project declaration; third-party review is separate",
        }
    }
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if name in seen and extras <= seen[name]:
            continue
        seen.setdefault(name, set()).update(extras)
        dist = distribution(name)
        if name != "space-env":
            versions[name] = dist.version
        licenses[name] = {
            "version": dist.version,
            "requires_python": dist.metadata.get("Requires-Python"),
            "license_expression": dist.metadata.get("License-Expression"),
            "license_files": dist.metadata.get_all("License-File", []),
            "license_field": dist.metadata.get("License"),
            "classifiers": [
                x
                for x in dist.metadata.get_all("Classifier", [])
                if x.startswith("License ::")
            ],
        }
        for raw in dist.requires or []:
            req = Requirement(raw)
            env = default_environment()
            if req.marker and not any(
                req.marker.evaluate(env | {"extra": e}) for e in ("", *seen[name])
            ):
                continue
            installed = distribution(req.name).version
            if req.specifier and not req.specifier.contains(
                installed, prereleases=True
            ):
                raise ValueError(
                    f"inconsistent installed dependency: {name} requires {req}, found {installed}; resolve a clean environment before freezing"
                )
            pending.append((req.name, set(req.extras)))
    return versions, licenses


if __name__ == "__main__":
    versions, licenses = freeze()
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-py311-py312-linux.txt").write_text(
        "# Tested dependency closure; Snapshot from Linux CPython 3.12; 3.11-3.12 candidates require CI matrix validation.\n"
        + "\n".join(f"{k}=={v}" for k, v in sorted(versions.items()))
        + "\n"
    )
    (output / "release-licenses.json").write_text(
        json.dumps(licenses, sort_keys=True, indent=2) + "\n"
    )
