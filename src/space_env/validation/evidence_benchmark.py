"""Local four-track evidence benchmark; no universal ranking or remote CI.

Run: python -m space_env.validation.evidence_benchmark --output artifacts/evidence
Audit: python -m space_env.validation.evidence_benchmark --audit artifacts/evidence
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import (
    LLMEpisode,
    seal,
    sources,
    verify_record,
)
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import TaskSession
from space_env.tasks.catalog import _recipes
from space_env.validation.learning_evidence import learning_comparison
from space_env.validation.trajectory_exchange import (
    compare_external,
    compare_states,
    native_trajectories,
    trajectory_manifest,
)

POLICIES = ("wait", "public_memory", "reference_schedule")
SMOKE_CASES = (
    "physical/delivery",
    "physical/attitude-nominal",
    "physical/orbit-nominal",
    "physical/relative-nominal",
    "finite/finite-j2",
    "navigation/no-contact",
)
ENGINES = {
    "Basilisk": "Basilisk",
    "BSK-RL": "bsk_rl",
    "Orekit": "orekit_jpype",
    "Tudat": "tudatpy",
}


def benchmark_manifest(profile="full"):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown benchmark profile")
    cases = _recipes()
    selected = list(cases) if profile == "full" else list(SMOKE_CASES)
    return {
        "schema": "space-env.evidence-manifest/1",
        "profile": profile,
        "missions": [
            {"id": key, "recipe_sha256": digest(cases[key])} for key in selected
        ],
        "policies": list(POLICIES),
        "max_tool_calls": 128,
        "scale": {
            "actors": [2, 4, 8, 16] if profile == "full" else [2, 4],
            "products_per_actor": 4,
            "repeats": 2,
            "worker_threads": 1,
        },
        "trajectory_manifest": trajectory_manifest(profile),
        "learning": "not measured without separate paired held-out external LLM records",
        "scope": "Frozen synthetic regression matrix; full means this matrix, not complete physics or industrial qualification.",
    }


def _write(root, name, value):
    atomic_write(root / name, canonical_json(value))


def _read(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes benchmark directory")
    return json.loads(path.read_text())


def _environment():
    distributions = {}
    for name in ("space-env", "numpy", "scipy", "pydantic", "boinor", "pyerfa"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "distributions": distributions,
    }


def _inventory():
    rows = []
    for engine, module in ENGINES.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        rows.append(
            {
                "engine": engine,
                "module_discoverable": available,
                "status": "not_executed" if available else "not_installed",
                "scope": "module discovery only; not an engine execution or capability score",
            }
        )
    return rows


def _summary(rows):
    return {
        policy: {
            "attempts": sum(r["policy"] == policy for r in rows),
            "errors": sum(
                r["policy"] == policy and r["status"] == "error" for r in rows
            ),
            "successes": sum(
                r["policy"] == policy and r.get("metrics", {}).get("success", False)
                for r in rows
            ),
            "successful_clean_episodes": sum(
                r["policy"] == policy
                and r.get("metrics", {}).get("sft_eligible", False)
                for r in rows
            ),
            "privileged": policy == "reference_schedule",
        }
        for policy in POLICIES
    }


def _missions(manifest, root):
    cases = _recipes()
    rows = []
    for item in manifest["missions"]:
        key = item["id"]
        case = cases[key]
        for policy in POLICIES:
            row = {"case": key, "policy": policy}
            started = time.perf_counter()
            episode = None
            try:
                session = TaskSession(case["scenario"], case["task"])
                episode = LLMEpisode(
                    session,
                    task_text=case.get(
                        "task_text", "Execute the declared mission goal."
                    ),
                    group=key,
                    split="test",
                    max_calls=manifest["max_tool_calls"],
                    policy_origin="scripted_privileged"
                    if policy == "reference_schedule"
                    else "scripted_public",
                )
                chooser = (
                    reference_policy(
                        case.get("reference_steps", case.get("steps", [])),
                        {a.id: a.role for a in session._task.actions},
                    )
                    if policy == "reference_schedule"
                    else public_policy(policy, session._task.decision_sec)
                )
                record = run_agent(episode, chooser)
                verify_record(record)
                name = "episodes/" + key.replace("/", "--") + "--" + policy
                _write(root, name + ".json", record)
                row.update(
                    status="completed",
                    artifact=name + ".json",
                    episode_sha256=record["sha256"],
                    metrics=record["metrics"],
                )
            except Exception as exc:  # noqa: BLE001 -- retain all predeclared attempts
                row.update(
                    status="error", error_type=type(exc).__name__, error=str(exc)[:1000]
                )
                if episode is not None:
                    partial = episode.record()
                    name = (
                        "episodes/"
                        + key.replace("/", "--")
                        + "--"
                        + policy
                        + "--partial.json"
                    )
                    _write(root, name, partial)
                    row["partial_artifact"] = name
            row["wall_sec"] = time.perf_counter() - started
            rows.append(row)
            _write(
                root,
                "mission-progress.json",
                {"expected": len(manifest["missions"]) * len(POLICIES), "rows": rows},
            )
    return {
        "rows": rows,
        "summary": _summary(rows),
        "scope": "scripted public policies and privileged reference schedule; no LLM calls or learned-agent superiority",
    }


def _scaling(manifest, root):
    from space_env.scaling.benchmark import run_scale
    from space_env.scaling.workloads import fleet_workload

    scale = manifest["scale"]
    workloads = []
    for actors in scale["actors"]:
        workload = fleet_workload(
            {
                "id": f"evidence-{actors}",
                "actors": actors,
                "planes": min(4, actors),
                "products": actors * scale["products_per_actor"],
                "repeats": scale["repeats"],
            }
        )
        workload["worker_threads"] = scale["worker_threads"]
        workloads.append(workload)
    return run_scale(
        {"id": "evidence-" + manifest["profile"], "workloads": workloads, "workers": 1},
        root / "scale",
    )


def run(output, *, profile="full", external_paths=(), learning_path=None):
    root = Path(output)
    if root.exists():
        raise ValueError("benchmark output must be new")
    frozen = benchmark_manifest(profile)
    before = sources()
    root.mkdir(parents=True)
    (root / "episodes").mkdir()
    _write(root, "manifest.json", seal(frozen))
    _write(root, "provenance.json", {"sources": before, "environment": _environment()})
    tracks = {}

    def measure(name, function):
        started = time.perf_counter()
        try:
            value = function()
            _write(root, name + ".json", value)
            tracks[name] = {
                "status": "completed",
                "artifact": name + ".json",
                "sha256": digest(value),
            }
        except Exception as exc:  # noqa: BLE001 -- track failure is evidence, never a missing row
            tracks[name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        tracks[name]["wall_sec"] = time.perf_counter() - started
        _write(root, "progress.json", {"tracks": tracks})

    def physics():
        from space_env.validation.ai_reference_suite import run_suite
        from space_env.validation.force_crosscheck import force_crosscheck

        analytic = run_suite()
        forces = force_crosscheck()
        trajectories = native_trajectories(frozen["trajectory_manifest"])
        return {
            "analytic": analytic,
            "force_formula_crosscheck": forces,
            "trajectories": trajectories,
            "passed": analytic["passed"]
            and forces["passed"]
            and trajectories["passed"],
            "scope": "analytic oracles plus boinor formula/LSODA checks; shared constants, frames, atmosphere and ephemerides are not independently qualified",
        }

    measure("physics", physics)
    measure("missions", lambda: _missions(frozen, root))
    measure("scaling", lambda: _scaling(frozen, root))
    submissions = []
    engines = set()
    for index, path in enumerate(external_paths):
        value = json.loads(Path(path).read_text())
        comparison = compare_external(frozen["trajectory_manifest"], value)
        if comparison["engine"] in engines:
            raise ValueError("duplicate external engine submission")
        engines.add(comparison["engine"])
        filename = f"external-{index}.json"
        _write(root, filename, value)
        submissions.append({"artifact": filename, "comparison": comparison})
    learning = {
        "status": "not_measured",
        "reason": "No paired held-out before/after external LLM evaluation was supplied.",
    }
    if learning_path is not None:
        data = json.loads(Path(learning_path).read_text())
        result = learning_comparison(
            data["before_records"], data["after_records"], data["disclosure"]
        )
        _write(root, "learning-input.json", data)
        _write(root, "learning.json", result)
        learning = {
            "status": "paired_native_outcomes_verified",
            "artifact": "learning.json",
            "sha256": digest(result),
        }
    if before != sources():
        raise ValueError("source files changed during benchmark")
    report = seal(
        {
            "schema": "space-env.evidence-benchmark/1",
            "manifest_sha256": digest(frozen),
            "model_sources": before,
            "environment": _environment(),
            "tracks": tracks,
            "external_engines": _inventory(),
            "external_submissions": submissions,
            "learning": learning,
            "superiority_established": False,
            "scope": "No aggregate ranking. Unexecuted engines are unmeasured, not defeated. Synthetic regression success is not industrial/flight qualification.",
        }
    )
    _write(root, "report.json", report)
    return report


def audit(directory):
    root = Path(directory)
    report, manifest = _read(root, "report.json"), _read(root, "manifest.json")
    if (
        report != seal({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
    ):
        raise ValueError("report source/seal mismatch")
    frozen = {k: v for k, v in manifest.items() if k != "sha256"}
    if (
        manifest != seal(frozen)
        or frozen != benchmark_manifest(frozen["profile"])
        or digest(frozen) != report["manifest_sha256"]
    ):
        raise ValueError("benchmark manifest mismatch")
    for track in report["tracks"].values():
        if (
            track["status"] == "completed"
            and digest(_read(root, track["artifact"])) != track["sha256"]
        ):
            raise ValueError("track artifact mismatch")
    if (
        report.get("schema") != "space-env.evidence-benchmark/1"
        or report.get("superiority_established") is not False
        or set(report["tracks"]) != {"physics", "missions", "scaling"}
    ):
        raise ValueError("invalid benchmark claims or track matrix")
    checked = 0
    if report["tracks"]["missions"]["status"] == "completed":
        missions = _read(root, "missions.json")
        expected = [(c["id"], p) for c in frozen["missions"] for p in POLICIES]
        if [(r["case"], r["policy"]) for r in missions["rows"]] != expected or missions[
            "summary"
        ] != _summary(missions["rows"]):
            raise ValueError("mission denominator/summary mismatch")
        recipes = _recipes()

        for row in missions["rows"]:
            if row["status"] == "error":
                if "partial_artifact" in row:
                    verify_record(_read(root, row["partial_artifact"]))
                continue
            record = _read(root, row["artifact"])
            verify_record(record)
            case = recipes[row["case"]]
            expected_initial = TaskSession(case["scenario"], case["task"]).checkpoint(
                compact=True
            )
            if (
                record["metrics"] != row["metrics"]
                or record["sha256"] != row["episode_sha256"]
                or record["initial_checkpoint"] != expected_initial
            ):
                raise ValueError("native mission recipe/outcome binding mismatch")
            expected_config = {
                "task_text": case.get(
                    "task_text", "Execute the declared mission goal."
                ),
                "group": row["case"],
                "split": "test",
                "max_calls": frozen["max_tool_calls"],
                "max_dialogue_bytes": 16000000,
                "policy_origin": "scripted_privileged"
                if row["policy"] == "reference_schedule"
                else "scripted_public",
            }
            if record["configuration"] != expected_config:
                raise ValueError("baseline configuration/provenance mismatch")
            chooser = (
                reference_policy(
                    case.get("reference_steps", case.get("steps", [])),
                    {a["id"]: a["role"] for a in case["task"]["actions"]},
                )
                if row["policy"] == "reference_schedule"
                else public_policy(row["policy"], case["task"]["decision_sec"])
            )
            for index, event in enumerate(record["events"]):
                expected_call = chooser(
                    record["messages"][: 2 + 2 * index], record["tools"]
                )
                if (
                    expected_call is None
                    or expected_call["name"] != event["name"]
                    or expected_call["arguments"] != event["arguments"]
                ):
                    raise ValueError("baseline policy/action binding mismatch")
            # Held-out benchmark episodes are evidence, never training exports.
            if any(
                (root / (row["artifact"][:-5] + "-" + fmt + ".jsonl")).exists()
                for fmt in ("rl", "sft")
            ):
                raise ValueError("held-out training export present")
            checked += 1
    for item in report["external_submissions"]:
        if (
            compare_external(
                frozen["trajectory_manifest"], _read(root, item["artifact"])
            )
            != item["comparison"]
        ):
            raise ValueError("external comparison mismatch")
    if report["learning"]["status"] != "not_measured":
        data = _read(root, "learning-input.json")
        actual = learning_comparison(
            data["before_records"], data["after_records"], data["disclosure"]
        )
        if (
            actual != _read(root, "learning.json")
            or digest(actual) != report["learning"]["sha256"]
        ):
            raise ValueError("learning comparison mismatch")
    if report["tracks"]["physics"]["status"] == "completed":
        physics = _read(root, "physics.json")
        rows = [
            compare_states(c, t["states_si"])
            for c, t in zip(
                frozen["trajectory_manifest"]["cases"],
                physics["trajectories"]["trajectories"],
                strict=True,
            )
        ]
        if rows != physics["trajectories"]["rows"]:
            raise ValueError("physics trajectory outcome mismatch")
        from space_env.validation.ai_reference_suite import verify_report

        analytic = physics["analytic"]
        forces = physics["force_formula_crosscheck"]
        if not verify_report(analytic) or analytic["passed"] != all(
            r["passed"] for r in analytic["checks"]
        ):
            raise ValueError("analytic reference receipt mismatch")
        force_passed = all(
            r["error_m_s2"] <= r["tolerance_m_s2"] and r["passed"]
            for r in forces["forces"]
        ) and all(
            r["position_error_m"] <= r["position_tolerance_m"]
            and r["velocity_error_m_s"] <= r["velocity_tolerance_m_s"]
            and r["passed"]
            for r in forces["j2_arcs"]
        )
        trajectory_passed = all(r["passed"] for r in rows)
        if (
            forces["passed"] != force_passed
            or physics["trajectories"]["passed"] != trajectory_passed
            or physics["passed"]
            != (analytic["passed"] and force_passed and trajectory_passed)
        ):
            raise ValueError("physics gate summary mismatch")
    return {
        "valid": True,
        "native_episodes_replayed": checked,
        "scope": "native episodes and analytic trajectory metrics replayed; scale artifacts integrity checked, timing and force checks not rerun",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--external", type=Path, action="append", default=[])
    parser.add_argument("--learning", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.audit:
        if args.external or args.learning or args.manifest_only:
            parser.error("audit does not accept new inputs")
        print(canonical_json(audit(args.audit)))
    elif args.manifest_only:
        if args.output.exists():
            parser.error("manifest output must be new")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(
            args.output.parent, args.output.name, seal(benchmark_manifest(args.profile))
        )
    else:
        report = run(
            args.output,
            profile=args.profile,
            external_paths=args.external,
            learning_path=args.learning,
        )
        print(
            canonical_json(
                {
                    "report": str(args.output / "report.json"),
                    "tracks": report["tracks"],
                    "superiority_established": False,
                }
            )
        )


if __name__ == "__main__":
    main()
