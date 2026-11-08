"""Offline RF verification and fresh-process measurements of bounded network missions.

No competitor, hardware calibration or operational-accuracy claims follow from
these synthetic workloads. Run explicitly; no CI or remote service is involved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from space_env.reference.radio import RFReceiver, RFTransmitter, rf_budget
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

MODULE = "space_env.validation.network_evidence"


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "validation/network_evidence.py",
            "api/network_workspace.py",
            "api/network_dynamic.py",
            "api/workspace.py",
            "planning/joint_network.py",
            "planning/network_benchmark.py",
            "planning/network_optimal.py",
            "planning/terminal_sequence.py",
            "reference/radio_network.py",
            "planning/network_demand_benchmark.py",
            "planning/network_selector.py",
            "planning/network_selector_study.py",
            "reference/orbit.py",
            "reference/frames.py",
            "reference/radio.py",
            "reference/dtn.py",
            "reference/operations.py",
            "reference/lifecycle.py",
            "reference/spacecraft_bus.py",
            "runtime/mission.py",
        )
    }


def seal(body):
    return body | {"sha256": digest(body)}


def peak_rss():
    """Linux mm high-water mark resets at exec, unlike inherited ru_maxrss."""
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            fields = line.split()
            if (
                fields
                and fields[0] == "VmHWM:"
                and len(fields) == 3
                and fields[2] == "kB"
            ):
                return int(fields[1]) * 1024, "linux-proc-VmHWM-after-exec"
        raise RuntimeError("Linux process VmHWM unavailable")
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(
        peak if sys.platform == "darwin" else peak * 1024
    ), "getrusage-may-include-pre-exec-peak"


def rf_evidence():
    """Separate linear Friis/noise arithmetic, no production budget helpers."""
    # SI definitions, deliberately not imported from the production RF module.
    light_speed, boltzmann = 299792458.0, 1.380649e-23
    rows = []
    inputs = []
    for ghz in (2.2, 8.4, 26.0):
        for km in (500.0, 2000.0, 10000.0):
            inputs.append(
                {"id": f"f{ghz}/r{km}", "frequency_hz": ghz * 1e9, "range_m": km * 1000}
            )
    inputs += [
        {"id": name, "frequency_hz": 2.2e9, "range_m": 2e6, **options}
        for name, options in (
            ("threshold-above", {"threshold_offset_db": 0.01}),
            ("threshold-below", {"threshold_offset_db": -0.01}),
            ("narrow-bandwidth", {"bandwidth_hz": 100000.0}),
            ("doppler-outside", {"doppler_hz": 2000000.0}),
            ("rate-cap", {"rate_cap_bps": 10000.0}),
            ("spectrum-outside", {"frequency_half_band_hz": 100000.0}),
            (
                "fade-and-loss",
                {"fade_db": 3.0, "feeder_loss_db": 2.0, "other_loss_db": 1.0},
            ),
        )
    ]
    for c in inputs:
        tx_input = {
            "frequency_hz": c["frequency_hz"],
            "rf_power_w": 1.0,
            "gain_dbi": 6.0,
            "feeder_loss_db": c.get("feeder_loss_db", 0.0),
            "modes": [
                {
                    "id": str(rate),
                    "information_rate_bps": float(rate),
                    "bits_per_symbol": 2,
                    "code_rate": 0.5,
                    "required_eb_n0_db": 5.0,
                    "implementation_loss_db": 1.0,
                }
                for rate in (250000, 1000000)
            ],
        }
        rx_input = {
            "gt_db_k": 0.0,
            "min_frequency_hz": c["frequency_hz"]
            - c.get("frequency_half_band_hz", 1e8),
            "max_frequency_hz": c["frequency_hz"]
            + c.get("frequency_half_band_hz", 1e8),
            "bandwidth_hz": c.get("bandwidth_hz", 2e6),
            "other_loss_db": c.get("other_loss_db", 0.0),
        }
        wavelength = light_speed / c["frequency_hz"]
        free_space_factor = (wavelength / (4 * math.pi * c["range_m"])) ** 2
        eirp_w = tx_input["rf_power_w"] * 10 ** (
            (tx_input["gain_dbi"] - tx_input["feeder_loss_db"]) / 10
        )
        if "threshold_offset_db" in c:
            desired = 10 ** ((5.0 + 1.0 + c["threshold_offset_db"]) / 10)
            required_gt = desired * boltzmann * 250000 / (eirp_w * free_space_factor)
            rx_input["gt_db_k"] = 10 * math.log10(required_gt)
        cn0_linear = (
            eirp_w
            * free_space_factor
            * 10 ** (rx_input["gt_db_k"] / 10)
            / (
                boltzmann
                * 10 ** ((rx_input["other_loss_db"] + c.get("fade_db", 0.0)) / 10)
            )
        )
        expected_modes = []
        for m in tx_input["modes"]:
            bandwidth = (
                m["information_rate_bps"]
                / (m["bits_per_symbol"] * m["code_rate"])
                * 1.25
            )
            margin = 10 * math.log10(
                cn0_linear
                / m["information_rate_bps"]
                / 10 ** ((m["required_eb_n0_db"] + m["implementation_loss_db"]) / 10)
            )
            eligible = (
                margin >= 0
                and bandwidth <= rx_input["bandwidth_hz"]
                and abs(c.get("doppler_hz", 0.0)) <= 1e6
                and c["frequency_hz"] + c.get("doppler_hz", 0.0) - bandwidth / 2
                >= rx_input["min_frequency_hz"]
                and c["frequency_hz"] + c.get("doppler_hz", 0.0) + bandwidth / 2
                <= rx_input["max_frequency_hz"]
                and m["information_rate_bps"] <= c.get("rate_cap_bps", math.inf)
            )
            expected_modes.append(
                {"id": m["id"], "margin_db": margin, "eligible": eligible}
            )
        selected = next(
            (m["id"] for m in reversed(expected_modes) if m["eligible"]), None
        )
        actual = rf_budget(
            RFTransmitter.model_validate(tx_input),
            RFReceiver.model_validate(rx_input),
            c["range_m"],
            fade_db=c.get("fade_db", 0.0),
            doppler_hz=c.get("doppler_hz", 0.0),
            rate_cap_bps=c.get("rate_cap_bps"),
        )
        fspl = -10 * math.log10(free_space_factor)
        errors = {
            "fspl_db": abs(actual["fspl_db"] - fspl),
            "cn0_db_hz": abs(actual["cn0_db_hz"] - 10 * math.log10(cn0_linear)),
            "margin_db": max(
                abs(a["margin_db"] - b["margin_db"])
                for a, b in zip(actual["modes"], expected_modes)
            ),
        }
        # ITU's rounded GHz/km form is an additional, lower-precision check.
        rounded_fspl = (
            92.44
            + 20 * math.log10(c["frequency_hz"] / 1e9)
            + 20 * math.log10(c["range_m"] / 1000)
        )
        rounded_error = abs(actual["fspl_db"] - rounded_fspl)
        rows.append(
            {
                "id": c["id"],
                "input": {"case": c, "transmitter": tx_input, "receiver": rx_input},
                "expected": {
                    "fspl_db": fspl,
                    "cn0_db_hz": 10 * math.log10(cn0_linear),
                    "selected_mode": selected,
                },
                "actual": actual,
                "absolute_errors": errors,
                "tolerance_db": 1e-10,
                "rounded_itu_error_db": rounded_error,
                "rounded_itu_tolerance_db": 0.02,
                "passed": all(e <= 1e-10 for e in errors.values())
                and rounded_error <= 0.02
                and actual["selected_mode"] == selected
                and [m["eligible"] for m in actual["modes"]]
                == [m["eligible"] for m in expected_modes],
            }
        )
    return seal(
        {
            "schema": "space-env.network-rf-evidence/1",
            "cases": rows,
            "passed": all(r["passed"] for r in rows),
            "oracle": "separate linear Friis power/noise computation; shared SI constants and physical equations",
            "sources": [
                "https://www.itu.int/rec/R-REC-P.525-5-202411-I/en",
                "https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-RA.2510-2022-PDF-E.pdf",
            ],
            "scope": "isolated constant-gain free-space budget and declared modem gates; not antenna, atmosphere, waveform, pointing, sampled-contact or hardware validation",
            "hardware_calibrated": False,
        }
    )


def workloads():
    """Fixed synthetic resource workloads, not an operational mission sample."""
    from space_env.api.network_workspace import default_network_configuration

    result = []
    for actors, requests in ((1, 1), (2, 4), (4, 16)):
        cfg = default_network_configuration()
        cfg["mission"]["planes"][0]["phases_deg"] = [i * 0.5 for i in range(actors)]
        target = cfg["mission"]["requests"][0]
        longitude = target["lon_deg"] + 17.0
        cfg["mission"]["stations"][0]["lon_deg"] = (longitude + 180) % 360 - 180
        cfg["mission"]["requests"] = [
            {**copy.deepcopy(target), "id": f"request-{i:02d}", "priority": i + 1}
            for i in range(requests)
        ]
        cfg["links"] = [
            {"source": f"fleet-{i:02d}", "destination": f"fleet-{i + 1:02d}"}
            for i in range(actors - 1)
        ]
        result.append(
            {
                "id": f"actors-{actors}-requests-{requests}",
                "configuration": cfg,
                "recover_after": None,
            }
        )
    cfg = default_network_configuration()
    cfg["mission"]["planes"][0]["phases_deg"].append(40.0)
    cfg["links"].append(
        {"source": "fleet-01", "destination": "fleet-03", "max_range_m": 5000000.0}
    )
    result.append(
        {"id": "custody-reroute-4", "configuration": cfg, "recover_after": 22}
    )
    return result


def worker(case):
    """One fresh process; timings include SDK/operator projection overhead."""
    from space_env import __version__
    from space_env.workspace import MissionWorkspace

    t0 = time.perf_counter()
    mission = MissionWorkspace.network(
        case["configuration"],
        request_order=case.get("request_order"),
        planning_method=case.get("planning_method", "serial"),
    )
    planning = time.perf_counter() - t0
    source = None
    steps = []
    recovery_sec = None
    source_unchanged = None
    try:
        while not mission.operator.session._done:
            if (
                case["recover_after"] is not None
                and source is None
                and mission.operator.session.revision == case["recover_after"]
            ):
                source = mission
                before = digest(source.operator.session.checkpoint(compact=True))
                start = time.perf_counter()
                mission = source.evolve(
                    "Measured receiver fault",
                    idempotency_key="measurement",
                    events=[
                        {
                            "type": "spacecraft_fault",
                            "payload": {"actor": "fleet-02", "subsystem": "radio"},
                        }
                    ],
                )
                recovery_sec = time.perf_counter() - start
                source_unchanged = (
                    digest(source.operator.session.checkpoint(compact=True)) == before
                )
            start = time.perf_counter()
            mission.step()
            steps.append(time.perf_counter() - start)
        result = mission.state()["network"]
        native_requests = mission.operator.session._runtime._backend.report()[
            "lifecycle"
        ]["requests"]
        schedule = mission.operator.session._scenario["metadata"]["network_workspace"][
            "schedule"
        ]
        forecast_ids = {a["request"] for a in schedule["allocations"]} | set(
            schedule.get("completed", [])
        )
        if source is not None:
            source_unchanged = (
                digest(source.operator.session.checkpoint(compact=True)) == before
            )
        start = time.perf_counter()
        checkpoint = mission.operator.session.checkpoint(compact=True)
        checkpoint_sec = time.perf_counter() - start
        peak, peak_method = peak_rss()
        return seal(
            {
                "schema": "space-env.network-scale-row/1",
                "case_id": case["id"],
                "input_sha256": digest(case),
                "source_sha256": digest(source_manifest()),
                "status": "completed",
                "native_fulfilled": result["fulfilled"],
                "weighted_fulfilled": sum(
                    r["priority"]
                    for r in native_requests.values()
                    if r["status"] == "completed"
                ),
                "priority_total": sum(r["priority"] for r in native_requests.values()),
                "request_outcomes": {
                    rid: {k: r[k] for k in ("status", "priority", "reason")}
                    for rid, r in native_requests.items()
                },
                "forecast_sha256": mission.operator.session._scenario["metadata"][
                    "network_workspace"
                ]["forecast"]["sha256"],
                "total": result["total"],
                "predicted_fulfilled": result["planned"],
                "forecast_weighted_fulfilled": sum(
                    native_requests[rid]["priority"] for rid in forecast_ids
                ),
                "optimizer": {
                    k: schedule[k]
                    for k in (
                        "objective_score",
                        "upper_bound_score",
                        "optimal",
                        "enumeration_complete",
                        "search_complete",
                        "candidates",
                        "max_search_nodes",
                    )
                }
                if schedule.get("planner") == "exact_small"
                else None,
                "terminal_sequence": result.get("terminal_sequence"),
                "schedule_sha256": schedule["sha256"],
                "search_nodes": result["search_nodes"],
                "search_budget_exhausted": result["budget_exhausted"],
                "planning_sec": planning,
                "recovery_sec": recovery_sec,
                "source_unchanged": source_unchanged,
                "sdk_steps": len(steps),
                "step_sec": {
                    "p50": float(np.percentile(steps, 50)),
                    "p95": float(np.percentile(steps, 95)),
                    "max": max(steps),
                },
                "checkpoint_sec": checkpoint_sec,
                "checkpoint_bytes": len(canonical_json(checkpoint).encode()),
                "checkpoint_sha256": checkpoint["sha256"],
                "elapsed_sec": time.perf_counter() - t0,
                "peak_rss_bytes": peak,
                "peak_rss_method": peak_method,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "space_env": __version__,
                    "numpy": np.__version__,
                    "threads": {
                        k: os.environ.get(k)
                        for k in (
                            "OPENBLAS_NUM_THREADS",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                        )
                    },
                },
                "scope": "RSS method explicit; no OS memory isolation; measured SDK/operator overhead, not bare integrator throughput",
            }
        )
    finally:
        mission.close()
        if source is not None:
            source.close()


def validate_row(row, case, *, expected_source_sha256=None):
    if (
        not isinstance(row, dict)
        or row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
        or row.get("input_sha256") != digest(case)
        or row.get("case_id") != case["id"]
        or row.get("source_sha256")
        != (expected_source_sha256 or digest(source_manifest()))
        or row.get("schema") != "space-env.network-scale-row/1"
        or row.get("status") != "completed"
    ):
        raise ValueError("Worker identity or result checksum mismatch")
    if not isinstance(row.get("step_sec"), dict):
        raise TypeError("Invalid worker timing structure")
    if row.get("status") == "completed" and (
        type(row.get("total")) is not int
        or row["total"] != len(case["configuration"]["mission"]["requests"])
        or type(row.get("native_fulfilled")) is not int
        or not 0 <= row["native_fulfilled"] <= row["total"]
        or row.get("source_unchanged") is False
    ):
        raise ValueError("Invalid native denominator or changed source")
    outcomes = row.get("request_outcomes")
    priorities = {
        r["id"]: r["priority"] for r in case["configuration"]["mission"]["requests"]
    }
    if (outcomes is not None or case.get("request_order") is not None) and (
        not isinstance(outcomes, dict)
        or set(outcomes) != set(priorities)
        or any(
            not isinstance(o, dict)
            or o.get("priority") != priorities[rid]
            or type(o.get("priority")) is not int
            or not isinstance(o.get("status"), str)
            for rid, o in outcomes.items()
        )
        or type(row.get("weighted_fulfilled")) is not int
        or type(row.get("priority_total")) is not int
        or row["priority_total"] != sum(priorities.values())
        or row["weighted_fulfilled"]
        != sum(o["priority"] for o in outcomes.values() if o["status"] == "completed")
        or row["native_fulfilled"]
        != sum(o["status"] == "completed" for o in outcomes.values())
    ):
        raise ValueError("Invalid native request score")
    measures = [row.get(k) for k in ("planning_sec", "checkpoint_sec", "elapsed_sec")]
    if case.get("planning_method") == "exact_small":
        cert = row.get("optimizer")
        if (
            not isinstance(cert, dict)
            or any(
                type(cert.get(k)) is not int or cert[k] < 0
                for k in (
                    "objective_score",
                    "upper_bound_score",
                    "candidates",
                    "max_search_nodes",
                )
            )
            or any(
                type(cert.get(k)) is not bool
                for k in ("optimal", "enumeration_complete", "search_complete")
            )
            or cert["objective_score"]
            != row.get("forecast_weighted_fulfilled", -1) * (row["total"] + 1)
            + row["predicted_fulfilled"]
            or not cert["objective_score"]
            <= cert["upper_bound_score"]
            <= row["priority_total"] * (row["total"] + 1) + row["total"]
            or cert["optimal"] != (cert["objective_score"] == cert["upper_bound_score"])
            or cert["max_search_nodes"] != case["configuration"]["max_search_nodes"]
        ):
            raise ValueError("Invalid forecast optimization witness")
    measures += [row.get("step_sec", {}).get(k) for k in ("p50", "p95", "max")]
    if (
        any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in measures
        )
        or not row["step_sec"]["p50"]
        <= row["step_sec"]["p95"]
        <= row["step_sec"]["max"]
        or any(
            type(row.get(k)) is not int or row[k] <= 0
            for k in ("peak_rss_bytes", "checkpoint_bytes", "sdk_steps")
        )
        or type(row.get("predicted_fulfilled")) is not int
        or not 0 <= row["predicted_fulfilled"] <= row["total"]
        or type(row.get("search_nodes")) is not int
        or row["search_nodes"] < 0
        or type(row.get("search_budget_exhausted")) is not bool
        or row.get("peak_rss_method")
        not in {"linux-proc-VmHWM-after-exec", "getrusage-may-include-pre-exec-peak"}
        or (
            case["recover_after"] is not None
            and (
                row.get("source_unchanged") is not True
                or type(row.get("recovery_sec")) not in (int, float)
                or not math.isfinite(row["recovery_sec"])
                or row["recovery_sec"] < 0
            )
        )
    ):
        raise ValueError("Invalid worker measurements")
    return row


def run_scale_case(case, *, timeout_sec=120):
    if type(timeout_sec) is not int or not 1 <= timeout_sec <= 600:
        raise ValueError("timeout_sec must be an integer in 1..600")
    start = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, "-m", MODULE, "--worker"],
            input=canonical_json(case),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        if process.returncode:
            raise ValueError(
                f"worker exited {process.returncode}: {process.stderr[-2000:]}"
            )
        return validate_row(json.loads(process.stdout), case)
    except (subprocess.TimeoutExpired, ValueError, TypeError, OSError) as exc:
        return seal(
            {
                "schema": "space-env.network-scale-row/1",
                "case_id": case["id"],
                "input_sha256": digest(case),
                "status": "timeout"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "error",
                "error": str(exc)[:2500],
                "total": len(case["configuration"]["mission"]["requests"]),
                "native_fulfilled": None,
                "elapsed_sec": time.perf_counter() - start,
            }
        )


def audit_evidence(directory):
    """Artifact consistency only: no native replay or new measurement."""
    directory = Path(directory)

    def read(name, sealed=True):
        obj = json.loads((directory / name).read_text())
        if not isinstance(obj, dict) or (
            sealed
            and obj.get("sha256")
            != digest({k: v for k, v in obj.items() if k != "sha256"})
        ):
            raise ValueError(f"Invalid artifact checksum: {name}")
        return obj

    report, inputs = read("report.json"), read("inputs.json")
    rf, analytical = read("rf.json"), read("analytical.json")
    if (
        report.get("schema") != "space-env.network-evidence/1"
        or report.get("inputs_sha256") != inputs["sha256"]
        or report.get("rf_sha256") != rf["sha256"]
        or report.get("analytical_sha256") != analytical["sha256"]
    ):
        raise ValueError("Evidence artifact binding mismatch")
    cases = {c["id"]: c for c in inputs["cases"]}
    repeats = inputs["repeats"]
    if (
        len(cases) != len(inputs["cases"])
        or type(repeats) is not int
        or not 1 <= repeats <= 3
        or not cases
        or any(not re.fullmatch(r"[a-z0-9-]{1,80}", k) for k in cases)
    ):
        raise ValueError("Invalid measured workload catalogue")
    expected = {(k, r) for k in cases for r in range(repeats)}
    observed = set()
    for entry in report["rows"]:
        row, repeat = entry["result"], entry["repeat"]
        key = row["case_id"], repeat
        if type(repeat) is not int or key not in expected or key in observed:
            raise ValueError("Missing/duplicate/foreign scale row")
        observed.add(key)
        if read(f"{key[0]}-{repeat}.json") != row:
            raise ValueError("Stored row differs from report")
        case = cases[key[0]]
        if row["status"] == "completed":
            validate_row(
                row, case, expected_source_sha256=digest(report["model_sources"])
            )
        elif (
            row["status"] not in {"error", "timeout"}
            or row.get("input_sha256") != digest(case)
            or row.get("native_fulfilled") is not None
            or row.get("total") != len(case["configuration"]["mission"]["requests"])
        ):
            raise ValueError("Invalid retained failure row")
    if observed != expected:
        raise ValueError("Scale denominator incomplete")
    external_passed = True
    if report["external_forces_included"]:
        external = read("external-forces.json", sealed=False)
        if digest(external) != report["external_forces_sha256"]:
            raise ValueError("External force artifact binding mismatch")
        external_passed = external["passed"]
    elif report["external_forces_sha256"] is not None:
        raise ValueError("Unexpected external comparison")
    calculated = (
        rf["passed"]
        and analytical["passed"]
        and external_passed
        and all(e["result"]["status"] == "completed" for e in report["rows"])
    )
    if report["passed"] != calculated:
        raise ValueError("Recorded evidence outcome inconsistent")
    return seal(
        {
            "schema": "space-env.network-evidence-audit/1",
            "consistent": True,
            "recorded_checks_passed": calculated,
            "source_matches_current": report["model_sources"] == source_manifest(),
            "rows": len(observed),
            "report_sha256": report["sha256"],
            "scope": "artifact consistency; not native replay, independent workload validation or hardware calibration",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--audit", type=Path, help="Check saved artifacts without re-running simulation"
    )
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument(
        "--external-forces",
        action="store_true",
        help="Include existing Boinor force comparison (shared frames/data)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit is not None:
        if args.worker or args.output is not None:
            parser.error("--audit is exclusive with --worker/--output")
        print(canonical_json(audit_evidence(args.audit)))
        return 0
    if args.worker:
        print(canonical_json(worker(json.load(sys.stdin))))
        return 0
    if args.output is None or not 1 <= args.timeout_sec <= 600:
        parser.error("--output new-directory and timeout 1..600 required")
    args.output.mkdir(parents=True, exist_ok=False)
    from space_env.validation.profile_evidence import run_evidence

    rf, analytical = rf_evidence(), run_evidence()
    atomic_write(args.output / "rf.json", canonical_json(rf) + "\n")
    atomic_write(args.output / "analytical.json", canonical_json(analytical) + "\n")
    external = None
    if args.external_forces:
        try:
            from space_env.validation.force_crosscheck import force_crosscheck

            external = force_crosscheck()
        except (ImportError, ValueError, RuntimeError) as exc:
            external = {
                "schema": "space-env.force-crosscheck-error/1",
                "passed": False,
                "error": str(exc),
            }
        atomic_write(
            args.output / "external-forces.json", canonical_json(external) + "\n"
        )
    cases = workloads()
    rows = []
    inputs = seal({"cases": cases, "repeats": args.repeats})
    atomic_write(args.output / "inputs.json", canonical_json(inputs) + "\n")
    for case in cases:
        for repeat in range(args.repeats):
            row = run_scale_case(case, timeout_sec=args.timeout_sec)
            rows.append({"repeat": repeat, "result": row})
            atomic_write(
                args.output / f"{case['id']}-{repeat}.json", canonical_json(row) + "\n"
            )
            print(
                f"{case['id']} repeat={repeat}: {row['status']}; native={row['native_fulfilled']}/{row['total']}",
                flush=True,
            )
    passed = (
        rf["passed"]
        and analytical["passed"]
        and (external is None or external["passed"])
        and all(r["result"]["status"] == "completed" for r in rows)
    )
    report = seal(
        {
            "schema": "space-env.network-evidence/1",
            "passed": passed,
            "rf_sha256": rf["sha256"],
            "analytical_sha256": analytical["sha256"],
            "external_forces_included": external is not None,
            "external_forces_sha256": digest(external)
            if external is not None
            else None,
            "inputs_sha256": inputs["sha256"],
            "rows": rows,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_sources": source_manifest(),
            "scope": "numerical verification plus four frozen synthetic workloads; repeats are not independent missions; pass means checks completed, not all requests delivered or performance superiority",
            "hardware_calibrated": False,
            "competitor_comparison": False,
            "unqualified": [
                "operational orbit accuracy",
                "independent EOP/atmosphere truth",
                "calibrated sensors/ADCS/RF",
                "large constellations",
                "training or sim-to-real performance",
            ],
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report) + "\n")
    print(f"Evidence checks passed: {passed}; {args.output / 'report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
