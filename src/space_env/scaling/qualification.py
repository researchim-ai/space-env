"""Separate observed runtime evidence from mission targets without waiving gates."""

PERFORMANCE = frozenset(
    {"checkpoint_budget", "execution_budget", "peak_rss_budget", "step_latency_budget"}
)
INTEGRITY = frozenset({"native_replay", "quality"})
MISSION = frozenset({"completion"})


def row_domains(row):
    gates = row.get("gates")
    if row.get("error") is not None or not isinstance(gates, dict):
        return {
            name: "not_observed"
            for name in (
                "technical_runtime",
                "native_integrity_and_witnesses",
                "mission_expectations",
                "other_gates",
            )
        }

    def state(required):
        if any(type(gates.get(k)) is not bool for k in required):
            return "not_observed"
        return "passed" if all(gates[k] for k in required) else "failed"

    native = (
        INTEGRITY
        | ({"campaign_witnesses"} if "campaign_witnesses" in gates else set())
        | ({"process_restart"} if "process_restart" in gates else set())
    )
    mission = MISSION | (
        {"completion_upper_bound"} if "completion_upper_bound" in gates else set()
    )
    other = set(gates) - PERFORMANCE - native - mission
    return {
        "technical_runtime": state(PERFORMANCE),
        "native_integrity_and_witnesses": state(native),
        "mission_expectations": state(mission),
        "other_gates": state(other),
    }


def qualification_domains(spec, rows, comparisons):
    expected = sum(w.repeats for w in spec.workloads)
    projected = [row_domains(r) for r in rows]
    domains = {}
    for name in (
        "technical_runtime",
        "native_integrity_and_witnesses",
        "mission_expectations",
        "other_gates",
    ):
        counts = {
            state: sum(r[name] == state for r in projected)
            for state in ("passed", "failed", "not_observed")
        }
        domains[name] = {
            "expected_repeats": expected,
            "observed_repeats": len(rows),
            **counts,
            "all_expected_passed": len(rows) == expected
            and counts["passed"] == expected,
        }
    raw = {}
    for name, formal in comparisons.items():
        ids = {w.id for w in spec.workloads if w.comparison_group == name}
        selected = [r for r in rows if r["id"] in ids]
        observed = [
            r
            for r in selected
            if r.get("error") is None
            and isinstance(r.get("native_backend_sha256"), str)
            and isinstance(r.get("trace_sha256"), str)
        ]
        signatures = {(r["native_backend_sha256"], r["trace_sha256"]) for r in observed}
        complete = len(observed) == len(selected) == formal["expected_repeats"]
        raw[name] = {
            "expected_repeats": formal["expected_repeats"],
            "observed_signatures": len(observed),
            "distinct_signature_pairs": len(signatures),
            "all_expected_signatures_observed": complete,
            "observed_signatures_identical": complete and len(signatures) == 1,
            "formal_group_passed": formal["passed"],
        }
    return {
        "schema": "space-env.qualification-domains/1",
        "domains": domains,
        "observed_raw_repeatability": raw,
        "scope": "mission_expectations includes expected negative outcomes; observed signatures do not waive failed gates or change overall ok",
    }
