from space_env.simulation.snapshot import digest
from space_env.validation.profile_evidence import profile_contracts, run_evidence


def test_analytical_profile_evidence_is_sealed_and_not_hardware_calibration():
    report = run_evidence()
    assert report["passed"] and len(report["cases"]) == 9
    assert report["sha256"] == digest(
        {k: v for k, v in report.items() if k != "sha256"}
    )
    assert not report["hardware_calibrated"]
    contracts = profile_contracts()
    assert {p["id"] for p in contracts["profiles"]} == {
        "workspace-eo",
        "workspace-dynamic",
        "relay-reference",
        "rigid-reference",
        "joint-network-workspace",
    }
    assert contracts["open_gaps"]
    assert all(p["limits"] for p in contracts["profiles"])
