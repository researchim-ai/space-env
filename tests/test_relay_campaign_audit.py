import json

import pytest

from examples.modeling.relay_campaign_audit import audit
from space_env.simulation.snapshot import digest


def test_audit_rejects_unknown_paths(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"rows": [{"id": "../unknown"}]}))
    with pytest.raises(ValueError, match="matrix"):
        audit(tmp_path)


def test_audit_rejects_recorded_terminal_overallocation(tmp_path):
    directory = tmp_path / "one-terminal"
    directory.mkdir()
    record = {
        "policy_origin": "scripted_public",
        "task_text": "brief",
        "metrics": {},
        "events": [],
        "messages": [{}, {"content": json.dumps({"observation": {}})}],
    }
    record["sha256"] = digest(record)
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"id": "one-terminal", "sha256": record["sha256"], "metrics": {}}
                ]
            }
        )
    )
    (directory / "episode.json").write_text(json.dumps(record))
    (directory / "privileged-case.json").write_text(
        json.dumps(
            {
                "public_brief": "brief",
                "public_contract": {"minimum_completed_requests": 0},
                "configuration": {
                    "scenario": {
                        "definition": {
                            "operations": {
                                "stations": [{"name": "G", "terminals": 1}],
                                "radio_network": {
                                    "receivers": {"G": {"aggregate_rate_bps": 10.0}}
                                },
                            }
                        }
                    }
                },
            }
        )
    )
    (directory / "native-network.json").write_text(
        json.dumps(
            {
                "radio": {
                    "intervals": [
                        {
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "grants": {
                                "a": {"station": "G", "rate_bps": 5.0},
                                "b": {"station": "G", "rate_bps": 5.0},
                            },
                            "station_bits": {"G": 10.0},
                        }
                    ]
                }
            }
        )
    )
    with pytest.raises(ValueError, match="capacity"):
        audit(tmp_path)
