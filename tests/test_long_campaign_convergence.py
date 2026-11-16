import copy
import json

import pytest

from examples.modeling.long_campaign_convergence import compare_records
from space_env.simulation.snapshot import digest


def sealed(fine=False):
    def view(t, delta):
        return {
            "native": {
                "time_sec": t,
                "entities": [
                    {
                        "kind": "actor",
                        "id": "a",
                        "values": {
                            "r_x_m": delta,
                            "r_y_m": 0.0,
                            "r_z_m": 0.0,
                            "v_x_m_s": delta / 10.0,
                            "v_y_m_s": 0.0,
                            "v_z_m_s": 0.0,
                        },
                    }
                ],
            }
        }

    body = {
        "initial": {
            "configuration": {
                "scenario": {
                    "definition": {
                        "orbits": {
                            "a": {
                                "integration": {
                                    "sample_sec": 1.0 if fine else 30.0,
                                    "max_step_sec": 5.0 if fine else 60.0,
                                }
                            }
                        }
                    }
                }
            }
        },
        "messages": [
            {},
            {"content": json.dumps({"observation": view(0.0, 0.1 if fine else 0.0)})},
        ],
        "events": [{"response": {"observation": view(1.0, 0.2 if fine else 0.0)}}],
    }
    return {**body, "sha256": digest(body)}


def test_maximum_over_common_observations_not_just_initial_state():
    result = compare_records(sealed(), sealed(True))
    assert result["matched_time_grid"] and result["common_observations"] == 2
    assert result["actors"]["a"]["max_position_m"] == pytest.approx(0.2)
    assert result["actors"]["a"]["max_velocity_m_s"] == pytest.approx(0.02)


def test_seal_and_nonorbit_changes_rejected():
    bad = copy.deepcopy(sealed(True))
    bad["initial"]["configuration"]["unexpected"] = "changed"
    with pytest.raises(ValueError, match="seal"):
        compare_records(sealed(), bad)
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="beyond orbit"):
        compare_records(sealed(), bad)
