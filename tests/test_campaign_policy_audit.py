import copy
import json

import pytest

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from tests.test_campaign_policy import CONTRACT, observation


def test_public_trace_check_rejects_an_unrelated_action():
    view = observation()
    record = {
        "policy_origin": "scripted_public",
        "messages": [{}, {"content": json.dumps({"observation": view})}],
        "events": [
            {
                "arguments": json.dumps(
                    {"action": {"choices": {"@central": ["wait"]}, "advance_sec": 0.25}}
                ),
                "response": {"observation": view},
            }
        ],
    }
    assert verify_policy_trace(record, CONTRACT)["decisions"] == 1
    altered = copy.deepcopy(record)
    altered["events"][0]["arguments"] = json.dumps(
        {"action": {"choices": {"@central": ["secret-action"]}, "advance_sec": 0.25}}
    )
    with pytest.raises(ValueError, match="differs"):
        verify_policy_trace(altered, CONTRACT)
    altered = copy.deepcopy(record)
    altered["policy_origin"] = "scripted_privileged"
    with pytest.raises(ValueError, match="provenance"):
        verify_policy_trace(altered, CONTRACT)
