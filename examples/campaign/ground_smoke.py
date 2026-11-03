"""Installed native 2s windows plus ACK-only cancellation, replay and RL/SFT."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from space_env.planning import PlanningRun
from space_env.scaling.benchmark import run_scale
from space_env.scaling.campaign import campaign_workload
from space_env.tasks import VerifiedEpisode


def main():
    short = campaign_workload(
        {
            "id": "installed-short-ground",
            "cameras": 1,
            "relays": 1,
            "waves": 1,
            "horizon_sec": 20.0,
            "repeats": 1,
            "ground_control_sec": 2.0,
        }
    )
    short.update(id="straight", comparison_group="short-ground")
    resumed = copy.deepcopy(short)
    resumed.update(id="resumed", resume_after_decisions=2)
    with tempfile.TemporaryDirectory(prefix="ground-installed-") as directory:
        result = run_scale(
            {"id": "installed-ground", "workers": 2, "workloads": [short, resumed]},
            Path(directory) / "scale",
        )
        assert (
            result["ok"]
            and result["qualification_domains"]["domains"]["mission_expectations"][
                "all_expected_passed"
            ]
        )
        w = campaign_workload(
            {
                "id": "installed-ack-release",
                "cameras": 2,
                "relays": 1,
                "waves": 1,
                "horizon_sec": 40.0,
                "mode": "ground_recovery",
                "repeats": 1,
                "ground_control_sec": 8.0,
                "ack_release_actions": True,
            }
        )
        run = PlanningRun(w["scenario"], w["task"], w["policy"])
        run.run()
        assert run.metrics()["fulfilled"] == 2
        assert any(
            "release-acked-relay-00" in t["decision"]["action_ids"]
            and any("-ground-" in a for a in t["decision"]["action_ids"])
            for t in run.trace
        )
        verified = VerifiedEpisode.from_planning_checkpoint(
            run.checkpoint(compact=True)
        )
        assert verified.verification["quality"]["passed"]
        bundle = verified.export_bundle(Path(directory) / "ack-release-dataset")
        assert all(f["rows"] > 0 for f in bundle["files"].values())
    print(
        json.dumps(
            {
                "ok": True,
                "short_ground_paired_recovery": True,
                "observed_ack_cancel_and_resubmit": True,
                "native_replay_and_verified_rl_sft": True,
                "scope": "small installed smoke, not large mission qualification",
            }
        )
    )


if __name__ == "__main__":
    main()
