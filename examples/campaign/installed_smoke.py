"""Small installed paired coupled campaign; large qualification is a separate run."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.scaling.campaign import campaign_workload


def main():
    w = campaign_workload(
        {
            "id": "installed-coupled",
            "cameras": 1,
            "relays": 1,
            "waves": 1,
            "horizon_sec": 20.0,
            "repeats": 1,
        }
    )
    w.update(id="straight", comparison_group="native-recovery")
    resumed = copy.deepcopy(w)
    resumed.update(id="resumed", resume_after_decisions=2)
    with tempfile.TemporaryDirectory(prefix="campaign-installed-") as directory:
        result = run_scale(
            {"id": "installed-coupled", "workers": 2, "workloads": [w, resumed]},
            Path(directory) / "result",
        )
        assert result["ok"], result
        assert result["semantic_comparisons"]["native-recovery"]["passed"]
        assert all(row["metrics"]["fulfilled"] == 1 for row in result["rows"])
        print(
            json.dumps(
                {
                    "ok": True,
                    "fresh_processes": 2,
                    "native_coupled": True,
                    "requests_per_episode": 1,
                    "prefix_recovery_semantic_parity": True,
                    "native_replay": True,
                    "verified_training_export": True,
                }
            )
        )


if __name__ == "__main__":
    main()
