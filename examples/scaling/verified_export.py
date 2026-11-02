"""One policy/native cold replay, then an atomic multi-format training bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from space_env.planning import PlanningRun
from space_env.scaling.workloads import fleet_workload
from space_env.tasks import VerifiedEpisode


def main(output):
    w = fleet_workload({"actors": 4, "products": 12, "planes": 2, "repeats": 1})
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    assert run.run()["fulfilled"] == 12
    verified = VerifiedEpisode.from_planning_checkpoint(run.checkpoint(compact=True))
    receipt = verified.export_bundle(output)
    assert receipt["verification"]["quality"]["passed"]
    assert sum(f["rows"] for f in receipt["files"].values()) == 8
    print(
        json.dumps(
            {
                "ok": True,
                "episode_sha256": receipt["episode_sha256"],
                "files": len(receipt["files"]),
                "rows": sum(f["rows"] for f in receipt["files"].values()),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
