"""Read-only synthetic observations for the 0.98 working tree."""
import json
import runpy
import tempfile
from pathlib import Path

import yaml

from space_env import __version__
from space_env.mission.tasking import TaskRequest, delivery_score
from space_env.mission.scenario_v2 import world_from_scenario
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import save_world, load_world


def main():
    helpers = runpy.run_path(str(Path(__file__).with_name("v086_audit_probes.py")))
    w = helpers["world"]()
    before = len(w.commands.export())
    try:
        w.accept([MissionCommand("same", "a", "wait"), MissionCommand("same", "a", "imaging")])
    except ValueError:
        pass
    out = {"version": __version__, "batch_conflict": {
        "before_records": before, "after_records": len(w.commands.export()), "now": w.now_sec()
    }}
    w = helpers["world"]()
    orders = w.accept([MissionCommand("ghost-request", "a", "imaging", request_id="not-a-request")])
    w.step(1, orders)
    w.settle()
    out["unknown_request_command"] = w.commands.get("ghost-request")
    request = TaskRequest("r", 0, 0, 0, 10, 100, 1, "optical")
    slips = [
        {"request_id": "r", "product_id": "early", "offset_bits": 0, "bits": 100, "time_sec": 5},
        {"request_id": "r", "product_id": "late", "offset_bits": 0, "bits": 120, "time_sec": 20},
    ]
    out["early_valid_late_larger"] = delivery_score([request], slips, [])["on_time"]
    with tempfile.TemporaryDirectory() as tmp:
        scenario = Path(tmp) / "source.yaml"
        scenario.write_text(yaml.safe_dump({
            "version": 2, "horizon_sec": 60,
            "instances": [{"id": "a", "model": "cubesat-6u"}],
            "stations": ["Kiruna"], "requests": [],
        }))
        w = world_from_scenario(scenario, attach_stations=True, refresh_passes=False)
        snap = Path(tmp) / "state.json"
        save_world(snap, w, scenario)
        loaded = load_world(snap)
        out["json_station_restore"] = {
            "before": [s.name for s in w.engines["a"].stations],
            "after": [s.name for s in loaded.engines["a"].stations],
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
