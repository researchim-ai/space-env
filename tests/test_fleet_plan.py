"""План цепочки пишется из того же шага, что и пересылка пакета."""
import json

from space_env.data_generators.sft import generate_sft_from_fleet_plan
from space_env.mission.fleet_plan import build_relay_schedule
from space_env.rl.trajectories import write_trajectories


def test_relay_schedule_only_keeps_hops_the_link_accepts():
    schedule = build_relay_schedule()
    relays = [item for item in schedule.tasks if item.kind == "relay"]
    assert schedule.arrived
    assert schedule.source != schedule.gateway
    assert relays
    assert all(item.done and item.bits_mbit > 0.0 for item in relays)
    assert all("один луч" in item.reason for item in relays)
    assert any(item.kind == "arrive" and item.actor == schedule.gateway for item in schedule.tasks)
    text = "\n".join(schedule.lines())
    assert schedule.source in text and schedule.gateway in text


def test_wide_spacing_records_a_blocked_hop():
    schedule = build_relay_schedule(separation_deg=40.0, steps=2)
    blocked = [item for item in schedule.tasks if item.kind == "relay" and not item.done]
    assert blocked
    assert "не пересылает" in blocked[0].reason
    assert schedule.arrived is False


def test_sft_and_rl_rows_carry_the_chain(tmp_path):
    records = list(generate_sft_from_fleet_plan())
    assert records[0]["meta"]["arrived"] is True
    assert "пересылает" in records[0]["output"]
    count = write_trajectories(
        tmp_path / "rl.jsonl",
        stages=[
            {
                "name": "constellation",
                "env": "space_env/ConstellationOps-v0",
                "episodes": 1,
                "max_episode_steps": 4,
            }
        ],
        seed=0,
        use_expert=True,
    )
    assert count >= 1
    row = json.loads((tmp_path / "rl.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(row["action"], list)
    assert row["env"] == "space_env/ConstellationOps-v0"
