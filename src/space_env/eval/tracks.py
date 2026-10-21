"""Пять дорожек бенчмарка: обучение и отложенный сид раздельно."""
from __future__ import annotations

from space_env.eval.holdout import compare_policies
from space_env.rl.report import training_report

# Дорожка, среда, потолок шагов, учить ли softmax на этом бюджете.
TRACKS: tuple[tuple[str, str, int, bool], ...] = (
    ("safety", "space_env/DebrisAvoid-v0", 1, False),
    ("tasking", "space_env/PriorityCoverage-v0", 1, False),
    ("constellation", "space_env/FleetShift-v0", 8, True),
    ("relative", "space_env/RendezvousHold-v0", 1, False),
    ("faults", "space_env/PassFault-v0", 6, False),
)


def run_benchmark_tracks(*, seed: int = 0, held_out_seed: int = 100) -> list[dict]:
    """
    На каждой дорожке эвристика и случайная политика на сиде обучения и на отложенном.
    У группировки дополнительно хвост короткого обучения. learned — хвост выше случайной награды.
    """
    rows = []
    for name, env_id, steps, learn in TRACKS:
        train = compare_policies(env_id, episodes=1, max_episode_steps=steps, seed=seed)
        held = compare_policies(env_id, episodes=1, max_episode_steps=steps, seed=held_out_seed)
        row: dict = {
            "track": name,
            "env": env_id,
            "train_expert_success": float(train["expert_success_rate"]),
            "held_expert_success": float(held["expert_success_rate"]),
            "held_random_success": float(held["random_success_rate"]),
            "held_random_reward": float(held["random_mean_reward"]),
        }
        if learn:
            trained = training_report(
                env_id,
                episodes=2,
                max_episode_steps=steps,
                seed=seed,
                eval_episodes=1,
            )
            row["trained_mean_last"] = float(trained["trained_mean_last"])
            row["learned"] = bool(trained["learned"])
        rows.append(row)
    return rows
