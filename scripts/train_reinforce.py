#!/usr/bin/env python3
"""Короткое обучение линейного REINFORCE на зарегистрированной среде."""
import argparse

from space_env.rl.reinforce import train_linear_reinforce


def main():
    parser = argparse.ArgumentParser(description="Train a linear REINFORCE policy")
    parser.add_argument("--env", default="space_env/AnomalyResponse-v0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    result = train_linear_reinforce(
        args.env,
        episodes=args.episodes,
        max_episode_steps=args.steps,
        seed=args.seed,
        lr=args.lr,
    )
    print(f"episodes={args.episodes} mean_last={result['mean_last']:.3f}")
    print("returns=" + ",".join(f"{value:.3f}" for value in result["returns"]))


if __name__ == "__main__":
    main()
