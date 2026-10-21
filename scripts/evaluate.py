#!/usr/bin/env python3
"""Случайный бенчмарк четырёх сред и, по желанию, рубрика готового SFT-ответа."""
import argparse

from space_env.eval.benchmarks import run_registered_benchmarks
from space_env.eval.holdout import compare_policies
from space_env.eval.sft_rubric import term_recall


def main():
    parser = argparse.ArgumentParser(description="Evaluate random agents and an optional SFT answer")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--answer", default="", help="Текст ответа для рубрики")
    parser.add_argument("--terms", default="AOS,eclipse,safe mode", help="Термины через запятую")
    args = parser.parse_args()
    rows = run_registered_benchmarks(episodes=args.episodes, max_episode_steps=args.steps, seed=args.seed)
    held = compare_policies("space_env/ResourceManager-v0", episodes=args.episodes, max_episode_steps=args.steps, seed=args.seed)
    print(
        "holdout ResourceManager: "
        f"random_success={held['random_success_rate']:.2f} expert_success={held['expert_success_rate']:.2f}"
    )
    for row in rows:
        print(
            f"{row['env']}: mean_reward={row['mean_reward']:.3f} "
            f"success_rate={row['success_rate']:.2f} mean_steps={row['mean_steps']:.1f}"
        )
    if args.answer:
        terms = [term.strip() for term in args.terms.split(",") if term.strip()]
        print(f"sft_term_recall={term_recall(args.answer, terms):.2f}")


if __name__ == "__main__":
    main()
