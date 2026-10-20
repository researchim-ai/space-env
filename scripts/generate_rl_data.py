#!/usr/bin/env python3
"""Случайные траектории RL по configs/curriculum.yaml."""
import argparse

from space_env.rl.trajectories import write_trajectories


def main():
    parser = argparse.ArgumentParser(description="Generate RL trajectory JSONL")
    parser.add_argument("-o", "--output", default="datasets/rl_trajectories.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    count = write_trajectories(args.output, seed=args.seed)
    print(f"Wrote {count} steps to {args.output}")


if __name__ == "__main__":
    main()
