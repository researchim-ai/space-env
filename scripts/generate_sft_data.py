#!/usr/bin/env python3
"""Генерация данных для SFT (JSONL: instruction, output)."""
import argparse
from pathlib import Path

from space_env.data_generators.sft import generate_sft_records


def main():
    p = argparse.ArgumentParser(description="Generate SFT JSONL")
    p.add_argument("-o", "--output", default="datasets/sft.jsonl", help="Output JSONL path")
    p.add_argument("--no-yaml", action="store_true", help="Skip Q&A from data/sft_pairs.yaml")
    p.add_argument("--no-scenarios", action="store_true", help="Skip scenario procedure pairs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    count = generate_sft_records(
        args.output,
        include_yaml=not args.no_yaml,
        include_scenarios=not args.no_scenarios,
        seed=args.seed,
    )
    print(f"Wrote {count} SFT records to {args.output}")


if __name__ == "__main__":
    main()
