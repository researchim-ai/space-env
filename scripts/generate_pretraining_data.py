#!/usr/bin/env python3
"""Генерация данных для претренинга (JSONL)."""
import argparse
from pathlib import Path

from space_env.data_generators.pretraining import generate_pretraining_chunks


def main():
    p = argparse.ArgumentParser(description="Generate pretraining JSONL")
    p.add_argument("-o", "--output", default="datasets/pretraining.jsonl", help="Output JSONL path")
    p.add_argument("-n", "--num-templates", type=int, default=100, help="Number of template samples")
    p.add_argument("-s", "--sim-steps", type=int, default=150, help="Simulation steps for sim-driven text")
    p.add_argument("--no-glossary", action="store_true", help="Skip glossary entries")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    count = generate_pretraining_chunks(
        args.output,
        num_template_samples=args.num_templates,
        num_simulation_steps=args.sim_steps,
        include_glossary=not args.no_glossary,
        seed=args.seed,
    )
    print(f"Wrote {count} chunks to {args.output}")


if __name__ == "__main__":
    main()
