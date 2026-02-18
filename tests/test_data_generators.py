"""Тесты генераторов данных для претренинга и SFT."""
import json
import tempfile
from pathlib import Path

import pytest

from space_env.data_generators.pretraining import (
    generate_pretraining_chunks,
    generate_from_templates,
    generate_from_simulation,
    generate_glossary_entries,
)
from space_env.data_generators.sft import generate_sft_records, generate_sft_from_yaml


def test_pretraining_templates():
    chunks = list(generate_from_templates(3, seed=0))
    assert len(chunks) == 3
    for c in chunks:
        assert "text" in c and "meta" in c
        assert "domain" in c["meta"]


def test_pretraining_simulation():
    chunks = list(generate_from_simulation(num_steps=5, step_sec=60.0, seed=0))
    assert len(chunks) >= 1
    for c in chunks:
        assert "text" in c and "meta" in c


def test_glossary():
    chunks = list(generate_glossary_entries())
    assert len(chunks) >= 5
    for c in chunks:
        assert "text" in c
        assert "AOS" in " ".join(c["text"] for c in chunks) or "eclipse" in " ".join(c["text"] for c in chunks)


def test_pretraining_jsonl(tmp_path):
    out = tmp_path / "pt.jsonl"
    n = generate_pretraining_chunks(out, num_template_samples=5, num_simulation_steps=8, include_glossary=True, seed=0)
    assert n >= 5 + 19  # templates + glossary, plus some sim
    lines = out.read_text().strip().split("\n")
    for line in lines:
        rec = json.loads(line)
        assert "text" in rec


def test_sft_yaml():
    recs = list(generate_sft_from_yaml(seed=0))
    assert len(recs) >= 1
    for r in recs:
        assert "instruction" in r and "output" in r


def test_sft_jsonl(tmp_path):
    out = tmp_path / "sft.jsonl"
    n = generate_sft_records(out, include_yaml=True, include_scenarios=True, seed=0)
    assert n >= 1
    lines = out.read_text().strip().split("\n")
    for line in lines:
        rec = json.loads(line)
        assert "instruction" in rec and "output" in rec
