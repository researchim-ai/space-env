"""Replay-backed corpus isolation, split leakage and transaction failure tests."""

import json
import sqlite3

import pytest

from space_env.factory.registry import _register_claims
from space_env.factory.semantic import split_group
from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus
from space_env.llm.episode import LLMEpisode, seal
from space_env.simulation.snapshot import canonical_json, digest
from tests.test_llm_episode import session


def write_record(tmp_path, name, **kwargs):
    record = LLMEpisode(
        session(),
        task_text="Deliver the products to Ground.",
        group=kwargs.pop("group", "test-geometry"),
        **kwargs,
    ).record()
    path = tmp_path / name
    path.write_text(canonical_json(record))
    return path, record


def reseal_bundle(root, change):
    path = root / "corpus.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(
        canonical_json(seal({k: v for k, v in manifest.items() if k != "sha256"}))
    )


def test_failure_is_rl_only_and_evaluation_is_never_training(tmp_path):
    path, _record = write_record(tmp_path, "train.json")
    result = build_corpus([path], tmp_path / "corpus")
    assert result["summary"]["rows"] == {"rl": 1, "sft": 0, "wire-sft": 0}
    assert not result["summary"]["successful_train_episodes"]
    assert result["registry_status"] == "not_registered"
    assert verify_corpus(tmp_path / "corpus")["ok"]
    evaluation, _ = write_record(tmp_path, "test.json", split="test")
    result = build_corpus([evaluation], tmp_path / "evaluation")
    assert result["summary"]["evaluation_episodes"] == 1
    assert result["summary"]["rows"] == {"rl": 0, "sft": 0, "wire-sft": 0}
    assert verify_corpus(tmp_path / "evaluation")["ok"]


def test_geometry_leakage_ignores_user_supplied_group_and_prose(tmp_path):
    a, _ = write_record(tmp_path, "a.json", group="invented-train")
    b, _ = write_record(tmp_path, "b.json", split="test", group="invented-test")
    with pytest.raises(ValueError, match="geometry crosses"):
        build_corpus([a, b], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".llm-corpus-*"))


def test_duplicate_trajectory_cannot_be_hidden_by_new_dialogue_group(tmp_path):
    a, _ = write_record(tmp_path, "a.json", group="g1")
    b, _ = write_record(tmp_path, "b.json", group="g2")
    with pytest.raises(ValueError, match="duplicate semantic"):
        build_corpus([a, b], tmp_path / "duplicate")


def test_shared_factory_registry_blocks_llm_train_after_heldout_geometry(tmp_path):
    path, record = write_record(tmp_path, "episode.json")
    group = split_group(record["initial_checkpoint"]["scenario"])
    registry = tmp_path / "registry.sqlite"
    _register_claims(
        registry,
        content=digest("factory-heldout"),
        manifest=digest("factory-manifest"),
        groups={(group, "holdout")},
        trajectories=set(),
    )
    build_corpus([path], tmp_path / "corpus")
    with pytest.raises(ValueError, match="geometry group crosses"):
        register_corpus(tmp_path / "corpus", registry)
    with sqlite3.connect(registry) as conn:
        assert conn.execute("SELECT count(*) FROM bundles").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM trajectories").fetchone() == (0,)


def test_registration_is_idempotent_but_new_wrapper_is_not_new_trajectory(tmp_path):
    a, _ = write_record(tmp_path, "a.json", group="g1")
    b, _ = write_record(tmp_path, "b.json", group="g2")
    build_corpus([a], tmp_path / "first")
    build_corpus([b], tmp_path / "second")
    registry = tmp_path / "r.sqlite"
    assert register_corpus(tmp_path / "first", registry)["status"] == "registered"
    assert (
        register_corpus(tmp_path / "first", registry)["status"] == "already_registered"
    )
    with pytest.raises(ValueError, match="trajectory already registered"):
        register_corpus(tmp_path / "second", registry)
    with pytest.raises(ValueError, match="outside"):
        register_corpus(tmp_path / "first", tmp_path / "first" / "registry.sqlite")


@pytest.mark.parametrize("fault", ["row", "summary", "entry", "extra", "symlink"])
def test_even_resealed_corpus_cannot_invent_rows_or_semantic_claims(tmp_path, fault):
    path, _ = write_record(tmp_path, "input.json")
    root = tmp_path / "corpus"
    build_corpus([path], root)
    if fault == "row":
        out = root / "data/rl.jsonl"
        row = json.loads(out.read_text())
        row["split"] = "test"
        out.write_text(canonical_json(row) + "\n")
        from space_env.simulation.streaming import file_digest

        reseal_bundle(
            root,
            lambda m: m["files"].update(
                {
                    "data/rl.jsonl": {
                        "sha256": file_digest(out),
                        "bytes": out.stat().st_size,
                    }
                }
            ),
        )
    elif fault == "summary":
        reseal_bundle(root, lambda m: m["summary"].update(independent_geometries=999))
    elif fault == "entry":
        reseal_bundle(root, lambda m: m["entries"][0].update(split="test"))
    elif fault == "extra":
        (root / "data/leaked.jsonl").write_text("{}")
    else:
        (root / "linked").symlink_to(path)
    with pytest.raises(ValueError):
        verify_corpus(root)


def test_invalid_episode_does_not_publish_or_remove_existing_output(tmp_path):
    path, record = write_record(tmp_path, "episode.json")
    record["metrics"]["success"] = True
    path.write_text(
        canonical_json(seal({k: v for k, v in record.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        build_corpus([path], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("precious")
    with pytest.raises(ValueError):
        build_corpus([path], output)
    assert (output / "keep").read_text() == "precious"


def test_empty_input_and_episode_budget_are_fail_closed(tmp_path, monkeypatch):
    from space_env.llm import corpus

    with pytest.raises(ValueError, match="empty"):
        build_corpus([], tmp_path / "empty")
    path, _ = write_record(tmp_path, "input.json")
    monkeypatch.setattr(corpus, "MAX_EPISODES", 1)
    with pytest.raises(ValueError, match="episode budget"):
        build_corpus([path, path], tmp_path / "too-many")


@pytest.mark.parametrize("claims", ["internal-split", "invalid-hash", "invalid-split"])
def test_registry_rejects_invalid_claims_before_any_write(tmp_path, claims):
    registry = tmp_path / "r.sqlite"
    groups = {(digest("g"), "train")}
    if claims == "internal-split":
        groups.add((digest("g"), "test"))
    elif claims == "invalid-hash":
        groups = {("not-a-hash", "train")}
    else:
        groups = {(digest("g"), "evaluation-but-misspelled")}
    with pytest.raises(ValueError):
        _register_claims(
            registry,
            content=digest("c"),
            manifest=digest("m"),
            groups=groups,
            trajectories=set(),
        )
    assert not registry.exists()


@pytest.fixture(scope="module")
def local_training_input(tmp_path_factory):
    """Simulated provider transport, backed by actual mission execution."""
    import copy

    from space_env.llm.episode import _training_row
    from space_env.llm.local_data import wire_sft_rows
    from space_env.planning.network_benchmark import geometry_key
    from space_env.simulation.streaming import atomic_jsonl
    from tests.test_llm_wire_data import successful_wire

    record, trace = copy.deepcopy(successful_wire.__wrapped__())
    config = record["initial_checkpoint"]["scenario"]["metadata"]["network_workspace"][
        "configuration"
    ]
    record["configuration"]["group"] = geometry_key(config)
    record = seal({k: v for k, v in record.items() if k != "sha256"})
    root = tmp_path_factory.mktemp("local-provider-input")
    (root / "episode.json").write_text(canonical_json(record))
    (root / "inference.json").write_text(canonical_json(seal({"trace": trace})))
    exports = {}
    for name, rows in [
        ("rl.jsonl", [_training_row(record, format="rl")]),
        ("sft.jsonl", [_training_row(record, format="sft")]),
        ("wire-sft.jsonl", wire_sft_rows(record, trace, model="qwen3.5:4b")),
    ]:
        exports[name] = atomic_jsonl(root / name, rows, max_bytes=64000000)
    (root / "manifest.json").write_text(
        canonical_json(
            seal(
                {
                    "schema": "space-env.local-llm-data/1",
                    "configuration": config,
                    "episode_sha256": record["sha256"],
                    "metrics": record["metrics"],
                    "exports": exports,
                    "provider_error": None,
                    "inference": {"model": "qwen3.5:4b"},
                }
            )
        )
    )
    return root


def test_provider_wire_bundle_is_preserved_and_reaudited(
    local_training_input, tmp_path
):
    root = tmp_path / "corpus"
    result = build_corpus([local_training_input], root)
    assert result["summary"]["rows"] == {"rl": 1, "sft": 1, "wire-sft": 30}
    assert result["summary"]["independent_geometries"] == 1
    assert result["summary"]["all_wait_decisions"] == 24
    assert result["summary"]["policy_origins"] == {"scripted_public": 1}
    assert (root / "audit/0000/inference.json").is_file()
    assert verify_corpus(root)["ok"]


def test_resealed_provider_context_tampering_is_rejected(
    local_training_input, tmp_path
):
    import shutil

    root = tmp_path / "input"
    shutil.copytree(local_training_input, root)
    path = root / "inference.json"
    inference = json.loads(path.read_text())
    inference["trace"][0]["request"]["messages"][0]["content"] = "invented context"
    path.write_text(
        canonical_json(seal({k: v for k, v in inference.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError, match="Wire context"):
        build_corpus([root], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "arguments", ["{", "[1]", '{"x":NaN}', '{"x":1,"x":2}', '{"choices":{"a":"wait"}}']
)
def test_malformed_failed_calls_are_preserved_for_rl(tmp_path, arguments):
    from space_env.llm.episode import STEP

    agent = LLMEpisode(session(), task_text="Deliver.", group="failure", max_calls=1)
    assert not agent.invoke(STEP, arguments, call_id="bad")["ok"]
    path = tmp_path / "failed.json"
    path.write_text(canonical_json(agent.record()))
    root = tmp_path / "corpus"
    result = build_corpus([path], root)
    assert result["summary"]["rows"] == {"rl": 1, "sft": 0, "wire-sft": 0}
    assert result["summary"]["all_wait_decisions"] == 0
    assert verify_corpus(root)["ok"]


def test_successful_and_failed_native_inputs_are_distinct_trajectories(tmp_path):
    from space_env.llm.episode import OBSERVE

    first, _ = write_record(tmp_path, "empty.json")
    agent = LLMEpisode(session(), task_text="Deliver.", group="observe", max_calls=1)
    agent.invoke(OBSERVE, "{}", call_id="observe")
    second = tmp_path / "observed.json"
    second.write_text(canonical_json(agent.record()))
    result = build_corpus([first, second], tmp_path / "corpus")
    assert result["summary"]["episodes"] == 2
    assert result["summary"]["independent_geometries"] == 1


def test_real_factory_bundle_and_llm_corpus_share_geometry_protection(tmp_path):
    from space_env.factory import build_dataset
    from space_env.factory.registry import register_bundle
    from space_env.tasks import TaskSession
    from tests.test_scaling import registry_spec

    spec = registry_spec()
    factory = tmp_path / "factory"
    assert build_dataset(spec, factory)["ok"]
    registry = tmp_path / "registry.sqlite"
    register_bundle(factory, registry)
    template = spec["templates"][0]
    agent = LLMEpisode(
        TaskSession(template["scenario"], template["task"]),
        task_text="Reserved evaluation.",
        group="different-label",
        split="test",
    )
    path = tmp_path / "episode.json"
    path.write_text(canonical_json(agent.record()))
    build_corpus([path], tmp_path / "evaluation")
    with pytest.raises(ValueError, match="geometry group crosses"):
        register_corpus(tmp_path / "evaluation", registry)


def test_concurrent_registry_claims_serialize_and_conflict_atomically(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    registry = tmp_path / "shared.sqlite"
    barrier = Barrier(2)

    def claim(split):
        barrier.wait(timeout=10)
        try:
            return _register_claims(
                registry,
                content=digest(split),
                manifest=digest(split),
                groups={(digest("same-geometry"), split)},
                trajectories=set(),
            )["status"]
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["train", "test"]))
    assert results.count("registered") == 1
    assert sum("crosses registry splits" in r for r in results) == 1
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT count(*) FROM groups").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM bundles").fetchone() == (1,)


def test_artifact_and_total_budgets_fail_without_publishing(tmp_path, monkeypatch):
    from space_env.llm import corpus

    path, _ = write_record(tmp_path, "episode.json")
    monkeypatch.setattr(corpus, "MAX_FILE_BYTES", 100)
    with pytest.raises(ValueError, match="oversized"):
        build_corpus([path], tmp_path / "large-file")
    assert not (tmp_path / "large-file").exists()
    monkeypatch.setattr(corpus, "MAX_FILE_BYTES", 64000000)
    monkeypatch.setattr(corpus, "MAX_BUNDLE_BYTES", 100)
    with pytest.raises(ValueError, match="byte budget"):
        build_corpus([path], tmp_path / "large-corpus")
    assert not (tmp_path / "large-corpus").exists()
