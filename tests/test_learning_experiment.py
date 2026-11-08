"""Native isolation tests and small unit tests; no fake model result claims."""

import json

import pytest

from space_env.llm.corpus import build_corpus
from space_env.llm.episode import LLMEpisode
from space_env.llm.learning_experiment import (
    encode_example,
    model_manifest,
    prepare,
    run,
)
from space_env.simulation.snapshot import canonical_json
from space_env.tasks import TaskSession
from tests.test_llm_episode import session as original_session


def session():
    native = original_session()
    return TaskSession(
        native._scenario, native._task.model_copy(update={"decision_sec": 5.0})
    )


def corpus(tmp_path, name, split):
    record = LLMEpisode(
        session(), task_text="Deliver.", group=name, split=split
    ).record()
    path = tmp_path / f"{name}.json"
    path.write_text(canonical_json(record))
    root = tmp_path / name
    build_corpus([path], root)
    return root


def test_global_geometry_isolation_ignores_caller_group(tmp_path):
    a = corpus(tmp_path, "train", "train")
    b = corpus(tmp_path, "heldout", "test")
    with pytest.raises(ValueError, match="geometry crosses experiment"):
        prepare([a, b], evaluation_split="test")


def test_duplicate_and_empty_training_rejected(tmp_path):
    root = corpus(tmp_path, "train", "train")
    with pytest.raises(ValueError, match="duplicate experiment"):
        prepare([root, root])
    with pytest.raises(ValueError, match="eligible train"):
        prepare([root])
    with pytest.raises(ValueError, match="held-out"):
        prepare([root], evaluation_split="train")


def test_model_manifest_hashes_resolved_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    weights = tmp_path / "weights"
    weights.write_bytes(b"test fixture, not model weights")
    (tmp_path / "model.safetensors").symlink_to(weights)
    first = model_manifest(tmp_path)
    weights.write_bytes(b"changed test fixture")
    assert model_manifest(tmp_path)["sha256"] != first["sha256"]
    (tmp_path / "config.json").write_text('{"model_type":"other"}')
    with pytest.raises(ValueError, match="Qwen"):
        model_manifest(tmp_path)


class TokenizerFixture:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return [4, 5]


def test_assistant_only_loss_and_no_silent_truncation():
    row = {"messages": [], "target": "fixture"}
    ids, labels = encode_example(TokenizerFixture(), row, 6)
    assert ids == [1, 2, 3, 4, 5, 99]
    assert labels == [-100, -100, -100, 4, 5, 99]
    with pytest.raises(ValueError, match="truncation"):
        encode_example(TokenizerFixture(), row, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"output_tokens": 8192},
        {"learning_rate": float("nan")},
        {"max_calls": 0},
    ],
)
def test_bad_budgets_fail_before_model_import(tmp_path, kwargs):
    with pytest.raises(ValueError, match="budgets"):
        run([], tmp_path / "model", tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_tampered_native_corpus_fails_closed(tmp_path):
    root = corpus(tmp_path, "train", "train")
    (root / "data/sft.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="content mismatch"):
        prepare([root])


def test_successful_native_demonstrations_become_causal_train_rows(tmp_path):
    from space_env.llm.learning_experiment import PROFILE
    from space_env.llm.ollama import decode_calls, wire_context
    from tests.test_llm_wire_data import successful_wire

    # Scripted native demonstration, explicitly not real model evidence.
    record, _trace = successful_wire.__wrapped__()
    path = tmp_path / "successful.json"
    path.write_text(canonical_json(record))
    heldout = LLMEpisode(
        session(), task_text="Held out.", group="heldout", split="test"
    ).record()
    other = tmp_path / "heldout.json"
    other.write_text(canonical_json(heldout))
    root = tmp_path / "corpus"
    build_corpus([path, other], root)
    data = prepare([root], evaluation_split="test")
    assert len(data["evaluations"]) == 1
    assert data["rows"]
    assert {r["origin"] for r in data["rows"]} == {"scripted_public"}
    for row in data["rows"]:
        assert row["episode_sha256"] == record["sha256"]
        i = row["event"]
        expected, tools, _ = wire_context(
            record["messages"][: 2 + 2 * i], record["tools"], 2, PROFILE, 5.0
        )
        assert row["messages"] == expected
        target = json.loads(row["target"])
        assert "expected_revision" not in target and "idempotency_key" not in target
        calls = decode_calls(
            {"message": {"content": row["target"]}},
            tools,
            PROFILE,
            current_revision=json.loads(expected[1]["content"])["current_revision"],
            request_index=i,
            fixed_decision_sec=5.0,
        )
        assert calls[0]["name"] == record["events"][i]["name"]
        semantic = lambda a: {
            k: v
            for k, v in a.items()
            if k not in ("expected_revision", "idempotency_key")
        }
        assert semantic(json.loads(calls[0]["arguments"])) == semantic(
            json.loads(record["events"][i]["arguments"])
        )


def test_requested_qwen35_cannot_silently_use_qwen2(tmp_path):
    from space_env.llm.learning_experiment import feasibility

    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    (tmp_path / "model.safetensors").write_bytes(b"unit fixture")
    report = feasibility(tmp_path)
    assert not report["training_imports_ready"]
    assert any("not substitutes" in b for b in report["blockers"])


def test_partial_provider_failure_is_retained_as_native_failure(tmp_path, monkeypatch):
    from space_env.llm import learning_experiment as experiment
    from space_env.llm.ollama import InferenceStopped

    original = LLMEpisode(
        session(), task_text="Deliver.", group="g", split="test"
    ).record()

    class StoppedFixture:
        def __init__(self, *args, **kwargs):
            self.trace = [{"fixture": "synthetic stop, not model output"}]

        def __call__(self, *args):
            raise InferenceStopped("fixture_stop")

    monkeypatch.setattr(experiment, "LocalPolicy", StoppedFixture)
    rows = experiment.evaluate(
        None,
        None,
        {"evaluations": [original]},
        tmp_path / "eval",
        max_calls=1,
        max_tokens=20,
        output_tokens=5,
    )
    assert rows[0]["provider_error"] == "fixture_stop"
    artifact = json.loads((tmp_path / "eval/0000.json").read_text())
    assert artifact["episode"]["events"] == []
    assert artifact["episode"]["configuration"]["split"] == "test"
    assert not artifact["episode"]["metrics"]["success"]


def test_missing_eos_is_rejected():
    tokenizer = TokenizerFixture()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        encode_example(tokenizer, {"messages": [], "target": "fixture"}, 20)


def test_nonfixed_checkpoint_cadence_is_rejected(tmp_path):
    record = LLMEpisode(original_session(), task_text="Deliver.", group="g").record()
    path = tmp_path / "nonfixed.json"
    path.write_text(canonical_json(record))
    root = tmp_path / "corpus"
    build_corpus([path], root)
    with pytest.raises(ValueError, match="cadence"):
        prepare([root])


@pytest.fixture(scope="module")
def completed_audit_fixture(tmp_path_factory):
    """Entirely scripted artifact fixture, never evidence of real model training."""
    from space_env.llm import learning_experiment as experiment
    from space_env.llm.episode import STEP, seal
    from space_env.llm.ollama import decode_calls, wire_context
    from space_env.simulation.snapshot import digest
    from space_env.simulation.streaming import file_digest
    from tests.test_llm_wire_data import successful_wire

    root = tmp_path_factory.mktemp("scripted-learning-audit")
    teacher, _trace = successful_wire.__wrapped__()
    train_path = root / "teacher.json"
    train_path.write_text(canonical_json(teacher))
    heldout = LLMEpisode(
        session(), task_text="Held out.", group="audit-heldout", split="test"
    ).record()
    eval_path = root / "heldout.json"
    eval_path.write_text(canonical_json(heldout))
    data_path = root / "corpus"
    build_corpus([train_path, eval_path], data_path)
    prepared = prepare([data_path], evaluation_split="test")
    model_path = root / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    (model_path / "model.safetensors").write_bytes(
        b"SCRIPTED TEST FIXTURE: not weights"
    )
    identity = model_manifest(model_path)
    output = root / "experiment"
    output.mkdir()
    settings = {
        "max_calls": 1,
        "max_tokens": 8192,
        "output_tokens": 256,
        "steps": 1,
        "seed": 0,
        "learning_rate": 1e-5,
        "protocol": experiment.PROFILE,
        "evaluation_split": "test",
        "enable_thinking": False,
        "do_sample": False,
        "use_cache": False,
        "window_pairs": 2,
        "fixed_decision_sec": experiment.CADENCE,
        "max_dialogue_bytes": experiment.DIALOGUE_BYTES,
        "optimizer": "SGD",
        "adapter": {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]},
        "allow_other_qwen": False,
        "device": "cpu",
        "dtype": "float32",
    }
    manifest = {
        "schema": "space-env.learning-experiment/1",
        "status": "completed",
        "model": identity,
        "sources": prepared["sources"],
        "corpora": prepared["corpora"],
        "data_sha256": digest(prepared["rows"]),
        "geometry_splits": prepared["geometry_splits"],
        "training_origins": sorted({r["origin"] for r in prepared["rows"]}),
        "train_rows": len(prepared["rows"]),
        "decision_counts": prepared["decision_counts"],
        "unrepresentable_decisions": prepared["unrepresentable_decisions"],
        "settings": settings,
        "losses": [1.0],
        "scope": "scripted unit fixture, not model results",
    }
    for phase in ("base", "after"):
        episode = LLMEpisode(
            TaskSession.from_checkpoint(heldout["initial_checkpoint"]),
            task_text="Held out.",
            group="audit-heldout",
            split="test",
            max_calls=1,
            max_dialogue_bytes=experiment.DIALOGUE_BYTES,
            policy_origin="external_llm",
        )
        messages, tools, schema = wire_context(
            episode.messages,
            episode.tool_schemas(),
            2,
            experiment.PROFILE,
            experiment.CADENCE,
        )
        roles = sorted(episode.roles)
        raw = canonical_json(
            {"tool": STEP, **{f"action_{i}": "wait" for i in range(len(roles))}}
        )
        calls = decode_calls(
            {"message": {"content": raw}},
            tools,
            experiment.PROFILE,
            current_revision=json.loads(messages[1]["content"])["current_revision"],
            request_index=0,
            fixed_decision_sec=experiment.CADENCE,
        )
        episode.invoke(calls[0]["name"], calls[0]["arguments"], call_id="model-1")
        record = episode.record()
        artifact = seal(
            {
                "episode": record,
                "provider_error": None,
                "trace": [
                    {
                        "messages": messages,
                        "format": schema,
                        "raw": raw,
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "generation_wall_sec": 0.1,
                        "generation_complete": True,
                        "delivered_calls": calls,
                    }
                ],
            }
        )
        (output / phase).mkdir()
        (output / phase / "0000.json").write_text(canonical_json(artifact))
        manifest[phase] = [
            {
                "initial_checkpoint_sha256": record["initial_checkpoint"]["sha256"],
                "episode_sha256": record["sha256"],
                "artifact_sha256": artifact["sha256"],
                "metrics": record["metrics"],
                "provider_error": None,
            }
        ]
    tuned = output / "tuned"
    tuned.mkdir()
    (tuned / "adapter_config.json").write_text(
        canonical_json(
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    (tuned / "adapter_model.safetensors").write_bytes(
        b"SCRIPTED TEST FIXTURE: not an adapter"
    )
    manifest["tuned_model"] = seal(
        {
            "kind": "PEFT LoRA adapter",
            "base_model_sha256": identity["sha256"],
            "files": {
                p.name: {"sha256": file_digest(p), "bytes": p.stat().st_size}
                for p in tuned.iterdir()
            },
        }
    )
    (output / "manifest.json").write_text(canonical_json(seal(manifest)))
    return output


def copy_audit_fixture(source, tmp_path):
    import shutil

    output = tmp_path / "experiment"
    shutil.copytree(source, output)
    return output


def reseal_artifact(output, mutate):
    from space_env.llm.episode import seal

    path = output / "base/0000.json"
    artifact = json.loads(path.read_text())
    mutate(artifact)
    artifact = seal({k: v for k, v in artifact.items() if k != "sha256"})
    path.write_text(canonical_json(artifact))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base"][0]["artifact_sha256"] = artifact["sha256"]
    manifest_path.write_text(
        canonical_json(seal({k: v for k, v in manifest.items() if k != "sha256"}))
    )


def test_completed_audit_is_offline_consistency_only(
    completed_audit_fixture, monkeypatch
):
    from space_env.llm import learning_experiment as experiment

    def forbid_model_call(*args, **kwargs):
        pytest.fail("offline audit attempted inference")

    monkeypatch.setattr(experiment, "LocalPolicy", forbid_model_call)
    report = experiment.audit(completed_audit_fixture)
    assert report["valid"]
    assert report["native_episodes_replayed"] == 2
    assert report["model_calls_bound_to_native"] == 2
    assert report["llm_resampled"] is False
    assert "does not prove" in report["scope"]


@pytest.mark.parametrize(
    "fault",
    ["prompt", "schema", "raw", "delivery", "missing", "tokens", "time", "error"],
)
def test_audit_rejects_resealed_trace_tampering(
    completed_audit_fixture, tmp_path, fault
):
    from space_env.llm.learning_experiment import audit

    output = copy_audit_fixture(completed_audit_fixture, tmp_path)

    def mutate(artifact):
        event = artifact["trace"][0]
        if fault == "prompt":
            event["messages"][0]["content"] += " invented instruction"
        elif fault == "schema":
            event["format"]["required"] = []
        elif fault == "raw":
            value = json.loads(event["raw"])
            value["tool"] = "space_env_actor_finish"
            event["raw"] = canonical_json(value)
        elif fault == "delivery":
            event["delivered_calls"] = []
        elif fault == "missing":
            artifact["trace"] = []
        elif fault == "tokens":
            event["output_tokens"] = 257
        elif fault == "time":
            event["generation_wall_sec"] = -1
        else:
            artifact["provider_error"] = "generation_incomplete"

    reseal_artifact(output, mutate)
    with pytest.raises(ValueError):
        audit(output)


@pytest.mark.parametrize(
    "fault",
    ["data", "sources", "metrics", "budget", "incomplete", "pair", "decision_counts"],
)
def test_audit_rejects_resealed_manifest_tampering(
    completed_audit_fixture, tmp_path, fault
):
    from space_env.llm.episode import seal
    from space_env.llm.learning_experiment import audit

    output = copy_audit_fixture(completed_audit_fixture, tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "data":
        manifest["data_sha256"] = "0" * 64
    elif fault == "sources":
        manifest["sources"] = {}
    elif fault == "decision_counts":
        manifest["decision_counts"]["nonwait"] = 0
    elif fault == "metrics":
        manifest["base"][0]["metrics"]["success"] = True
    elif fault == "budget":
        manifest["settings"]["max_calls"] = 2
    elif fault == "incomplete":
        manifest["status"] = "incomplete"
    else:
        manifest["after"] = []
    path.write_text(
        canonical_json(seal({k: v for k, v in manifest.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        audit(output)


@pytest.mark.parametrize("fault", ["adapter", "extra", "symlink", "base"])
def test_audit_rejects_file_tampering(completed_audit_fixture, tmp_path, fault):
    from space_env.llm.episode import seal
    from space_env.llm.learning_experiment import audit

    output = copy_audit_fixture(completed_audit_fixture, tmp_path)
    if fault == "adapter":
        (output / "tuned/adapter_model.safetensors").write_bytes(b"tampered")
    elif fault == "extra":
        (output / "unbound.json").write_text("{}")
    elif fault == "symlink":
        (output / "link").symlink_to(output / "manifest.json")
    else:
        # Point the manifest at a changed copy, leaving the shared source fixture intact.
        import shutil

        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        source = manifest["model"]["path"]
        copied = tmp_path / "changed-model"
        shutil.copytree(source, copied)
        (copied / "model.safetensors").write_bytes(b"tampered model")
        manifest["model"]["path"] = str(copied)
        manifest["model"] = seal(
            {k: v for k, v in manifest["model"].items() if k != "sha256"}
        )
        path.write_text(
            canonical_json(seal({k: v for k, v in manifest.items() if k != "sha256"}))
        )
    with pytest.raises(ValueError):
        audit(output)


def test_cli_preflight_blockers_exit_one(monkeypatch, capsys):
    import sys

    from space_env.llm import learning_experiment as experiment

    monkeypatch.setattr(
        experiment, "feasibility", lambda *args, **kwargs: {"blockers": ["fixture"]}
    )
    monkeypatch.setattr(sys, "argv", ["learning_experiment", "preflight"])
    assert experiment.main() == 1
    assert json.loads(capsys.readouterr().out)["blockers"] == ["fixture"]


def test_default_native_call_budget_is_64():
    import inspect

    assert inspect.signature(run).parameters["max_calls"].default == 64


@pytest.mark.parametrize(
    "stop",
    ["context_budget", "generation_incomplete", "invalid_structured_action_response"],
)
def test_offline_audit_retains_valid_provider_stops(
    completed_audit_fixture, tmp_path, stop
):
    from space_env.llm import learning_experiment as experiment
    from space_env.llm.episode import seal

    output = copy_audit_fixture(completed_audit_fixture, tmp_path)
    path = output / "base/0000.json"
    artifact = json.loads(path.read_text())
    record = artifact["episode"]
    empty = LLMEpisode(
        TaskSession.from_checkpoint(record["initial_checkpoint"]),
        **record["configuration"],
    ).record()
    artifact["episode"] = empty
    artifact["provider_error"] = stop
    trace = artifact["trace"][0]
    trace.pop("delivered_calls")
    trace["raw"] = "{"
    trace["stop_error"] = stop
    trace["generation_complete"] = stop == "invalid_structured_action_response"
    if stop == "context_budget":
        trace.update(
            input_tokens=8192, output_tokens=0, raw="", generation_wall_sec=0.0
        )
    artifact = seal({k: v for k, v in artifact.items() if k != "sha256"})
    path.write_text(canonical_json(artifact))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base"][0].update(
        episode_sha256=empty["sha256"],
        artifact_sha256=artifact["sha256"],
        metrics=empty["metrics"],
        provider_error=stop,
    )
    manifest_path.write_text(
        canonical_json(seal({k: v for k, v in manifest.items() if k != "sha256"}))
    )
    report = experiment.audit(output)
    assert report["valid"]
    assert report["model_calls_bound_to_native"] == 1


def test_resealed_raw_and_delivery_cannot_change_native_action(
    completed_audit_fixture, tmp_path
):
    from space_env.llm import learning_experiment as experiment
    from space_env.llm.ollama import decode_calls, wire_context

    output = copy_audit_fixture(completed_audit_fixture, tmp_path)

    def mutate(artifact):
        event = artifact["trace"][0]
        target = json.loads(event["raw"])
        target["tool"] = "space_env_actor_finish"
        event["raw"] = canonical_json(target)
        _, tools, _ = wire_context(
            artifact["episode"]["messages"][:2],
            artifact["episode"]["tools"],
            2,
            experiment.PROFILE,
            experiment.CADENCE,
        )
        event["delivered_calls"] = decode_calls(
            {"message": {"content": event["raw"]}},
            tools,
            experiment.PROFILE,
            current_revision=0,
            request_index=0,
            fixed_decision_sec=experiment.CADENCE,
        )

    reseal_artifact(output, mutate)
    with pytest.raises(ValueError, match="native action"):
        experiment.audit(output)


def test_audit_cli_reports_no_resampling(completed_audit_fixture, monkeypatch, capsys):
    import sys

    from space_env.llm import learning_experiment as experiment

    monkeypatch.setattr(
        sys,
        "argv",
        ["learning_experiment", "audit", "--output", str(completed_audit_fixture)],
    )
    assert experiment.main() == 0
    assert json.loads(capsys.readouterr().out)["llm_resampled"] is False


@pytest.fixture(scope="module")
def reference_sequence_teacher():
    from tests.test_sequence_lanes import native_sequence

    # This existing fixture runs the real three-role sequence_lanes planner and
    # native reference_policy, with sparse role choices; no provider is involved.
    _meta, record = native_sequence.__wrapped__()
    assert record["metrics"]["sft_eligible"]
    return record


def build_teacher_corpus(tmp_path, teacher):
    path = tmp_path / "teacher.json"
    path.write_text(canonical_json(teacher))
    heldout = LLMEpisode(
        session(), task_text="Held out.", group="g", split="test"
    ).record()
    other = tmp_path / "heldout.json"
    other.write_text(canonical_json(heldout))
    root = tmp_path / "corpus"
    build_corpus([path, other], root)
    return root


def test_sparse_reference_sequence_preserves_nonwait_targets_and_native_effects(
    reference_sequence_teacher, tmp_path
):
    from space_env.llm.episode import STEP, verify_record
    from space_env.llm.learning_experiment import CADENCE, PROFILE
    from space_env.llm.ollama import decode_calls, wire_context

    teacher = reference_sequence_teacher
    root = build_teacher_corpus(tmp_path, teacher)
    data = prepare([root], evaluation_split="test")
    roles = sorted(
        LLMEpisode(
            TaskSession.from_checkpoint(teacher["initial_checkpoint"]),
            **teacher["configuration"],
        ).roles
    )
    assert len(roles) == 3
    assert len(data["rows"]) == len(teacher["events"])
    assert data["unrepresentable_decisions"] == 0
    native = LLMEpisode(
        TaskSession.from_checkpoint(teacher["initial_checkpoint"]),
        **teacher["configuration"],
    )
    normalized = LLMEpisode(
        TaskSession.from_checkpoint(teacher["initial_checkpoint"]),
        **teacher["configuration"],
    )
    nonwait_count, wait_count, sparse_nonwait = 0, 0, 0
    for row, event in zip(data["rows"], teacher["events"]):
        original = json.loads(event["arguments"])
        target = json.loads(row["target"])
        assert event["name"] == STEP
        assert set(target) == {"tool", "action_0", "action_1", "action_2"}
        expected_commands = {
            r: [a for a in actions if a != "wait"]
            for r, actions in original["choices"].items()
            if any(a != "wait" for a in actions)
        }
        target_commands = {
            r: [target[f"action_{i}"]]
            for i, r in enumerate(roles)
            if target[f"action_{i}"] != "wait"
        }
        assert target_commands == expected_commands
        if expected_commands:
            nonwait_count += 1
            sparse_nonwait += set(original["choices"]) != set(roles)
        else:
            wait_count += 1
        # The decoder gets native schemas narrowed from the actual causal prefix.
        messages, tools, _ = wire_context(
            normalized.messages, normalized.tool_schemas(), 2, PROFILE, CADENCE
        )
        decoded = decode_calls(
            {"message": {"content": row["target"]}},
            tools,
            PROFILE,
            current_revision=json.loads(messages[1]["content"])["current_revision"],
            request_index=row["event"],
            fixed_decision_sec=CADENCE,
        )[0]
        native.invoke(event["name"], event["arguments"], call_id=event["call_id"])
        normalized.invoke(
            decoded["name"], decoded["arguments"], call_id=f"normalized-{row['event']}"
        )
        # Task journal/SDK nonce receipts differ; the actual native mission state
        # and public actor observation must remain exactly equal at every step.
        assert normalized.observe() == native.observe()
        assert normalized._session._runtime.checkpoint(
            compact=True
        ) == native._session._runtime.checkpoint(compact=True)
    assert sparse_nonwait > 0
    assert nonwait_count > 0 and wait_count > 0
    assert data["decision_counts"] == {
        "wait": wait_count,
        "nonwait": nonwait_count,
        "finish": 0,
    }
    assert normalized.record()["metrics"]["success"]
    verify_record(normalized.record())


def test_positive_allwait_tail_is_refused(reference_sequence_teacher, tmp_path):
    from space_env.llm.episode import STEP

    teacher = reference_sequence_teacher
    prefix = LLMEpisode(
        TaskSession.from_checkpoint(teacher["initial_checkpoint"]),
        **teacher["configuration"],
    )
    last_command = max(
        i
        for i, e in enumerate(teacher["events"])
        if e["name"] == STEP
        and any(
            a != "wait"
            for ids in json.loads(e["arguments"])["choices"].values()
            for a in ids
        )
    )
    for event in teacher["events"][: last_command + 1]:
        prefix.invoke(event["name"], event["arguments"], call_id=event["call_id"])
    tail = LLMEpisode(prefix._session, **teacher["configuration"])
    for event in teacher["events"][last_command + 1 :]:
        tail.invoke(event["name"], event["arguments"], call_id=event["call_id"])
    record = tail.record()
    assert record["events"] and record["metrics"]["sft_eligible"]
    root = build_teacher_corpus(tmp_path, record)
    with pytest.raises(ValueError, match="at least one non-wait"):
        prepare([root], evaluation_split="test")


@pytest.mark.parametrize(
    "choices", [{"unknown": ["wait"]}, {"a": "wait"}, {"a": [None]}]
)
def test_noop_normalization_rejects_unknown_or_malformed_choices(choices):
    from space_env.llm.learning_experiment import _nonwait_choices

    with pytest.raises(ValueError):
        _nonwait_choices(choices, ["a", "b", "c"])


def test_noop_normalization_retains_command_multiplicity():
    from space_env.llm.learning_experiment import _nonwait_choices

    assert _nonwait_choices({"a": ["wait", "capture"], "c": []}, ["a", "b", "c"]) == {
        "a": ["capture"]
    }
    assert _nonwait_choices({"a": ["capture", "capture"]}, ["a", "b", "c"]) == {
        "a": ["capture", "capture"]
    }
