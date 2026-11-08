"""Offline native Qwen SFT experiment. No downloads or synthetic model fallback."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from space_env.llm.corpus import _read, verify_corpus
from space_env.llm.episode import STEP, LLMEpisode, seal, sources, verify_record
from space_env.llm.ollama import InferenceStopped, decode_calls, wire_context
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest
from space_env.tasks import TaskSession

PROFILE = "json_fixed_v1"
CADENCE = 5.0
DIALOGUE_BYTES = 16000000


def model_manifest(directory):
    """Hash actual resolved local files (HF snapshot symlinks are supported)."""
    root = Path(directory).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    if not str(config.get("model_type", "")).startswith("qwen"):
        raise ValueError("local Qwen model required")
    files = {
        str(p.relative_to(root)): {"sha256": file_digest(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    if not any(n.endswith(".safetensors") for n in files):
        raise ValueError("local safetensors weights required")
    return seal({"path": str(root), "config": config, "files": files})


def feasibility(model=None, *, allow_other_qwen=False):
    result = {"packages": {}, "blockers": [], "cuda": False}
    for name in ("torch", "transformers", "peft", "datasets", "trl"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch

        result.update(
            cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads()
        )
        from peft import LoraConfig, get_peft_model  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- preflight must report optional import/ABI failures
        result["blockers"].append(f"{type(exc).__name__}: {exc}")
    if Path("/proc/meminfo").exists():
        result["memory"] = Path("/proc/meminfo").read_text().splitlines()[:3]
    if model:
        try:
            result["model"] = model_manifest(model)
            if (
                not allow_other_qwen
                and "qwen3_5" not in result["model"]["config"]["model_type"]
            ):
                result["blockers"].append(
                    "requested Qwen3.5 requires local HF safetensors; other Qwen versions are not substitutes"
                )
        except (OSError, ValueError, KeyError) as exc:
            result["blockers"].append(f"{type(exc).__name__}: {exc}")
    else:
        result["blockers"].append(
            "local Qwen3.5 HF directory required; Ollama GGUF is not a PEFT training checkpoint"
        )
    result["training_imports_ready"] = not result["blockers"]
    result["scope"] = (
        "import and file checks only; allocation/context feasibility checked at execution"
    )
    return result


def _nonwait_choices(choices, roles):
    """Native omitted roles, empty lists and explicit waits perform no command.

    Unknown roles and malformed lists are never normalized into valid commands.
    Keep action multiplicity/order, so duplicate or multi-command targets cannot
    silently become a single-command fixed-protocol demonstration.
    """
    if not isinstance(choices, dict) or set(choices) - set(roles):
        raise ValueError("unknown or malformed actor choices")
    result = {}
    for role, actions in choices.items():
        if not isinstance(actions, list) or any(
            not isinstance(a, str) for a in actions
        ):
            raise ValueError("malformed actor action list")
        nonwait = [a for a in actions if a != "wait"]
        if nonwait:
            result[role] = nonwait
    return result


def _decision_counts(rows):
    counts = {"wait": 0, "nonwait": 0, "finish": 0}
    for row in rows:
        target = json.loads(row["target"])
        if target["tool"] == STEP:
            key = (
                "nonwait"
                if any(
                    v != "wait" for k, v in target.items() if k.startswith("action_")
                )
                else "wait"
            )
        else:
            key = "finish"
        counts[key] += 1
    return counts


def prepare(corpora, *, evaluation_split="development"):
    """Replay corpora; enforce global geometry/trajectory isolation; causal rows only."""
    if evaluation_split not in ("development", "test"):
        raise ValueError("held-out evaluation split required")
    groups, trajectories, records, manifests = {}, set(), [], []
    for directory in corpora:
        directory = Path(directory)
        receipt = verify_corpus(directory)
        manifest = _read(directory / "corpus.json")
        manifests.append(
            {"path": str(directory.resolve()), "sha256": receipt["sha256"]}
        )
        for entry in manifest["entries"]:
            g, split = entry["geometry_sha256"], entry["split"]
            if g in groups and groups[g] != split:
                raise ValueError("geometry crosses experiment splits")
            groups[g] = split
            if entry["trajectory_sha256"] in trajectories:
                raise ValueError("duplicate experiment trajectory")
            trajectories.add(entry["trajectory_sha256"])
            record = _read(
                directory / "audit" / f"{entry['index']:04d}" / "episode.json"
            )
            if (
                TaskSession.from_checkpoint(
                    record["initial_checkpoint"]
                )._task.decision_sec
                != CADENCE
            ):
                raise ValueError("checkpoint cadence must equal fixed 5 seconds")
            records.append((entry, record))
    rows, evaluations, seen_eval, rejected = [], [], set(), 0
    for entry, record in records:
        if entry["split"] == evaluation_split:
            # One task per checkpoint, independent of demonstration policy/trajectory.
            key = record["initial_checkpoint"]["sha256"]
            if key not in seen_eval:
                evaluations.append(record)
                seen_eval.add(key)
        if entry["split"] != "train" or not entry["native_sft_eligible"]:
            continue
        for i, event in enumerate(record["events"]):
            messages, tools, _schema = wire_context(
                record["messages"][: 2 + 2 * i], record["tools"], 2, PROFILE, CADENCE
            )
            args = json.loads(event["arguments"])
            roles = sorted(
                next(
                    t["function"]["parameters"]["properties"]["choices"]["properties"]
                    for t in tools
                    if t["function"]["name"] == STEP
                )
            )
            revision = json.loads(messages[1]["content"])["current_revision"]
            target = {
                "tool": event["name"],
                **{f"action_{j}": "wait" for j in range(len(roles))},
            }
            if event["name"] == STEP:
                choices = args.get("choices", {})
                try:
                    normalized = _nonwait_choices(choices, roles)
                except ValueError:
                    rejected += 1
                    continue
                if any(len(actions) > 1 for actions in normalized.values()):
                    rejected += 1
                    continue
                if args["advance_sec"] != CADENCE:
                    rejected += 1
                    continue
                target.update(
                    {
                        f"action_{j}": (normalized.get(r) or ["wait"])[0]
                        for j, r in enumerate(roles)
                    }
                )
            if event["name"] not in (STEP, "space_env_actor_finish"):
                rejected += 1
                continue
            decoded = decode_calls(
                {"message": {"content": canonical_json(target)}},
                tools,
                PROFILE,
                current_revision=revision,
                request_index=i,
                fixed_decision_sec=CADENCE,
            )

            def semantic(arguments, roles=roles):
                result = {
                    k: v
                    for k, v in arguments.items()
                    if k not in ("expected_revision", "idempotency_key")
                }
                if "choices" in result:
                    result["choices"] = _nonwait_choices(result["choices"], roles)
                return result

            if decoded[0]["name"] != event["name"] or semantic(
                json.loads(decoded[0]["arguments"])
            ) != semantic(args):
                rejected += 1
                continue
            rows.append(
                {
                    "messages": messages,
                    "target": canonical_json(target),
                    "episode_sha256": record["sha256"],
                    "origin": entry["policy_origin"],
                    "geometry_sha256": entry["geometry_sha256"],
                    "event": i,
                }
            )
    if not rows or not evaluations:
        raise ValueError("need eligible train decisions and held-out native tasks")
    decision_counts = _decision_counts(rows)
    if not decision_counts["nonwait"]:
        raise ValueError(
            "positive SFT requires at least one non-wait decision; all-wait data refused; "
            + canonical_json(decision_counts)
        )
    return {
        "decision_counts": decision_counts,
        "corpora": manifests,
        "rows": rows,
        "evaluations": evaluations,
        "geometry_splits": groups,
        "unrepresentable_decisions": rejected,
        "protocol": PROFILE,
        "sources": sources(),
    }


def encode_example(tokenizer, row, max_tokens):
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS token required")
    prompt = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target = tokenizer.encode(row["target"], add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(prompt) + len(target) > max_tokens:
        raise ValueError("SFT context exceeds budget; truncation is forbidden")
    return prompt + target, [-100] * len(prompt) + target


class LocalPolicy:
    def __init__(self, model, tokenizer, *, max_tokens, output_tokens):
        self.model, self.tokenizer = model, tokenizer
        self.max_tokens, self.output_tokens = max_tokens, output_tokens
        self.trace = []

    def __call__(self, messages, tools):
        import torch

        selected, narrowed, schema = wire_context(messages, tools, 2, PROFILE, CADENCE)
        ids = self.tokenizer.apply_chat_template(
            selected, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(ids) + self.output_tokens > self.max_tokens:
            self.trace.append(
                {
                    "messages": selected,
                    "format": schema,
                    "input_tokens": len(ids),
                    "output_tokens": 0,
                    "raw": "",
                    "generation_wall_sec": 0.0,
                    "generation_complete": False,
                    "stop_error": "context_budget",
                }
            )
            raise InferenceStopped("context_budget")
        tensor = torch.tensor([ids], device=next(self.model.parameters()).device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                do_sample=False,
                max_new_tokens=self.output_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, len(ids) :].tolist()
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        event = {
            "messages": selected,
            "format": schema,
            "input_tokens": len(ids),
            "output_tokens": len(tokens),
            "raw": raw,
            "generation_wall_sec": time.perf_counter() - started,
            "generation_complete": bool(tokens)
            and tokens[-1] == self.tokenizer.eos_token_id,
        }
        self.trace.append(event)
        if not tokens or tokens[-1] != self.tokenizer.eos_token_id:
            event["stop_error"] = "generation_incomplete"
            raise InferenceStopped("generation_incomplete")
        try:
            calls = decode_calls(
                {"message": {"content": raw}},
                narrowed,
                PROFILE,
                current_revision=json.loads(selected[1]["content"])["current_revision"],
                request_index=len(self.trace) - 1,
                fixed_decision_sec=CADENCE,
            )
        except InferenceStopped as exc:
            event["stop_error"] = str(exc)
            raise
        event["delivered_calls"] = calls
        return calls[0]


def evaluate(
    model, tokenizer, prepared, output, *, max_calls, max_tokens, output_tokens
):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, original in enumerate(prepared["evaluations"]):
        cfg = original["configuration"]
        episode = LLMEpisode(
            TaskSession.from_checkpoint(original["initial_checkpoint"]),
            task_text=json.loads(original["messages"][1]["content"])["task"],
            group=cfg["group"],
            split=cfg["split"],
            max_calls=max_calls,
            max_dialogue_bytes=DIALOGUE_BYTES,
            policy_origin="external_llm",
        )
        policy = LocalPolicy(
            model, tokenizer, max_tokens=max_tokens, output_tokens=output_tokens
        )
        error = None
        try:
            while not episode.done:
                call = policy(episode.messages, episode.tool_schemas())
                episode.invoke(
                    call["name"],
                    call["arguments"],
                    call_id=f"model-{len(policy.trace)}",
                )
        except InferenceStopped as exc:
            error = str(exc)
        record = episode.record()
        verify_record(record)
        artifact = seal(
            {"episode": record, "trace": policy.trace, "provider_error": error}
        )
        atomic_write(output / f"{i:04d}.json", canonical_json(artifact))
        rows.append(
            {
                "initial_checkpoint_sha256": record["initial_checkpoint"]["sha256"],
                "episode_sha256": record["sha256"],
                "artifact_sha256": artifact["sha256"],
                "metrics": record["metrics"],
                "provider_error": error,
            }
        )
    return rows


def run(
    corpora,
    model_path,
    output,
    *,
    steps=1,
    max_tokens=8192,
    output_tokens=256,
    max_calls=64,
    seed=0,
    learning_rate=1e-5,
    evaluation_split="development",
    allow_other_qwen=False,
):
    """CPU LoRA SFT, greedy paired native rollouts; fails closed on errors."""
    if (
        type(steps) is not int
        or steps < 1
        or type(seed) is not int
        or not 1 <= max_calls <= 4096
        or not 0 < output_tokens < max_tokens
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("invalid experiment budgets")
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("output must not exist")
    for source in [*corpora, model_path]:
        if output.is_relative_to(Path(source).resolve()) or Path(
            source
        ).resolve().is_relative_to(output):
            raise ValueError("output must be separate from immutable inputs")
    prepared = prepare(corpora, evaluation_split=evaluation_split)
    identity = model_manifest(model_path)
    preflight = feasibility(model_path, allow_other_qwen=allow_other_qwen)
    if preflight["blockers"]:
        raise RuntimeError(canonical_json(preflight))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("local tokenizer needs chat template and EOS")
    encoded = [encode_example(tokenizer, r, max_tokens) for r in prepared["rows"]]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    ).cpu()
    if max_tokens > model.config.max_position_embeddings:
        raise ValueError("context exceeds model positions")
    output.mkdir(parents=True)
    settings = {
        "max_calls": max_calls,
        "max_tokens": max_tokens,
        "output_tokens": output_tokens,
    }
    manifest = {
        "schema": "space-env.learning-experiment/1",
        "status": "incomplete",
        "model": identity,
        "sources": prepared["sources"],
        "corpora": prepared["corpora"],
        "data_sha256": digest(prepared["rows"]),
        "geometry_splits": prepared["geometry_splits"],
        "training_origins": sorted({r["origin"] for r in prepared["rows"]}),
        "train_rows": len(encoded),
        "decision_counts": prepared["decision_counts"],
        "unrepresentable_decisions": prepared["unrepresentable_decisions"],
        "settings": {
            **settings,
            "steps": steps,
            "seed": seed,
            "learning_rate": learning_rate,
            "protocol": PROFILE,
            "evaluation_split": evaluation_split,
            "enable_thinking": False,
            "do_sample": False,
            "use_cache": False,
            "window_pairs": 2,
            "fixed_decision_sec": CADENCE,
            "max_dialogue_bytes": DIALOGUE_BYTES,
            "optimizer": "SGD",
            "adapter": {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]},
            "allow_other_qwen": allow_other_qwen,
            "device": "cpu",
            "dtype": "float32",
        },
        "feasibility": preflight,
    }
    atomic_write(output / "manifest.json", canonical_json(seal(manifest)))
    model.config.use_cache = False
    model.eval()
    before = evaluate(model, tokenizer, prepared, output / "base", **settings)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable adapter parameters")
    initial_adapter = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    losses = []
    for step in range(steps):
        ids, labels = encoded[step % len(encoded)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.save_pretrained(output / "tuned", safe_serialization=True)
    tokenizer.save_pretrained(output / "tuned")
    tuned = seal(
        {
            "kind": "PEFT LoRA adapter",
            "base_model_sha256": identity["sha256"],
            "files": {
                p.name: {"sha256": file_digest(p), "bytes": p.stat().st_size}
                for p in sorted((output / "tuned").iterdir())
                if p.is_file()
            },
        }
    )
    if all(torch.equal(a, b.detach()) for a, b in zip(initial_adapter, trainable)):
        raise RuntimeError("adapter weights did not change")
    model.eval()
    after = evaluate(model, tokenizer, prepared, output / "after", **settings)
    if sources() != prepared["sources"] or model_manifest(model_path) != identity:
        raise RuntimeError("source or base model changed during experiment")
    if [r["initial_checkpoint_sha256"] for r in before] != [
        r["initial_checkpoint_sha256"] for r in after
    ]:
        raise RuntimeError("unpaired native initial tasks")
    for receipt in prepared["corpora"]:
        if verify_corpus(receipt["path"])["sha256"] != receipt["sha256"]:
            raise RuntimeError("corpus changed during experiment")
    manifest.update(
        status="completed",
        base=before,
        after=after,
        tuned_model=tuned,
        losses=losses,
        scope="real local model generations and LoRA SGD; descriptive paired outcomes, no superiority claim",
    )
    atomic_write(output / "manifest.json", canonical_json(seal(manifest)))
    return seal(manifest)


def _sealed_read(path):
    value = _read(path)
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("invalid experiment artifact seal")
    return value


def _audit_trace(record, trace, error, settings):
    """Bind each public prompt and decoded action to a cold-replayed native event."""
    if not isinstance(trace, list):
        raise TypeError("trace must be a list")
    cursor, stopped = 0, None
    for index, attempt in enumerate(trace):
        if cursor >= settings["max_calls"] or stopped is not None:
            raise ValueError("trace exceeds native call budget or continues after stop")
        if (
            cursor == len(record["events"])
            and record["metrics"]["stop_reason"] != "caller_stopped"
        ):
            raise ValueError("trace continues after native termination")
        messages, tools, schema = wire_context(
            record["messages"][: 2 + 2 * cursor], record["tools"], 2, PROFILE, CADENCE
        )
        if attempt.get("messages") != messages or attempt.get("format") != schema:
            raise ValueError("trace causal prompt/schema mismatch")
        inputs, outputs = attempt.get("input_tokens"), attempt.get("output_tokens")
        wall = attempt.get("generation_wall_sec")
        complete = attempt.get("generation_complete")
        if (
            type(inputs) is not int
            or inputs < 1
            or type(outputs) is not int
            or not 0 <= outputs <= settings["output_tokens"]
            or type(wall) not in (float, int)
            or not math.isfinite(wall)
            or wall < 0
            or type(complete) is not bool
            or not isinstance(attempt.get("raw"), str)
        ):
            raise ValueError("invalid trace accounting")
        stop = attempt.get("stop_error")
        if inputs + settings["output_tokens"] > settings["max_tokens"]:
            if (
                stop != "context_budget"
                or outputs != 0
                or complete
                or attempt["raw"] != ""
                or wall != 0
            ):
                raise ValueError("context budget evidence mismatch")
            stopped = stop
        elif not complete:
            if stop != "generation_incomplete":
                raise ValueError("incomplete generation evidence mismatch")
            stopped = stop
        else:
            if outputs < 1:
                raise ValueError("completed generation needs output tokens")
            try:
                calls = decode_calls(
                    {"message": {"content": attempt["raw"]}},
                    tools,
                    PROFILE,
                    current_revision=json.loads(messages[1]["content"])[
                        "current_revision"
                    ],
                    request_index=index,
                    fixed_decision_sec=CADENCE,
                )
            except InferenceStopped as exc:
                if stop != str(exc):
                    raise ValueError("decoder stop evidence mismatch") from exc
                stopped = str(exc)
            else:
                if (
                    stop is not None
                    or attempt.get("delivered_calls") != calls
                    or len(calls) != 1
                ):
                    raise ValueError("decoded/delivered call mismatch")
                if cursor >= len(record["events"]):
                    raise ValueError("generated call has no native event")
                event = record["events"][cursor]
                call = calls[0]
                if (
                    event["name"] != call["name"]
                    or json.loads(event["arguments"]) != json.loads(call["arguments"])
                    or event["call_id"] != f"model-{index + 1}"
                ):
                    raise ValueError("raw generation/native action mismatch")
                cursor += 1
        if stopped is not None and attempt.get("delivered_calls"):
            raise ValueError("stopped generation delivered native calls")
    if cursor != len(record["events"]) or stopped != error:
        raise ValueError("trace/native length or provider error mismatch")
    if error is None and record["metrics"]["stop_reason"] == "caller_stopped":
        raise ValueError("unexplained early rollout stop")
    return cursor


def audit(directory):
    """Offline consistency audit only: never loads or resamples a language model.

    A self-consistent fabricated/resealed generation cannot be distinguished from
    a genuine one by hashes, native replay, or this audit.
    """
    directory = Path(directory).absolute()
    if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("experiment symlinks are not allowed")
    manifest = _sealed_read(directory / "manifest.json")
    if (
        manifest.get("schema") != "space-env.learning-experiment/1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("completed experiment required")
    if manifest["sources"] != sources():
        raise ValueError("experiment source identity mismatch")
    if model_manifest(manifest["model"]["path"]) != manifest["model"]:
        raise ValueError("base model file manifest mismatch")
    settings = manifest["settings"]
    fixed = {
        "protocol": PROFILE,
        "fixed_decision_sec": CADENCE,
        "max_dialogue_bytes": DIALOGUE_BYTES,
        "optimizer": "SGD",
        "adapter": {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]},
        "device": "cpu",
        "dtype": "float32",
        "enable_thinking": False,
        "do_sample": False,
        "use_cache": False,
        "window_pairs": 2,
    }
    if any(settings.get(k) != v for k, v in fixed.items()) or any(
        type(settings.get(k)) is not bool
        for k in ("enable_thinking", "do_sample", "use_cache", "allow_other_qwen")
    ):
        raise ValueError("experiment protocol/settings mismatch")
    if (
        type(settings["max_calls"]) is not int
        or not 1 <= settings["max_calls"] <= 4096
        or type(settings["output_tokens"]) is not int
        or type(settings["max_tokens"]) is not int
        or not 0 < settings["output_tokens"] < settings["max_tokens"]
        or type(settings["steps"]) is not int
        or settings["steps"] < 1
        or type(settings["seed"]) is not int
        or type(settings["learning_rate"]) not in (int, float)
        or not math.isfinite(settings["learning_rate"])
        or settings["learning_rate"] <= 0
    ):
        raise ValueError("invalid experiment settings/budgets")
    if (
        not settings["allow_other_qwen"]
        and "qwen3_5" not in manifest["model"]["config"]["model_type"]
    ):
        raise ValueError("unapproved substitute Qwen model")
    prepared = prepare(
        [r["path"] for r in manifest["corpora"]],
        evaluation_split=settings["evaluation_split"],
    )
    expected = {
        "corpora": prepared["corpora"],
        "data_sha256": digest(prepared["rows"]),
        "geometry_splits": prepared["geometry_splits"],
        "training_origins": sorted({r["origin"] for r in prepared["rows"]}),
        "train_rows": len(prepared["rows"]),
        "decision_counts": prepared["decision_counts"],
        "unrepresentable_decisions": prepared["unrepresentable_decisions"],
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("corpus/preparation manifest mismatch")
    losses = manifest["losses"]
    if (
        not isinstance(losses, list)
        or len(losses) != settings["steps"]
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in losses
        )
    ):
        raise ValueError("invalid training loss accounting")
    tuned = manifest["tuned_model"]
    actual_tuned = seal(
        {
            "kind": "PEFT LoRA adapter",
            "base_model_sha256": manifest["model"]["sha256"],
            "files": {
                str(p.relative_to(directory / "tuned")): {
                    "sha256": file_digest(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted((directory / "tuned").rglob("*"))
                if p.is_file()
            },
        }
    )
    if tuned != actual_tuned or not {
        "adapter_config.json",
        "adapter_model.safetensors",
    } <= set(tuned["files"]):
        raise ValueError("adapter file manifest mismatch")
    adapter = _read(directory / "tuned" / "adapter_config.json")
    if (
        adapter.get("r") != 8
        or adapter.get("lora_alpha") != 16
        or adapter.get("lora_dropout") != 0.0
        or adapter.get("task_type") != "CAUSAL_LM"
        or sorted(adapter.get("target_modules", [])) != ["q_proj", "v_proj"]
    ):
        raise ValueError("adapter configuration mismatch")
    expected_files = {"manifest.json", *(f"tuned/{n}" for n in tuned["files"])}
    bound = 0
    for phase in ("base", "after"):
        rows = manifest[phase]
        if len(rows) != len(prepared["evaluations"]):
            raise ValueError("incomplete native evaluation pairing")
        for i, (row, original) in enumerate(zip(rows, prepared["evaluations"])):
            name = f"{phase}/{i:04d}.json"
            expected_files.add(name)
            artifact = _sealed_read(directory / name)
            record = artifact["episode"]
            verify_record(record)
            cfg = original["configuration"]
            expected_config = {
                "task_text": json.loads(original["messages"][1]["content"])["task"],
                "group": cfg["group"],
                "split": cfg["split"],
                "max_calls": settings["max_calls"],
                "max_dialogue_bytes": DIALOGUE_BYTES,
                "policy_origin": "external_llm",
            }
            if (
                record["initial_checkpoint"] != original["initial_checkpoint"]
                or record["configuration"] != expected_config
                or record["messages"][:2] != original["messages"][:2]
                or record["tools"] != original["tools"]
            ):
                raise ValueError("native initial checkpoint/task/budgets mismatch")
            expected_row = {
                "initial_checkpoint_sha256": record["initial_checkpoint"]["sha256"],
                "episode_sha256": record["sha256"],
                "artifact_sha256": artifact["sha256"],
                "metrics": record["metrics"],
                "provider_error": artifact["provider_error"],
            }
            if row != expected_row:
                raise ValueError("native metrics/receipt mismatch")
            bound += _audit_trace(
                record, artifact["trace"], artifact["provider_error"], settings
            )
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    } != expected_files:
        raise ValueError("experiment file inventory mismatch")
    if sources() != manifest["sources"]:
        raise ValueError("sources changed during audit")
    return {
        "valid": True,
        "manifest_sha256": manifest["sha256"],
        "native_episodes_replayed": 2 * len(prepared["evaluations"]),
        "model_calls_bound_to_native": bound,
        "llm_resampled": False,
        "scope": "artifact consistency and native replay only; does not prove generation or training provenance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "run", "audit"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-split", choices=("development", "test"), default="development"
    )
    parser.add_argument(
        "--allow-other-qwen",
        action="store_true",
        help="explicitly permit a non-Qwen3.5 experiment",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-calls", type=int, default=64)
    args = parser.parse_args()
    if args.command == "preflight":
        result = feasibility(args.model, allow_other_qwen=args.allow_other_qwen)
    elif args.command == "prepare":
        data = prepare(args.corpus, evaluation_split=args.evaluation_split)
        result = {k: v for k, v in data.items() if k not in ("rows", "evaluations")}
        result.update(
            train_rows=len(data["rows"]),
            eval_tasks=len(data["evaluations"]),
            data_sha256=digest(data["rows"]),
        )
    elif args.command == "audit":
        if args.output is None:
            parser.error("audit requires --output experiment directory")
        result = audit(args.output)
    else:
        if args.model is None or args.output is None:
            parser.error("run requires --model and --output")
        result = run(
            args.corpus,
            args.model,
            args.output,
            steps=args.steps,
            max_tokens=args.max_tokens,
            output_tokens=args.output_tokens,
            max_calls=args.max_calls,
            evaluation_split=args.evaluation_split,
            allow_other_qwen=args.allow_other_qwen,
        )
    print(canonical_json(result))
    return 1 if args.command == "preflight" and result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
