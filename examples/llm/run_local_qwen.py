"""Explicit local Qwen baseline; no downloads or remote API, all raw calls retained."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from space_env.llm.benchmark import public_policy, run_agent
from space_env.llm.episode import LLMEpisode, seal, sources, verify_record
from space_env.llm.ollama import (
    PROFILES,
    InferenceStopped,
    OllamaPolicy,
    decode_calls,
    wire_context,
)
from space_env.planning.network_benchmark import geometry_key
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.workspace import MissionWorkspace


def api(opener, endpoint, route):
    with opener.open(endpoint + route, timeout=10) as response:
        return json.load(response)


def identity(manifest):
    value = json.loads(manifest.read_text())
    model = next(
        x
        for x in value["layers"]
        if x["mediaType"] == "application/vnd.ollama.image.model"
    )
    cache = manifest.parents[4]
    blob = cache / "blobs" / model["digest"].replace(":", "-")
    checksum = hashlib.sha256()
    with blob.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    if (
        model["digest"] != "sha256:" + checksum.hexdigest()
        or blob.stat().st_size != model["size"]
    ):
        raise ValueError("Local model blob checksum mismatch")
    return {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "weights_sha256": checksum.hexdigest(),
        "weights_bytes": blob.stat().st_size,
    }


def audit(directory):
    def read(name):
        value = json.loads((directory / name).read_text())
        if value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("Invalid artifact checksum")
        return value

    report = read("report.json")
    trace = read("inference.json")
    if (
        report["model_sources"] != sources()
        or report["inference_trace_sha256"] != trace["sha256"]
    ):
        raise ValueError("Source/trace identity mismatch")
    if len(report["rows"]) != 3 or {r["policy"] for r in report["rows"]} != {
        "wait",
        "public_memory",
        "qwen3.5-4b",
    }:
        raise ValueError("Incomplete baseline pairing")
    model_record = None
    for row in report["rows"]:
        record = read(row["policy"] + ".json")
        verify_record(record)
        if (
            row["episode_sha256"] != record["sha256"]
            or row["metrics"] != record["metrics"]
            or record["initial_checkpoint"]["sha256"]
            != report["initial_checkpoint_sha256"]
            or record["configuration"]["split"]
            != report.get("case", {}).get("split", "development")
        ):
            raise ValueError("Native episode/metric binding mismatch")
        if "case" in report:
            config = record["initial_checkpoint"]["scenario"]["metadata"][
                "network_workspace"
            ]["configuration"]
            if (
                record["initial_checkpoint"]["scenario"]["metadata"][
                    "network_workspace"
                ]["planning_method"]
                != report["case"].get("planning_method", "serial")
                or config != report["case"]["configuration"]
                or geometry_key(config) != report["case"]["geometry_sha256"]
                or record["configuration"]["group"] != report["case"]["group"]
            ):
                raise ValueError("Case geometry/group mismatch")
        if row["policy"] == "qwen3.5-4b":
            model_record = record
            inference = row["inference"]
    cursor = 0
    for request_index, event in enumerate(trace["trace"]):
        request = event["request"]
        expected_messages, expected_tools, expected_format = wire_context(
            model_record["messages"][: 2 + 2 * cursor],
            model_record["tools"],
            inference["context_window_pairs"],
            inference.get("protocol_profile", "baseline_v1"),
            inference.get("fixed_decision_sec") or 5.0,
        )
        if (
            request["messages"] != expected_messages
            or request["tools"] != ([] if expected_format else expected_tools)
            or request.get("format") != expected_format
        ):
            raise ValueError("Inference prompt differs from causal actor context")
        if (
            request["model"] != inference["model"]
            or request["think"] is not False
            or request["stream"] is not False
            or request["options"]
            != {
                "temperature": 0.0,
                "seed": 0,
                "num_ctx": inference["context_tokens"],
                "num_predict": inference["max_output_tokens_per_request"],
            }
        ):
            raise ValueError("Inference settings mismatch")
        response = event.get("response", {})
        if response.get("done_reason") == "length":
            continue
        try:
            decoded = (
                decode_calls(
                    response,
                    expected_tools,
                    inference.get("protocol_profile", "baseline_v1"),
                    current_revision=json.loads(expected_messages[1]["content"])[
                        "current_revision"
                    ]
                    if inference.get("protocol_profile") == "json_fixed_v1"
                    else None,
                    request_index=request_index,
                    fixed_decision_sec=inference.get("fixed_decision_sec") or 5.0,
                )
                if "response" in event
                else []
            )
        except InferenceStopped as exc:
            if event.get("decode_error") != str(exc) or event.get("delivered_calls"):
                raise ValueError("Decoder error evidence mismatch") from exc
            continue
        if "decoded_calls" in event and event["decoded_calls"] != decoded:
            raise ValueError("Decoded call evidence mismatch")
        delivered = event.get("delivered_calls", decoded)
        if delivered != decoded[: len(delivered)]:
            raise ValueError("Delivered calls are not a generated prefix")
        for call in delivered:
            if cursor >= len(model_record["events"]):
                raise ValueError("Generated/native tool sequence length mismatch")
            actual = model_record["events"][cursor]
            if actual["name"] != call["name"] or json.loads(
                actual["arguments"]
            ) != json.loads(call["arguments"]):
                raise ValueError("Native action was not generated by the model")
            cursor += 1
    if cursor != len(model_record["events"]):
        raise ValueError("Missing generated calls")
    for key, provider_key in (
        ("input_tokens", "prompt_eval_count"),
        ("output_tokens", "eval_count"),
    ):
        if inference[key] != sum(
            t.get("response", {}).get(provider_key, 0) for t in trace["trace"]
        ):
            raise ValueError("Inference token accounting mismatch")
    return {
        "valid": True,
        "native_episodes_replayed": 3,
        "model_calls_bound_to_native": cursor,
        "llm_resampled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11519")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home()
        / ".ollama/models/manifests/registry.ollama.ai/library/qwen3.5/4b",
    )
    parser.add_argument("--configuration", type=Path)
    parser.add_argument(
        "--planning-method",
        default="serial",
        choices=(
            "serial",
            "exact_small",
            "sequence_compact",
            "pointing_bounded",
            "sequence_lanes",
        ),
    )
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    parser.add_argument("--case-group", default="default-network-development")
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--protocol-profile", choices=PROFILES, default="baseline_v1")
    parser.add_argument("--wall-sec", type=float, default=600.0)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    policy = OllamaPolicy(
        endpoint=args.endpoint,
        model=args.model,
        max_requests=args.max_calls,
        max_wall_sec=args.wall_sec,
        protocol_profile=args.protocol_profile,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    model_identity = identity(args.manifest)
    version = api(policy._opener, policy.endpoint, "/api/version")
    tags = api(policy._opener, policy.endpoint, "/api/tags")
    selected = next(m for m in tags["models"] if m["name"] == args.model)
    if selected["digest"] != model_identity["manifest_sha256"]:
        raise ValueError("Server model does not match local manifest")
    configuration = (
        json.loads(args.configuration.read_text()) if args.configuration else None
    )
    mission = MissionWorkspace.network(
        configuration, planning_method=args.planning_method
    )
    if (
        args.protocol_profile == "json_fixed_v1"
        and policy.fixed_decision_sec != mission.operator.session._task.decision_sec
    ):
        raise ValueError(
            "Benchmark fixed cadence must equal the native baseline cadence"
        )
    task_text = "Deliver all released mission requests before their deadlines. Use tools to execute commands and advance physical mission time; only delivered and acknowledged products count. Coordinate all spacecraft actor roles."
    rows = []
    initial_hashes = set()
    try:
        for name in ("wait", "public_memory", "qwen3.5-4b"):
            episode = LLMEpisode(
                mission.operator.session,
                task_text=task_text,
                group=args.case_group,
                split=args.split,
                max_calls=args.max_calls,
                policy_origin="external_llm"
                if name == "qwen3.5-4b"
                else "scripted_public",
            )
            start = time.monotonic()
            error = None
            if name != "qwen3.5-4b":
                record = run_agent(
                    episode,
                    public_policy(name, mission.operator.session._task.decision_sec),
                )
            else:
                try:
                    while not episode.done:
                        call = policy(episode.messages, episode.tool_schemas())
                        atomic_write(
                            args.output / "inference.json",
                            canonical_json(seal({"trace": policy.trace})),
                        )
                        if call is None:
                            break
                        result = episode.invoke(
                            call["name"],
                            call["arguments"],
                            call_id=f"llm-{len(episode._events)}",
                        )
                        print(
                            "qwen",
                            len(episode._events),
                            call["name"],
                            result["ok"],
                            result.get("feedback"),
                            flush=True,
                        )
                        atomic_write(
                            args.output / "partial-episode.json",
                            canonical_json(episode.record()),
                        )
                except InferenceStopped as exc:
                    error = str(exc)
                finally:
                    atomic_write(
                        args.output / "inference.json",
                        canonical_json(seal({"trace": policy.trace})),
                    )
                record = episode.record()
            verify_record(record)
            atomic_write(args.output / (name + ".json"), canonical_json(record))
            initial_hashes.add(record["initial_checkpoint"]["sha256"])
            rows.append(
                {
                    "policy": name,
                    "status": "provider_stopped" if error else "completed",
                    "provider_error": error,
                    "episode_sha256": record["sha256"],
                    "metrics": record["metrics"],
                    "wall_sec": time.monotonic() - start,
                    "inference": policy.metrics() if name == "qwen3.5-4b" else None,
                }
            )
            print(name, rows[-1], flush=True)
        if len(initial_hashes) != 1:
            raise ValueError("Policies received different native initial tasks")
        latest = next(
            m
            for m in api(policy._opener, policy.endpoint, "/api/tags")["models"]
            if m["name"] == args.model
        )
        if latest["digest"] != selected["digest"]:
            raise ValueError("Model identity changed during run")
        report = seal(
            {
                "schema": "space-env.local-qwen-baseline/1",
                "case": {
                    "configuration": mission.operator.session._scenario["metadata"][
                        "network_workspace"
                    ]["configuration"],
                    "geometry_sha256": geometry_key(
                        mission.operator.session._scenario["metadata"][
                            "network_workspace"
                        ]["configuration"]
                    ),
                    "planning_method": args.planning_method,
                    "group": args.case_group,
                    "split": args.split,
                },
                "model_identity": model_identity,
                "server_version": version,
                "server_model": selected,
                "model_sources": sources(),
                "initial_checkpoint_sha256": next(iter(initial_hashes)),
                "rows": rows,
                "inference_trace_sha256": json.loads(
                    (args.output / "inference.json").read_text()
                )["sha256"],
                "scope": "one previously known synthetic development geometry, quantized local model, explicit two-pair LLM context window, scripted baseline memory differs; native cold replay verifies execution, not model resampling; no training or competitor superiority claim",
            }
        )
        atomic_write(args.output / "report.json", canonical_json(report))
    finally:
        mission.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
