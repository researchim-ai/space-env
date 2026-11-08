"""Replay and provider-wire checks for local LLM training bundles, without inference."""

import json
from pathlib import Path

from space_env.llm.episode import _training_row, verify_record
from space_env.llm.ollama import decode_calls, wire_context
from space_env.planning.network_benchmark import geometry_key
from space_env.simulation.snapshot import digest
from space_env.simulation.streaming import file_digest, iter_jsonl


def wire_sft_rows(record, trace, *, model, cadence=5.0):
    verify_record(record)
    if (
        record["configuration"]["split"] != "train"
        or not record["metrics"]["sft_eligible"]
    ):
        raise ValueError("Wire SFT requires a successful error-free train episode")
    rows, cursor = [], 0
    for index, event in enumerate(trace):
        request, response = event["request"], event.get("response")
        if (
            not response
            or not response.get("done")
            or response.get("done_reason") == "length"
        ):
            raise ValueError("Incomplete provider generation is not positive SFT")
        messages, tools, schema = wire_context(
            record["messages"][: 2 + 2 * cursor],
            record["tools"],
            2,
            "json_fixed_v1",
            cadence,
        )
        if (
            request["messages"] != messages
            or request["tools"] != []
            or request["format"] != schema
            or request["model"] != model
        ):
            raise ValueError("Wire context/provider binding mismatch")
        calls = decode_calls(
            response,
            tools,
            "json_fixed_v1",
            current_revision=json.loads(messages[1]["content"])["current_revision"],
            request_index=index,
            fixed_decision_sec=cadence,
        )
        if (
            len(calls) != 1
            or event.get("delivered_calls") != calls
            or cursor >= len(record["events"])
        ):
            raise ValueError("Wire/native call matrix mismatch")
        native = record["events"][cursor]
        if native["name"] != calls[0]["name"] or json.loads(
            native["arguments"]
        ) != json.loads(calls[0]["arguments"]):
            raise ValueError("Target action did not generate the native transition")
        rows.append(
            {
                "schema": "space-env.llm-wire-sft/1",
                "episode_sha256": record["sha256"],
                "group": record["configuration"]["group"],
                "split": "train",
                "index": index,
                "messages": [
                    *messages,
                    {"role": "assistant", "content": response["message"]["content"]},
                ],
                "message_loss_mask": [False, False, True],
                "response_format": schema,
                "protocol_profile": "json_fixed_v1",
                "model": model,
                "native_replay_verified": True,
                "quality": "successful_execution_not_expert",
                "target": "model_generated_semantic_json_only; no SDK revision/nonces in target",
            }
        )
        cursor += 1
    if cursor != len(record["events"]):
        raise ValueError("Incomplete wire/native episode")
    return rows


def verify_local_bundle(directory):
    directory = Path(directory)

    def read(name):
        value = json.loads((directory / name).read_text())
        if value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("Invalid artifact checksum")
        return value

    manifest, record, inference = (
        read("manifest.json"),
        read("episode.json"),
        read("inference.json"),
    )
    if manifest.get("schema") != "space-env.local-llm-data/1":
        raise ValueError("Unsupported local training bundle schema")
    exports = manifest.get("exports")
    if not isinstance(exports, dict) or not set(exports) <= {
        "rl.jsonl",
        "sft.jsonl",
        "wire-sft.jsonl",
    }:
        raise ValueError("Unsafe local training export inventory")
    verify_record(record)
    from space_env.api.network_workspace import NetworkConfiguration

    normalized = NetworkConfiguration.model_validate(
        manifest["configuration"]
    ).model_dump(mode="json", by_alias=True)
    native_config = record["initial_checkpoint"]["scenario"]["metadata"][
        "network_workspace"
    ]["configuration"]
    if (
        normalized != native_config
        or geometry_key(normalized) != record["configuration"]["group"]
        or record["configuration"]["split"] != "train"
        or manifest["metrics"] != record["metrics"]
        or manifest["episode_sha256"] != record["sha256"]
    ):
        raise ValueError("Training geometry/episode/metric mismatch")
    expected_files = (
        {"rl.jsonl", "sft.jsonl", "wire-sft.jsonl"}
        if record["metrics"]["sft_eligible"] and manifest["provider_error"] is None
        else {"rl.jsonl"}
    )
    if set(manifest["exports"]) != expected_files:
        raise ValueError("Training format matrix mismatch")
    for name, receipt in manifest["exports"].items():
        path = directory / name
        actual = list(iter_jsonl(path, max_row_bytes=64000000))
        expected = (
            wire_sft_rows(
                record, inference["trace"], model=manifest["inference"]["model"]
            )
            if name == "wire-sft.jsonl"
            else [_training_row(record, format="sft" if name == "sft.jsonl" else "rl")]
        )
        if (
            actual != expected
            or len(actual) != receipt["rows"]
            or file_digest(path) != receipt["file_sha256"]
            or path.stat().st_size != receipt["bytes"]
        ):
            raise ValueError("Training target/context checksum mismatch")
    return {
        "valid": True,
        "native_goal_verified": record["metrics"]["success"],
        "wire_sft_rows": manifest["exports"].get("wire-sft.jsonl", {}).get("rows", 0),
        "llm_resampled": False,
    }
