"""Verified episode export; control-plane scenarios/checkpoints never enter prompts."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path

from space_env.eval.statistics import assign_split
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl
from space_env.tasks.session import TaskSession


def iter_dataset(record, *, format="rl", role="@central", include_rejected=False):
    """Verify the entire episode first, then pull owned public rows one at a time."""
    _options(format, role, include_rejected)
    return VerifiedEpisode(record).iter_rows(
        format=format, role=role, include_rejected=include_rejected
    )


def _options(format, role, include_rejected):
    if type(include_rejected) is not bool:
        raise TypeError("include_rejected must be boolean")
    if format not in {"rl", "sft"}:
        raise ValueError("format must be rl or sft")
    if format == "sft" and role != "@central":
        raise ValueError(
            "SFT export currently requires the centralized joint controller"
        )


def _rows(session, episode_sha, format, role, include_rejected):
    lineage = session._original["lineage"]
    split = assign_split(lineage, "")
    action_roles = {a.id: a.role for a in session._task.actions}
    for row in session._transitions:
        r = row["transition"]
        request = row["request"]
        if format == "sft" and not include_rejected and not r["feedback"]["accepted"]:
            continue
        observation = r["role_observations"][role]
        next_observation = r["next_role_observations"][role]
        choices = (
            request["choices"]
            if role == "@central"
            else {role: request["choices"].get(role, ["wait"])}
        )
        if role != "@central" and "@central" in request["choices"]:
            choices = {
                role: [
                    a
                    for a in request["choices"]["@central"]
                    if action_roles.get(a) == role
                ]
                or ["wait"]
            }
        common = {
            "schema": "space-env.task-dataset/1",
            "episode_sha256": episode_sha,
            "lineage": lineage,
            "split": split,
            "view_role": role,
            "index": r["index"],
            "elapsed_sec": r["elapsed_sec"],
            "discount": r["discount"],
            "reward": r["reward"],
            "terminated": r["terminated"],
            "truncated": r["truncated"],
            "replay_verified": True,
            "policy_origin": session._task.policy_origin,
            "quality": "verified_execution_not_expert_qualification",
        }
        feedback = r["feedback"]
        if role != "@central" and feedback["role"] not in {None, role}:
            feedback = {"accepted": False, "code": "joint_rejected"}
        if format == "rl":
            common.update(
                observation=observation,
                action={"choices": choices, "advance_sec": request["advance_sec"]},
                next_observation=next_observation,
                feedback=feedback,
            )
        else:
            common["messages"] = [
                {"role": "user", "content": canonical_json(observation)},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "decision",
                            "type": "function",
                            "function": {
                                "name": "space_env_task_joint_step",
                                "arguments": canonical_json(
                                    {
                                        "choices": choices,
                                        "expected_revision": request[
                                            "expected_revision"
                                        ],
                                        "idempotency_key": request["idempotency_key"],
                                        "advance_sec": request["advance_sec"],
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "decision",
                    "content": canonical_json(
                        {"observation": next_observation, "feedback": r["feedback"]}
                    ),
                },
            ]
        yield copy.deepcopy(common)


def export_dataset(
    record, output, *, format="rl", role="@central", include_rejected=False
):
    _options(format, role, include_rejected)
    return VerifiedEpisode(record).export(
        output, format=format, role=role, include_rejected=include_rejected
    )


class VerifiedEpisode:
    """Owned replay-verified episode; repeated public exports never trust live state.

    Private storage is not an application mutation API. Each yielded row is owned.
    """

    def __init__(self, record):
        record = copy.deepcopy(record)
        self._session = TaskSession.from_checkpoint(record)
        self._sha = record["sha256"]
        self._proof = {"kind": "task_cold_replay", "episode_sha256": self._sha}

    @classmethod
    def from_planning_checkpoint(cls, record):
        from space_env.factory.quality import quality_report
        from space_env.planning.runner import PlanningRun

        record = copy.deepcopy(record)
        run = PlanningRun.from_checkpoint(record)
        result = object.__new__(cls)
        result._planning_run = run  # trusted internal trainer/factory consumers only
        result._session = run.session
        result._sha = record["episode"]["sha256"]
        result._proof = {
            "kind": "planning_and_native_cold_replay",
            "episode_sha256": result._sha,
            "trace_sha256": digest(run.trace),
            "quality": quality_report(run),
        }
        return result

    @property
    def verification(self):
        return copy.deepcopy(self._proof)

    def iter_rows(self, *, format="rl", role="@central", include_rejected=False):
        _options(format, role, include_rejected)
        if (
            format == "sft"
            and self._session._task.goal is not None
            and (
                not self._session.goal_status()["success"]
                or any(
                    not r["transition"]["feedback"]["accepted"]
                    for r in self._session._transitions
                )
            )
        ):
            raise ValueError(
                "SFT requires successful native goal and error-free execution"
            )
        if role not in self._session._views:
            raise ValueError("unknown export role")
        return _rows(self._session, self._sha, format, role, include_rejected)

    def export(self, output, *, format="rl", role="@central", include_rejected=False):
        receipt = atomic_jsonl(
            output,
            self.iter_rows(format=format, role=role, include_rejected=include_rejected),
        )
        return {
            "schema": "space-env.task-export/1",
            "rows": receipt["rows"],
            "format": format,
            "split": assign_split(self._session._original["lineage"], ""),
            "episode_sha256": self._sha,
            "rows_sha256": receipt["rows_sha256"],
        }

    def export_bundle(
        self,
        output,
        *,
        formats=("rl", "sft"),
        roles=("@central",),
        max_bytes=4294967296,
    ):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("invalid export byte budget")
        if (
            not 1 <= len(formats) <= 2
            or len(set(formats)) != len(formats)
            or not 1 <= len(roles) <= 65
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("invalid/duplicate export formats or roles")
        for role in roles:
            for format in formats:
                self.iter_rows(format=format, role=role)
        output = Path(output)
        if output.exists():
            raise ValueError("export bundle output must not exist")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent)
        )
        try:
            files, total = {}, 0
            for i, role in enumerate(roles):
                for format in formats:
                    path = f"role-{i:03}/{format}.jsonl"
                    target = stage / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    meta = atomic_jsonl(
                        target,
                        self.iter_rows(format=format, role=role),
                        max_bytes=max_bytes,
                    )
                    total += meta["bytes"]
                    if total > max_bytes:
                        raise ValueError("export bundle byte budget exceeded")
                    files[path] = {**meta, "role": role, "format": format}
            body = {
                "schema": "space-env.task-export-bundle/1",
                "episode_sha256": self._sha,
                "verification": self.verification,
                "files": files,
            }
            receipt = {**body, "sha256": digest(body)}
            if total + len(canonical_json(receipt).encode()) > max_bytes:
                raise ValueError("export bundle byte budget exceeded including receipt")
            atomic_write(stage / "bundle.json", canonical_json(receipt))
            if output.exists():
                raise ValueError("export bundle output appeared during build")
            os.rename(stage, output)
            return receipt
        finally:
            if stage.exists():
                shutil.rmtree(stage)
