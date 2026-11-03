"""Bounded process-local artifact spooling; native imports remain guarded."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import secrets
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel

CHUNK_BYTES = 1024 * 1024
FILE_BYTES = 32 * 1024 * 1024


def failure(code, message, status=400):
    from space_env.api.operator import OperatorError

    return OperatorError(code, message, status)


class TransferSpec(StrictModel):
    direction: Literal["upload", "download"]
    format: Literal["archive", "checkpoint", "training"]
    bytes: int | None = Field(default=None, ge=1, le=FILE_BYTES, strict=True)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    takeover: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def contract(self):
        if self.direction == "upload":
            if self.format == "training" or self.bytes is None or self.sha256 is None:
                raise ValueError("upload requires archive/checkpoint bytes and SHA-256")
        elif self.bytes is not None or self.sha256 is not None or self.takeover:
            raise ValueError("download does not take upload metadata")
        if self.takeover and self.format != "checkpoint":
            raise ValueError("takeover applies only to checkpoint import")
        return self


@dataclass
class ArtifactChunk:
    offset: int
    length: int
    stream: object


class BoundedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise ValueError("artifact byte budget exceeded")
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class FileBody:
    def __init__(self, stream, job, release):
        self.stream, self.size, self.sha256 = stream, job["received"], job["sha256"]
        self.content_type = (
            "application/zip"
            if job["spec"].format == "training"
            else "application/json; charset=utf-8"
        )
        self.filename = {
            "training": "mission-training.zip",
            "checkpoint": "mission-checkpoint.json",
            "archive": "mission-archive.json",
        }[job["spec"].format]
        self.release = release
        self.closed = False

    def __len__(self):
        return self.size

    def iter_chunks(self, size=65536):
        if type(size) is not int or not 1 <= size <= CHUNK_BYTES:
            raise ValueError("invalid download read size")
        while data := self.stream.read(size):
            yield data

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stream.close()
            finally:
                self.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ArtifactStore:
    def __init__(
        self, *, parent=None, quota_bytes=128 * 1024 * 1024, max_jobs=8, ttl_sec=600
    ):
        if (
            type(quota_bytes) is not int
            or not 1 <= quota_bytes <= 128 * 1024 * 1024
            or type(max_jobs) is not int
            or not 1 <= max_jobs <= 8
            or isinstance(ttl_sec, bool)
            or not isinstance(ttl_sec, (int, float))
            or not math.isfinite(ttl_sec)
            or not 0.01 <= ttl_sec <= 86400
        ):
            raise ValueError("invalid artifact-store budget")
        self.parent = parent
        if parent is not None and Path(parent).resolve().is_relative_to(
            Path(__file__).resolve().parents[1]
        ):
            raise ValueError("artifact spool must be outside package source/assets")
        self.quota, self.max_jobs, self.ttl = quota_bytes, max_jobs, ttl_sec
        self.lock = threading.RLock()
        self.jobs, self.reserved = {}, 0
        self.temp = None

    def _root(self):
        if self.temp is None:
            self.temp = tempfile.TemporaryDirectory(
                prefix="operator-artifacts-", dir=self.parent
            )
        return Path(self.temp.name)

    def _drop(self, id):
        job = self.jobs[id]
        shutil.rmtree(job["path"].parent, ignore_errors=False)
        self.reserved -= job["reservation"]
        del self.jobs[id]

    def cleanup(self):
        with self.lock:
            expired = [
                id
                for id, job in self.jobs.items()
                if not job["busy"] and time.monotonic() - job["touched"] >= self.ttl
            ]
            for id in expired:
                self._drop(id)
            return len(expired)

    def close(self):
        with self.lock:
            if any(job["busy"] for job in self.jobs.values()):
                raise failure(
                    "transfer_busy", "an artifact operation is in progress", 409
                )
            for id in list(self.jobs):
                self._drop(id)
            if self.temp is not None:
                self.temp.cleanup()
                self.temp = None

    @staticmethod
    @contextmanager
    def _owner(store, session_id, token):
        # Keep the existing store→operator→native lock order. Revocation is checked
        # at operation boundaries, not retroactively halfway through a response.
        with store.lock:
            session = store.get(session_id)
            with session.lock, session.session._lock:
                session._guard()
                if not session.permission(token)[2]:
                    raise failure(
                        "forbidden", "artifact operations require admin capability", 403
                    )
                yield session

    def _find(self, id, token):
        self.cleanup()
        with self.lock:
            job = self.jobs.get(id)
            if job is None:
                raise failure("not_found", "no such artifact transfer", 404)
            if (
                not isinstance(token, str)
                or hashlib.sha256(token.encode()).hexdigest() != job["owner"]
            ):
                raise failure("unauthorized", "transfer capability mismatch", 401)
            return job

    @contextmanager
    def _lease(self, id, token):
        with self.lock:
            job = self._find(id, token)
            if job["busy"]:
                raise failure(
                    "transfer_busy", "artifact already in use; retry later", 409
                )
            job["busy"] = True
        try:
            yield job
        finally:
            with self.lock:
                if id in self.jobs:
                    job["busy"], job["touched"] = False, time.monotonic()

    @staticmethod
    def _status(id, job):
        return {
            "schema": "space-env.operator-transfer/1",
            "transfer_id": id,
            "session_id": job["session_id"],
            "direction": job["spec"].direction,
            "format": job["spec"].format,
            "state": job["state"],
            "bytes": job["expected"],
            "received_bytes": job["received"],
            "sha256": job["sha256"],
            "busy": job["busy"],
            "chunk_bytes": CHUNK_BYTES,
            "result_session_id": job.get("installed_id")
            or (job["receipt"]["session_id"] if job["receipt"] else None),
        }

    def create(self, store, session_id, token, payload):
        spec = TransferSpec.model_validate(payload)
        with self._owner(store, session_id, token) as session:
            with self.lock:
                self.cleanup()
                reservation = (
                    spec.bytes
                    if spec.direction == "upload"
                    else FILE_BYTES * (2 if spec.format == "training" else 1)
                )
                if (
                    len(self.jobs) >= self.max_jobs
                    or self.reserved + reservation > self.quota
                ):
                    raise failure(
                        "artifact_capacity", "artifact spool quota exhausted", 429
                    )
                id = secrets.token_urlsafe(18)
                directory = self._root() / id
                directory.mkdir(mode=0o700)
                path = directory / "payload"
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)
                except Exception:
                    shutil.rmtree(directory)
                    raise
                job = {
                    "session_id": session_id,
                    "owner": hashlib.sha256(token.encode()).hexdigest(),
                    "spec": spec,
                    "path": path,
                    "reservation": reservation,
                    "expected": spec.bytes,
                    "received": 0,
                    "sha256": spec.sha256,
                    "state": "uploading",
                    "busy": True,
                    "touched": time.monotonic(),
                    "receipt": None,
                }
                self.jobs[id] = job
                self.reserved += reservation
            try:
                if spec.direction == "download":
                    if spec.format == "training":
                        path.unlink()
                        session.export_to(token, path, max_bytes=FILE_BYTES)
                    else:
                        value = (
                            session.archive(token)
                            if spec.format == "archive"
                            else session.checkpoint(token)
                        )
                        encoder = json.JSONEncoder(
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        with path.open("wb") as output:
                            bounded = BoundedWriter(output, FILE_BYTES)
                            for chunk in encoder.iterencode(value):
                                bounded.write(chunk.encode())
                            output.flush()
                            os.fsync(output.fileno())
                    job["received"] = job["expected"] = path.stat().st_size
                    job["sha256"] = self._sha(path)
                    job["state"] = "ready"
                    with self.lock:
                        self.reserved += job["received"] - job["reservation"]
                        job["reservation"] = job["received"]
            except Exception:
                with self.lock:
                    self._drop(id)
                raise
            finally:
                with self.lock:
                    job["busy"], job["touched"] = False, time.monotonic()
            return self._status(id, job)

    @staticmethod
    def _sha(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                value.update(chunk)
        return value.hexdigest()

    def status(self, store, id, token):
        job = self._find(id, token)
        with self._owner(store, job["session_id"], token), self.lock:
            return self._status(id, self._find(id, token))

    def cancel(self, store, id, token):
        job = self._find(id, token)
        with self._owner(store, job["session_id"], token), self.lock:
            job = self._find(id, token)
            if job["busy"]:
                raise failure(
                    "transfer_busy", "retry cancellation at an operation boundary", 409
                )
            self._drop(id)
            return {"ok": True}

    def append(self, store, id, token, chunk):
        if (
            not isinstance(chunk, ArtifactChunk)
            or type(chunk.offset) is not int
            or chunk.offset < 0
            or type(chunk.length) is not int
            or not 1 <= chunk.length <= CHUNK_BYTES
        ):
            raise ValueError("invalid artifact chunk offset/length")
        job = self._find(id, token)
        with self._owner(store, job["session_id"], token):
            pass  # Authorize once; socket IO must not hold the global mission lock.
        with self._lease(id, token) as job:
            if job["spec"].direction != "upload" or job["state"] not in {
                "uploading",
                "ready",
            }:
                raise failure("transfer_state", "transfer does not accept chunks", 409)
            prior = job["received"]
            replay = chunk.offset < prior
            if chunk.offset > prior or chunk.offset + chunk.length > (
                prior if replay else job["expected"]
            ):
                raise failure(
                    "chunk_offset",
                    "chunk must append or exactly repeat accepted bytes",
                    409,
                )
            try:
                with job["path"].open("r+b") as output:
                    output.seek(chunk.offset)
                    remaining = chunk.length
                    while remaining:
                        data = chunk.stream.read(min(65536, remaining))
                        if (
                            not isinstance(data, bytes)
                            or not data
                            or len(data) > remaining
                        ):
                            raise failure(
                                "short_chunk", "incomplete artifact chunk", 400
                            )
                        if replay:
                            if output.read(len(data)) != data:
                                raise failure(
                                    "chunk_conflict",
                                    "retry bytes differ from accepted chunk",
                                    409,
                                )
                        else:
                            output.write(data)
                        remaining -= len(data)
                    output.flush()
                    os.fsync(output.fileno())
                if not replay:
                    job["received"] = prior + chunk.length
                if job["received"] == job["expected"]:
                    if self._sha(job["path"]) != job["sha256"]:
                        with self.lock:
                            self._drop(id)
                        raise failure(
                            "artifact_digest",
                            "uploaded bytes do not match SHA-256",
                            422,
                        )
                    job["state"] = "ready"
            except Exception:
                if id in self.jobs and not replay:
                    try:
                        with job["path"].open("r+b") as output:
                            output.truncate(prior)
                        job["received"] = prior
                    except OSError:
                        with self.lock:
                            self._drop(id)
                raise
            return {**self._status(id, job), "busy": False}

    def commit(self, store, id, token):
        job = self._find(id, token)
        with (
            self._owner(store, job["session_id"], token),
            self._lease(id, token) as job,
        ):
            if job["state"] == "consumed":
                return copy.deepcopy(job["receipt"])
            if job["spec"].direction != "upload" or job["state"] != "ready":
                raise failure("transfer_state", "a complete upload is required", 409)
            if self._sha(job["path"]) != job["sha256"]:
                raise failure("artifact_digest", "spooled bytes changed", 422)
            with job["path"].open(encoding="utf-8") as stream:
                value = json.load(stream)
            payload = (
                {"archive": value}
                if job["spec"].format == "archive"
                else {"record": value, "takeover": job["spec"].takeover}
            )
            # Never blindly retry an import whose commit outcome was not acknowledged.
            # The native store can have committed before an IO/readback failure.
            before = set(store.sessions)
            job["state"] = "committing"
            try:
                result = store.restore(payload)
            except BaseException:
                job["state"] = "failed"
                added = set(store.sessions) - before
                if len(added) == 1:
                    job["installed_id"] = next(iter(added))
                raise
            # Publish the receipt before cleanup: a retry must not create a duplicate session.
            job["receipt"], job["state"] = copy.deepcopy(result), "consumed"
            try:
                job["path"].unlink(missing_ok=True)
            except OSError:
                pass  # Keep the reservation until later cancellation/expiry cleans it.
            else:
                with self.lock:
                    self.reserved -= job["reservation"]
                    job["reservation"] = 0
            return result

    def download(self, store, id, token):
        job = self._find(id, token)
        with self._owner(store, job["session_id"], token):
            lease = self._lease(id, token)
            job = lease.__enter__()
            try:
                if job["spec"].direction != "download" or job["state"] != "ready":
                    raise failure("transfer_state", "download is not ready", 409)
                stream = job["path"].open("rb")
                if self._sha(job["path"]) != job["sha256"]:
                    stream.close()
                    raise failure("artifact_digest", "spooled bytes changed", 422)
                return FileBody(stream, job, lambda: lease.__exit__(None, None, None))
            except Exception:
                lease.__exit__(None, None, None)
                raise


ARTIFACTS = ArtifactStore()
