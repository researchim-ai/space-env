"""Bounded pull-based JSONL IO with atomic publication and compatible array hashes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from space_env.simulation.snapshot import canonical_json


def array_digest(rows):
    checksum = hashlib.sha256(b"[")
    for i, row in enumerate(rows):
        if i:
            checksum.update(b",")
        checksum.update(canonical_json(row).encode())
    checksum.update(b"]")
    return checksum.hexdigest()


def iter_jsonl(path, *, max_row_bytes=33554432):
    if type(max_row_bytes) is not int or max_row_bytes < 1:
        raise ValueError("invalid row byte limit")
    with Path(path).open("rb") as stream:
        while line := stream.readline(max_row_bytes + 1):
            if len(line) > max_row_bytes:
                raise ValueError("JSONL row byte budget exceeded")
            value = json.loads(line)
            canonical_json(value)  # json.loads alone accepts NaN/Infinity.
            yield value


def atomic_jsonl(
    path, rows, *, max_row_bytes=33554432, max_rows=1000000, max_bytes=4294967296
):
    if any(type(v) is not int or v < 1 for v in [max_row_bytes, max_rows, max_bytes]):
        raise ValueError("invalid JSONL budget")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    checksum, file_hash = hashlib.sha256(b"["), hashlib.sha256()
    count, size = 0, 0
    try:
        with os.fdopen(fd, "wb") as stream:
            for row in rows:
                payload = canonical_json(row).encode()
                line = payload + b"\n"
                if (
                    len(line) > max_row_bytes
                    or count + 1 > max_rows
                    or size + len(line) > max_bytes
                ):
                    raise ValueError("JSONL budget exceeded")
                if count:
                    checksum.update(b",")
                checksum.update(payload)
                file_hash.update(line)
                stream.write(line)
                count += 1
                size += len(line)
            checksum.update(b"]")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return {
            "rows": count,
            "bytes": size,
            "rows_sha256": checksum.hexdigest(),
            "file_sha256": file_hash.hexdigest(),
        }
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1048576):
            checksum.update(chunk)
    return checksum.hexdigest()
