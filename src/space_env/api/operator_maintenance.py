"""Local OS-owner online backups, replay validation and exclusive restore copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from space_env.api.operator_storage import OperatorRepository


def _verify_snapshot(database):
    """Validate active native sessions and all lifecycle/journal heads; expose no secrets."""
    from space_env.api.operator import OperatorStore

    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError("operator database does not exist")
    store = OperatorStore(path)
    with store.repository.connect() as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("operator SQLite integrity check failed")
        ids = [row[0] for row in db.execute("SELECT id FROM sessions ORDER BY id")]
    sessions = []
    for id in ids:
        saved = store.repository.load(id, include_closed=True)
        sessions.append(
            {
                "session_id": id,
                "active": saved["active"],
                "generation": saved["generation"],
                "revision": len(saved["state"]["checkpoint"]["transitions"]),
                "control_epoch": saved["state"]["control"]["epoch"],
            }
        )
    return {
        "schema": "space-env.operator-store-verification/1",
        "ok": True,
        "sessions": sessions,
        "active_native_replayed": len(store.sessions),
        "closed_journals_checked": sum(not item["active"] for item in sessions),
        "scope": "compatible-build active native replay and all stored journal/snapshot integrity; not flight calibration or authenticated audit",
    }


def _file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def backup_operator_store(database, output, *, timeout_sec=30.0):
    """Copy a coherent live WAL database, validate privately, publish without overwrite.

    The deadline bounds SQLite copying. Cold native verification has its own native
    numerical budgets and is deliberately not represented as a wall-time guarantee.
    """
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(timeout_sec)
        or not 0.01 <= timeout_sec <= 600
    ):
        raise ValueError("copy timeout must be finite and within 0.01–600 seconds")
    source = Path(database).resolve()
    requested = Path(output).absolute()
    if not source.is_file():
        raise FileNotFoundError("operator database does not exist")
    # Do not follow an existing output symlink or replace any existing file/directory.
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("backup output exists; choose a new path")
    destination = requested.parent.resolve() / requested.name
    package = Path(__file__).resolve().parents[1]
    if destination.is_relative_to(package):
        raise ValueError("backup must be outside package source/assets")
    if not destination.parent.is_dir():
        raise FileNotFoundError("backup parent directory does not exist")
    repository = OperatorRepository(source)
    stage = Path(tempfile.mkdtemp(prefix=".operator-backup-", dir=destination.parent))
    staged = stage / "snapshot.sqlite3"
    try:
        fd = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        deadline = time.monotonic() + timeout_sec

        def progress(status, remaining, total):
            if total > 1048576 or time.monotonic() > deadline:
                raise TimeoutError("SQLite copy exceeded its page/time budget")

        target = sqlite3.connect(staged)
        try:
            with repository.connect() as connection:
                connection.backup(target, pages=32, progress=progress, sleep=0.01)
        finally:
            target.close()
        verification = _verify_snapshot(staged)
        checksum = _file_sha256(staged)
        size = staged.stat().st_size
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        # Same-directory staging keeps this on one filesystem. link is exclusive
        # even if another process creates the final output after our initial check.
        os.link(staged, destination, follow_symlinks=False)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return {
            "schema": "space-env.operator-backup/1",
            "ok": True,
            "output": str(destination),
            "sha256": checksum,
            "bytes": size,
            "verification": verification,
            "scope": "local SQLite online snapshot with native replay; capability hashes and session IDs retained",
        }
    finally:
        shutil.rmtree(stage)


def verify_operator_store(database, *, timeout_sec=30.0):
    """Verify a private online snapshot, never mix heads from a live writer."""
    with tempfile.TemporaryDirectory(prefix="operator-verify-") as directory:
        receipt = backup_operator_store(
            database, Path(directory) / "snapshot.sqlite3", timeout_sec=timeout_sec
        )
        return receipt["verification"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["backup", "restore"]:
        command = commands.add_parser(name)
        command.add_argument("database")
        command.add_argument("output")
        command.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="SQLite copy deadline in seconds; excludes native replay",
        )
    verify = commands.add_parser("verify")
    verify.add_argument("database")
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_operator_store(args.database)
    else:
        result = backup_operator_store(
            args.database, args.output, timeout_sec=args.timeout
        )
        result["operation"] = args.command
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
