"""Local SQLite operator snapshots, generation CAS and hash-chained commit journal."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from space_env.simulation.snapshot import canonical_json, digest


class StorageConflict(ValueError):
    pass


class OperatorRepository:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError(
                "operator database must be outside the package source/assets"
            )
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA max_page_count=1048576")
            db.execute("BEGIN IMMEDIATE")
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not tables:
                db.execute("CREATE TABLE meta (schema TEXT)")
                db.execute("INSERT INTO meta VALUES ('space-env.operator-store/1')")
                db.execute(
                    "CREATE TABLE sessions (id TEXT PRIMARY KEY, generation INTEGER, active INTEGER, payload TEXT, sha TEXT)"
                )
                db.execute(
                    "CREATE TABLE commits (id TEXT, generation INTEGER, payload TEXT, sha TEXT, PRIMARY KEY(id,generation))"
                )
            elif db.execute("SELECT schema FROM meta").fetchall() != [
                ("space-env.operator-store/1",)
            ]:
                raise ValueError("unsupported operator storage schema")
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=3)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=3000")
        db.execute("PRAGMA max_page_count=1048576")
        try:
            with db:
                yield db
        finally:
            db.close()

    def load(self, id, *, include_closed=False):
        with self.connect() as db:
            row = db.execute(
                "SELECT generation,active,payload,sha FROM sessions WHERE id=?", (id,)
            ).fetchone()
            if row is None or not row[1] and not include_closed:
                return None
            generation, active, payload, sha = row
            value = json.loads(payload)
            if digest(value) != sha:
                raise ValueError("operator snapshot digest mismatch")
            previous = "0" * 64
            commits = db.execute(
                "SELECT generation,payload,sha FROM commits WHERE id=? ORDER BY generation",
                (id,),
            ).fetchall()
            for seq, (number, body, checksum) in enumerate(commits, 1):
                entry = json.loads(body)
                if (
                    number != seq
                    or digest(entry) != checksum
                    or entry["previous"] != previous
                    or entry["generation"] != seq
                ):
                    raise ValueError("operator journal chain mismatch")
                previous = checksum
            if (
                len(commits) != generation
                or not commits
                or json.loads(commits[-1][1])["state_sha256"] != sha
            ):
                raise ValueError("operator snapshot/journal head mismatch")
            if bool(active) == bool(value["closed"]):
                raise ValueError("operator lifecycle state mismatch")
            return {
                "generation": generation,
                "active": bool(active),
                "state": value,
                "sha256": sha,
            }

    def active(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM sessions ORDER BY id")]
        rows = [(id, self.load(id, include_closed=True)) for id in ids]
        return [(id, value) for id, value in rows if value["active"]]

    def generation(self, id):
        with self.connect() as db:
            row = db.execute(
                "SELECT generation,active FROM sessions WHERE id=?", (id,)
            ).fetchone()
            return None if row is None or not row[1] else row[0]

    def save(self, id, state, expected, kind, *, active=True):
        payload = canonical_json(state)
        if len(payload.encode()) > 32 * 1024 * 1024:
            raise ValueError("operator snapshot exceeds 32 MiB")
        checksum = digest(state)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT generation FROM sessions WHERE id=?", (id,)
            ).fetchone()
            if (old[0] if old else 0) != expected:
                raise StorageConflict(
                    "durable generation changed; refresh before retry"
                )
            if (
                expected == 0
                and active
                and db.execute(
                    "SELECT count(*) FROM sessions WHERE active=1"
                ).fetchone()[0]
                >= 8
            ):
                raise StorageConflict("durable active-session budget reached")
            prior = db.execute(
                "SELECT sha FROM commits WHERE id=? AND generation=?", (id, expected)
            ).fetchone()
            entry = {
                "schema": "space-env.operator-commit/1",
                "generation": expected + 1,
                "kind": kind,
                "revision": len(state["checkpoint"]["transitions"]),
                "control_epoch": state["control"]["epoch"],
                "state_sha256": checksum,
                "previous": prior[0] if prior else "0" * 64,
            }
            db.execute(
                "INSERT INTO commits VALUES (?,?,?,?)",
                (id, expected + 1, canonical_json(entry), digest(entry)),
            )
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET generation=excluded.generation,active=excluded.active,payload=excluded.payload,sha=excluded.sha",
                (id, expected + 1, int(active), payload, checksum),
            )
        return expected + 1
