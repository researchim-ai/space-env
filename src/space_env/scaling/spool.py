"""Disk-backed bounded row and duplicate state with per-episode transactions."""

from __future__ import annotations

import json
import sqlite3

from space_env.simulation.snapshot import canonical_json


class SpoolBudget(ValueError):
    pass


class RowSpool:
    def __init__(self, path, *, max_bytes=536870912, max_row_bytes=33554432):
        if (
            type(max_bytes) is not int
            or max_bytes < 65536
            or type(max_row_bytes) is not int
            or max_row_bytes < 1
        ):
            raise ValueError("invalid spool budget")
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(f"PRAGMA max_page_count={max_bytes // 4096}")
        self.connection.execute(
            "CREATE TABLE rows (stream TEXT, position INTEGER, payload TEXT, PRIMARY KEY(stream,position))"
        )
        self.connection.execute(
            "CREATE TABLE seen (key TEXT PRIMARY KEY, payload TEXT)"
        )
        self.lengths = {}
        self.max_row_bytes = max_row_bytes
        self.saved = None

    def rows(self, name):
        self.lengths.setdefault(name, 0)
        return SpoolRows(self, name)

    def begin(self):
        self.connection.execute("SAVEPOINT episode")
        self.saved = self.lengths.copy()

    def finish(self, rollback=False):
        if rollback:
            self.connection.execute("ROLLBACK TO episode")
            self.lengths = self.saved
        self.connection.execute("RELEASE episode")
        self.saved = None

    def __contains__(self, key):
        return (
            self.connection.execute(
                "SELECT 1 FROM seen WHERE key=?", (canonical_json(key),)
            ).fetchone()
            is not None
        )

    def __getitem__(self, key):
        value = self.connection.execute(
            "SELECT payload FROM seen WHERE key=?", (canonical_json(key),)
        ).fetchone()
        if value is None:
            raise KeyError(key)
        return json.loads(value[0])

    def __setitem__(self, key, value):
        try:
            self.connection.execute(
                "INSERT INTO seen VALUES (?,?)",
                (canonical_json(key), canonical_json(value)),
            )
        except sqlite3.OperationalError as exc:
            raise SpoolBudget("spool storage budget or IO failure") from exc

    def close(self):
        self.connection.close()


class SpoolRows:
    def __init__(self, spool, name):
        self.spool, self.name = spool, name

    def __len__(self):
        return self.spool.lengths[self.name]

    def append(self, row):
        payload = canonical_json(row)
        if len(payload.encode()) + 1 > self.spool.max_row_bytes:
            raise SpoolBudget("spool row byte budget exceeded")
        try:
            self.spool.connection.execute(
                "INSERT INTO rows VALUES (?,?,?)", (self.name, len(self), payload)
            )
        except sqlite3.OperationalError as exc:
            raise SpoolBudget("spool storage budget or IO failure") from exc
        self.spool.lengths[self.name] += 1

    def __iter__(self):
        cursor = self.spool.connection.execute(
            "SELECT payload FROM rows WHERE stream=? ORDER BY position", (self.name,)
        )
        for row in cursor:
            yield json.loads(row[0])


def close_rows(data):
    next(iter(data.values())).spool.close()
