"""Transactional local cross-bundle split/trajectory registry; never uploads data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from space_env.factory.bundle import verify_bundle


def register_bundle(bundle, registry):
    bundle, registry = Path(bundle).resolve(), Path(registry).resolve()
    if registry.is_relative_to(bundle):
        raise ValueError("registry must be outside the immutable bundle")
    verified = verify_bundle(bundle)
    if not verified["ok"]:
        raise ValueError("quarantined bundle cannot enter registry")
    report = json.loads((bundle / "report.json").read_text())
    catalogue = json.loads((bundle / "manifest.json").read_text())
    groups = {(c["group_sha256"], c["split"]) for c in catalogue["cases"]}
    trajectories = {
        (e["trajectory_sha256"], a["split"])
        for a in report["audit"]
        for e in a["exports"]
        if e["rows"] > 0
    }
    return _register_claims(
        registry,
        content=verified["content_sha256"],
        manifest=catalogue["sha256"],
        groups=groups,
        trajectories=trajectories,
    )


def _register_claims(registry, *, content, manifest, groups, trajectories):
    """Commit validated producers' claims together in the shared v1 database.

    Callers must verify their artifacts before using this private primitive.
    The transaction is local; it does not publish or move bundle files.
    """
    import re

    groups, trajectories = set(groups), set(trajectories)
    identities = [
        content,
        manifest,
        *(i for i, _ in groups),
        *(i for i, _ in trajectories),
    ]
    if any(
        not isinstance(i, str) or not re.fullmatch(r"[0-9a-f]{64}", i)
        for i in identities
    ):
        raise ValueError("registry identities must be SHA-256 strings")
    for claims in (groups, trajectories):
        seen = {}
        for identity, split in claims:
            if split not in {"train", "development", "test", "holdout", "ood"}:
                raise ValueError("unknown registry split")
            if identity in seen and seen[identity] != split:
                raise ValueError("geometry or trajectory crosses splits inside bundle")
            seen[identity] = split
    registry = Path(registry).resolve()
    registry.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(registry, timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not tables:
            conn.execute("CREATE TABLE meta (schema TEXT)")
            conn.execute("INSERT INTO meta VALUES ('space-env.corpus-registry/1')")
            conn.execute(
                "CREATE TABLE bundles (content TEXT PRIMARY KEY, manifest TEXT)"
            )
            conn.execute("CREATE TABLE groups (identity TEXT PRIMARY KEY, split TEXT)")
            conn.execute(
                "CREATE TABLE trajectories (identity TEXT PRIMARY KEY, split TEXT, content TEXT)"
            )
        elif conn.execute("SELECT schema FROM meta").fetchall() != [
            ("space-env.corpus-registry/1",)
        ]:
            raise ValueError("unknown registry schema")
        if conn.execute("SELECT 1 FROM bundles WHERE content=?", (content,)).fetchone():
            conn.rollback()
            return {
                "ok": True,
                "status": "already_registered",
                "content_sha256": content,
            }
        for identity, split in groups:
            old = conn.execute(
                "SELECT split FROM groups WHERE identity=?", (identity,)
            ).fetchone()
            if old and old[0] != split:
                raise ValueError("geometry group crosses registry splits")
        for identity, split in trajectories:
            old = conn.execute(
                "SELECT split FROM trajectories WHERE identity=?", (identity,)
            ).fetchone()
            if old:
                raise ValueError(
                    "trajectory already registered"
                    if old[0] == split
                    else "trajectory crosses registry splits"
                )
        conn.execute("INSERT INTO bundles VALUES (?,?)", (content, manifest))
        conn.executemany("INSERT OR IGNORE INTO groups VALUES (?,?)", sorted(groups))
        conn.executemany(
            "INSERT INTO trajectories VALUES (?,?,?)",
            [(i, s, content) for i, s in sorted(trajectories)],
        )
        conn.commit()
        return {
            "ok": True,
            "status": "registered",
            "content_sha256": content,
            "groups": len(groups),
            "trajectories": len(trajectories),
        }
    except ValueError:
        conn.rollback()
        raise
    except sqlite3.DatabaseError as exc:
        conn.rollback()
        raise ValueError("invalid registry or registry IO failure") from exc
    finally:
        conn.close()
