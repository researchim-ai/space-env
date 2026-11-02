"""Explicit local OS-owner recovery; never an unauthenticated HTTP capability route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from space_env.api.operator import OperatorStore


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("session_id", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        if args.session_id or args.output:
            parser.error("--list takes only database")
        store = OperatorStore(args.database)
        print(
            json.dumps(
                {
                    "sessions": [
                        {
                            "id": id,
                            "revision": op.session.revision,
                            "control": op.control,
                            "generation": op.generation,
                        }
                        for id, op in store.sessions.items()
                    ]
                }
            )
        )
        return
    if not args.session_id or not args.output:
        parser.error("session_id and private output file are required")
    output = Path(args.output)
    if output.exists():
        raise ValueError("connection output already exists; choose a new private file")
    store = OperatorStore(args.database)
    # Directory/file reservation failures occur before revoking existing capabilities.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        result = store.recover(args.session_id)
        connection = {
            "schema": "space-env.operator-connection/1",
            "session_id": result["session_id"],
            "tokens": result["tokens"],
        }
        with os.fdopen(fd, "w") as stream:
            fd = -1
            json.dump(connection, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if fd >= 0:
            os.close(fd)
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "control_epoch": result["state"]["control"]["epoch"],
            }
        )
    )


if __name__ == "__main__":
    main()
