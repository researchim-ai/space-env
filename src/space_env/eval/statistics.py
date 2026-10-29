"""
Семя и разбиение. Сбой остаётся в списке и в знаменателе.
Варианты одного задания попадают в один и тот же кусок.
"""
from __future__ import annotations

import time
import zlib


def assign_split(task_id: str, variant: str, *, holdout_every: int = 5) -> str:
    """Кусок зависит только от задания. variant в хэш не входит."""
    del variant
    if holdout_every < 2:
        raise ValueError("holdout_every должен быть не меньше 2")
    bucket = zlib.adler32(task_id.encode()) % int(holdout_every)
    return "holdout" if bucket == 0 else "train"


def summarize_seeds(rows: list[dict]) -> dict:
    """Доля успеха считается по всем строкам, включая исключения."""
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success") and not row.get("error"))
    return {
        "seeds": [int(row["seed"]) for row in rows],
        "rows": list(rows),
        "successes": successes,
        "failures": total - successes,
        "success_rate": (successes / total) if total else 0.0,
        "wall_sec": sum(float(row.get("wall_sec", 0.0)) for row in rows),
    }


def run_seeded(function, seeds: list[int]) -> dict:
    """function(seed) → bool. Исключение записывается и не выбрасывается наружу."""
    rows = []
    for seed in seeds:
        started = time.perf_counter()
        try:
            value = function(int(seed))
            rows.append(
                {
                    "seed": int(seed),
                    "success": bool(value),
                    "error": "",
                    "wall_sec": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "seed": int(seed),
                    "success": False,
                    "error": type(exc).__name__,
                    "wall_sec": time.perf_counter() - started,
                }
            )
    return summarize_seeds(rows)
