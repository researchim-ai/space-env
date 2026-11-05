"""Bounded fleet planning: priority-aware ordering and just-in-time reservations.

Uses the same forecast, paid stages and resource guards as the legacy scheduler.
No native rollout, admission oracle or guessed success is used during search.
"""

from __future__ import annotations

from pydantic import Field

from space_env.planning.chain import schedule_chains, validate_view
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class FleetSettings(StrictModel):
    max_search_nodes: int = Field(default=500000, ge=1, le=2000000, strict=True)
    max_orders: int = Field(default=12, ge=1, le=32, strict=True)
    edge_guard_sec: float = Field(default=0.01, ge=0, le=1, strict=True)


def _orders(chains, limit):
    keys = [
        lambda c: (c["deadline_sec"], -c["priority"], c["id"]),
        lambda c: (-c["priority"], c["deadline_sec"], c["id"]),
        lambda c: (sum(c["durations_sec"]), c["deadline_sec"], -c["priority"], c["id"]),
        lambda c: (c["release_sec"], c["deadline_sec"], c["id"]),
    ]
    result = []
    for key in keys:
        order = [c["id"] for c in sorted(chains, key=key)]
        if order not in result:
            result.append(order)
    base = result[0]
    candidates = [list(reversed(base))] + [
        base[i:] + base[:i] for i in range(1, len(base))
    ]
    for i in range(len(base) - 1):
        order = base.copy()
        order[i], order[i + 1] = order[i + 1], order[i]
        candidates.append(order)
    for order in candidates:
        if order not in result:
            result.append(order)
        if len(result) >= limit:
            break
    return result[:limit]


def schedule_fleet(view, settings=None):
    validate_view(view)
    cfg = FleetSettings.model_validate(settings or {})
    orders = _orders(view["chains"], cfg.max_orders)
    by_id = {c["id"]: c for c in view["chains"]}
    trials = []
    used = 0
    best = None
    baseline = None

    def score(schedule):
        return (
            sum(by_id[r["request"]]["priority"] for r in schedule["allocations"]),
            schedule["predicted_fulfilled"],
            -sum(r["finish_sec"] for r in schedule["allocations"]),
        )

    # A complete legacy proposal is retained before trying improvements.
    candidates = [(None, "earliest")] + [
        (order, placement) for order in orders for placement in ("compact", "earliest")
    ]
    for i, (order, placement) in enumerate(candidates):
        remaining = cfg.max_search_nodes - used
        if remaining <= 0:
            break
        allowance = remaining if i == 0 else max(1, remaining // (len(candidates) - i))
        try:
            schedule = schedule_chains(
                view,
                {"edge_guard_sec": cfg.edge_guard_sec, "max_search_nodes": allowance},
                order=order,
                placement=placement,
            )
        except ValueError as exc:
            if "search budget exceeded" not in str(exc):
                raise
            used += allowance
            trials.append(
                {
                    "placement": placement,
                    "budget_exhausted": True,
                    "charged_nodes": allowance,
                }
            )
            if best is None:
                raise ValueError(
                    "Лимит поиска слишком мал даже для исходного плана; увеличьте max_search_nodes."
                ) from exc
            continue
        used += schedule["search_nodes"]
        value = score(schedule)
        trials.append(
            {
                "placement": placement,
                "order": order,
                "score": list(value),
                "search_nodes": schedule["search_nodes"],
            }
        )
        if i == 0:
            baseline = {
                "predicted_fulfilled": schedule["predicted_fulfilled"],
                "score": list(value),
            }
        if best is None or value > best[0]:
            best = (value, schedule)
    if best is None:
        raise ValueError("No complete fleet proposal within search budget")
    body = {
        "schema": "space-env.fleet-schedule/1",
        "forecast_sha256": view["sha256"],
        "schedule": best[1],
        "settings": cfg.model_dump(mode="json"),
        "search_nodes": used,
        "baseline": baseline,
        "selected_score": list(best[0]),
        "trials": trials,
        "scope": "bounded serial resource packing; compact reservations and priority objective; native execution decides actual success, no global optimum claim",
    }
    return body | {"sha256": digest(body)}
