"""Bounded assignment search over explicit eligible workspace assets.

Only null assignments may change. Candidates use authorized forecasts, never
native execution outcomes. Native simulation remains the success authority.
"""

from __future__ import annotations

import copy
from itertools import product

from space_env.planning.chain import collect_chain_forecast
from space_env.planning.fleet_scheduler import schedule_fleet
from space_env.runtime import AccessPolicy, MissionRuntime


def assign_configuration(configuration):
    # Local import keeps the public workspace module free of an import cycle.
    from space_env.api.workspace import compile_configuration

    source = configuration.model_dump(mode="json", by_alias=True)
    actors = [
        f"{p.id}-{i:02d}"
        for p in configuration.planes
        for i in range(len(p.phases_deg))
    ]
    stations = [s.name for s in configuration.stations]
    free = []
    initial = copy.deepcopy(source)
    for i, request in enumerate(source["requests"]):
        for key, choices in (("actor_id", actors), ("station", stations)):
            if request[key] is None:
                free.append((i, key, choices))
                initial["requests"][i][key] = choices[i % len(choices)]

    limit = configuration.max_assignment_trials if free else 1
    remaining = configuration.planner.max_search_nodes
    best = None
    seen = set()
    trials = []
    pending = [initial]
    # Coordinate improvements are generated from each best complete proposal.
    # Never mutate caller configuration or change an explicit assignment.
    while pending and len(trials) < limit and remaining > 0:
        candidate = pending.pop(0)
        signature = tuple((r["actor_id"], r["station"]) for r in candidate["requests"])
        if signature in seen:
            continue
        seen.add(signature)
        allowance = max(1, remaining // (limit - len(trials)))
        settings = configuration.planner.model_dump(mode="json")
        settings["max_search_nodes"] = allowance
        try:
            cfg, scenario, task = compile_configuration(candidate)
            runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
            view = collect_chain_forecast(runtime, scenario, task)
            result = schedule_fleet(view, settings)
        except ValueError as exc:
            # Failed scheduler calls can consume the allowance; conservatively
            # charge it even if compilation failed before search.
            remaining -= allowance
            trials.append(
                {
                    "assignments": [list(x) for x in signature],
                    "error": str(exc),
                    "charged_nodes": allowance,
                }
            )
        else:
            charged = result["search_nodes"]
            remaining -= charged
            score = tuple(result["selected_score"])
            trials.append(
                {
                    "assignments": [list(x) for x in signature],
                    "score": list(score),
                    "charged_nodes": charged,
                }
            )
            if best is None or score > best[0]:
                best = (score, cfg, scenario, task, runtime, view, result)
                # Put improvements of this solution ahead of older proposals.
                pending = []
        base = best[1].model_dump(mode="json", by_alias=True) if best else candidate
        if not pending:
            for index in sorted({i for i, _, _ in free}):
                if len(pending) >= limit - len(trials):
                    break
                choices = {key: options for i, key, options in free if i == index}
                for actor, station in product(
                    choices.get("actor_id", [base["requests"][index]["actor_id"]]),
                    choices.get("station", [base["requests"][index]["station"]]),
                ):
                    if len(pending) >= limit - len(trials):
                        break
                    trial = copy.deepcopy(base)
                    trial["requests"][index].update(actor_id=actor, station=station)
                    trial_signature = tuple(
                        (r["actor_id"], r["station"]) for r in trial["requests"]
                    )
                    if trial_signature not in seen:
                        pending.append(trial)

    if best is None:
        raise ValueError(
            "Не найден полный допустимый план назначения в заданном бюджете. "
            + (trials[-1]["error"] if trials else "Увеличьте бюджет поиска.")
        )
    audit = {
        "schema": "space-env.workspace-assignment/1",
        "automatic": bool(free),
        "max_trials": limit,
        "trials": trials,
        "search_nodes": configuration.planner.max_search_nodes - remaining,
        "max_search_nodes": configuration.planner.max_search_nodes,
        "scope": "bounded coordinate search; fixed assignments preserved; forecast only; no global optimum guarantee",
    }
    return (*best[1:], audit)
