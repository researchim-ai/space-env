"""Validated, same-actor processing prerequisites with stable topological order."""
from __future__ import annotations

import heapq


def dependency_children(products: dict) -> dict[str, list[str]]:
    result = {name: [] for name in products}
    for p in products.values():
        for parent in p.depends_on:
            result[parent].append(p.id)
    return result


def dependency_order(products: dict) -> list[str]:
    incoming, children = {}, {p: [] for p in products}
    for name, product in products.items():
        parents = product.depends_on
        if len(parents) != len(set(parents)) or name in parents:
            raise ValueError("duplicate or self dependency")
        incoming[name] = len(parents)
        for parent in parents:
            if parent not in products or products[parent].actor != product.actor:
                raise ValueError("dependency must reference a known product on the same actor")
            children[parent].append(name)
    ready = [p for p, count in incoming.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        parent = heapq.heappop(ready)
        result.append(parent)
        for child in children[parent]:
            incoming[child] -= 1
            if incoming[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(products):
        raise ValueError("processing dependency cycle")
    return result
