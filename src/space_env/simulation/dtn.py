"""
Следующий хоп DTN. Закрытый или выключенный луч не кандидат.
Сначала уходит продукт с большим приоритетом. Истёкший продукт не уходит.
"""
from __future__ import annotations


def next_hop(pairs: list, source: str, *, down: set[frozenset[str]] | None = None, queues: dict[str, float] | None = None) -> str | None:
    """Сосед с замкнутой линией и самой короткой очередью. При равенстве — меньшее имя."""
    blocked = down or set()
    load = queues or {}
    best: tuple[float, str] | None = None
    for pair in pairs:
        ends = {pair.leader, pair.follower}
        if source not in ends:
            continue
        other = pair.follower if pair.leader == source else pair.leader
        if frozenset(ends) in blocked or pair.earth_blocked or pair.isl_margin_db < 0.0:
            continue
        rank = (float(load.get(other, 0.0)), other)
        if best is None or rank < best:
            best = rank
    return None if best is None else best[1]


def product_to_send(products: list[dict]) -> dict | None:
    """Живой продукт с наибольшим приоритетом. expired и пустые пропускаются."""
    alive = [
        item
        for item in products
        if not item.get("delivered_to") and float(item.get("onboard_bits", 0.0)) > 1e-9
    ]
    if not alive:
        return None
    return order_for_forward(alive)[0]


def order_for_forward(products: list[dict]) -> list[dict]:
    """Порядок отдачи: приоритет по убыванию, затем идентификатор."""
    return sorted(
        products,
        key=lambda item: (-int(item.get("priority", 0)), str(item.get("product_id", ""))),
    )
