"""Structural progress over the already-public product-specific action graph.

Distances count relay hops to a declared ground sender, not predicted delivery.
Strict decrease rejects cycles even when the catalogue contains reverse links.
"""

from collections import defaultdict, deque


def route_progress(catalogue):
    edges = defaultdict(list)
    sinks = defaultdict(set)
    for row in catalogue:
        for command in row.get("commands", []):
            product = command.get("product_id")
            if not product:
                continue
            if command["kind"] == "relay":
                edges[product].append(
                    (row["id"], command["actor_id"], command["destination_id"])
                )
            elif command["kind"] == "downlink":
                sinks[product].add(command["actor_id"])
    result = {}
    for product, routes in edges.items():
        predecessors = defaultdict(set)
        for _, source, destination in routes:
            predecessors[destination].add(source)
        distances = {actor: 0 for actor in sorted(sinks[product])}
        queue = deque(distances)
        while queue:
            destination = queue.popleft()
            for source in sorted(predecessors[destination]):
                if source not in distances:
                    distances[source] = distances[destination] + 1
                    queue.append(source)
        for action, source, destination in routes:
            left, right = distances.get(source), distances.get(destination)
            result[action] = {
                "source_hops_to_ground_sender": left,
                "destination_hops_to_ground_sender": right,
                "makes_progress": left is not None
                and right is not None
                and right < left,
            }
    return result
