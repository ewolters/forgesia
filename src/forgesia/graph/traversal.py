"""Graph traversal — BFS causal chains, DFS cycles, Dijkstra distance, counterfactuals."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from .model import CausalGraph, EdgeType


@dataclass
class CausalPath:
    """A directed path through the causal graph."""

    nodes: list[str]  # node ids in order
    edges: list[str]  # edge ids in order
    total_weight: float = 1.0
    path_confidence: float = 1.0  # product of edge confidences


@dataclass
class CycleInfo:
    """Detected cycle in the graph."""

    nodes: list[str]
    is_contradiction: bool = False  # cycle contains contradicting edges


@dataclass
class CounterfactualResult:
    """Impact of removing a node from the graph."""

    removed_node: str
    affected_nodes: list[str] = field(default_factory=list)
    broken_chains: int = 0
    estimated_impact: float = 0.0  # 0-1


def find_causal_chains(
    graph: CausalGraph,
    source: str,
    target: str,
    max_depth: int = 50,
    edge_types: set[EdgeType] | None = None,
) -> list[CausalPath]:
    """Find all causal paths from source to target via BFS.

    Args:
        graph: The causal graph.
        source: Starting node id.
        target: Destination node id.
        max_depth: Maximum path length.
        edge_types: Only follow these edge types (default: all).

    Returns:
        List of CausalPath objects, sorted by total_weight descending.
    """
    if source not in graph.nodes or target not in graph.nodes:
        return []

    paths = []
    # BFS queue: (current_node, path_nodes, path_edges, weight, confidence)
    queue = deque([(source, [source], [], 1.0, 1.0)])
    visited_states: set[tuple[str, ...]] = set()

    while queue:
        current, path_nodes, path_edges, weight, conf = queue.popleft()

        if len(path_nodes) > max_depth:
            continue

        if current == target and len(path_nodes) > 1:
            paths.append(CausalPath(
                nodes=path_nodes,
                edges=path_edges,
                total_weight=weight,
                path_confidence=conf,
            ))
            continue

        for edge, child in graph.get_children(current):
            if edge_types and edge.edge_type not in edge_types:
                continue
            if child.id in path_nodes:
                continue  # avoid cycles in path

            state = tuple(path_nodes + [child.id])
            if state in visited_states:
                continue
            visited_states.add(state)

            queue.append((
                child.id,
                path_nodes + [child.id],
                path_edges + [edge.id],
                weight * edge.weight,
                conf * edge.confidence,
            ))

    paths.sort(key=lambda p: p.total_weight, reverse=True)
    return paths


def detect_cycles(graph: CausalGraph) -> list[CycleInfo]:
    """Detect all cycles using DFS with recursion stack.

    Also flags contradictions: cycles containing 'contradicts' edges.

    Returns:
        List of CycleInfo for each detected cycle.
    """
    cycles = []
    visited = set()
    rec_stack = set()
    path: list[str] = []

    def _dfs(node_id: str):
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for edge, child in graph.get_children(node_id):
            if child.id not in visited:
                _dfs(child.id)
            elif child.id in rec_stack:
                # Found cycle: extract from child.id position to end of path
                cycle_start = path.index(child.id)
                cycle_nodes = path[cycle_start:] + [child.id]

                # Check for contradictions in cycle edges
                has_contradiction = False
                for i in range(len(cycle_nodes) - 1):
                    for e, _ in graph.get_children(cycle_nodes[i]):
                        if _ and _.id == cycle_nodes[i + 1] and e.edge_type == EdgeType.CONTRADICTS:
                            has_contradiction = True

                cycles.append(CycleInfo(
                    nodes=cycle_nodes,
                    is_contradiction=has_contradiction,
                ))

        path.pop()
        rec_stack.discard(node_id)

    for node_id in graph.nodes:
        if node_id not in visited:
            _dfs(node_id)

    return cycles


def causal_distance(
    graph: CausalGraph,
    source: str,
    target: str,
) -> tuple[float, list[str]]:
    """Confidence-weighted shortest path via Dijkstra.

    Distance = sum(1 / max(0.01, edge.confidence)) — low confidence = high distance.

    Args:
        graph: The causal graph.
        source: Starting node id.
        target: Destination node id.

    Returns:
        (distance, path_node_ids). Distance is inf if unreachable.
    """
    if source not in graph.nodes or target not in graph.nodes:
        return float("inf"), []

    dist = {source: 0.0}
    prev: dict[str, str | None] = {source: None}
    heap = [(0.0, source)]
    visited = set()

    while heap:
        d, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == target:
            # Reconstruct path
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = prev.get(node)
            return d, list(reversed(path))

        for edge, child in graph.get_children(current):
            if child.id in visited:
                continue
            edge_dist = 1.0 / max(0.01, edge.confidence)
            new_dist = d + edge_dist
            if new_dist < dist.get(child.id, float("inf")):
                dist[child.id] = new_dist
                prev[child.id] = current
                heapq.heappush(heap, (new_dist, child.id))

    return float("inf"), []


def counterfactual_impact(
    graph: CausalGraph,
    node_id: str,
) -> CounterfactualResult:
    """Estimate impact of removing a node (counterfactual analysis).

    Counts how many downstream nodes lose all their causal paths
    from root nodes if this node were removed.

    Args:
        graph: The causal graph.
        node_id: Node to hypothetically remove.

    Returns:
        CounterfactualResult with affected nodes and impact score.
    """
    if node_id not in graph.nodes:
        return CounterfactualResult(removed_node=node_id)

    # Find all nodes reachable from this node (downstream)
    downstream = set()
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        for edge, child in graph.get_children(current):
            if child.id not in downstream and child.id != node_id:
                downstream.add(child.id)
                queue.append(child.id)

    # For each downstream node, check if it has alternative paths from roots
    # (paths that don't go through the removed node)
    roots = graph.get_roots()
    root_ids = {r.id for r in roots} - {node_id}

    affected = []
    for dn in downstream:
        # BFS from roots avoiding the removed node
        reachable = False
        for root_id in root_ids:
            visit_queue = deque([root_id])
            visited = {node_id}  # treat removed node as visited (blocked)
            while visit_queue:
                c = visit_queue.popleft()
                if c == dn:
                    reachable = True
                    break
                if c in visited:
                    continue
                visited.add(c)
                for _, child in graph.get_children(c):
                    if child.id not in visited:
                        visit_queue.append(child.id)
            if reachable:
                break

        if not reachable:
            affected.append(dn)

    n_total = graph.n_nodes - 1  # exclude removed node
    impact = len(affected) / n_total if n_total > 0 else 0.0

    return CounterfactualResult(
        removed_node=node_id,
        affected_nodes=affected,
        broken_chains=len(affected),
        estimated_impact=impact,
    )
