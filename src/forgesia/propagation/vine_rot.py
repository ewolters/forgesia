"""Vine rot propagation — upstream failure decay through causal graph.

When a node fails, propagate failure evidence upstream to all causes,
decaying strength at each hop. Named after how vine diseases propagate
through root systems.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..graph.model import CausalGraph


@dataclass
class PropagationEffect:
    """Effect on one node from propagation."""

    node_id: str
    strength: float  # propagated evidence strength
    depth: int  # hops from origin
    confidence_before: float
    confidence_after: float
    flagged: bool = False  # confidence dropped below threshold


@dataclass
class PropagationResult:
    """Result of vine rot propagation."""

    origin: str
    effects: list[PropagationEffect] = field(default_factory=list)
    flagged_nodes: list[str] = field(default_factory=list)
    max_depth_reached: int = 0


def propagate_failure(
    graph: CausalGraph,
    origin_node: str,
    initial_strength: float = 1.0,
    decay_factor: float = 0.1,
    min_strength: float = 0.05,
    max_depth: int = 10,
    threshold: float = 0.3,
) -> PropagationResult:
    """Propagate failure evidence upstream through causal graph.

    BFS from failed node, applying decaying failure evidence to each
    upstream parent. Evidence weight decreases with distance.

    Args:
        graph: The causal graph.
        origin_node: The node that failed.
        initial_strength: Starting evidence strength (1.0 = full failure).
        decay_factor: Fraction of strength lost per hop (0.1 = 10% loss).
        min_strength: Stop propagating below this strength.
        max_depth: Maximum hops upstream.
        threshold: Flag nodes whose confidence drops below this.

    Returns:
        PropagationResult with effects on each visited node.
    """
    if origin_node not in graph.nodes:
        return PropagationResult(origin=origin_node)

    result = PropagationResult(origin=origin_node)
    visited = set()

    # Queue: (node_id, strength, depth)
    queue = deque([(origin_node, initial_strength, 0)])

    while queue:
        node_id, strength, depth = queue.popleft()

        if node_id in visited or strength < min_strength or depth > max_depth:
            continue
        visited.add(node_id)

        if depth > result.max_depth_reached:
            result.max_depth_reached = depth

        node = graph.nodes.get(node_id)
        if node is None:
            continue

        # Apply failure evidence (increase β)
        conf_before = node.confidence
        evidence_weight = strength * (1 - decay_factor)
        node.beta = min(10000, node.beta + evidence_weight)
        conf_after = node.confidence

        flagged = conf_after < threshold and strength >= min_strength
        effect = PropagationEffect(
            node_id=node_id,
            strength=strength,
            depth=depth,
            confidence_before=conf_before,
            confidence_after=conf_after,
            flagged=flagged,
        )
        result.effects.append(effect)

        if flagged:
            result.flagged_nodes.append(node_id)

        # Propagate upstream to parents
        if depth < max_depth:
            for edge, parent in graph.get_parents(node_id):
                parent_strength = strength * (1 - decay_factor) * edge.weight
                if parent_strength >= min_strength and parent.id not in visited:
                    queue.append((parent.id, parent_strength, depth + 1))

    return result
