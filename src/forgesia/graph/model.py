"""CausalGraph — the core data structure for belief networks.

A directed graph where nodes carry beliefs (Beta distributions)
and edges carry causal relationships with weights.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    CAUSES = "causes"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    PRECEDES = "precedes"
    INHIBITS = "inhibits"
    AMPLIFIES = "amplifies"
    REQUIRES = "requires"


class NodeType(Enum):
    EVENT = "event"
    CAUSE = "cause"
    HYPOTHESIS = "hypothesis"
    SYMPTOM = "symptom"
    EVIDENCE = "evidence"
    ACTION = "action"
    OUTCOME = "outcome"
    CONDITION = "condition"
    FACTOR = "factor"


@dataclass
class Node:
    """A node in the causal graph carrying a Beta belief."""

    id: str
    node_type: NodeType = NodeType.FACTOR
    label: str = ""
    alpha: float = 1.0  # Beta prior success count
    beta: float = 1.0  # Beta prior failure count
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Current belief: E[Beta(α, β)] = α / (α + β)."""
        total = self.alpha + self.beta
        if total == 0:
            return 0.5
        return self.alpha / total

    @property
    def uncertainty(self) -> float:
        """Uncertainty: std(Beta) normalized to [0, 1]."""
        total = self.alpha + self.beta
        if total == 0:
            return 1.0
        var = (self.alpha * self.beta) / (total ** 2 * (total + 1))
        return min(1.0, var ** 0.5 / 0.5)

    @property
    def sample_size(self) -> float:
        """Effective observations: α + β - 2 (subtract prior)."""
        return max(0.0, self.alpha + self.beta - 2.0)


@dataclass
class Edge:
    """A directed edge in the causal graph."""

    id: str
    source: str  # node id
    target: str  # node id
    edge_type: EdgeType = EdgeType.CAUSES
    weight: float = 1.0  # strength of relationship
    confidence: float = 0.5  # how certain we are this edge exists
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalGraph:
    """Directed graph with Beta-distributed node beliefs and weighted edges.

    This is the core data structure. All other modules operate on it.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._children: dict[str, list[str]] = defaultdict(list)  # node_id → [edge_ids]
        self._parents: dict[str, list[str]] = defaultdict(list)  # node_id → [edge_ids]

    def add_node(self, node: Node) -> Node:
        """Add a node. Replaces if id already exists."""
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        """Add a directed edge. Validates nodes exist and no self-loops."""
        if edge.source == edge.target:
            raise ValueError(f"Self-loop not allowed: {edge.source}")
        if edge.source not in self.nodes:
            raise ValueError(f"Source node not found: {edge.source}")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node not found: {edge.target}")

        self.edges[edge.id] = edge
        self._children[edge.source].append(edge.id)
        self._parents[edge.target].append(edge.id)
        return edge

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all connected edges."""
        if node_id not in self.nodes:
            return
        # Remove connected edges
        edge_ids = list(self._children.get(node_id, [])) + list(self._parents.get(node_id, []))
        for eid in set(edge_ids):
            self.remove_edge(eid)
        del self.nodes[node_id]
        self._children.pop(node_id, None)
        self._parents.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge."""
        if edge_id not in self.edges:
            return
        edge = self.edges[edge_id]
        self._children[edge.source] = [e for e in self._children[edge.source] if e != edge_id]
        self._parents[edge.target] = [e for e in self._parents[edge.target] if e != edge_id]
        del self.edges[edge_id]

    def get_children(self, node_id: str) -> list[tuple[Edge, Node]]:
        """Get downstream (edge, child_node) pairs."""
        result = []
        for eid in self._children.get(node_id, []):
            edge = self.edges.get(eid)
            if edge:
                child = self.nodes.get(edge.target)
                if child:
                    result.append((edge, child))
        return result

    def get_parents(self, node_id: str) -> list[tuple[Edge, Node]]:
        """Get upstream (edge, parent_node) pairs."""
        result = []
        for eid in self._parents.get(node_id, []):
            edge = self.edges.get(eid)
            if edge:
                parent = self.nodes.get(edge.source)
                if parent:
                    result.append((edge, parent))
        return result

    def get_roots(self) -> list[Node]:
        """Nodes with no parents."""
        return [n for n in self.nodes.values() if not self._parents.get(n.id)]

    def get_leaves(self) -> list[Node]:
        """Nodes with no children."""
        return [n for n in self.nodes.values() if not self._children.get(n.id)]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        """Serialize to plain dict (JSON-safe)."""
        from dataclasses import asdict
        return {
            "nodes": [asdict(n) | {"node_type": n.node_type.value} for n in self.nodes.values()],
            "edges": [asdict(e) | {"edge_type": e.edge_type.value} for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CausalGraph:
        """Deserialize from dict."""
        g = cls()
        for nd in data.get("nodes", []):
            nt = nd.pop("node_type", "factor")
            meta = nd.pop("metadata", {})
            g.add_node(Node(node_type=NodeType(nt), metadata=meta, **nd))
        for ed in data.get("edges", []):
            et = ed.pop("edge_type", "causes")
            meta = ed.pop("metadata", {})
            g.add_edge(Edge(edge_type=EdgeType(et), metadata=meta, **ed))
        return g
