"""Tests for graph module — model, traversal, cycles, counterfactuals."""

from forgesia.graph.model import CausalGraph, Edge, Node
from forgesia.graph.traversal import (
    causal_distance,
    counterfactual_impact,
    detect_cycles,
    find_causal_chains,
)


def _make_chain_graph():
    """A → B → C → D linear chain."""
    g = CausalGraph()
    for n in ["A", "B", "C", "D"]:
        g.add_node(Node(id=n, label=n))
    g.add_edge(Edge(id="e1", source="A", target="B", weight=0.9, confidence=0.8))
    g.add_edge(Edge(id="e2", source="B", target="C", weight=0.8, confidence=0.7))
    g.add_edge(Edge(id="e3", source="C", target="D", weight=0.7, confidence=0.9))
    return g


def _make_diamond_graph():
    """A → B, A → C, B → D, C → D (diamond)."""
    g = CausalGraph()
    for n in ["A", "B", "C", "D"]:
        g.add_node(Node(id=n, label=n))
    g.add_edge(Edge(id="e1", source="A", target="B"))
    g.add_edge(Edge(id="e2", source="A", target="C"))
    g.add_edge(Edge(id="e3", source="B", target="D"))
    g.add_edge(Edge(id="e4", source="C", target="D"))
    return g


class TestCausalGraph:
    def test_add_node(self):
        g = CausalGraph()
        g.add_node(Node(id="n1", label="Test"))
        assert g.n_nodes == 1
        assert g.nodes["n1"].label == "Test"

    def test_add_edge(self):
        g = CausalGraph()
        g.add_node(Node(id="a"))
        g.add_node(Node(id="b"))
        g.add_edge(Edge(id="e1", source="a", target="b"))
        assert g.n_edges == 1

    def test_zero_alpha_beta_node(self):
        """Node(alpha=0, beta=0) should not crash on property access."""
        n = Node(id="z", label="zero", alpha=0.0, beta=0.0)
        assert n.confidence == 0.5  # uninformative prior
        assert n.uncertainty == 1.0  # maximum uncertainty
        assert n.sample_size == 0.0

    def test_self_loop_rejected(self):
        g = CausalGraph()
        g.add_node(Node(id="a"))
        import pytest
        with pytest.raises(ValueError, match="Self-loop"):
            g.add_edge(Edge(id="e1", source="a", target="a"))

    def test_children_parents(self):
        g = _make_chain_graph()
        children = g.get_children("A")
        assert len(children) == 1
        assert children[0][1].id == "B"
        parents = g.get_parents("D")
        assert len(parents) == 1
        assert parents[0][1].id == "C"

    def test_roots_leaves(self):
        g = _make_chain_graph()
        roots = g.get_roots()
        assert len(roots) == 1 and roots[0].id == "A"
        leaves = g.get_leaves()
        assert len(leaves) == 1 and leaves[0].id == "D"

    def test_remove_node(self):
        g = _make_chain_graph()
        g.remove_node("B")
        assert g.n_nodes == 3
        assert "B" not in g.nodes
        assert g.n_edges == 1  # only C→D remains

    def test_node_confidence(self):
        n = Node(id="test", alpha=8, beta=2)
        assert abs(n.confidence - 0.8) < 0.01

    def test_serialization(self):
        g = _make_chain_graph()
        d = g.to_dict()
        g2 = CausalGraph.from_dict(d)
        assert g2.n_nodes == 4
        assert g2.n_edges == 3


class TestTraversal:
    def test_find_chain(self):
        g = _make_chain_graph()
        paths = find_causal_chains(g, "A", "D")
        assert len(paths) == 1
        assert paths[0].nodes == ["A", "B", "C", "D"]

    def test_diamond_two_paths(self):
        g = _make_diamond_graph()
        paths = find_causal_chains(g, "A", "D")
        assert len(paths) == 2

    def test_no_path(self):
        g = _make_chain_graph()
        paths = find_causal_chains(g, "D", "A")  # reverse — no path
        assert len(paths) == 0

    def test_path_weight(self):
        g = _make_chain_graph()
        paths = find_causal_chains(g, "A", "D")
        assert paths[0].total_weight < 1.0  # product of <1 weights

    def test_causal_distance(self):
        g = _make_chain_graph()
        dist, path = causal_distance(g, "A", "D")
        assert dist < float("inf")
        assert path == ["A", "B", "C", "D"]

    def test_distance_unreachable(self):
        g = _make_chain_graph()
        dist, path = causal_distance(g, "D", "A")
        assert dist == float("inf")


class TestCycles:
    def test_no_cycles(self):
        g = _make_chain_graph()
        cycles = detect_cycles(g)
        assert len(cycles) == 0

    def test_simple_cycle(self):
        g = CausalGraph()
        for n in ["A", "B", "C"]:
            g.add_node(Node(id=n))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        g.add_edge(Edge(id="e2", source="B", target="C"))
        g.add_edge(Edge(id="e3", source="C", target="A"))
        cycles = detect_cycles(g)
        assert len(cycles) >= 1


class TestCounterfactual:
    def test_remove_middle(self):
        g = _make_chain_graph()
        result = counterfactual_impact(g, "B")
        # Removing B breaks the only path from A to C and D
        assert "C" in result.affected_nodes or "D" in result.affected_nodes
        assert result.estimated_impact > 0

    def test_remove_leaf(self):
        g = _make_chain_graph()
        result = counterfactual_impact(g, "D")
        assert len(result.affected_nodes) == 0  # D has no downstream

    def test_diamond_redundancy(self):
        g = _make_diamond_graph()
        result = counterfactual_impact(g, "B")
        # D still reachable via C, so impact should be lower
        assert "D" not in result.affected_nodes
