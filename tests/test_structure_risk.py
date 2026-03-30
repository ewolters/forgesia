"""Tests for structure learning and risk scoring."""

from forgesia.graph.model import CausalGraph, Edge, EdgeType, Node, NodeType
from forgesia.structure.learning import (
    compute_energy,
    failure_diversity,
    mutual_information,
    phi_coefficient,
    propose_edges,
)
from forgesia.risk.scoring import (
    PreferenceType,
    RiskLevel,
    differential_diagnosis,
    filter_by_preference,
    score_risk,
)


def _make_observations(n=100):
    """Generate correlated binary observations."""
    import random
    random.seed(42)
    obs = []
    for _ in range(n):
        a = random.random() < 0.6
        b = a if random.random() < 0.8 else (not a)  # correlated with A
        c = random.random() < 0.5  # independent
        obs.append({"A": a, "B": b, "C": c})
    return obs


class TestPhiCoefficient:
    def test_correlated(self):
        obs = _make_observations()
        phi = phi_coefficient(obs, "A", "B")
        assert phi > 0.3  # correlated

    def test_independent(self):
        obs = _make_observations()
        phi = phi_coefficient(obs, "A", "C")
        assert abs(phi) < 0.3  # independent


class TestMutualInformation:
    def test_correlated(self):
        obs = _make_observations()
        mi = mutual_information(obs, "A", "B")
        assert mi > 0.1

    def test_independent(self):
        obs = _make_observations()
        mi = mutual_information(obs, "A", "C")
        assert mi < 0.1


class TestEdgeProposals:
    def test_proposes_correlated(self):
        g = CausalGraph()
        g.add_node(Node(id="A"))
        g.add_node(Node(id="B"))
        g.add_node(Node(id="C"))
        obs = _make_observations()
        proposals = propose_edges(g, obs)
        # Should propose A→B (correlated) but not A→C
        proposed_pairs = {(p.source, p.target) for p in proposals}
        assert any("A" in pair and "B" in pair for pair in proposed_pairs)

    def test_no_duplicates(self):
        g = CausalGraph()
        g.add_node(Node(id="A"))
        g.add_node(Node(id="B"))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        obs = _make_observations()
        proposals = propose_edges(g, obs)
        assert not any(p.source == "A" and p.target == "B" for p in proposals)


class TestEnergy:
    def test_empty_graph(self):
        g = CausalGraph()
        energy = compute_energy(g)
        assert energy.total == 0.0

    def test_complexity_increases(self):
        g1 = CausalGraph()
        g1.add_node(Node(id="A"))
        e1 = compute_energy(g1)

        g2 = CausalGraph()
        for n in ["A", "B", "C"]:
            g2.add_node(Node(id=n))
        g2.add_edge(Edge(id="e1", source="A", target="B"))
        g2.add_edge(Edge(id="e2", source="B", target="C"))
        e2 = compute_energy(g2)

        assert e2.complexity > e1.complexity

    def test_governance_penalty(self):
        g = CausalGraph()
        g.add_node(Node(id="A"))
        e_clean = compute_energy(g, governance_violations=0)
        e_dirty = compute_energy(g, governance_violations=3)
        assert e_dirty.total > e_clean.total


class TestFailureDiversity:
    def test_concentrated(self):
        g = CausalGraph()
        g.add_node(Node(id="root"))
        g.add_node(Node(id="leaf1", alpha=1, beta=10))  # failed
        g.add_node(Node(id="leaf2", alpha=10, beta=1))  # healthy
        g.add_edge(Edge(id="e1", source="root", target="leaf1"))
        g.add_edge(Edge(id="e2", source="root", target="leaf2"))
        div = failure_diversity(g)
        assert div < 0.7  # concentrated in leaf1

    def test_uniform_failure(self):
        g = CausalGraph()
        g.add_node(Node(id="root"))
        for i in range(5):
            g.add_node(Node(id=f"leaf{i}", alpha=1, beta=5))  # all equally failed
            g.add_edge(Edge(id=f"e{i}", source="root", target=f"leaf{i}"))
        div = failure_diversity(g)
        assert div > 0.8  # spread across all leaves


class TestRiskScoring:
    def test_high_risk(self):
        g = CausalGraph()
        g.add_node(Node(id="cause", node_type=NodeType.CAUSE, alpha=8, beta=2))
        g.add_node(Node(id="ev1", node_type=NodeType.EVIDENCE))
        g.add_node(Node(id="ev2", node_type=NodeType.EVIDENCE))
        g.add_node(Node(id="effect1"))
        g.add_node(Node(id="effect2"))
        g.add_edge(Edge(id="e1", source="ev1", target="cause", edge_type=EdgeType.SUPPORTS))
        g.add_edge(Edge(id="e2", source="ev2", target="cause", edge_type=EdgeType.SUPPORTS))
        g.add_edge(Edge(id="e3", source="cause", target="effect1", edge_type=EdgeType.CAUSES))
        g.add_edge(Edge(id="e4", source="cause", target="effect2", edge_type=EdgeType.CAUSES))

        risk = score_risk(g, "cause")
        assert risk.score > 0.5
        assert risk.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_low_risk(self):
        g = CausalGraph()
        g.add_node(Node(id="weak", alpha=1, beta=5))  # low confidence
        risk = score_risk(g, "weak")
        assert risk.level in (RiskLevel.LOW, RiskLevel.MEDIUM)


class TestDiagnosis:
    def test_ranking(self):
        g = CausalGraph()
        g.add_node(Node(id="h1", node_type=NodeType.HYPOTHESIS, alpha=8, beta=2, label="Hypothesis A"))
        g.add_node(Node(id="h2", node_type=NodeType.HYPOTHESIS, alpha=3, beta=7, label="Hypothesis B"))
        g.add_node(Node(id="ev"))
        g.add_edge(Edge(id="e1", source="ev", target="h1", edge_type=EdgeType.SUPPORTS, confidence=0.9))

        result = differential_diagnosis(g)
        assert len(result.rankings) == 2
        assert result.rankings[0].node_id == "h1"  # more evidence support
        assert result.confidence_spread > 0

    def test_empty(self):
        g = CausalGraph()
        g.add_node(Node(id="x", node_type=NodeType.FACTOR))
        result = differential_diagnosis(g)
        assert len(result.rankings) == 0  # no hypotheses


class TestPreferenceFiltering:
    def test_speed_prefers_confident(self):
        g = CausalGraph()
        g.add_node(Node(id="sure", alpha=9, beta=1))
        g.add_node(Node(id="unsure", alpha=2, beta=2))
        ranked = filter_by_preference(g, ["sure", "unsure"], PreferenceType.SPEED)
        assert ranked[0][0] == "sure"

    def test_compliance_prefers_evidence(self):
        g = CausalGraph()
        g.add_node(Node(id="documented"))
        g.add_node(Node(id="undocumented"))
        g.add_node(Node(id="ev1", node_type=NodeType.EVIDENCE))
        g.add_node(Node(id="ev2", node_type=NodeType.EVIDENCE))
        g.add_edge(Edge(id="e1", source="ev1", target="documented"))
        g.add_edge(Edge(id="e2", source="ev2", target="documented"))
        ranked = filter_by_preference(g, ["documented", "undocumented"], PreferenceType.COMPLIANCE)
        assert ranked[0][0] == "documented"
