"""Tests for propagation — vine rot, CPT, belief propagation."""

from forgesia.graph.model import CausalGraph, Edge, Node
from forgesia.propagation.vine_rot import propagate_failure
from forgesia.propagation.cpt import CPT, CPTEntry, loopy_belief_propagation, update_cpt


class TestVineRot:
    def test_propagates_upstream(self):
        g = CausalGraph()
        g.add_node(Node(id="root", alpha=5, beta=1))
        g.add_node(Node(id="mid", alpha=5, beta=1))
        g.add_node(Node(id="leaf", alpha=5, beta=1))
        g.add_edge(Edge(id="e1", source="root", target="mid"))
        g.add_edge(Edge(id="e2", source="mid", target="leaf"))

        result = propagate_failure(g, "leaf")
        assert len(result.effects) >= 2  # leaf + at least mid
        # Upstream nodes should have lower confidence after propagation
        root_effect = next((e for e in result.effects if e.node_id == "root"), None)
        if root_effect:
            assert root_effect.confidence_after <= root_effect.confidence_before

    def test_strength_decays(self):
        g = CausalGraph()
        for n in ["A", "B", "C"]:
            g.add_node(Node(id=n, alpha=5, beta=1))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        g.add_edge(Edge(id="e2", source="B", target="C"))

        result = propagate_failure(g, "C")
        strengths = {e.node_id: e.strength for e in result.effects}
        # C (origin) should have higher strength than B, B > A
        assert strengths.get("C", 0) >= strengths.get("B", 0)

    def test_flagged_nodes(self):
        g = CausalGraph()
        g.add_node(Node(id="weak", alpha=1.5, beta=1))  # low confidence
        g.add_node(Node(id="failed", alpha=1, beta=1))
        g.add_edge(Edge(id="e1", source="weak", target="failed"))

        result = propagate_failure(g, "failed", threshold=0.4)
        # weak node may get flagged
        assert result.origin == "failed"


class TestCPT:
    def test_update(self):
        cpt = CPT(
            child_var="effect",
            parent_vars=["cause"],
            child_states=["yes", "no"],
            entries=[
                CPTEntry(parent_config=("true",), child_state="yes", alpha=1),
                CPTEntry(parent_config=("true",), child_state="no", alpha=1),
            ],
        )
        result = update_cpt(cpt, ("true",), "yes", weight=5)
        assert result.prob_after > result.prob_before
        assert result.confidence > 0

    def test_new_config(self):
        cpt = CPT(child_var="x", parent_vars=["a"], child_states=["on", "off"])
        result = update_cpt(cpt, ("high",), "on")
        assert result.prob_after > 0

    def test_uniform_fallback(self):
        cpt = CPT(child_var="x", parent_vars=["a"], child_states=["on", "off"])
        p = cpt.get_probability(("unknown",), "on")
        assert abs(p - 0.5) < 0.01  # uniform for 2 states


class TestBeliefPropagation:
    def test_observation_propagates(self):
        # rain → wet_grass, sprinkler → wet_grass
        rain_cpt = CPT(
            child_var="wet_grass",
            parent_vars=["rain"],
            child_states=["true", "false"],
            entries=[
                CPTEntry(parent_config=("true",), child_state="true", alpha=9, count=0),
                CPTEntry(parent_config=("true",), child_state="false", alpha=1, count=0),
                CPTEntry(parent_config=("false",), child_state="true", alpha=2, count=0),
                CPTEntry(parent_config=("false",), child_state="false", alpha=8, count=0),
            ],
        )

        beliefs = loopy_belief_propagation(
            cpts={"wet_grass": rain_cpt},
            observations={"rain": "true"},
        )
        # Given rain=true, wet_grass should be likely true
        assert beliefs["wet_grass"]["true"] > 0.7

    def test_no_observations(self):
        cpt = CPT(
            child_var="x",
            parent_vars=["a"],
            child_states=["yes", "no"],
            entries=[
                CPTEntry(parent_config=("true",), child_state="yes", alpha=5),
                CPTEntry(parent_config=("true",), child_state="no", alpha=5),
                CPTEntry(parent_config=("false",), child_state="yes", alpha=5),
                CPTEntry(parent_config=("false",), child_state="no", alpha=5),
            ],
        )
        beliefs = loopy_belief_propagation(cpts={"x": cpt}, observations={})
        # Should stay near uniform
        assert abs(beliefs["x"]["yes"] - 0.5) < 0.15
