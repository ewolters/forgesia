"""Tests for inference module — belief updates, Bayes rule, information gain."""


from forgesia.graph.model import CausalGraph, Node
from forgesia.inference.belief import (
    Evidence,
    EvidenceDirection,
    EvidenceType,
    apply_evidence,
    bayes_update,
    credible_interval,
    information_gain,
    suggest_likelihood_ratio,
    temporal_decay,
)


class TestBetaUpdate:
    def test_supporting_evidence(self):
        g = CausalGraph()
        g.add_node(Node(id="h1", alpha=1, beta=1))  # uniform prior
        result = apply_evidence(g, "h1", Evidence(direction=EvidenceDirection.SUPPORTS))
        assert result.confidence_after > result.confidence_before
        assert result.alpha_after > result.alpha_before
        assert result.beta_after == result.beta_before

    def test_refuting_evidence(self):
        g = CausalGraph()
        g.add_node(Node(id="h1", alpha=5, beta=1))  # strong belief
        result = apply_evidence(g, "h1", Evidence(direction=EvidenceDirection.REFUTES))
        assert result.confidence_after < result.confidence_before

    def test_evidence_type_weighting(self):
        g1 = CausalGraph()
        g1.add_node(Node(id="h1", alpha=1, beta=1))
        r1 = apply_evidence(g1, "h1", Evidence(evidence_type=EvidenceType.PHYSICAL, direction=EvidenceDirection.SUPPORTS))

        g2 = CausalGraph()
        g2.add_node(Node(id="h1", alpha=1, beta=1))
        r2 = apply_evidence(g2, "h1", Evidence(evidence_type=EvidenceType.CIRCUMSTANTIAL, direction=EvidenceDirection.SUPPORTS))

        # Physical evidence (weight 1.0) should move belief more than circumstantial (0.5)
        assert r1.evidence_applied > r2.evidence_applied

    def test_neutral_no_change(self):
        g = CausalGraph()
        g.add_node(Node(id="h1", alpha=3, beta=3))
        result = apply_evidence(g, "h1", Evidence(direction=EvidenceDirection.NEUTRAL))
        assert result.confidence_after == result.confidence_before


class TestBayesRule:
    def test_strong_evidence_shifts_belief(self):
        posterior = bayes_update(0.5, likelihood_ratio=10.0)
        assert posterior > 0.9

    def test_refuting_evidence(self):
        posterior = bayes_update(0.5, likelihood_ratio=0.1)
        assert posterior < 0.1

    def test_neutral_lr(self):
        posterior = bayes_update(0.5, likelihood_ratio=1.0)
        assert abs(posterior - 0.5) < 0.01

    def test_confidence_dampens(self):
        full = bayes_update(0.5, likelihood_ratio=10.0, confidence=1.0)
        damped = bayes_update(0.5, likelihood_ratio=10.0, confidence=0.3)
        assert damped < full

    def test_bounds(self):
        assert bayes_update(0.01, 100.0) <= 0.99
        assert bayes_update(0.99, 0.01) >= 0.01


class TestLikelihoodRatio:
    def test_significant(self):
        lr = suggest_likelihood_ratio(p_value=0.001)
        assert lr >= 5

    def test_not_significant(self):
        lr = suggest_likelihood_ratio(p_value=0.5)
        assert lr < 1

    def test_large_effect(self):
        lr = suggest_likelihood_ratio(effect_size=1.0)
        assert lr > 1

    def test_combined(self):
        lr = suggest_likelihood_ratio(p_value=0.001, effect_size=0.9, sample_size=1500)
        assert lr > 5  # 5 * 1.5 * 1.1 = 8.25


class TestCredibleInterval:
    def test_uniform_prior(self):
        lo, hi = credible_interval(1, 1, level=0.95)
        assert lo < 0.1 and hi > 0.9  # very wide

    def test_strong_belief(self):
        lo, hi = credible_interval(80, 20, level=0.95)
        assert lo > 0.7  # concentrated around 0.8

    def test_level(self):
        lo_95, hi_95 = credible_interval(10, 10, level=0.95)
        lo_99, hi_99 = credible_interval(10, 10, level=0.99)
        assert (hi_99 - lo_99) > (hi_95 - lo_95)  # 99% wider


class TestInformationGain:
    def test_uncertain_node_high_gain(self):
        n = Node(id="h1", alpha=1, beta=1)  # max uncertainty
        gain = information_gain(n, likelihood_ratio=5.0)
        assert gain > 0.3

    def test_certain_node_low_gain(self):
        n = Node(id="h1", alpha=100, beta=1)  # very certain
        gain = information_gain(n, likelihood_ratio=5.0)
        assert gain < 0.1  # already know the answer


class TestTemporalDecay:
    def test_recent_no_decay(self):
        n = Node(id="h1", alpha=10, beta=2)
        conf = temporal_decay(n, days_since_evidence=0)
        assert abs(conf - n.confidence) < 0.01

    def test_old_evidence_decays(self):
        n = Node(id="h1", alpha=10, beta=2)
        conf_new = temporal_decay(n, days_since_evidence=0)
        conf_old = temporal_decay(n, days_since_evidence=180)
        assert conf_old < conf_new  # decayed toward 0.5

    def test_very_old_approaches_prior(self):
        n = Node(id="h1", alpha=100, beta=1)
        conf = temporal_decay(n, days_since_evidence=10000)
        assert abs(conf - 0.5) < 0.1  # should be near prior
