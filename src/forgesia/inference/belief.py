"""Belief updates — Beta-Binomial conjugate, evidence weighting, Bayes rule.

The core inference engine. Every observation updates node beliefs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..graph.model import CausalGraph, Node


class EvidenceType(Enum):
    PHYSICAL = "physical"
    DOCUMENTARY = "documentary"
    ANALYTICAL = "analytical"
    STATISTICAL = "statistical"
    TESTIMONIAL = "testimonial"
    CIRCUMSTANTIAL = "circumstantial"


class EvidenceDirection(Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


# Reliability weights by evidence type (BOOT-001 §6.1)
EVIDENCE_WEIGHTS = {
    EvidenceType.PHYSICAL: 1.0,
    EvidenceType.DOCUMENTARY: 0.9,
    EvidenceType.ANALYTICAL: 0.8,
    EvidenceType.STATISTICAL: 0.75,
    EvidenceType.TESTIMONIAL: 0.7,
    EvidenceType.CIRCUMSTANTIAL: 0.5,
}


@dataclass
class Evidence:
    """A piece of evidence to apply to a node."""

    evidence_type: EvidenceType = EvidenceType.ANALYTICAL
    direction: EvidenceDirection = EvidenceDirection.SUPPORTS
    reliability: float = 0.8  # 0-1, how trustworthy
    weight: float = 1.0  # observation count equivalent


@dataclass
class BeliefUpdate:
    """Result of updating a node's belief."""

    node_id: str
    alpha_before: float
    beta_before: float
    alpha_after: float
    beta_after: float
    confidence_before: float
    confidence_after: float
    uncertainty_before: float
    uncertainty_after: float
    evidence_applied: float  # effective weight applied


def apply_evidence(
    graph: CausalGraph,
    node_id: str,
    evidence: Evidence,
) -> BeliefUpdate:
    """Apply evidence to a node's Beta belief (conjugate update).

    Supporting evidence increases α. Refuting evidence increases β.
    Weight is scaled by evidence type reliability and stated reliability.

    Args:
        graph: The causal graph.
        node_id: Target node.
        evidence: Evidence to apply.

    Returns:
        BeliefUpdate with before/after beliefs.
    """
    node = graph.nodes.get(node_id)
    if node is None:
        raise ValueError(f"Node not found: {node_id}")

    # Effective weight = base weight × type reliability × stated reliability
    type_weight = EVIDENCE_WEIGHTS.get(evidence.evidence_type, 0.5)
    effective_weight = evidence.weight * type_weight * evidence.reliability

    alpha_before = node.alpha
    beta_before = node.beta
    conf_before = node.confidence
    unc_before = node.uncertainty

    if evidence.direction == EvidenceDirection.SUPPORTS:
        node.alpha = min(10000, node.alpha + effective_weight)
    elif evidence.direction == EvidenceDirection.REFUTES:
        node.beta = min(10000, node.beta + effective_weight)
    # NEUTRAL: no update

    return BeliefUpdate(
        node_id=node_id,
        alpha_before=alpha_before,
        beta_before=beta_before,
        alpha_after=node.alpha,
        beta_after=node.beta,
        confidence_before=conf_before,
        confidence_after=node.confidence,
        uncertainty_before=unc_before,
        uncertainty_after=node.uncertainty,
        evidence_applied=effective_weight,
    )


def bayes_update(
    prior_prob: float,
    likelihood_ratio: float,
    confidence: float = 1.0,
) -> float:
    """Bayes' rule with confidence-adjusted likelihood ratio.

    posterior_odds = prior_odds × adjusted_LR
    adjusted_LR = 1 + (LR - 1) × confidence

    Args:
        prior_prob: Prior probability (0-1).
        likelihood_ratio: P(evidence | H1) / P(evidence | H0).
        confidence: How much to trust this LR (0-1). At 0, LR becomes 1 (no update).

    Returns:
        Posterior probability, clamped to [0.01, 0.99].
    """
    prior_prob = max(0.01, min(0.99, prior_prob))
    adjusted_lr = 1 + (likelihood_ratio - 1) * confidence

    prior_odds = prior_prob / (1 - prior_prob)
    posterior_odds = prior_odds * adjusted_lr
    posterior = posterior_odds / (1 + posterior_odds)

    return max(0.01, min(0.99, posterior))


def suggest_likelihood_ratio(
    p_value: float | None = None,
    effect_size: float | None = None,
    sample_size: int | None = None,
) -> float:
    """Heuristic likelihood ratio from statistical evidence.

    Maps p-values, effect sizes, and sample sizes to an approximate LR
    for use with bayes_update().

    Args:
        p_value: Statistical significance.
        effect_size: Cohen's d or equivalent.
        sample_size: Number of observations.

    Returns:
        Likelihood ratio (>1 supports, <1 refutes, 1 = no information).
    """
    lr = 1.0

    if p_value is not None:
        if p_value < 0.001:
            lr = 10.0
        elif p_value < 0.01:
            lr = 5.0
        elif p_value < 0.05:
            lr = 2.0
        elif p_value < 0.10:
            lr = 1.3
        else:
            lr = 0.5

    if effect_size is not None:
        if abs(effect_size) > 0.8:
            lr *= 1.5
        elif abs(effect_size) > 0.5:
            lr *= 1.2
        elif abs(effect_size) < 0.2:
            lr *= 0.8

    if sample_size is not None:
        if sample_size > 1000:
            lr *= 1.1
        elif sample_size < 30:
            lr *= 0.9

    return lr


def credible_interval(
    alpha: float,
    beta_param: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Credible interval for a Beta(α, β) distribution.

    Uses scipy.stats.beta.ppf for exact quantiles, falls back to
    normal approximation if scipy unavailable.

    Args:
        alpha: Beta shape parameter.
        beta_param: Beta shape parameter.
        level: Credible level (default 0.95).

    Returns:
        (lower, upper) bounds.
    """
    try:
        from scipy.stats import beta as beta_dist
        lo = float(beta_dist.ppf((1 - level) / 2, alpha, beta_param))
        hi = float(beta_dist.ppf(1 - (1 - level) / 2, alpha, beta_param))
        return (lo, hi)
    except ImportError:
        mean = alpha / (alpha + beta_param)
        var = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
        std = math.sqrt(var)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
        return (max(0.0, mean - z * std), min(1.0, mean + z * std))


def information_gain(
    node: Node,
    likelihood_ratio: float,
) -> float:
    """Expected information gain from an experiment on this node.

    Shannon entropy reduction if we were to observe evidence with this LR.

    Args:
        node: Node to evaluate.
        likelihood_ratio: Expected LR if the experiment confirms.

    Returns:
        Expected reduction in entropy (bits). Higher = more informative.
    """
    p = node.confidence

    def _entropy(prob):
        if prob <= 0 or prob >= 1:
            return 0.0
        return -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))

    h_current = _entropy(p)

    # If positive result (LR as given)
    p_pos = bayes_update(p, likelihood_ratio)
    h_pos = _entropy(p_pos)

    # If negative result (LR inverted)
    p_neg = bayes_update(p, 1 / likelihood_ratio)
    h_neg = _entropy(p_neg)

    # Expected entropy after experiment (weighted by current belief)
    h_after = p * h_pos + (1 - p) * h_neg

    return max(0.0, h_current - h_after)


def temporal_decay(
    node: Node,
    days_since_evidence: float,
    half_life_days: float = 90.0,
) -> float:
    """Apply temporal decay to a node's belief — INNOVATION.

    Old evidence counts less. Beliefs drift toward the prior (0.5)
    as time passes without new observations.

    Args:
        node: Node to decay.
        days_since_evidence: Days since last evidence was applied.
        half_life_days: Half-life for belief decay.

    Returns:
        New confidence after decay (does NOT mutate node).
    """
    decay = 0.5 ** (days_since_evidence / half_life_days)

    # Effective α and β decay toward 1 (prior)
    effective_alpha = 1.0 + (node.alpha - 1.0) * decay
    effective_beta = 1.0 + (node.beta - 1.0) * decay

    return effective_alpha / (effective_alpha + effective_beta)
