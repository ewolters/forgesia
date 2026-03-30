"""Risk scoring, differential diagnosis, and preference-based filtering.

4-factor risk model + diagnostic reasoning for root cause prioritization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..graph.model import CausalGraph, EdgeType, NodeType


class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PreferenceType(Enum):
    SAFETY = "safety"
    COST = "cost"
    SPEED = "speed"
    COMPLIANCE = "compliance"


@dataclass
class RiskScore:
    """Risk assessment for a single hypothesis/node."""

    node_id: str
    score: float = 0.0  # 0-1
    level: RiskLevel = RiskLevel.LOW
    confidence_score: float = 0.0
    evidence_score: float = 0.0
    severity_score: float = 0.0
    cascading_score: float = 0.0


@dataclass
class DiagnosticRanking:
    """Ranked hypothesis with evidence assessment."""

    node_id: str
    label: str
    confidence: float
    support_score: float
    refute_score: float
    net_evidence: float
    rank: int = 0


@dataclass
class DiagnosticResult:
    """Differential diagnosis result."""

    rankings: list[DiagnosticRanking] = field(default_factory=list)
    confidence_spread: float = 0.0  # max - min confidence


def score_risk(
    graph: CausalGraph,
    node_id: str,
    w_confidence: float = 0.3,
    w_evidence: float = 0.3,
    w_severity: float = 0.2,
    w_cascading: float = 0.2,
) -> RiskScore:
    """Compute 4-factor risk score for a node.

    Factors:
    - confidence: how likely this is the cause (from Beta belief)
    - evidence: how much evidence supports it
    - severity: metadata-driven or default
    - cascading: how many downstream nodes it affects

    Args:
        graph: The causal graph.
        node_id: Node to score.
        w_*: Factor weights (must sum to 1.0).

    Returns:
        RiskScore with component scores and level classification.
    """
    node = graph.nodes.get(node_id)
    if node is None:
        return RiskScore(node_id=node_id)

    # Confidence score (from Beta belief)
    conf = node.confidence

    # Evidence score: count supporting edges
    parents = graph.get_parents(node_id)
    supporting = [e for e, _ in parents if e.edge_type in (EdgeType.SUPPORTS, EdgeType.CAUSES)]
    evidence = min(1.0, len(supporting) / 5.0)

    # Severity score (from metadata or default by node type)
    severity_map = {
        NodeType.EVENT: 0.7, NodeType.CAUSE: 0.6, NodeType.HYPOTHESIS: 0.5,
        NodeType.SYMPTOM: 0.4, NodeType.CONDITION: 0.3, NodeType.FACTOR: 0.3,
    }
    severity = node.metadata.get("severity", severity_map.get(node.node_type, 0.5))

    # Cascading score: count downstream children
    children = graph.get_children(node_id)
    cascading_edges = [e for e, _ in children if e.edge_type in (EdgeType.CAUSES, EdgeType.AMPLIFIES)]
    cascading = min(1.0, len(cascading_edges) / 3.0)

    # Weighted total
    total = w_confidence * conf + w_evidence * evidence + w_severity * severity + w_cascading * cascading

    # Level classification
    if total >= 0.8:
        level = RiskLevel.CRITICAL
    elif total >= 0.6:
        level = RiskLevel.HIGH
    elif total >= 0.4:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return RiskScore(
        node_id=node_id,
        score=total,
        level=level,
        confidence_score=conf,
        evidence_score=evidence,
        severity_score=severity,
        cascading_score=cascading,
    )


def differential_diagnosis(
    graph: CausalGraph,
    hypothesis_nodes: list[str] | None = None,
) -> DiagnosticResult:
    """Rank hypotheses by net evidence for root cause prioritization.

    For each hypothesis node, compute:
    - support_score: sum of reliability scores from supporting evidence
    - refute_score: sum of reliability scores from refuting evidence
    - net_evidence: support - refute

    Args:
        graph: The causal graph.
        hypothesis_nodes: Specific node ids to rank (default: all hypotheses).

    Returns:
        DiagnosticResult with ranked hypotheses.
    """
    if hypothesis_nodes is None:
        hypothesis_nodes = [
            n.id for n in graph.nodes.values()
            if n.node_type in (NodeType.HYPOTHESIS, NodeType.CAUSE)
        ]

    rankings = []
    for node_id in hypothesis_nodes:
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        parents = graph.get_parents(node_id)
        support = sum(
            e.confidence * e.weight
            for e, _ in parents
            if e.edge_type in (EdgeType.SUPPORTS, EdgeType.CAUSES)
        )
        refute = sum(
            e.confidence * e.weight
            for e, _ in parents
            if e.edge_type == EdgeType.CONTRADICTS
        )

        rankings.append(DiagnosticRanking(
            node_id=node_id,
            label=node.label or node_id,
            confidence=node.confidence,
            support_score=support,
            refute_score=refute,
            net_evidence=support - refute,
        ))

    # Sort by net evidence descending, then confidence descending
    rankings.sort(key=lambda r: (r.net_evidence, r.confidence), reverse=True)
    for i, r in enumerate(rankings):
        r.rank = i + 1

    spread = 0.0
    if len(rankings) >= 2:
        confs = [r.confidence for r in rankings]
        spread = max(confs) - min(confs)

    return DiagnosticResult(rankings=rankings, confidence_spread=spread)


def filter_by_preference(
    graph: CausalGraph,
    node_ids: list[str],
    preference: PreferenceType,
) -> list[tuple[str, float]]:
    """Re-rank nodes by a preference criterion.

    Safety: count preventive downstream actions.
    Cost: simpler causes (fewer edges) preferred.
    Speed: higher confidence preferred (less investigation needed).
    Compliance: more evidence preferred.

    Args:
        graph: The causal graph.
        node_ids: Nodes to rank.
        preference: Preference type.

    Returns:
        List of (node_id, preference_score) sorted descending.
    """
    scored = []

    for node_id in node_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if preference == PreferenceType.SAFETY:
            children = graph.get_children(node_id)
            preventive = sum(1 for e, c in children if c.node_type == NodeType.ACTION)
            score = min(1.0, preventive / 2.0)

        elif preference == PreferenceType.COST:
            parents = graph.get_parents(node_id)
            score = 0.8 if len(parents) <= 2 else 0.4

        elif preference == PreferenceType.SPEED:
            score = node.confidence

        elif preference == PreferenceType.COMPLIANCE:
            parents = graph.get_parents(node_id)
            evidence_count = sum(1 for _, p in parents if p.node_type == NodeType.EVIDENCE)
            score = min(1.0, evidence_count / 5.0)

        else:
            score = 0.5

        scored.append((node_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
