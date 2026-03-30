"""Structure learning — discover causal edges from observations.

Phi coefficient for correlation, mutual information for dependency,
energy function for graph quality, edge proposal and scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..graph.model import CausalGraph, EdgeType


@dataclass
class EdgeProposal:
    """A proposed new edge with evidence score."""

    source: str
    target: str
    correlation: float  # phi coefficient
    mutual_information: float
    energy_delta: float  # negative = improves graph
    required_evidence: float  # how much more evidence needed
    edge_type: EdgeType = EdgeType.CAUSES


@dataclass
class GraphEnergy:
    """Energy (cost) of the current graph state.

    Lower energy = better graph. Components:
    - accuracy: how well the graph explains observations
    - complexity: penalizes too many nodes/edges
    - churn: penalizes frequent structural changes
    - governance: penalizes standard violations
    """

    total: float = 0.0
    accuracy: float = 0.0
    complexity: float = 0.0
    churn: float = 0.0
    governance: float = 0.0
    avg_path_length: float = 0.0


def phi_coefficient(
    observations: list[dict[str, bool]],
    node_a: str,
    node_b: str,
) -> float:
    """Phi coefficient (binary correlation) between two nodes.

    Args:
        observations: List of observation dicts {node_id: True/False}.
        node_a: First node id.
        node_b: Second node id.

    Returns:
        Phi coefficient in [-1, 1]. |φ| > 0.3 suggests dependency.
    """
    n = len(observations)
    if n == 0:
        return 0.0

    both = sum(1 for o in observations if o.get(node_a) and o.get(node_b))
    a_only = sum(1 for o in observations if o.get(node_a) and not o.get(node_b))
    b_only = sum(1 for o in observations if not o.get(node_a) and o.get(node_b))
    neither = sum(1 for o in observations if not o.get(node_a) and not o.get(node_b))

    a_total = both + a_only
    b_total = both + b_only
    not_a_total = n - a_total
    not_b_total = n - b_total

    denom = math.sqrt(a_total * not_a_total * b_total * not_b_total)
    if denom == 0:
        return 0.0

    return (both * neither - a_only * b_only) / denom


def mutual_information(
    observations: list[dict[str, bool]],
    node_a: str,
    node_b: str,
) -> float:
    """Mutual information between two binary nodes.

    MI = Σ p(a,b) * log(p(a,b) / (p(a) * p(b)))

    Args:
        observations: Observation dicts.
        node_a: First node.
        node_b: Second node.

    Returns:
        MI in bits. Higher = stronger dependency.
    """
    n = len(observations)
    if n == 0:
        return 0.0

    # Joint distribution
    counts = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
    for o in observations:
        a = o.get(node_a, False)
        b = o.get(node_b, False)
        key = ("T" if a else "F") + ("T" if b else "F")
        counts[key] += 1

    p_a = (counts["TT"] + counts["TF"]) / n
    p_b = (counts["TT"] + counts["FT"]) / n

    mi = 0.0
    for key, count in counts.items():
        if count == 0:
            continue
        p_joint = count / n
        a_val = key[0] == "T"
        b_val = key[1] == "T"
        p_marginal_a = p_a if a_val else (1 - p_a)
        p_marginal_b = p_b if b_val else (1 - p_b)
        if p_marginal_a > 0 and p_marginal_b > 0:
            mi += p_joint * math.log2(p_joint / (p_marginal_a * p_marginal_b))

    return mi


def compute_energy(
    graph: CausalGraph,
    accuracy_loss: float = 0.0,
    churn_count: int = 0,
    governance_violations: int = 0,
    lambda_complexity: float = 0.1,
    gamma_churn: float = 0.2,
    rho_governance: float = 10.0,
    tau_path: float = 0.05,
) -> GraphEnergy:
    """Compute graph energy function (BOOT-001 §5).

    E = accuracy + λ·complexity + γ·churn + ρ·governance + τ·avg_path_length

    Lower is better. Used to evaluate whether structural changes improve the graph.

    Args:
        graph: Current graph state.
        accuracy_loss: -log p(held-out data | graph). Lower = better fit.
        churn_count: Number of recent structural changes.
        governance_violations: Number of standard violations.
        lambda_complexity: Weight for complexity penalty.
        gamma_churn: Weight for churn penalty.
        rho_governance: Weight for governance violations.
        tau_path: Weight for path length.
    """
    complexity = lambda_complexity * (graph.n_nodes + graph.n_edges)
    churn = gamma_churn * churn_count
    governance = rho_governance * governance_violations

    avg_path = (graph.n_edges / max(1, graph.n_nodes)) * tau_path

    total = accuracy_loss + complexity + churn + governance + avg_path

    return GraphEnergy(
        total=total,
        accuracy=accuracy_loss,
        complexity=complexity,
        churn=churn,
        governance=governance,
        avg_path_length=graph.n_edges / max(1, graph.n_nodes),
    )


def propose_edges(
    graph: CausalGraph,
    observations: list[dict[str, bool]],
    min_correlation: float = 0.3,
    complexity_cost: float = 0.1,
) -> list[EdgeProposal]:
    """Propose new edges based on observed co-occurrence.

    For each unconnected node pair, compute phi coefficient and MI.
    Propose edges where |φ| exceeds threshold.

    Args:
        graph: Current graph (to avoid proposing existing edges).
        observations: Binary observation dicts.
        min_correlation: Minimum |φ| to propose an edge.
        complexity_cost: Energy cost of adding an edge.

    Returns:
        List of EdgeProposal, sorted by MI descending.
    """
    # Get all node pairs without existing edges
    existing_edges = {(e.source, e.target) for e in graph.edges.values()}
    node_ids = list(graph.nodes.keys())

    proposals = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a, b = node_ids[i], node_ids[j]
            if (a, b) in existing_edges or (b, a) in existing_edges:
                continue

            phi = phi_coefficient(observations, a, b)
            if abs(phi) < min_correlation:
                continue

            mi = mutual_information(observations, a, b)
            energy_delta = -mi + complexity_cost
            required_evidence = 0.3 / max(0.01, mi) if mi > 0 else float("inf")

            # Direction: positive phi → a causes b; negative → b inhibits a
            if phi > 0:
                source, target = a, b
                etype = EdgeType.CAUSES
            else:
                source, target = b, a
                etype = EdgeType.INHIBITS

            proposals.append(EdgeProposal(
                source=source,
                target=target,
                correlation=phi,
                mutual_information=mi,
                energy_delta=energy_delta,
                required_evidence=required_evidence,
                edge_type=etype,
            ))

    proposals.sort(key=lambda p: p.mutual_information, reverse=True)
    return proposals


def failure_diversity(
    graph: CausalGraph,
) -> float:
    """Entropy of failure distribution across leaf nodes — INNOVATION.

    High entropy = failures spread across many leaves (structural problem).
    Low entropy = failures concentrated in few leaves (local problem).

    Args:
        graph: The causal graph.

    Returns:
        Normalized entropy [0, 1]. >0.7 = structural, <0.3 = local.
    """
    leaves = graph.get_leaves()
    if len(leaves) < 2:
        return 0.0

    # Use (1 - confidence) as failure probability
    failure_probs = [1 - leaf.confidence for leaf in leaves]
    total = sum(failure_probs)
    if total == 0:
        return 0.0

    # Normalize to distribution
    dist = [p / total for p in failure_probs]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in dist if p > 0)
    max_entropy = math.log2(len(leaves))

    return entropy / max_entropy if max_entropy > 0 else 0.0
