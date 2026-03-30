"""ForgeSIA — Bayesian causal reasoning engine.

The Synara computation core. Pure Python. No web framework, no database.

Modules:
    graph       — causal graph data structure, traversal, cycle detection
    inference   — Beta-Binomial updates, evidence weighting, Bayes rule
    propagation — vine rot failure propagation, CPT, belief propagation
    structure   — edge proposals, mutual information, energy, learning
    risk        — risk scoring, differential diagnosis, preference filtering
"""

__version__ = "0.1.0"

__all__ = [
    # graph.model
    "CausalGraph",
    "Edge",
    "EdgeType",
    "Node",
    "NodeType",
    # graph.traversal
    "CausalPath",
    "CycleInfo",
    "CounterfactualResult",
    "find_causal_chains",
    "detect_cycles",
    "causal_distance",
    "counterfactual_impact",
    # inference.belief
    "Evidence",
    "EvidenceType",
    "EvidenceDirection",
    "BeliefUpdate",
    "apply_evidence",
    "bayes_update",
    "suggest_likelihood_ratio",
    "credible_interval",
    "information_gain",
    "temporal_decay",
    # propagation.cpt
    "CPT",
    "CPTEntry",
    "CPTUpdateResult",
    "update_cpt",
    "loopy_belief_propagation",
    # propagation.vine_rot
    "PropagationEffect",
    "PropagationResult",
    "propagate_failure",
    # structure.learning
    "EdgeProposal",
    "GraphEnergy",
    "phi_coefficient",
    "mutual_information",
    "compute_energy",
    "propose_edges",
    "failure_diversity",
    # risk.scoring
    "RiskLevel",
    "PreferenceType",
    "RiskScore",
    "DiagnosticRanking",
    "DiagnosticResult",
    "score_risk",
    "differential_diagnosis",
    "filter_by_preference",
    # calibration
    "calibrate",
    "GOLDEN_CASES",
    "get_calibration_adapter",
]
