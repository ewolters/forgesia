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
