# ForgeSIA

Bayesian causal reasoning engine. Builds directed causal graphs, propagates beliefs through them, learns structure from data, and scores risk. The Synara computation core.

## Install

```bash
pip install forgesia
```

## Quick Start

```python
from forgesia.graph.model import CausalGraph, Node, Edge
from forgesia.inference.belief import apply_evidence, Evidence, EvidenceDirection
from forgesia.risk.scoring import score_risk

# Build a causal graph
g = CausalGraph()
g.add_node(Node(id="contamination", alpha=1, beta=1))
g.add_node(Node(id="off_spec"))
g.add_edge(Edge(id="e1", source="contamination", target="off_spec"))

# Update beliefs with evidence
result = apply_evidence(g, "contamination", Evidence(direction=EvidenceDirection.SUPPORTS))

# Score risk
risk = score_risk(g, "contamination")
print(risk.level, risk.score)
```

## Modules

| Module | Purpose |
|---|---|
| `graph.model` | `CausalGraph`, `Node`, `Edge`, `EdgeType`, `NodeType` |
| `graph.traversal` | Path finding, cycle detection, causal distance, counterfactuals |
| `inference.belief` | Beta-Binomial updates, Bayes rule, credible intervals, information gain |
| `propagation.cpt` | Conditional probability tables, loopy belief propagation |
| `propagation.vine_rot` | Failure propagation with upstream decay |
| `structure.learning` | Phi coefficient, mutual information, edge proposals, energy |
| `risk.scoring` | 4-factor risk model, differential diagnosis, preference filtering |
| `calibration` | Golden reference cases, ForgeCal adapter |

## Dependencies

- Python >= 3.10
- `scipy` >= 1.10 (credible intervals)

## Tests

```bash
python3 -m pytest tests/ -q
```

66 tests covering all modules.

## License

MIT
