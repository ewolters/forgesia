# ForgeSIA -- Bayesian Causal Reasoning Engine

## What It Is

Pure-Python causal graph engine for the Forge ecosystem. Nodes carry Beta-distributed beliefs (alpha/beta parameters). Evidence updates shift those beliefs via conjugate updating. Edges encode causal, evidential, or temporal relationships. Used by Synara (SVEND's analysis engine) for root cause analysis.

## Architecture

```
forgesia/
  graph/model.py       -- CausalGraph, Node, Edge dataclasses
  graph/traversal.py   -- BFS path finding, cycle detection, counterfactuals
  inference/belief.py  -- Beta-Binomial conjugate updates, Bayes rule
  propagation/cpt.py   -- Conditional probability tables, loopy BP
  propagation/vine_rot.py -- Failure propagation with decay
  structure/learning.py -- Phi coefficient, MI, edge proposals
  risk/scoring.py      -- 4-factor risk model, differential diagnosis
  calibration.py       -- Golden cases + ForgeCal adapter
```

## Running Tests

```bash
cd ~/forgesia
python3 -m pytest tests/ -q          # 66 tests
python3 -m ruff check .              # lint
```

## Key Design Decisions

- **Stdlib only** -- scipy is the sole external dep (credible intervals only). No numpy.
- **Beta-Binomial conjugate model** -- alpha/beta on each node. Evidence increments alpha (supports) or beta (contradicts). Simple, interpretable, analytically tractable.
- **No mutation of graph during propagation** -- propagation functions return result objects, not modified graphs.
- **Calibration built-in** -- `calibration.py` has 12 golden cases that double as regression tests and ForgeCal integration.
- **Vine rot propagation** -- failure at a leaf propagates upstream with configurable decay. Models cascading failure in manufacturing process chains.
- **Loopy BP** -- iterative message passing for graphs with cycles. Convergence not guaranteed but bounded by max iterations.
