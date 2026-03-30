"""Calibration adapter for ForgeSIA.

Golden reference cases for Bayesian causal reasoning:
- Graph traversal (BFS paths, cycle detection)
- Belief updates (Beta conjugate, Bayes rule)
- Propagation (vine rot decay)
- Structure learning (MI, phi coefficient)
- Risk scoring (4-factor model)
"""

from __future__ import annotations


GOLDEN_CASES = [
    {
        "case_id": "CAL-SIA-001",
        "description": "BFS finds chain A→B→C→D",
        "test": "causal_chain",
        "input": {},
        "expected": {"n_paths": 1, "path_length": 4},
    },
    {
        "case_id": "CAL-SIA-002",
        "description": "Diamond graph has 2 paths A→D",
        "test": "diamond_paths",
        "input": {},
        "expected": {"n_paths": 2},
    },
    {
        "case_id": "CAL-SIA-003",
        "description": "Cycle detection finds A→B→C→A",
        "test": "cycle_detection",
        "input": {},
        "expected": {"has_cycle": True},
    },
    {
        "case_id": "CAL-SIA-004",
        "description": "Supporting evidence increases confidence",
        "test": "evidence_update",
        "input": {},
        "expected": {"confidence_increased": True},
    },
    {
        "case_id": "CAL-SIA-005",
        "description": "Bayes update with LR=10 shifts 0.5 → >0.9",
        "test": "bayes_rule",
        "input": {"prior": 0.5, "lr": 10.0},
        "expected": {"posterior_gt": 0.9},
    },
    {
        "case_id": "CAL-SIA-006",
        "description": "Vine rot propagates and decays upstream",
        "test": "vine_rot",
        "input": {},
        "expected": {"propagated": True},
    },
    {
        "case_id": "CAL-SIA-007",
        "description": "CPT update increases probability",
        "test": "cpt_update",
        "input": {},
        "expected": {"prob_increased": True},
    },
    {
        "case_id": "CAL-SIA-008",
        "description": "Phi coefficient > 0.3 for correlated nodes",
        "test": "phi_coefficient",
        "input": {},
        "expected": {"phi_gt": 0.3},
    },
    {
        "case_id": "CAL-SIA-009",
        "description": "Risk scoring: high-confidence node with evidence → HIGH/CRITICAL",
        "test": "risk_score",
        "input": {},
        "expected": {"score_gt": 0.5},
    },
    {
        "case_id": "CAL-SIA-010",
        "description": "Temporal decay: old evidence drifts toward 0.5",
        "test": "temporal_decay",
        "input": {},
        "expected": {"decayed": True},
    },
    {
        "case_id": "CAL-SIA-011",
        "description": "Information gain: uncertain node has higher gain",
        "test": "info_gain",
        "input": {},
        "expected": {"uncertain_higher": True},
    },
    {
        "case_id": "CAL-SIA-012",
        "description": "Credible interval: Beta(80,20) 95% CI lower > 0.7",
        "test": "credible_interval",
        "input": {"alpha": 80, "beta": 20},
        "expected": {"ci_lower_gt": 0.7},
    },
]


def calibrate():
    """Run all golden reference cases. Standalone entry point."""
    results = []
    for case in GOLDEN_CASES:
        case_id = case["case_id"]
        test = case["test"]
        inp = case["input"]
        exp = case["expected"]
        try:
            actual = _run_case(case_id, test, inp)
            passed = _check_case(actual, exp)
            results.append({"case_id": case_id, "passed": passed, "actual": actual})
        except Exception as e:
            results.append({"case_id": case_id, "passed": False, "error": str(e)})

    passed = sum(1 for r in results if r["passed"])
    return {
        "package": "forgesia",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "is_calibrated": passed == len(results),
    }


def _run_case(case_id: str, test: str, inp: dict) -> dict:
    from .graph.model import CausalGraph, Edge, EdgeType, Node, NodeType
    from .graph.traversal import find_causal_chains, detect_cycles
    from .inference.belief import (
        apply_evidence, bayes_update, credible_interval,
        information_gain, temporal_decay,
        Evidence, EvidenceDirection,
    )
    from .propagation.vine_rot import propagate_failure
    from .propagation.cpt import CPT, CPTEntry, update_cpt
    from .structure.learning import phi_coefficient
    from .risk.scoring import score_risk

    if test == "causal_chain":
        g = CausalGraph()
        for n in ["A", "B", "C", "D"]:
            g.add_node(Node(id=n))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        g.add_edge(Edge(id="e2", source="B", target="C"))
        g.add_edge(Edge(id="e3", source="C", target="D"))
        paths = find_causal_chains(g, "A", "D")
        return {"n_paths": len(paths), "path_length": len(paths[0].nodes) if paths else 0}

    elif test == "diamond_paths":
        g = CausalGraph()
        for n in ["A", "B", "C", "D"]:
            g.add_node(Node(id=n))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        g.add_edge(Edge(id="e2", source="A", target="C"))
        g.add_edge(Edge(id="e3", source="B", target="D"))
        g.add_edge(Edge(id="e4", source="C", target="D"))
        paths = find_causal_chains(g, "A", "D")
        return {"n_paths": len(paths)}

    elif test == "cycle_detection":
        g = CausalGraph()
        for n in ["A", "B", "C"]:
            g.add_node(Node(id=n))
        g.add_edge(Edge(id="e1", source="A", target="B"))
        g.add_edge(Edge(id="e2", source="B", target="C"))
        g.add_edge(Edge(id="e3", source="C", target="A"))
        cycles = detect_cycles(g)
        return {"has_cycle": len(cycles) > 0}

    elif test == "evidence_update":
        g = CausalGraph()
        g.add_node(Node(id="h1", alpha=1, beta=1))
        result = apply_evidence(g, "h1", Evidence(direction=EvidenceDirection.SUPPORTS))
        return {"confidence_increased": result.confidence_after > result.confidence_before}

    elif test == "bayes_rule":
        posterior = bayes_update(inp["prior"], inp["lr"])
        return {"posterior": posterior}

    elif test == "vine_rot":
        g = CausalGraph()
        g.add_node(Node(id="root", alpha=5, beta=1))
        g.add_node(Node(id="leaf", alpha=5, beta=1))
        g.add_edge(Edge(id="e1", source="root", target="leaf"))
        result = propagate_failure(g, "leaf")
        return {"propagated": len(result.effects) >= 1}

    elif test == "cpt_update":
        cpt = CPT(child_var="x", parent_vars=["a"], child_states=["yes", "no"],
                   entries=[CPTEntry(parent_config=("t",), child_state="yes"),
                            CPTEntry(parent_config=("t",), child_state="no")])
        result = update_cpt(cpt, ("t",), "yes", weight=5)
        return {"prob_increased": result.prob_after > result.prob_before}

    elif test == "phi_coefficient":
        import random
        random.seed(42)
        obs = []
        for _ in range(100):
            a = random.random() < 0.6
            b = a if random.random() < 0.8 else (not a)
            obs.append({"A": a, "B": b})
        phi = phi_coefficient(obs, "A", "B")
        return {"phi": phi}

    elif test == "risk_score":
        g = CausalGraph()
        g.add_node(Node(id="cause", node_type=NodeType.CAUSE, alpha=8, beta=2))
        for i in range(3):
            g.add_node(Node(id=f"ev{i}", node_type=NodeType.EVIDENCE))
            g.add_edge(Edge(id=f"sup{i}", source=f"ev{i}", target="cause", edge_type=EdgeType.SUPPORTS))
        for i in range(3):
            g.add_node(Node(id=f"eff{i}"))
            g.add_edge(Edge(id=f"cas{i}", source="cause", target=f"eff{i}", edge_type=EdgeType.CAUSES))
        risk = score_risk(g, "cause")
        return {"score": risk.score}

    elif test == "temporal_decay":
        n = Node(id="h1", alpha=10, beta=2)
        conf_new = temporal_decay(n, days_since_evidence=0)
        conf_old = temporal_decay(n, days_since_evidence=365)
        return {"decayed": conf_old < conf_new}

    elif test == "info_gain":
        uncertain = Node(id="u", alpha=1, beta=1)
        certain = Node(id="c", alpha=100, beta=1)
        g_u = information_gain(uncertain, likelihood_ratio=5.0)
        g_c = information_gain(certain, likelihood_ratio=5.0)
        return {"uncertain_higher": g_u > g_c}

    elif test == "credible_interval":
        lo, hi = credible_interval(inp["alpha"], inp["beta"])
        return {"ci_lower": lo}

    raise ValueError(f"Unknown test: {test}")


def _check_case(actual: dict, expected: dict) -> bool:
    for key, val in expected.items():
        if key.endswith("_gt"):
            if actual.get(key[:-3], 0) <= val:
                return False
        elif key.endswith("_lt"):
            if actual.get(key[:-3], 0) >= val:
                return False
        elif isinstance(val, bool):
            if actual.get(key) != val:
                return False
        elif isinstance(val, (int, float)):
            actual_val = actual.get(key)
            if actual_val is None or abs(float(actual_val) - val) > 0.01:
                return False
    return True


def get_calibration_adapter():
    """ForgeCal adapter protocol."""
    try:
        from forgecal.core import CalibrationAdapter, CalibrationCase, Expectation
    except ImportError:
        return None

    cases = []
    for gc in GOLDEN_CASES:
        expectations = []
        for key, val in gc["expected"].items():
            if key.endswith("_gt"):
                expectations.append(Expectation(key=key[:-3], expected=val, comparison="greater_than"))
            elif key.endswith("_lt"):
                expectations.append(Expectation(key=key[:-3], expected=val, comparison="less_than"))
            elif isinstance(val, bool):
                expectations.append(Expectation(key=key, expected=val, comparison="equals"))
            elif isinstance(val, str):
                expectations.append(Expectation(key=key, expected=val, comparison="equals"))
            else:
                expectations.append(Expectation(key=key, expected=val, tolerance=0.01, comparison="abs_within"))
        cases.append(CalibrationCase(
            case_id=gc["case_id"],
            package="forgesia",
            category="causal_reasoning",
            analysis_type="synara",
            analysis_id=gc["test"],
            config=gc["input"],
            data={},
            expectations=expectations,
            description=gc["description"],
        ))

    def _run(case):
        gc = next(g for g in GOLDEN_CASES if g["case_id"] == case.case_id)
        return _run_case(case.case_id, gc["test"], gc["input"])

    from forgesia import __version__
    return CalibrationAdapter(package="forgesia", version=__version__, cases=cases, runner=_run)
