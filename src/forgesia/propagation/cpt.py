"""Conditional Probability Tables — Dirichlet-Multinomial updates + belief propagation.

CPTs encode P(child_state | parent_states). Updated from observations
using the Dirichlet-Multinomial conjugate model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CPTEntry:
    """One row in a CPT: a specific parent config → child state."""

    parent_config: tuple  # tuple of parent state values
    child_state: str
    alpha: float = 1.0  # Dirichlet prior pseudo-count
    count: float = 0.0  # observed count

    @property
    def probability(self) -> float:
        return (self.alpha + self.count)  # raw — normalized by CPT.get_probability


@dataclass
class CPT:
    """Conditional Probability Table for one child node."""

    child_var: str
    parent_vars: list[str]
    child_states: list[str]
    entries: list[CPTEntry] = field(default_factory=list)

    def __post_init__(self):
        self._index: dict[tuple[tuple, str], int] = {}
        for i, entry in enumerate(self.entries):
            self._index[(entry.parent_config, entry.child_state)] = i

    def get_probability(self, parent_config: tuple, child_state: str) -> float:
        """P(child_state | parent_config), normalized."""
        total = sum(
            e.alpha + e.count
            for e in self.entries
            if e.parent_config == parent_config
        )
        idx = self._index.get((parent_config, child_state))
        if idx is None or total == 0:
            return 1.0 / max(1, len(self.child_states))  # uniform fallback
        entry = self.entries[idx]
        return (entry.alpha + entry.count) / total


@dataclass
class CPTUpdateResult:
    """Result of updating a CPT from an observation."""

    child_var: str
    parent_config: tuple
    child_state: str
    prob_before: float
    prob_after: float
    total_observations: float
    confidence: float  # how sure we are about this distribution


def update_cpt(
    cpt: CPT,
    parent_config: tuple,
    child_state: str,
    weight: float = 1.0,
    max_count: float = 10000,
) -> CPTUpdateResult:
    """Update CPT from an observation (Dirichlet-Multinomial conjugate).

    Args:
        cpt: The CPT to update.
        parent_config: Observed parent state configuration.
        child_state: Observed child state.
        weight: Observation weight (default 1).
        max_count: Maximum accumulated count per entry.

    Returns:
        CPTUpdateResult with before/after probabilities.
    """
    prob_before = cpt.get_probability(parent_config, child_state)

    key = (parent_config, child_state)
    idx = cpt._index.get(key)

    if idx is None:
        # New entry
        entry = CPTEntry(parent_config=parent_config, child_state=child_state)
        cpt.entries.append(entry)
        idx = len(cpt.entries) - 1
        cpt._index[key] = idx

    entry = cpt.entries[idx]
    bounded_weight = min(weight, max_count - entry.count)
    entry.count += max(0, bounded_weight)

    prob_after = cpt.get_probability(parent_config, child_state)

    total_obs = sum(e.alpha + e.count for e in cpt.entries if e.parent_config == parent_config)
    confidence = min(0.99, total_obs / (total_obs + 10))

    return CPTUpdateResult(
        child_var=cpt.child_var,
        parent_config=parent_config,
        child_state=child_state,
        prob_before=prob_before,
        prob_after=prob_after,
        total_observations=total_obs,
        confidence=confidence,
    )


def loopy_belief_propagation(
    cpts: dict[str, CPT],
    observations: dict[str, str],
    max_iterations: int = 100,
    convergence_threshold: float = 0.001,
) -> dict[str, dict[str, float]]:
    """Loopy belief propagation over a set of CPTs.

    Approximate marginal inference by iterative message passing.

    Args:
        cpts: {child_var: CPT} for each variable with parents.
        observations: {var_name: observed_state} for clamped variables.
        max_iterations: Max BP iterations.
        convergence_threshold: Stop when max belief change < this.

    Returns:
        {var_name: {state: probability}} marginal beliefs for each variable.
    """
    # Collect all variables and their states
    all_vars: dict[str, list[str]] = {}
    for cpt in cpts.values():
        if cpt.child_var not in all_vars:
            all_vars[cpt.child_var] = cpt.child_states
        for pv in cpt.parent_vars:
            if pv not in all_vars:
                all_vars[pv] = ["true", "false"]  # default binary

    # Initialize beliefs to uniform (or clamped)
    beliefs: dict[str, dict[str, float]] = {}
    for var, states in all_vars.items():
        if var in observations:
            beliefs[var] = {s: 1.0 if s == observations[var] else 0.0 for s in states}
        else:
            n = len(states)
            beliefs[var] = {s: 1.0 / n for s in states}

    # Iterative update
    for iteration in range(max_iterations):
        max_change = 0.0

        for cpt in cpts.values():
            if cpt.child_var in observations:
                continue  # don't update clamped variables

            new_belief = {s: 0.0 for s in cpt.child_states}

            # Enumerate parent configurations
            parent_configs = _enumerate_configs(cpt.parent_vars, all_vars, beliefs)

            for config, config_prob in parent_configs:
                for state in cpt.child_states:
                    p = cpt.get_probability(config, state)
                    new_belief[state] += p * config_prob

            # Normalize
            total = sum(new_belief.values())
            if total > 0:
                new_belief = {s: v / total for s, v in new_belief.items()}

            # Check convergence
            for s in cpt.child_states:
                change = abs(new_belief[s] - beliefs.get(cpt.child_var, {}).get(s, 0))
                max_change = max(max_change, change)

            beliefs[cpt.child_var] = new_belief

        if max_change < convergence_threshold:
            break

    return beliefs


def _enumerate_configs(
    parent_vars: list[str],
    all_vars: dict[str, list[str]],
    beliefs: dict[str, dict[str, float]],
) -> list[tuple[tuple, float]]:
    """Enumerate all parent state configurations with their joint probability."""
    if not parent_vars:
        return [((), 1.0)]

    configs = [((), 1.0)]
    for pv in parent_vars:
        states = all_vars.get(pv, ["true", "false"])
        new_configs = []
        for config, prob in configs:
            for state in states:
                state_prob = beliefs.get(pv, {}).get(state, 1.0 / len(states))
                new_configs.append((config + (state,), prob * state_prob))
        configs = new_configs

    return configs
