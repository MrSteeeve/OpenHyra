"""Development-only parent selection and behavior diagnostics for DSL search.

Behavior distance is a diagnostic and optional parent-selection rule. It never
changes the evaluator-owned score or makes an infeasible candidate eligible.
"""
from __future__ import annotations

from itertools import combinations
from math import sqrt


BEHAVIOR_TOLERANCE = 1e-9


def proposal_brief(round_index, slot, per_round):
    """Separate incumbent refinement from mechanism exploration, without a bonus."""
    index = round_index * per_round + slot
    if index % 2 == 0:
        return {"track": "refine", "instructions":
            "Improve the feasible incumbent's expected return under the fixed demand. "
            "Use constraint margins and independent price diagnostics; explain which "
            "cashflow change improves value, rather than only increasing leverage."}
    mechanisms = (
        "path summaries: average, running extrema, or occupation/count statistics",
        "barrier-conditioned cashflows or automatic termination",
        "a hybrid of path-dependent eligibility and terminal participation",
    )
    return {"track": "structure-exploration", "mechanism_focus": mechanisms[(index // 2) % len(mechanisms)],
        "instructions": "Propose a different economic payoff mechanism using only supported DSL syntax. "
        "A new name, strike, coefficient, or extra tail leverage alone is not a mechanism change. "
        "Explain in rationale the benefit for the supplied demand, a market history where "
        "the behavior differs from a terminal-only combination, and its trade-off. "
        "All original constraints and the expected-return objective still apply. "
        "If this mechanism adds no value, say so; structural complexity earns no reward."}


def behavior_distance(left, right):
    """RMS difference of normalized returns on the same diagnostic paths."""
    if not left or not right or len(left) != len(right):
        return None
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def choose_parent(results, round_index, policy, *, allow_infeasible=False):
    eligible = {cid: result for cid, result in results.items() if result.score is not None}
    if not eligible:
        if allow_infeasible:
            violations = {}
            for cid, result in results.items():
                value = result.metrics.get("constraint_violation_sum")
                if value is None:
                    value = (result.metrics.get("demand_checks") or {}).get("normalized_violation_sum")
                violations[cid] = value if value is not None else float("inf")
            if violations:
                cid = min(violations, key=violations.get)
                value = violations[cid]
                return {"candidate_id": cid, "selection": "least-constraint-violation",
                        "distance": None, "normalized_violation_sum": value if value != float("inf") else None}
        raise RuntimeError("no eligible candidate, including the seed")
    incumbent = max(eligible, key=lambda cid: eligible[cid].score)
    choice = {"candidate_id": incumbent, "selection": "incumbent", "distance": None}
    if policy == "alternating-diverse" and round_index % 2:
        reference = eligible[incumbent].metrics.get("behavior_vector")
        distances = {cid: behavior_distance(reference, result.metrics.get("behavior_vector"))
                     for cid, result in eligible.items() if cid != incumbent}
        distances = {cid: distance for cid, distance in distances.items() if distance is not None}
        if distances and max(distances.values()) > BEHAVIOR_TOLERANCE:
            cid = max(distances, key=distances.get)
            choice = {"candidate_id": cid, "selection": "behavior-distant-feasible", "distance": distances[cid]}
    return choice


def behavior_diagnostics(results):
    vectors = [result.metrics["behavior_vector"] for result in results.values()
               if result.score is not None and result.metrics.get("behavior_vector")]
    distances = [distance for left, right in combinations(vectors, 2)
                 if (distance := behavior_distance(left, right)) is not None]
    representatives = []
    for vector in vectors:
        if not any((distance := behavior_distance(vector, reference)) is not None
                   and distance <= BEHAVIOR_TOLERANCE for reference in representatives):
            representatives.append(vector)
    return {"eligible_vectors": len(vectors), "distinct_vectors": len(representatives),
            "comparable_pairs": len(distances),
            "mean_pairwise_rms_return_distance": sum(distances) / len(distances) if distances else None,
            "max_pairwise_rms_return_distance": max(distances) if distances else None,
            "rms_equivalence_tolerance": BEHAVIOR_TOLERANCE,
            "scope": "numeric probe diversity, not product count",
            "used_in_score": False}
