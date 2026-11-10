"""Paired study planning; market paths and proposals are never study repeats.

The unit is an independently allocated, fixed-budget pair of research runs.
Pilot and preregistered confirmatory results must remain separate. No adaptive
stopping for significance is supported. Financial novelty is auxiliary and
requires actual human review; it is not the optimization endpoint here.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binom, binomtest


def frontier_event(*, candidate_cost: float, candidate_fulfillment: float,
                   catalog_cost: float, catalog_fulfillment: float,
                   frontier_cost: float, fulfillment_target: float,
                   tolerance: float, candidate_market_id: str,
                   catalog_market_id: str, frontier_market_id: str,
                   frontier_status: str = "optimal", numerical_tolerance: float = 1e-8) -> bool:
    """Candidate closes more of the *same finite-tree* frontier gap by tolerance.

    All costs must refer to the same funding protocol, fixed absolute service
    demand and fulfillment target. For minimum-service-capital funding, cost is
    total upfront funded service capital K = Q contract cost C + required cash
    reserve R; outside capital B-K is excluded from servicing. The legacy
    fixed-budget contract-leg control cannot be mixed with this primary endpoint.
    This establishes an optimization event, not originality or continuous-market
    optimality. A catalog that fails the target does not establish the endpoint.
    """
    values = (candidate_cost, candidate_fulfillment, catalog_cost,
              catalog_fulfillment, frontier_cost, fulfillment_target, tolerance, numerical_tolerance)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Endpoint inputs must be finite")
    if not candidate_market_id or len({candidate_market_id, catalog_market_id,
                                      frontier_market_id}) != 1:
        raise ValueError("Endpoint requires the same finite-tree market")
    if frontier_status != "optimal":
        raise ValueError("Frontier optimum must be certified before scoring")
    if tolerance <= 0 or numerical_tolerance < 0 or not 0 <= fulfillment_target <= 1:
        raise ValueError("Positive preregistered tolerance and valid target required")
    if min(candidate_cost, catalog_cost, frontier_cost) < 0:
        raise ValueError("Costs must be nonnegative")
    if not 0 <= candidate_fulfillment <= 1 or not 0 <= catalog_fulfillment <= 1:
        raise ValueError("Fulfillment must be a probability")
    if catalog_fulfillment < fulfillment_target:
        raise ValueError("Optimized catalog must meet the fixed fulfillment target")
    if candidate_fulfillment < fulfillment_target:
        return False
    if min(candidate_cost, catalog_cost) < frontier_cost - numerical_tolerance:
        raise ValueError("Feasible cost below theoretical frontier: audit the evaluator")
    return (catalog_cost - frontier_cost) - (candidate_cost - frontier_cost) >= tolerance


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")


def _event(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or (type(value) is int and value in (0, 1)):
        return int(value)
    raise ValueError("Events must be reviewed/scored booleans or None; no imputation")


def paired_summary(rows: Sequence[Mapping], alpha: float = .05, *,
                   bootstrap_samples: int = 10000, seed: int = 20261012) -> dict:
    """Summarize one model and phase; rows hold both arms of unique pair_ids.

    Required keys: pair_id, model_group, phase (pilot/confirmatory),
    feedback_event and independent_event. Missing/None events cause attrition,
    never failures. Optional replicate_unit must be research_run_pair. Additional
    fields do not affect scoring. Preregistration is an external requirement;
    this function does not certify that it occurred.
    """
    _validate_alpha(alpha)
    rows = list(rows)
    if not rows:
        raise ValueError("Supply a single model/phase group")
    groups = {(r.get("model_group"), r.get("phase")) for r in rows}
    if len(groups) != 1:
        raise ValueError("Do not mix model groups or pilot/confirmatory phases")
    model, phase = next(iter(groups))
    if not model or phase not in {"pilot", "confirmatory"}:
        raise ValueError("model_group and pilot/confirmatory phase are required")
    ids, complete, missing = set(), [], []
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id is None or pair_id in ids:
            raise ValueError("Unique nonmissing research-run pair_id required")
        ids.add(pair_id)
        if row.get("replicate_unit", "research_run_pair") != "research_run_pair":
            raise ValueError("Candidate or market paths cannot serve as repetitions")
        if row.get("adaptive_stopping", False):
            raise ValueError("Adaptive stopping for significance is disallowed")
        f, i = _event(row.get("feedback_event")), _event(row.get("independent_event"))
        if f is None or i is None:
            missing.append({"pair_id": pair_id, "missing_arms":
                            [arm for arm, v in (("feedback", f), ("independent", i)) if v is None]})
        else:
            complete.append((pair_id, f, i))
    counts = {key: 0 for key in ("n00", "n01", "n10", "n11")}
    for _, f, i in complete:
        counts[f"n{f}{i}"] += 1
    n, discordant = len(complete), counts["n01"] + counts["n10"]
    pvalue = float(binomtest(counts["n10"], discordant, .5,
                            alternative="two-sided").pvalue) if discordant else 1.
    feedback_rate = (counts["n10"] + counts["n11"]) / n if n else None
    independent_rate = (counts["n01"] + counts["n11"]) / n if n else None
    difference, ci = None, None
    if n:
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        # Each array entry is a whole pair_id, preserving within-pair dependence.
        differences = np.array([f - i for _, f, i in complete], dtype=float)
        rng = np.random.default_rng(seed)
        means = np.empty(bootstrap_samples)
        for start in range(0, bootstrap_samples, 512):
            size = min(512, bootstrap_samples - start)
            means[start:start + size] = differences[rng.integers(0, n, (size, n))].mean(axis=1)
        difference = float(differences.mean())
        ci = [float(v) for v in np.quantile(means, [alpha / 2, 1 - alpha / 2])]
    return {"schema": "controlled-paired-study-v1", "model_group": model,
            "phase": phase, "n_attempted": len(rows), "n_complete": n,
            "n_missing": len(missing), "attrition_rate": len(missing) / len(rows),
            "missing_pairs": missing, "joint_counts": counts,
            "discordant_pairs": discordant, "feedback_rate": feedback_rate,
            "independent_rate": independent_rate, "rate_difference": difference,
            "paired_bootstrap_ci": ci, "bootstrap_seed": seed,
            "paired_bootstrap_ci_method": "Approximate percentile bootstrap resampling complete pair ids",
            "paired_bootstrap_ci_degenerate": bool(ci is not None and ci[0] == ci[1]),
            "paired_bootstrap_ci_caveat": "Small or uniform samples can yield a degenerate percentile interval; this does not establish zero population uncertainty. McNemar inference uses the exact test, not this interval.",
            "bootstrap_samples": bootstrap_samples, "alpha": alpha,
            "exact_mcnemar_two_sided_pvalue": pvalue,
            "reject_null": bool(n and pvalue <= alpha),
            "feedback_advantage_supported": bool(n and pvalue <= alpha and difference > 0),
            "confirmatory_phase": phase == "confirmatory",
            "preregistration_verified": False,
            "significance_claim_allowed": False,
            "claim_requirement": "Fixed-N preregistration, real independent run pairs, and multiplicity control must be verified externally",
            "endpoint": "Funded service capital cost closer than optimized catalog to same finite-tree frontier by registered tolerance at fixed absolute demand and fulfillment target under the same funding protocol",
            "novelty": "Auxiliary; unreviewed expert assessments remain missing",
            "replicate_unit": "research_run_pair", "adaptive_stopping_allowed": False}


def paired_summaries(rows: Sequence[Mapping], alpha: float = .05, **kwargs) -> dict:
    """Partition groups; Holm-adjust only confirmatory primary model groups.

    Every group is considered primary; for one prespecified primary model, pass
    only that model's rows. Pilot rows never enter the confirmatory family.
    """
    _validate_alpha(alpha)
    partitions = {}
    for row in rows:
        partitions.setdefault((row.get("model_group"), row.get("phase")), []).append(row)
    summaries = [paired_summary(group, alpha, **kwargs) for group in partitions.values()]
    confirmatory = [s for s in summaries if s["phase"] == "confirmatory"]
    ordered = sorted(confirmatory, key=lambda s: s["exact_mcnemar_two_sided_pvalue"])
    maximum = 0.
    for rank, summary in enumerate(ordered):
        maximum = max(maximum, min(1., (len(ordered) - rank) * summary["exact_mcnemar_two_sided_pvalue"]))
        summary["holm_adjusted_pvalue"] = maximum
        summary["holm_reject_null"] = bool(summary["n_complete"] and maximum <= alpha)
        summary["holm_feedback_advantage_supported"] = bool(summary["holm_reject_null"] and summary["rate_difference"] > 0)
    return {"groups": summaries, "primary_model_groups": [s["model_group"] for s in confirmatory],
            "multiplicity": "Holm across confirmatory primary model groups",
            "pilot_in_primary_family": False}


def _joint(joint_probs):
    if isinstance(joint_probs, Mapping):
        probs = tuple(float(joint_probs[k]) for k in ("p00", "p01", "p10", "p11"))
    else:
        probs = tuple(float(p) for p in joint_probs)
    if len(probs) != 4 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-10):
        raise ValueError("Specify full joint probabilities p00,p01,p10,p11 summing to one")
    return probs


@lru_cache(maxsize=128)
def _critical_values(n, alpha):
    # Exact two-sided binomial test at p=.5 is twice the smaller tail (capped 1).
    d = np.arange(n + 1)
    critical = binom.ppf(alpha / 2, d, .5).astype(int)
    # ppf is the first k with CDF >= alpha/2; equality is rejected too.
    critical -= (2 * binom.cdf(critical, d, .5) > alpha).astype(int)
    return critical


def _power(probs, n, alpha):
    if type(n) is not int or n < 1:
        raise ValueError("Sample sizes must be positive integers")
    discordance = probs[1] + probs[2]
    if discordance == 0:
        return 0.
    d = np.arange(n + 1)
    c = _critical_values(n, alpha)
    theta = probs[2] / discordance
    rejection = np.where(c >= 0, binom.cdf(c, d, theta) + binom.sf(d - c - 1, d, theta), 0.)
    return float(np.dot(binom.pmf(d, n, discordance), rejection))


def power_grid(joint_probs, sample_sizes=(20, 30, 40, 60), alpha=.05) -> dict:
    """Exact planning power: D~Bin(n,p01+p10), n10|D~Bin(D,p10/(p01+p10))."""
    _validate_alpha(alpha)
    probs = _joint(joint_probs)
    return {"joint_probs": dict(zip(("p00", "p01", "p10", "p11"), probs)),
            "alpha": alpha, "method": "exact conditional binomial enumeration",
            "alternative": "two-sided", "unit": "independent research-run pairs",
            "grid": [{"n_pairs": n, "power": _power(probs, n, alpha)} for n in sample_sizes],
            "assumption": "Power is conditional on this full joint table; unknown correlation requires sensitivity analysis",
            "adaptive_stopping_allowed": False}


def choose_n(joint_probs, target_power=.8, *, limit=1000, alpha=.05) -> dict:
    """First fixed sample size reaching target under one explicit joint table."""
    _validate_alpha(alpha)
    if not 0 < target_power < 1 or type(limit) is not int or limit < 1:
        raise ValueError("Valid target_power and positive integer limit required")
    probs = _joint(joint_probs)
    selected, achieved = None, 0.
    for n in range(1, limit + 1):
        achieved = _power(probs, n, alpha)
        if achieved >= target_power:
            selected = n
            break
    return {"n_pairs": selected, "power": achieved, "target_power": target_power,
            "limit": limit, "target_reached": selected is not None,
            "joint_probs": dict(zip(("p00", "p01", "p10", "p11"), probs)),
            "alpha": alpha, "assumptions_verified": False,
            "adaptive_stopping_allowed": False}
