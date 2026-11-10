"""Finite-grid optimization of familiar contract families under the task score.

These controls use exactly the model candidate's evaluator and constraints.
They are not a global optimum, an innovation claim, or generation feedback.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product, zip_longest

from stochopia.adapters.structurer_dsl import DSL_CONTRACT_VERSION
from structurerdsl.program import Program

from .equity_demands import demand_spec


BASELINE_SCOPE = "finite grid optimized standard families, not global optimum"
FAMILIES = ("capped-call-spread", "single-two-way-digital",
            "double-two-way-digital", "funded-downside-put-combination")


def baseline_config(overrides=None):
    config = {
        "schema": "stochopia-equity-standard-baselines.v1",
        "max_evaluations": 24,
        "parameters": {
            "capped-call-spread": {"participation": [.05, .105, .20], "cap": [.10, .20]},
            "single-two-way-digital": {"threshold": [.05, .095, .12], "coupon": [.012, .62]},
            "double-two-way-digital": {"inner_threshold": [.08, .095, .12],
                "coupon_pairs": [[.02, .02], [.62, .50]], "outer_threshold": .19},
            "funded-downside-put-combination": {"put_participation": [.10, .28],
                "coupon": [.01, .04, .10], "put_strike": .90, "threshold": .095,
                "call_cap": .10},
        },
        "scope": BASELINE_SCOPE,
        "feed_to_generator": False,
    }
    if overrides:
        config.update(deepcopy(overrides))
    return config


def _digital(threshold, coupon):
    spot = "(terminal (spot u))"
    return f"(when (or (gt {spot} {1 + threshold}) (lt {spot} {1 - threshold})) maturity (pay {coupon} maturity))"


def _call(participation, cap):
    return f"(pay (mul_const {participation} (clip (lin 1 (terminal (spot u)) -1) 0 {cap})) maturity)"


def _family_bodies(family, demand_family, params):
    if family == "capped-call-spread":
        return [_call(part, cap) for part, cap in product(params["participation"], params["cap"])]
    if family == "single-two-way-digital":
        return [_digital(threshold, coupon) for threshold, coupon in product(params["threshold"], params["coupon"])]
    if family == "double-two-way-digital":
        return [f"(both {_digital(threshold, coupons[0])} {_digital(params['outer_threshold'], coupons[1])})"
                for threshold, coupons in product(params["inner_threshold"], params["coupon_pairs"])]
    output = []
    for part, coupon in product(params["put_participation"], params["coupon"]):
        income = (_call(coupon, params["call_cap"])
                  if demand_family == "upside" else _digital(params["threshold"], coupon))
        put = f"(pay (mul_const {part} (max (lin -1 (terminal (spot u)) {params['put_strike']}) 0)) maturity)"
        output.append(f"(both {income} {put})")
    return output


def baseline_candidates(demand_id, config=None):
    """Round-robin standard families before truncating the fixed evaluation budget."""
    config = baseline_config(config)
    if config["max_evaluations"] < 1:
        raise ValueError("standard baseline budget must be positive")
    family_lists = []
    demand_family = demand_spec(demand_id)["family"]
    for family in FAMILIES:
        entries = []
        bodies = _family_bodies(family, demand_family, config["parameters"][family])
        for index, body in enumerate(bodies):
            sexpr = f"(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) (funded 1 $issue_price {body}))"
            artifact = {"protocol": "stochopia-structuring-search.v1", "contract": {
                "version": DSL_CONTRACT_VERSION, "program": Program.from_sexpr(sexpr).to_json(),
                "notional": 1_000_000., "pitch": BASELINE_SCOPE}}
            entries.append({"candidate_id": f"standard-{family}-{index:03d}",
                            "family": family, "artifact": artifact})
        family_lists.append(entries)
    ordered = [candidate for row in zip_longest(*family_lists) for candidate in row if candidate is not None]
    return ordered[:config["max_evaluations"]]


def _evaluation_fields(result):
    if isinstance(result, tuple):  # evaluate_submission's native return value.
        score, metrics = result[:2]
    elif isinstance(result, dict):
        score, metrics = result.get("score"), result.get("metrics", result)
    else:  # Harness EvaluationResult.
        return result.score, result.status, result.metrics
    return score, metrics.get("status", "ok" if score is not None else "rejected"), metrics


def optimize_standard_baselines(demand_id, evaluate, config=None):
    """Apply the caller's identical development evaluator; select only eligible scores."""
    config = baseline_config(config)
    attempts, best, by_family = [], None, {family: None for family in FAMILIES}
    for candidate in baseline_candidates(demand_id, config):
        score, status, metrics = _evaluation_fields(evaluate(candidate["artifact"]))
        eligible = score is not None and status in ("ok", "eligible", "accepted")
        row = {**candidate, "score": score, "status": status, "eligible": eligible,
               "constraints": metrics.get("demand_checks"), "metrics": metrics}
        attempts.append(row)
        if eligible:
            incumbent = by_family[candidate["family"]]
            if incumbent is None or score > incumbent["score"]:
                by_family[candidate["family"]] = row
            if best is None or score > best["score"]:
                best = row
    return {"schema": config["schema"], "scope": BASELINE_SCOPE, "demand_id": demand_id,
            "config": config, "budget": config["max_evaluations"], "evaluations": len(attempts),
            "attempts": attempts, "best": best, "best_by_family": by_family,
            "feed_to_generator": False}
