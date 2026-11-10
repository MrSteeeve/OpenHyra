"""Continuous-distribution contract evaluation, separate from the historical v1.

Cashflows and Q prices use the existing Stochopia/DSL interpreter. A small
representative subset also runs the quote/submit/account/replay workflow.
Scores measure economic returns; program novelty is reported separately.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import math
import statistics
import time

import torch

from stochopia.adapters.structurer_dsl import (
    DSLContract, conditional_paths, future_value,
)
from stochopia.environment import replay_and_evaluate, run_agent_episode
from stochopia.environment.dsl import StochopiaDSLEnv
from structurerdsl.interp import run
from structurerdsl.path import Paths

from .equity_demands import equity_fixture, normalize_and_check
from .equity_protocol_v2 import (
    demand_spec_v2, interval_check_paths, path_pair_diagnostics,
    protocol_config, scenario_paths,
)


def cash_returns(contract, snapshot, ratios, budget, residual):
    """Research convention: redemption cash stays uninvested through the horizon."""
    grid = Paths(
        spot={contract.program.underlyings[0]: torch.tensor(ratios, dtype=torch.float64)},
        t_years=torch.arange(4, dtype=torch.float64) / 12,
        rate=snapshot.risk_free_rate,
    )
    payments = run(contract.program, grid, contract.params()).cashflows[:, 1:].sum(dim=1)
    return ((payments * contract.notional + residual - budget) / budget).tolist()


def constraint_margins(checks):
    """Failures stay ineligible; margins are feedback for proposing a next design."""
    for item in checks:
        margin = (item["limit"] - item["observed"] if item["name"] == "terminal-loss-limit"
                  else item["observed"] - item["limit"])
        item["margin"] = margin
        item["normalized_violation"] = max(-margin, 0.) / max(abs(item["limit"]), .01)
    return sum(item["normalized_violation"] for item in checks)


def aggregate_returns(values, scenarios, config):
    distributions = {}
    for regime in config["distributions"]:
        selected = [value for value, row in zip(values, scenarios) if row["distribution_id"] == regime["id"]]
        distributions[regime["id"]] = statistics.mean(selected)
    expected = sum(value * row["weight"] for value, row in zip(values, scenarios))
    worst = min(distributions.values())
    return {
        "expected_net_return": expected,
        "worst_distribution_return": worst,
        "distribution_means": distributions,
        "distribution_return_std": {
            regime["id"]: statistics.stdev([value for value, row in zip(values, scenarios)
                if row["distribution_id"] == regime["id"]])
            for regime in config["distributions"]
        },
        # Stratification and antithetic partners invalidate an iid path SE.
        "distribution_standard_errors": {key: None for key in distributions},
        "sampling_uncertainty_scope": "dependent stratified-antithetic paths; report independent held-out gap, not iid SE",
        "score": expected if config["score_mode"] == "mean" else worst,
    }


def _shape(node):
    if isinstance(node, dict):
        return {key: _shape(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_shape(value) for value in node]
    return "number" if isinstance(node, (int, float)) else node


def _unit_price(contract, snapshot, paths, seed):
    unit = replace(contract, notional=1.)
    grid = conditional_paths(unit, snapshot, (snapshot.spot,), n_paths=paths, seed=seed)
    return future_value(unit, grid, 0)


def pricing_audit(contract, snapshot, checks, paths, pricing_seed, seeds):
    primary = checks["cash_outlay"] / contract.notional
    _, primary_se = _unit_price(contract, snapshot, paths, pricing_seed)
    rows = []
    for seed in seeds:
        value, se = _unit_price(contract, snapshot, paths, seed)
        rows.append({"seed": seed, "fair_value_per_unit": value, "mc_se_per_unit": se,
                     "relative_price_gap": value / primary - 1.})
    return {"quote_seed": pricing_seed, "quote_price_per_unit": primary,
            "quote_mc_se_per_unit": primary_se, "independent_estimates": rows,
            "scope": "fixed issued notional repriced under independent Q Monte Carlo draws; diagnostics do not change the trade price or score"}


def _behavior(contract, snapshot, checks, spec, paths, pricing_seed):
    terminals = [.60, .80, .90, .95, 1., 1.05, 1.10, 1.20, 1.40]
    probes = [[value ** (i / 3) for i in range(4)] for value in terminals]
    pairs = path_pair_diagnostics()
    probes += [path for pair in pairs for path in pair["paths"]]
    values = cash_returns(contract, snapshot, probes, spec["investment_budget_cny"], checks["uninvested_cash"])
    grid = Paths(spot={contract.program.underlyings[0]: torch.tensor(probes, dtype=torch.float64)},
                 t_years=torch.arange(4, dtype=torch.float64) / 12, rate=snapshot.risk_free_rate)
    # Unit-notional future payments on fixed public probes; no realized P data.
    payments = run(contract.program, grid, contract.params()).cashflows[:, 1:].tolist()
    names = [f"terminal-{value:g}" for value in terminals]
    names += [f"{pair['name']}-{side}" for pair in pairs for side in (0, 1)]
    seed = DSLContract.from_payload(equity_fixture(spec["family"]))
    seed_price, _ = _unit_price(seed, snapshot, paths, pricing_seed)
    seed = replace(seed, notional=spec["investment_budget_cny"] / seed_price)
    reference = cash_returns(seed, snapshot, probes, spec["investment_budget_cny"], 0.)
    distance = math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(values, reference)))
    return {"behavior_vector": values,
            "public_cashflow_probes": [dict(name=name, spot_ratios=path, cashflows=cash)
                for name, path, cash in zip(names, probes, payments)],
            "innovation": {"ast_shape_same_as_seed": _shape(contract.program.to_json()["body"]) == _shape(seed.program.to_json()["body"]),
                           "behavior_rms_vs_seed": distance,
                           "scope": "fixed public probes and number-erased AST shape; neither is proof of financial novelty",
                           "reward_bonus": 0.},
            "path_dependence_diagnostics": [
                {"name": pair["name"], "return_difference": values[9 + 2*i + 1] - values[9 + 2*i]}
                for i, pair in enumerate(pairs)]}


def evaluate_v2(submission, request, output):
    from .evaluator import ContractPolicy, PROTOCOL, task_suite

    started = time.monotonic()
    config = protocol_config(request.get("eval_config"))
    spec = demand_spec_v2(request["demand_id"], config)
    if request.get("objective", spec["objective"]) != spec["objective"]:
        raise ValueError("objective must match the configured v2 score_mode")
    split = request.get("split", "development")
    paths, pricing_seed = int(request.get("paths", 4096)), int(request.get("pricing_seed", 42))
    _, base = task_suite("public", spec["demand_id"])[0]
    snapshot = base.snapshots[0]
    checks = None
    stage = "syntax-and-parameters"
    try:
        if submission.get("protocol") != PROTOCOL:
            raise ValueError("unsupported structuring artifact protocol")
        if submission.get("generation_error"):
            raise ValueError(submission["generation_error"])
        parsed = DSLContract.from_payload(submission["contract"])
        stage = "demand-constraints"
        normalized, checks = normalize_and_check(parsed, base, spec, paths=paths, pricing_seed=pricing_seed)
        interval_paths = interval_check_paths(spec)
        interval_returns = cash_returns(normalized, snapshot, [p["spot_ratios"] for p in interval_paths],
            spec["investment_budget_cny"], checks["uninvested_cash"])
        for item, value in zip(interval_paths, interval_returns):
            checks["checks"].append({"name": item["name"], "interval_name": item["interval_name"],
                "observed": value, "limit": item["min_net_return"], "passed": value >= item["min_net_return"] - 1e-9})
        violation = constraint_margins(checks["checks"])
        checks.update(passed=all(c["passed"] for c in checks["checks"]), normalized_violation_sum=violation,
                      interval_check_scope=spec["interval_check_scope"])
        if not checks["passed"]:
            raise ValueError("equity demand not met: " + ", ".join(c["name"] for c in checks["checks"] if not c["passed"]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        metrics = {"status": "rejected", "error": str(exc), "rejection_stage": stage, "split": split,
                   "objective": spec["objective"], "demand": spec, "demand_checks": checks,
                   "constraint_violation_sum": checks.get("normalized_violation_sum") if checks else None,
                   "accepted_rate": 0., "episodes": [], "cost": {"evaluation_seconds": time.monotonic() - started}}
        return None, metrics, submission, {"split": split, "demand_protocol": "v2"}

    scenarios = scenario_paths(split, config)
    values = cash_returns(normalized, snapshot, [row["ratios"] for row in scenarios],
                          spec["investment_budget_cny"], checks["uninvested_cash"])
    aggregate = aggregate_returns(values, scenarios, config)
    normalized_payload = normalized.to_payload()

    # One representative from each P distribution checks full account semantics.
    # Cashflows for every scored path were evaluated above, not approximated by
    # these representative episodes.
    representatives = [next(i for i, row in enumerate(scenarios) if row["distribution_id"] == r["id"])
                       for r in config["distributions"]]

    async def replay_representatives():
        rows = []
        for index in representatives:
            row = scenarios[index]
            folder = output / row["name"]
            folder.mkdir(parents=True, exist_ok=True)
            task = replace(base, snapshots=tuple(replace(s, episode_id=row["name"], spot=6000. * ratio)
                for s, ratio in zip(base.snapshots, row["ratios"])), version="synthetic-equity-demand.v2")
            env = StochopiaDSLEnv(task, dsl_paths=paths, dsl_seed=pricing_seed)
            episode = await run_agent_episode(env, ContractPolicy(normalized_payload), trajectory_path=folder / "trajectory.json")
            result = replay_and_evaluate(folder / "trajectory.json", output_path=folder / "evaluation.json", replay=True)
            actual = result["metrics"]["client_flow_adjusted_return"]
            rows.append({"case": row["name"], "status": result["status"], "constraints": result["constraint_summary"],
                         "replay_verified": result["replay_verified"], "batch_return": values[index],
                         "account_return": actual, "cashflow_account_match": math.isclose(actual, values[index], abs_tol=1e-9),
                         "trajectory": str(folder / "trajectory.json"), "infrastructure_error": episode.infrastructure_error})
        return rows

    rows = asyncio.run(replay_representatives())
    eligible = all(row["status"] == "complete" and row["replay_verified"] and row["cashflow_account_match"]
        and row["constraints"]["accepted_submissions"] == 1 and row["constraints"]["invalid_actions"] == 0 for row in rows)
    audits = config.get("pricing_audit_seeds", {}).get(split, [43, 44] if split != "held_out" else [143, 144])
    metrics = {**aggregate, **_behavior(normalized, snapshot, checks, spec, paths, pricing_seed),
        "status": "ok" if eligible else "rejected", "split": split, "objective": spec["objective"],
        "score_mode": config["score_mode"], "demand": spec, "demand_checks": checks, "constraint_violation_sum": 0.,
        "accepted_rate": sum(row["constraints"]["accepted_submissions"] == 1 for row in rows) / len(rows),
        "expected_net_profit_cny": aggregate["expected_net_return"] * spec["investment_budget_cny"],
        "mean_client_return": aggregate["expected_net_return"],
        "pricing_audit": pricing_audit(normalized, snapshot, checks, paths, pricing_seed, audits),
        "episodes": rows, "scored_paths": len(scenarios), "representative_replay_count": len(rows),
        "evaluation_scope": "all P cashflows interpreted in batch; representative quote/submit/account episodes independently replayed",
        "scenario_probabilities": {row["name"]: row["weight"] for row in scenarios},
        "cost": {"evaluation_seconds": time.monotonic() - started, "episodes": len(rows),
                 "pricing_paths_per_quote": paths, "p_cashflow_paths": len(scenarios), "normalization_valuations": 2}}
    score = aggregate["score"] if eligible else None
    metrics["score"] = score
    if not eligible:
        metrics.update(error="representative environment or cashflow-account agreement failed", rejection_stage="environment-replay")
    return score, metrics, {"protocol": PROTOCOL, "contract": normalized_payload}, {
        "split": split, "demand_protocol": "v2", "pricing_seed": pricing_seed}
