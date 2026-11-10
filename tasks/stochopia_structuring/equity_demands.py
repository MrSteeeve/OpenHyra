"""Four public equity payoff demands; no simulated investor or negotiation."""
from __future__ import annotations

from dataclasses import replace

from stochopia.adapters.structurer_dsl import DSLContract, valuation
from structurerdsl.path import Paths
from structurerdsl.interp import run
from structurerdsl.program import Program
import torch


DEMAND_OBJECTIVE = "equity-demand-expected-return.v1"
DEMAND_IDS = ("upside-protected", "upside-risk20", "two-way-protected", "two-way-risk20")

# Equal probabilities define a finite synthetic P distribution, not market data.
# Every scenario starts from the same pricing snapshot. Q pricing stays GBM.
SCENARIOS = {
    "public": [("example", (1., 1.03, 1.06, 1.10))],
    "development": [
        ("dev-up", (1., 1.03, 1.06, 1.10)),
        ("dev-flat", (1., .98, 1.02, 1.)),
        ("dev-mild-down", (1., .97, .94, .90)),
        ("dev-rally", (1., 1.08, 1.15, 1.20)),
        ("dev-deep-down", (1., .90, .80, .65)),
        ("dev-rebound", (1., .80, .95, 1.10)),
        ("dev-round-trip", (1., 1.15, 1.05, .90)),
    ],
    "held_out": [
        ("hold-up", (1., 1.01, 1.09, 1.12)),
        ("hold-flat", (1., 1.06, .95, 1.01)),
        ("hold-mild-down", (1., .94, .98, .88)),
        ("hold-rally", (1., 1.02, 1.18, 1.25)),
        ("hold-deep-down", (1., .95, .70, .60)),
        ("hold-rebound", (1., .75, 1.03, 1.12)),
        ("hold-round-trip", (1., 1.20, 1.02, .88)),
    ],
}


def demand_spec(demand_id):
    if demand_id not in DEMAND_IDS:
        raise ValueError(f"unknown equity demand: {demand_id}")
    family = "upside" if demand_id.startswith("upside") else "two-way"
    protected = demand_id.endswith("protected")
    # These disclosed paths define finite anchor checks, not all-path shape proofs.
    anchors = [{"name": "up-10", "spot_ratios": [1., 1.03, 1.06, 1.10], "min_net_return": .01}]
    if family == "upside":
        anchors += [
            {"name": "flat", "spot_ratios": [1., .98, 1.02, 1.], "min_net_return": 0.},
            {"name": "down-5", "spot_ratios": [1., .98, .96, .95], "min_net_return": 0.},
            {"name": "down-10", "spot_ratios": [1., .97, .94, .90], "min_net_return": 0.},
        ]
    else:
        anchors += [{"name": "down-10", "spot_ratios": [1., .97, .94, .90], "min_net_return": .01}]
    return {
        "schema": "stochopia-equity-demand.v1", "demand_id": demand_id,
        "underlying_domain": "equity", "family": family,
        "description": "温和回撤情景收回投入，并参与上涨" if family == "upside" else "温和上涨和下跌情景均获得收益",
        "investment_budget_cny": 1_000_000., "horizon_months": 3,
        "principal_protection_required": protected,
        "max_terminal_principal_loss": 0. if protected else .20,
        "nonprotected_means": "loss is permitted up to the limit, not required",
        "anchor_requirements": anchors,
        "p_distribution": {"kind": "finite-synthetic-equity-scenarios", "probabilities": "equal within each split",
                           "regimes": ["up", "flat", "mild-down", "rally", "deep-down", "rebound", "round-trip"],
                           "development_cases": 7, "held_out_cases": 7, "market_calibrated": False},
        "cash_convention": "sum cash received through horizon; early redemption cash stays uninvested",
        "objective": DEMAND_OBJECTIVE,
    }


def demand_tasks(base, split, demand_id):
    spec = demand_spec(demand_id)
    # Reuse the environment's account/quote interface. This mechanical profile
    # supplies no persona or utility. Terminal loss is checked separately below;
    # the legacy max_loss_pct also measures interim stress and is not that limit.
    client = replace(base.client, id="equity-demand", name="Equity Demand",
        capital=spec["investment_budget_cny"], min_return_pct=0., min_hit_prob=0.,
        max_loss_pct=1., risk_appetite="moderate", preferences="", current_focus="",
        max_maturity_months=3, principal_protection_required=spec["principal_protection_required"])
    return [(name, replace(base, client=client,
        snapshots=tuple(replace(s, episode_id=name, spot=6000. * ratio, realized_vol_20d=.20)
                        for s, ratio in zip(base.snapshots, ratios)),
        version="synthetic-equity-demand.v1", task_seed=100 + i))
        for i, (name, ratios) in enumerate(SCENARIOS[split])]


def normalize_and_check(contract, task, spec, *, paths, pricing_seed):
    """Fix actual purchase budget, then check payoff bounds and public anchors."""
    if contract.program.maturity != spec["horizon_months"]:
        raise ValueError("equity demand requires exactly 3 months")
    snapshot = task.snapshots[0]
    unit = replace(contract, notional=1.)
    fair_per_unit = valuation(unit, snapshot, (snapshot.spot,), n_paths=paths, seed=pricing_seed)["fair_value"]
    if fair_per_unit <= 0:
        raise ValueError("equity demand requires positive fair value")
    budget = spec["investment_budget_cny"]
    # A sub-cent margin avoids a repeated MC multiplication exceeding cash by
    # floating-point roundoff. It is accounted for as uninvested cash below.
    normalized = replace(contract, notional=budget * (1. - 1e-10) / fair_per_unit)
    cash_outlay = valuation(normalized, snapshot, (snapshot.spot,), n_paths=paths, seed=pricing_seed)["fair_value"]
    residual = budget - cash_outlay
    floor = normalized.guaranteed_ratio * normalized.notional
    loss = max(cash_outlay - floor, 0.) / cash_outlay
    checks = [{"name": "terminal-loss-limit", "observed": loss,
               "limit": spec["max_terminal_principal_loss"],
               "passed": loss <= spec["max_terminal_principal_loss"] + 1e-9}]
    for anchor in spec["anchor_requirements"]:
        name = normalized.program.underlyings[0]
        grid = Paths(spot={name: torch.tensor([anchor["spot_ratios"]], dtype=torch.float64)},
                     t_years=torch.arange(4, dtype=torch.float64) / 12, rate=snapshot.risk_free_rate)
        result = run(normalized.program, grid, normalized.params())
        payments = float(result.cashflows[:, 1:].sum()) * normalized.notional
        net_return = (payments + residual - budget) / budget
        checks.append({"name": anchor["name"], "observed": net_return,
                       "limit": anchor["min_net_return"],
                       "passed": net_return >= anchor["min_net_return"] - 1e-9})
    return normalized, {
        "demand_id": spec["demand_id"], "passed": all(c["passed"] for c in checks), "checks": checks,
        "investment_budget_cny": budget, "cash_outlay": cash_outlay, "uninvested_cash": residual,
        "proposed_notional": contract.notional, "normalized_notional": normalized.notional,
        "guaranteed_payment_floor_cny": floor, "terminal_loss_bound": loss,
        "floor_method": "existing conservative DSL payment bound; not a general verifier proof",
    }


def equity_fixture(family, variant=0):
    """Handwritten controls only; offline proposals are not model designs."""
    if family == "upside":
        expression = "(mul_const 0.105 (clip (lin 1 (terminal (spot u)) -1) 0 0.10))"
        if variant == 1:
            expression = "(mul_const 0.08 (max (lin 1 (terminal (spot u)) -1) 0))"
        elif variant == 2:
            expression = "(mul_const 0.11 (clip (lin 1 (avg_arith (spot u) obs) -0.98) 0 0.12))"
        elif variant == 3:
            expression = "(mul_const 0.30 (clip (lin 1 (terminal (spot u)) -1) 0 0.10))"
        body = f"(funded 1 $issue_price (pay {expression} maturity))"
        if variant == 3:
            body = f"(funded 1 $issue_price (both (pay {expression} maturity) (give (pay (clip (lin -1 (terminal (spot u)) 0.9) 0 0.19) maturity))))"
    else:
        threshold, coupon = ((.095, .012), (.08, .012), (.10, .014), (.095, .011))[variant]
        body = f"(funded 1 $issue_price (when (or (gt (terminal (spot u)) {1 + threshold}) (lt (terminal (spot u)) {1 - threshold})) maturity (pay {coupon} maturity)))"
        if variant == 3:
            body = f"(funded 1 $issue_price (both (when (or (gt (terminal (spot u)) {1 + threshold}) (lt (terminal (spot u)) {1 - threshold})) maturity (pay {coupon} maturity)) (give (pay (clip (lin -1 (terminal (spot u)) 0.8) 0 0.19) maturity))))"
    program = Program.from_sexpr(f"(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) {body})")
    from stochopia.adapters.structurer_dsl import DSL_CONTRACT_VERSION
    return {"version": DSL_CONTRACT_VERSION, "program": program.to_json(), "notional": 1_000_000.}
