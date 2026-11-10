"""Scientific scoring checks: constraints, continuous P, and independent Q audit."""
import json
import math

import pytest
import torch

from tasks.stochopia_structuring.evaluator import PROTOCOL, evaluate_submission
from tasks.stochopia_structuring.equity_demands import equity_fixture
from tasks.stochopia_structuring.equity_protocol_v2 import protocol_config, scenario_paths
from tasks.stochopia_structuring.evaluator_v2 import aggregate_returns
from tasks.stochopia_structuring.generator import APIContractGenerator, SYSTEM_PROMPT, SYSTEM_PROMPT_V2


@pytest.fixture(autouse=True)
def cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config(mode="mean"):
    return protocol_config({"paths_per_distribution": 8, "score_mode": mode})


def test_mean_and_robust_scores_have_different_economic_order():
    cfg = config()
    cases = scenario_paths("development", cfg)
    risky = [{"central": .10, "bullish": .40, "bearish": -.05}[row["distribution_id"]] for row in cases]
    steady = [.03] * len(cases)
    assert aggregate_returns(risky, cases, cfg)["score"] > aggregate_returns(steady, cases, cfg)["score"]
    cfg["score_mode"] = "worst_distribution"
    assert aggregate_returns(risky, cases, cfg)["score"] < aggregate_returns(steady, cases, cfg)["score"]
    assert all(value is None for value in aggregate_returns(risky, cases, cfg)["distribution_standard_errors"].values())


@pytest.mark.parametrize("demand", ["upside-protected", "two-way-protected", "upside-risk20", "two-way-risk20"])
def test_continuous_batch_matches_authoritative_accounts(tmp_path, demand):
    artifact = {"protocol": PROTOCOL, "contract": equity_fixture("upside" if demand.startswith("upside") else "two-way")}
    score, metrics, normalized, _ = evaluate_submission(artifact, {
        "demand_protocol": "v2", "demand_id": demand, "eval_config": config(), "paths": 128, "output": str(tmp_path)})
    assert score is not None
    assert score == pytest.approx(sum(metrics["distribution_means"].values()) / 3)
    assert metrics["scored_paths"] == 24
    assert metrics["representative_replay_count"] == 3
    assert all(row["replay_verified"] and row["cashflow_account_match"] for row in metrics["episodes"])
    assert metrics["demand_checks"]["cash_outlay"] == pytest.approx(1e6, abs=.001)
    assert len(metrics["behavior_vector"]) == 13
    assert metrics["innovation"]["reward_bonus"] == 0
    assert metrics["innovation"]["ast_shape_same_as_seed"]
    assert metrics["innovation"]["behavior_rms_vs_seed"] == pytest.approx(0, abs=1e-9)
    assert [row["seed"] for row in metrics["pricing_audit"]["independent_estimates"]] == [43, 44]
    assert all(math.isfinite(row["fair_value_per_unit"]) for row in metrics["pricing_audit"]["independent_estimates"])
    assert normalized["contract"]["notional"] != 1e6


def test_interval_check_blocks_point_payoff_and_returns_failure_margin(tmp_path):
    # Same original +10% payoff, but no upside payment at the next interval point.
    artifact = {"protocol": PROTOCOL, "contract": equity_fixture("upside")}
    from structurerdsl.program import Program
    expression = "(when (and (gt (terminal (spot u)) 1.099) (lt (terminal (spot u)) 1.101)) maturity (pay 0.03 maturity))"
    artifact["contract"]["program"] = Program.from_sexpr(
        f"(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) (funded 1 $issue_price {expression}))").to_json()
    score, metrics, _, _ = evaluate_submission(artifact, {"demand_protocol": "v2", "demand_id": "upside-protected",
        "eval_config": config(), "paths": 4096, "output": str(tmp_path)})
    assert score is None and metrics["rejection_stage"] == "demand-constraints"
    checks = metrics["demand_checks"]["checks"]
    assert next(c for c in checks if c["name"] == "up-10")["passed"]
    assert any(not c["passed"] for c in checks if c.get("interval_name") == "participate-upside")
    assert metrics["constraint_violation_sum"] > 0
    assert metrics["episodes"] == []


def test_v2_prompt_is_versioned_without_changing_legacy_provider_prompt():
    generator = APIContractGenerator(base_url="https://example.invalid", model="test", api_key_env="DUMMY")
    old = generator.request_body({"demand": {"schema": "stochopia-equity-demand.v1"}})
    new = generator.request_body({"demand": {"schema": "stochopia-equity-demand.v2"}})
    assert old["messages"][0]["content"] == SYSTEM_PROMPT
    assert new["messages"][0]["content"] == SYSTEM_PROMPT_V2
    assert "numeric scalar" in new["messages"][0]["content"]
    assert "max_tokens" not in old and "max_tokens" not in new


def test_model_comparison_plan_routes_v2_and_independent_without_api():
    from experiments.run_stochopia_model_comparison import comparison_plan
    plan = comparison_plan(models=["claude-haiku-5.5", "deepseek-v4.1-flash"], demands=["two-way-risk20"],
        eval_config=config("worst_distribution"), search_mode="independent")
    assert plan["demand_protocol"] == "v2" and plan["search_mode"] == "independent"
    assert plan["parent_policy"] is None
    assert plan["eval_config"]["score_mode"] == "worst_distribution"
    assert plan["planned_max_api_calls"] == 8
    assert plan["max_tokens_per_call"] is None
    assert "credential_present" in plan["groups"][0]
