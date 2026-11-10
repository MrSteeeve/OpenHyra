"""Independent economic and lifecycle checks for the v2 research evaluator."""
from copy import deepcopy
import math

import pytest
import torch
from structurerdsl.program import Program
from stochopia.adapters.structurer_dsl import DSLContract, DSL_CONTRACT_VERSION

from tasks.stochopia_structuring.evaluator import PROTOCOL, evaluate_submission, task_suite
from tasks.stochopia_structuring import evaluator_v2 as evaluation
from tasks.stochopia_structuring.equity_protocol_v2 import protocol_config, demand_spec_v2


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def artifact(body):
    program = Program.from_sexpr(
        "(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) " + body + ")")
    return {"protocol": PROTOCOL, "contract": {
        "version": DSL_CONTRACT_VERSION, "program": program.to_json(), "notional": 1_000_000}}


def test_expected_and_robust_rankings_follow_distribution_weights():
    config = protocol_config({"distributions": [
        {"id": "good", "probability": .8}, {"id": "bad", "probability": .2}]})
    scenarios = [{"distribution_id": "good", "weight": .4},
                 {"distribution_id": "good", "weight": .4},
                 {"distribution_id": "bad", "weight": .1},
                 {"distribution_id": "bad", "weight": .1}]
    values = [.1, .3, -.2, -.4]
    mean = evaluation.aggregate_returns(values, scenarios, {**config, "score_mode": "mean"})
    robust = evaluation.aggregate_returns(values, scenarios, {**config, "score_mode": "worst_distribution"})
    assert mean["score"] == pytest.approx(.8 * .2 + .2 * -.3)
    assert robust["score"] == pytest.approx(-.3)
    assert robust["expected_net_return"] == mean["expected_net_return"]
    assert mean["distribution_standard_errors"] == {"good": None, "bad": None}


def test_demand_and_config_are_copied_instead_of_mutating_future_requests():
    source = protocol_config({"score_mode": "mean"})
    original = deepcopy(source)
    demand = demand_spec_v2("upside-protected", source)
    demand["protocol_config"]["split_seeds"].pop("held_out")
    demand["p_distribution"]["distributions"][0]["annual_drift"] = 10
    demand["interval_requirements"][0]["min_net_return"] = 99
    assert source == original
    assert demand_spec_v2("upside-protected", source)["interval_requirements"][0]["min_net_return"] == 0


def test_same_terminal_different_history_is_detected_in_executed_cashflows():
    _, task = task_suite("public", "upside-risk20")[0]
    histories = [[1., 1.03, 1.06, 1.10], [1., .75, .92, 1.10]]
    terminal = DSLContract.from_payload(artifact("(premium (pay (terminal (spot u)) maturity))")["contract"])
    asian = DSLContract.from_payload(artifact("(premium (pay (avg_arith (spot u) obs) maturity))")["contract"])
    t_returns = evaluation.cash_returns(terminal, task.snapshots[0], histories, 1_000_000, 0)
    a_returns = evaluation.cash_returns(asian, task.snapshots[0], histories, 1_000_000, 0)
    assert t_returns[0] == pytest.approx(t_returns[1])
    assert a_returns[0] != pytest.approx(a_returns[1])
    assert a_returns[0] - a_returns[1] == pytest.approx((1.03 + 1.06 - .75 - .92) / 3)


def test_syntax_rejection_has_no_economic_score_or_repairs(tmp_path):
    submission = artifact("(premium (pay 1 maturity))")
    submission["contract"]["params"] = {"description": "not a scalar"}
    score, metrics, normalized, _ = evaluate_submission(submission, {
        "demand_protocol": "v2", "demand_id": "upside-risk20", "paths": 32,
        "output": str(tmp_path), "eval_config": {"paths_per_distribution": 4}})
    assert score is None
    assert metrics["status"] == "rejected"
    assert metrics["rejection_stage"] == "syntax-and-parameters"
    assert metrics["episodes"] == [] and metrics["demand_checks"] is None
    assert normalized == submission


def test_implicit_objective_is_mean_and_conflicting_objective_is_rejected(tmp_path):
    assert demand_spec_v2("upside-risk20")["objective"] == "equity-demand-expected-return.v2"
    with pytest.raises(ValueError, match="objective"):
        evaluate_submission(artifact("(premium (pay 1 maturity))"), {
            "demand_protocol": "v2", "demand_id": "upside-risk20",
            "objective": "equity-demand-robust-return.v2", "output": str(tmp_path)})


def test_early_redemption_batch_cash_matches_full_account_replay(tmp_path, monkeypatch):
    # This is a lifecycle fixture, not a claim that a risk-free deposit solves
    # the actual 1% upside demand. Keep loss protection but loosen return asks.
    original_spec = evaluation.demand_spec_v2

    def lifecycle_demand(demand_id, config):
        spec = original_spec(demand_id, config)
        for anchor in spec["anchor_requirements"]:
            anchor["min_net_return"] = -1.
        for interval in spec["interval_requirements"]:
            interval["min_net_return"] = -1.
        return spec

    monkeypatch.setattr(evaluation, "demand_spec_v2", lifecycle_demand)
    submission = artifact("(funded 1 $issue_price (both (pay 1 maturity) (terminate_if (gt (spot u) 0) obs 0)))")
    score, metrics, _, _ = evaluate_submission(submission, {
        "demand_protocol": "v2", "demand_id": "upside-protected", "paths": 32,
        "output": str(tmp_path), "eval_config": {"paths_per_distribution": 4}})
    # The maturity leg is suppressed and principal repays after one month.
    # Early cash stays uninvested, so no extra two-month accrual belongs here.
    assert score == pytest.approx(math.expm1(.02 / 12), abs=1e-9)
    assert metrics["scored_paths"] == 12 and metrics["representative_replay_count"] == 3
    assert all(row["replay_verified"] and row["cashflow_account_match"] for row in metrics["episodes"])
    assert all(row["account_return"] == pytest.approx(score) for row in metrics["episodes"])
    assert metrics["pricing_audit"]["quote_price_per_unit"] == pytest.approx(math.exp(-.02 / 12))
    assert metrics["innovation"]["reward_bonus"] == 0


def test_account_replay_disagreement_cannot_receive_a_search_score(tmp_path, monkeypatch):
    original_spec = evaluation.demand_spec_v2
    original_replay = evaluation.replay_and_evaluate

    def loosen(demand_id, config):
        spec = original_spec(demand_id, config)
        for anchor in spec["anchor_requirements"]:
            anchor["min_net_return"] = -1.
        for interval in spec["interval_requirements"]:
            interval["min_net_return"] = -1.
        return spec

    def disagree(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        result["metrics"]["client_flow_adjusted_return"] += .01
        return result

    monkeypatch.setattr(evaluation, "demand_spec_v2", loosen)
    monkeypatch.setattr(evaluation, "replay_and_evaluate", disagree)
    score, metrics, _, _ = evaluate_submission(artifact("(premium (pay 1 maturity))"), {
        "demand_protocol": "v2", "demand_id": "upside-risk20", "paths": 32,
        "output": str(tmp_path), "eval_config": {"paths_per_distribution": 4}})
    assert score is None and metrics["score"] is None
    assert metrics["status"] == "rejected" and metrics["rejection_stage"] == "environment-replay"
    assert all(not row["cashflow_account_match"] for row in metrics["episodes"])
