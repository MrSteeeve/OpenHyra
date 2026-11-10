"""Integration checks for the independent research task, not Bermudan metrics."""
import json
import httpx
from pathlib import Path

import pytest
import torch

from tasks.stochopia_structuring.evaluator import PROTOCOL, evaluate_submission, public_condition, task_suite
from tasks.stochopia_structuring.generator import APIContractGenerator, FixtureGenerator, parse_generation
from experiments.run_stochopia_structuring import run_search
from examples.dsl_roundtrip import fixture_payload
from tasks.stochopia_structuring.equity_demands import DEMAND_IDS, DEMAND_OBJECTIVE, demand_spec, equity_fixture
from tasks.stochopia_structuring.generator import EquityFixtureGenerator


@pytest.fixture(autouse=True)
def cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_public_condition_and_authoritative_score(tmp_path):
    _, task = task_suite("development")[0]
    condition = public_condition(task, paths=128)
    assert set(condition) == {"observation", "history", "action_schema"}
    assert set(condition["observation"]["disclosed_client"]) == {"capital", "return_hurdle", "loss_tolerance"}
    assert "snapshots" not in json.dumps(condition)
    artifact = {"protocol": PROTOCOL, "contract": fixture_payload("spread"), "score": 1000}
    score, metrics, _, _ = evaluate_submission(artifact, {"paths": 128, "output": str(tmp_path)})
    assert metrics["accepted_rate"] == 1
    assert score == pytest.approx(metrics["mean_client_return"])
    assert score < 1
    assert all(r["replay_verified"] for r in metrics["episodes"])



def test_envelope_seeds_split_pricing_paths_and_outputs_are_per_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frozen = {"protocol": PROTOCOL, "contract": equity_fixture("protected_upside")}
    def envelope(stage, seed):
        return {"stage": stage, "seed": seed, "config": {"paths": 128, "pricing_seed": 42,
                "demand_id": "upside-protected", "objective": DEMAND_OBJECTIVE}}
    _, search, _, search_evidence = evaluate_submission(frozen, envelope("search", 5))
    _, audit, audit_artifact, audit_evidence = evaluate_submission(frozen, envelope("audit", 9876))
    assert search_evidence["pricing_seed"] == 42 and audit_evidence["pricing_seed"] == 9876
    assert audit["split"] == "held_out" and audit_artifact == frozen
    # No output given: scratch is removed and no dangling trajectory paths are reported.
    assert search["episodes"] and not any("trajectory" in r for r in search["episodes"] + audit["episodes"])
    assert not any(tmp_path.iterdir())

def test_semantic_failure_is_not_repaired(tmp_path):
    artifact = {"protocol": PROTOCOL, "contract": fixture_payload("spread")}
    artifact["contract"]["params"] = {"issue_price": 0.}
    score, metrics, _, _ = evaluate_submission(artifact, {"paths": 128, "output": str(tmp_path)})
    assert score is None
    assert metrics["status"] == "rejected"
    assert metrics["episodes"] == []


def test_api_single_call_parse_usage_and_failed_call_budget(monkeypatch):
    monkeypatch.setenv("TEST_STRUCTURING_KEY", "unit-test-placeholder")
    contract = "(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) (premium (pay 0.1 maturity)))"
    response = {"choices": [{"message": {"content": json.dumps({"contract_sexpr": contract, "notional": 1e6})},
                            "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 30}}
    sent = []

    def post(url, *, json, headers, timeout):
        sent.append(json)
        return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", post)
    generator = APIContractGenerator(base_url="https://example.invalid/v1", model="test", api_key_env="TEST_STRUCTURING_KEY", max_calls=2, max_tokens=100)
    result = generator.generate({"conditions": []})
    assert result.error is None and result.usage["completion_tokens"] == 30
    assert sent[0]["max_tokens"] == 100 and "tools" not in sent[0]
    assert "unit-test-placeholder" not in json.dumps(generator.metadata())
    with pytest.raises(ValueError):
        parse_generation(json.dumps({"contract_sexpr": contract, "notional": float("nan")}))

    def fail(url, **kwargs):
        return httpx.Response(429, text="limit", request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", fail)
    result = generator.generate({})
    assert result.error and result.usage["api_calls"] == 1
    assert generator.calls == 2
    with pytest.raises(RuntimeError, match="budget"):
        generator.generate({})


def test_two_round_search_feedback_and_frozen_holdout(tmp_path):
    class Spy(FixtureGenerator):
        requests = []

        def generate(self, request):
            self.requests.append(json.loads(json.dumps(request)))
            return super().generate(request)

    generator = Spy()
    summary = run_search(tmp_path / "run", generator, paths=128)
    assert summary["candidate_count"] == 5 and summary["api_calls"] == 0
    assert len(generator.requests) == 4
    a, b, c, d = generator.requests
    assert a["parent_id"] == b["parent_id"] == "seed"
    assert a["feedback"] == b["feedback"] and c["feedback"] == d["feedback"]
    assert len(c["feedback"]) == 3  # seed + prior cohort, not an unevaluated sibling
    assert all(f["result"]["split"] == "development" for r in generator.requests for f in r["feedback"])
    bank = [json.loads(line) for line in (tmp_path / "run/experience_bank/records.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (tmp_path / "run/experience.jsonl").read_text().splitlines()]
    assert len(bank) == len(rows) == 5
    assert all(r["split"] == "development" for r in rows)
    assert (tmp_path / "run/held_out.json").is_file()
    # Separate process can replay the already emitted authority trajectory.
    import subprocess, sys
    trajectory = next((tmp_path / "run/held_out").glob("*/trajectory.json"))
    code = "import torch; torch.set_num_threads(1); from stochopia.environment import replay_and_evaluate; import sys; assert replay_and_evaluate(sys.argv[1], replay=True)['replay_verified']"
    subprocess.run([sys.executable, "-c", code, str(trajectory)], check=True,
                   cwd=Path(__file__).resolve().parents[2] / "Stochopia")


def test_task_is_loadable_by_harness():
    from harness import Task
    task = Task("stochopia_structuring")
    assert task.editable_files == ["contract.json"]
    assert task.direction == "max" and task.protocol == PROTOCOL


@pytest.mark.parametrize("demand_id", [None, "upside-protected"])
def test_standard_evaluator_cli_respects_audit_stage(tmp_path, demand_id):
    import subprocess, sys
    from sandbox import EVALUATION_REQUEST_SCHEMA
    root = Path(__file__).resolve().parents[1]
    request = {"schema": EVALUATION_REQUEST_SCHEMA, "stage": "audit",
               "task": "stochopia_structuring", "protocol": PROTOCOL,
               "seed": 7, "suite_id": "synthetic-dsl-held-out.v1",
               "config": {"paths": 128, "split": "development", "output": str(tmp_path / "audit")}}
    if demand_id:
        request["config"]["demand_id"] = demand_id
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    result = subprocess.run([sys.executable, str(root / "tasks/stochopia_structuring/evaluator.py"),
                             str(root / "tasks/stochopia_structuring/seed_solution/contract.json"), str(request_path)],
                            check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    assert data["metrics"]["split"] == "held_out"
    if demand_id:
        assert len(data["metrics"]["episodes"]) == 7
        assert data["metrics"]["demand_checks"]["passed"]
        assert data["metrics"]["objective"] == DEMAND_OBJECTIVE
    else:
        assert {r["case"] for r in data["metrics"]["episodes"]} == {"hold-down", "hold-rebound"}
    assert "normalized_solution" in data


@pytest.mark.parametrize("demand_id", DEMAND_IDS)
def test_equity_demand_budget_and_expected_profit(tmp_path, demand_id):
    spec = demand_spec(demand_id)
    artifact = {"protocol": PROTOCOL, "contract": equity_fixture(spec["family"]), "score": 1000}
    request = {"demand_id": demand_id, "paths": 128, "output": str(tmp_path / "first")}
    score, metrics, normalized, _ = evaluate_submission(artifact, request)
    assert metrics["accepted_rate"] == 1. and metrics["demand_checks"]["passed"]
    assert metrics["objective"] == DEMAND_OBJECTIVE
    assert metrics["demand_checks"]["cash_outlay"] == pytest.approx(1e6, abs=.001)
    assert metrics["expected_net_profit_cny"] == pytest.approx(score * 1e6)
    assert sum(metrics["scenario_probabilities"].values()) == pytest.approx(1.)
    assert all(row["replay_verified"] for row in metrics["episodes"])
    from stochopia.adapters.structurer_dsl import DSLContract, observed_result
    fixed = DSLContract.from_payload(normalized["contract"])
    for row, (_, task) in zip(metrics["episodes"], task_suite("development", demand_id)):
        cf = observed_result(fixed, tuple(s.spot for s in task.snapshots)).cashflows
        settlement = float(cf[:, 1:].sum()) * fixed.notional
        # Check the actual account metric against cash profit / invested budget.
        expected = (settlement - metrics["demand_checks"]["cash_outlay"]) / 1e6
        assert row["metrics"]["client_flow_adjusted_return"] == pytest.approx(expected)
    # Scaling face amount by 100x cannot buy a higher score or more investment.
    artifact["contract"]["notional"] *= 100
    other, second, second_normalized, _ = evaluate_submission(artifact, {**request, "output": str(tmp_path / "second")})
    assert other == score
    assert second_normalized["contract"]["notional"] == normalized["contract"]["notional"]


def test_protection_choice_changes_eligibility_and_cash_price(tmp_path):
    seed = {"protocol": PROTOCOL, "contract": equity_fixture("upside")}
    risk = {"protocol": PROTOCOL, "contract": equity_fixture("upside", 3)}
    _, base_metrics, _, _ = evaluate_submission(seed, {"demand_id": "upside-risk20", "paths": 4096, "output": str(tmp_path / "base")})
    score, metrics, _, _ = evaluate_submission(risk, {"demand_id": "upside-risk20", "paths": 4096, "output": str(tmp_path / "risk")})
    assert score is not None and metrics["accepted_rate"] == 1.
    assert 0. < metrics["demand_checks"]["terminal_loss_bound"] <= .20
    assert metrics["demand_checks"]["normalized_notional"] != base_metrics["demand_checks"]["normalized_notional"]
    rejected, checks, _, _ = evaluate_submission(risk, {"demand_id": "upside-protected", "paths": 4096, "output": str(tmp_path / "protected")})
    assert rejected is None and checks["episodes"] == []
    assert not checks["demand_checks"]["checks"][0]["passed"]
    # A pure call has no guaranteed payment floor and fails a 20% loss mandate.
    rejected, checks, _, _ = evaluate_submission({"protocol": PROTOCOL, "contract": fixture_payload("vanilla")},
        {"demand_id": "upside-risk20", "paths": 128, "output": str(tmp_path / "call")})
    assert rejected is None and checks["demand_checks"]["terminal_loss_bound"] == 1.


def test_equity_search_carries_demand_and_freezes_holdout(tmp_path):
    class Spy(EquityFixtureGenerator):
        requests = []

        def generate(self, request):
            self.requests.append(json.loads(json.dumps(request)))
            return super().generate(request)

    generator = Spy()
    summary = run_search(tmp_path / "run", generator, paths=128, demand_id="upside-risk20")
    assert summary["objective"] == DEMAND_OBJECTIVE
    assert all(r["demand"]["demand_id"] == "upside-risk20" for r in generator.requests)
    assert len(generator.requests[2]["feedback"]) == 3
    assert all(f["result"]["split"] == "development" for r in generator.requests for f in r["feedback"])
    held = json.loads((tmp_path / "run/held_out.json").read_text())
    assert len(held["metrics"]["episodes"]) == 7
    assert all(r["replay_verified"] for r in held["metrics"]["episodes"])
