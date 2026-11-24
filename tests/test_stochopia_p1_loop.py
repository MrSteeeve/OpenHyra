"""Exercise two development rounds and the detached verifier with miniature days."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stochopia.data import SZSEDaySlice
from stochopia.neural.primitives import seed_lob_source

from tasks.stochopia_p1.entry import DevelopmentLoop
from tasks.stochopia_p1.gate import CandidateGate
from tasks.stochopia_p1.no_llm_mutator import mutate
from tasks.stochopia_p1.paired_stats import paired_readouts
from tasks.stochopia_p1.protocol import MAX_INPUT_TOKENS, MAX_REPAIRS, CallUsage, Messages, ProviderReply, TwoStepProposer
from tasks.stochopia_p1.store import (
    CandidateEvaluationFailed, DevelopmentExport, EvaluationRequest, Ledger, MetricRule,
    ModelRef, PairedEvaluation, RunBusyError, StudyPlan,
)
from tasks.stochopia_p1.verifier import SEED_TASK, freeze, run_verifier


TRAIN = (20240102, 20240103)
DEV = (20240105, 20240108, 20240109)
HOLDOUT = (20240111, 20240112, 20240115)
METRIC = "impact_peak"


def _day(date: int, bid: int) -> SZSEDaySlice:
    ticks = np.zeros((2, 2, 10), dtype=np.int32)
    ticks[:, 0, 0] = bid
    ticks[:, 1, 0] = bid + 2
    depth = np.zeros_like(ticks, dtype=np.int64)
    depth[:, :, 0] = 100
    return SZSEDaySlice(
        wind_code="000001.SZ", trading_date=date, prev_close_ticks=bid,
        session_bounds=((0, 1), (1, 2)), time_ms=np.array([36_000_000, 48_000_000]),
        seq_id=np.array([1, 2], dtype=np.uint64), dt_ms=np.zeros(2, dtype=np.int64),
        action_type=np.zeros(2, dtype=np.int8), rel_ticks=np.zeros(2, dtype=np.int32),
        volume_shares=np.full(2, 100), cancel_age_ms=np.zeros(2, dtype=np.int64),
        pre_book_ticks=ticks, pre_book_shares=depth, mid_ticks_x2=np.full(2, 2 * bid + 2),
        spread_ticks=np.full(2, 2, dtype=np.int32),
    )


class ScriptedEvaluator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[EvaluationRequest] = []

    def __call__(self, request: EvaluationRequest) -> PairedEvaluation:
        self.requests.append(request)
        assert request.train_dates == TRAIN
        assert request.train_seeds == (103, 137, 211)
        if request.split == "dev":
            assert request.eval_dates == DEV
            group = request.group
            index = request.trial_id[-2:]
            preregistered = self.root / group / f"round-{int(index):03}" / "preregistration.json"
            assert preregistered.is_file()
            assert preregistered.with_name("program.json").is_file()
        else:
            assert request.eval_dates == HOLDOUT

        gain = 0.125 if request.trial_id == "metrics-r01" else 0.25
        nll_gain = 0.02
        if request.parent == request.candidate:
            gain = 0.0
            nll_gain = 0.0
        parent = {}
        candidate = {}
        for index, date in enumerate(request.eval_dates):
            day = _day(date, 10_000 + 100 * index)
            # The level and each day's shift vary, so pairing is observable.
            peak = float(day.mid_ticks_x2[0]) / 2
            parent[date] = {METRIC: peak, "val_nll_total": 3.0, "lob_impact_score": 20.0}
            candidate[date] = {METRIC: peak + gain, "val_nll_total": 3.0 - nll_gain, "lob_impact_score": 21.0}
        before = {"val_nll_total": 3.0, "lob_impact_score": 20.0, "sign_acf_delta": 17.4321}
        after = {"val_nll_total": 3.0 - nll_gain, "lob_impact_score": 21.0, "sign_acf_delta": 28.6543}
        return PairedEvaluation(parent, candidate, before, after if gain else before)


def _fixture(root: Path) -> tuple[DevelopmentLoop, ScriptedEvaluator, dict[str, list[str]], dict[str, list[CallUsage]]]:
    packets: dict[str, list[str]] = {"evidence": [], "metrics": []}
    usage: dict[str, list[CallUsage]] = {"evidence": [], "metrics": []}
    plan = StudyPlan(
        seed_model=ModelRef("seed", seed_lob_source()),
        seed_evidence={"parameters": 21932, "state_dim": 48, "val_nll_total": 3.0,
                       "lob_impact_score": 20.0, "sign_acf_delta": 17.4321},
        train_dates=TRAIN, dev_dates=DEV,
        metric_rules=(MetricRule(METRIC, "increase", 0.01, (0.2, 0.3)),),
    )
    evaluate = ScriptedEvaluator(root)
    gate = CandidateGate()

    def proposer_for(group: str) -> TwoStepProposer:
        index = 0

        def transport(messages: list[dict[str, str]], *, response_json: bool) -> ProviderReply:
            nonlocal index
            packets[group].append("\n".join(item["content"] for item in messages))
            if response_json:
                index += 1
                parent = f"{group}-1" if group == "evidence" and index == 2 else None
                proposal = {
                    "hypothesis": {"hypothesis_id": f"{group}-{index}", "parent_hypothesis_id": parent,
                                   "observed_bottleneck": f"{group} diagnosis", "operator_diagnosis": "time-memory scale",
                                   "structural_prescription": "apply a short scan", "falsifier": "peak gain below interval"},
                    "predictions": [{"metric": METRIC, "direction": "increase", "interval": [0.2, 0.3]}],
                }
                return ProviderReply(json.dumps(proposal), 100, 55)
            registered = root / group / f"round-{index:03}" / "preregistration.json"
            assert registered.exists()
            source = mutate(index, family="lru" if index == 1 else "selective_scan").source
            return ProviderReply(f"```python\n{source}\n```", 125, 250)

        return TwoStepProposer(transport, gate, usage[group].append, count_tokens=lambda _packet: 200)

    proposers = {group: proposer_for(group) for group in packets}
    return DevelopmentLoop(root, plan, proposers=proposers, gate=gate, evaluate=evaluate), evaluate, packets, usage


def test_two_rounds_separate_evidence_and_verify_once(tmp_path: Path) -> None:
    dev = tmp_path / "development"
    verifier = tmp_path / "verification"
    loop, evaluate, messages, calls = _fixture(dev)
    first = loop.run_round()
    assert first["evidence"]["verdict"] == "VERIFIED_GAIN"
    assert first["metrics"]["verdict"] == "UNVERIFIED_GAIN"
    assert first["metrics"]["prediction_verdicts"][0]["status"] == "REFUTED"
    assert first["no_llm"]["verdict"] == "VERIFIED_GAIN"

    second = loop.run_round()
    assert second["evidence"]["parent"]["model_id"] == "evidence-r01"
    assert second["evidence"]["island"] == first["evidence"]["island"]
    assert second["evidence"]["hypothesis"]["parent_hypothesis_id"] == "evidence-1"
    assert second["metrics"]["parent"]["model_id"] == "seed"
    assert second["metrics"]["island"] != first["metrics"]["island"]
    assert second["no_llm"]["parent"]["model_id"] == "no_llm-r01"
    assert len(evaluate.requests) == 6
    assert "17.4321" in messages["evidence"][0]
    assert loop.plan.seed_model.source in messages["evidence"][0]
    assert "28.6543" in messages["evidence"][2]
    assert "evidence diagnosis" in messages["evidence"][2]
    assert "VERIFIED_GAIN; core=SUPPORTED" in messages["evidence"][2]
    assert "sign_acf_delta" not in "\n".join(messages["metrics"])
    assert "Recent group history" not in "\n".join(messages["metrics"])
    assert all(len(group_calls) == 4 for group_calls in calls.values())
    assert (dev / "evidence" / "round-001" / "usage" / "02.json").is_file()
    assert not (dev / "no_llm" / "round-001" / "usage").exists()
    with pytest.raises(RuntimeError, match="complete"):
        loop.run_round()

    exported = DevelopmentExport.from_dict(json.loads(json.dumps(loop.export().to_dict())))
    assert exported.plan == loop.plan
    with pytest.raises(ValueError, match="independent"):
        freeze(exported, dev / "forbidden", test_dates=HOLDOUT, rollout_seed=22)
    sealed = freeze(exported, verifier, test_dates=HOLDOUT, rollout_seed=22)
    assert freeze(exported, verifier, test_dates=HOLDOUT, rollout_seed=22) == sealed
    with pytest.raises(ValueError, match="sealed"):
        freeze(exported, verifier, test_dates=HOLDOUT, rollout_seed=23)
    before = {path: path.read_bytes() for path in dev.glob("**/*.json")}

    with Ledger(verifier).lock(), pytest.raises(RunBusyError):
        run_verifier(verifier, evaluate=evaluate)
    crashed = False

    def interrupt_once(request: EvaluationRequest) -> PairedEvaluation:
        nonlocal crashed
        if request.trial_id != SEED_TASK and not crashed:
            crashed = True
            raise RuntimeError("fixture interruption before evaluating the next frozen pair")
        return evaluate(request)

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_verifier(verifier, evaluate=interrupt_once)
    assert (verifier / "results" / f"{SEED_TASK}.json").exists()

    def verify_with_failed_candidate(request: EvaluationRequest) -> PairedEvaluation:
        if request.trial_id == "no_llm-r01":
            raise CandidateEvaluationFailed("candidate timeout in held-out repeat")
        return evaluate(request)

    report = run_verifier(verifier, evaluate=verify_with_failed_candidate)
    assert report["paired_runs"] == len(sealed["tasks"])
    assert report["groups"]["evidence"]["verified_mechanism_gain_rate"] == 1.0
    assert report["groups"]["metrics"]["first"]["paired_assessment"]["verdict"] == "UNVERIFIED_GAIN"
    assert report["groups"]["metrics"]["final"]["model_id"] == "metrics-r02"
    assert report["groups"]["no_llm"]["first"]["initial_nll_delta"] is None
    assert report["groups"]["no_llm"]["failed_candidates"] == 1
    assert report["groups"]["no_llm"]["verified_mechanism_gain_rate"] == 0.5
    assert sum(request.trial_id == SEED_TASK for request in evaluate.requests) == 1
    verified_requests = list(evaluate.requests)
    assert run_verifier(verifier, evaluate=evaluate) == report
    assert evaluate.requests == verified_requests
    assert {path: path.read_bytes() for path in dev.glob("**/*.json")} == before


def test_incomplete_readout_cannot_claim_an_interval() -> None:
    model = ModelRef("tiny", "")
    request = EvaluationRequest("test-readout", "evidence", model, model, TRAIN, DEV, (103, 137, 211), 12300, "dev")
    baseline = {day: {METRIC: 1000 * i, "complete": 500 * i} for i, day in enumerate(DEV)}
    candidate = {day: {METRIC: row[METRIC] + 0.1, "complete": row["complete"] + 0.25}
                 for day, row in baseline.items()}
    candidate[DEV[1]][METRIC] = None
    evaluated = PairedEvaluation(baseline, candidate, {}, {})
    measured = paired_readouts(evaluated, request, [METRIC, "complete"])
    assert measured[METRIC]["interval"] is None
    assert measured[METRIC]["reason"] == "missing paired day outcomes"
    assert measured[METRIC]["paired_days"] == 2
    assert measured["complete"]["interval"] == [0.25, 0.25]


def test_attempt_budget_and_candidate_failure_do_not_stall_other_groups(tmp_path: Path) -> None:
    loop, evaluate, _messages, calls = _fixture(tmp_path)
    evidence = loop.proposers["evidence"]
    underlying_transport = evidence.transport
    synthesis_requests = 0

    def interrupted_transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        nonlocal synthesis_requests
        if response_json:
            return underlying_transport(messages, response_json=True)
        synthesis_requests += 1
        raise RuntimeError("process stopped after submitting synthesis")

    def candidate_failure(request: EvaluationRequest) -> PairedEvaluation:
        if request.trial_id == "metrics-r01":
            raise CandidateEvaluationFailed("candidate exceeded its 60-second training budget")
        return evaluate(request)

    evidence.transport = interrupted_transport
    loop.evaluate = candidate_failure
    for _ in range(MAX_REPAIRS + 1):
        with pytest.raises(RuntimeError, match="process stopped"):
            loop.run_round()

    result = loop.run_round()
    assert synthesis_requests == MAX_REPAIRS + 1
    assert [call.stage for call in calls["evidence"]] == ["hypothesis"]
    assert result["evidence"]["verdict"] == "invalid_program"
    assert result["metrics"]["verdict"] == "evaluation_failed"
    assert result["metrics"]["training_budget_consumed"] is True
    assert result["metrics"]["prediction_verdicts"][0]["status"] == "INCONCLUSIVE"
    assert result["no_llm"]["verdict"] == "VERIFIED_GAIN"
    assert [request.trial_id for request in evaluate.requests] == ["no_llm-r01"]


def test_invalid_plans_spend_only_attempted_calls_and_allow_other_groups(tmp_path: Path) -> None:
    loop, evaluate, _messages, calls = _fixture(tmp_path)
    evidence = loop.proposers["evidence"]
    metrics = loop.proposers["metrics"]
    valid_transport = metrics.transport

    def malformed(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        assert f'Use one registered metric for the first forecast: ["{METRIC}"]' in messages[-1]["content"]
        return ProviderReply("not JSON", 50, 5)

    def wrong_metric(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        assert f'Use one registered metric for the first forecast: ["{METRIC}"]' in messages[-1]["content"]
        reply = valid_transport(messages, response_json=response_json)
        return ProviderReply(reply.text.replace(METRIC, "unregistered"), reply.input_tokens, reply.output_tokens)

    evidence.transport = malformed
    metrics.transport = wrong_metric
    result = loop.run_round()
    for group in ("evidence", "metrics"):
        assert result[group]["verdict"] == "invalid_plan"
        assert result[group]["predictions"] == []
        assert result[group]["candidate"] is None
        assert result[group]["training_budget_consumed"] is False
        assert [call.stage for call in calls[group]] == ["hypothesis"]
        assert (tmp_path / group / "round-001" / "requests" / "hypothesis-01.json").exists()
    assert "registered core metric" in result["metrics"]["error"]
    assert [request.trial_id for request in evaluate.requests] == ["no_llm-r01"]

    evidence.count_tokens = lambda _message: MAX_INPUT_TOKENS + 1
    second = loop.run_round()
    assert second["evidence"]["verdict"] == "invalid_plan"
    assert "required context exceed" in second["evidence"]["error"]
    assert not (tmp_path / "evidence" / "round-002" / "requests").exists()
    assert second["no_llm"]["verdict"] == "VERIFIED_GAIN"
