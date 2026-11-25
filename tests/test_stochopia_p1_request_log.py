"""Offline verification of metering, retry classification, and request replay."""

import json
from pathlib import Path
from threading import Event

import httpx
import pytest

from tasks.stochopia_p1.entry import DevelopmentLoop
from tasks.stochopia_p1.protocol import CallUsage, Messages, ProviderReply, TwoStepProposer
from tasks.stochopia_p1.provider import ProviderAttemptError, ProviderOutageError, provider_handshake, registry_transport
from tasks.stochopia_p1.request_log import HTTP_RETRY_DELAYS, ProposalBudgetExhausted, TrialCalls
from tasks.stochopia_p1.store import (
    CandidateEvaluationFailed, EvaluationRequest, Ledger, MetricRule, ModelRef, PairedEvaluation, StudyPlan,
)


# --- Transport classification and attempt metering ---

@pytest.mark.parametrize(("outcome", "retryable"), [(429, True), (502, True), (400, False), (None, True)])
def test_transport_errors_are_classified_without_implicit_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: int | None, retryable: bool,
) -> None:
    registry = tmp_path / "fixture.yaml"
    registry.write_text(
        "models:\n  tiny:\n    provider: openai-compatible\n    model: tiny\n"
        "    base_url: https://invalid.example/v1\n    api_key_env: P1_FIXTURE_ONLY\n", encoding="utf-8",
    )
    monkeypatch.setenv("P1_FIXTURE_ONLY", "offline fixture")
    attempts = 0

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if outcome is None:
            raise httpx.ConnectError("offline fixture connection failure")
        return httpx.Response(outcome, json={"error": "offline fixture"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ProviderAttemptError) as failure:
        registry_transport(registry, "tiny")([], response_json=True)
    assert failure.value.retryable is retryable
    assert attempts == 1


def test_transit_retries_repeat_the_same_request_and_replay_durable_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Messages] = []
    backoff: list[float] = []
    counted: list[CallUsage] = []
    monkeypatch.setattr("tasks.stochopia_p1.request_log.sleep", backoff.append)
    request: Messages = [{"role": "system", "content": "fixture"}]

    def transient(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        calls.append(messages)
        if len(calls) <= 3:
            raise ProviderAttemptError("Provider HTTP 502", retryable=True)
        return ProviderReply("{}", 45, 21)

    adapter = TwoStepProposer(transient, lambda source: None, counted.append, count_tokens=lambda _: 45)
    ledger = Ledger(tmp_path)
    path = ("metrics_ds", "round-001")
    usage = CallUsage("hypothesis", 45, 21)
    with ledger.lock():
        run = TrialCalls(ledger, path, adapter)
        assert run.transport(request, response_json=True).text == "{}"
        run.record(usage)
    assert calls == [request] * 4
    assert backoff == [2.0, 5.0, 15.0]
    failed = ledger.read(*path, "requests", "hypothesis-01-result.json")
    assert failed["input_tokens"] is None and failed["seconds"] >= 0
    success = ledger.read(*path, "requests", "hypothesis-01-http-04-result.json")
    assert success["input_tokens"] == 45 and success["output_tokens"] == 21
    for suffix in ("", "-http-02", "-http-03", "-http-04"):
        submission = ledger.read(*path, "requests", f"hypothesis-01{suffix}.json")
        assert submission["messages"] == request and submission["response_json"] is True
    with ledger.lock():
        resumed = TrialCalls(ledger, path, adapter)
        assert resumed.transport(request, response_json=True).text == "{}"
        resumed.record(usage)
    assert len(calls) == 4 and counted == [usage]


# --- Circuit breaker and exact logical-request resume ---

def test_exhausted_transport_batch_pauses_and_resume_keeps_attempt_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submissions: list[Messages] = []
    delays: list[float] = []
    usage: list[CallUsage] = []
    monkeypatch.setattr("tasks.stochopia_p1.request_log.sleep", delays.append)
    request: Messages = [{"role": "system", "content": "fixture"}]

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        submissions.append(messages)
        if len(submissions) <= 12:
            input_tokens = 3 if len(submissions) == 1 else None
            output_tokens = 0 if input_tokens is not None else None
            raise ProviderAttemptError(
                "Provider transport failure", retryable=True,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
        return ProviderReply('{"ok": true}', 11, 5)

    ledger = Ledger(tmp_path)
    adapter = TwoStepProposer(transport, lambda _source: None, usage.append, count_tokens=lambda _: 11)
    path = ("metrics_ds", "round-001")
    with ledger.lock():
        for _ in range(2):
            with pytest.raises(ProviderOutageError, match="5 transport retries"):
                TrialCalls(ledger, path, adapter).transport(request, response_json=True)
        assert len(submissions) == 12
        assert delays == list(HTTP_RETRY_DELAYS) * 2
        assert usage == []
        failed_path = ledger.file(*path, "requests", "hypothesis-01-result.json")
        first = ledger.read(*path, "requests", "hypothesis-01-result.json")
        assert first["input_tokens"] == 3 and first["output_tokens"] == 0
        failed_bytes = failed_path.read_bytes()
        recovered = TrialCalls(ledger, path, adapter)
        assert recovered.transport(request, response_json=True).text == '{"ok": true}'
        recovered.record(CallUsage("hypothesis", 11, 5))
        assert ledger.read(*path, "requests", "hypothesis-01-http-13.json")["attempt"] == 1
        assert ledger.read(*path, "requests", "hypothesis-01-http-13-result.json")["input_tokens"] == 11
        assert failed_path.read_bytes() == failed_bytes
        assert TrialCalls(ledger, path, adapter).transport(request, response_json=True).text == '{"ok": true}'
    assert submissions == [request] * 13
    assert usage == [CallUsage("hypothesis", 11, 5)]


def test_pause_from_other_trajectory_blocks_new_attempt_after_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = Event()
    attempted: list[Messages] = []

    def pause_on_backoff(delay: float) -> None:
        assert delay == HTTP_RETRY_DELAYS[0]
        stop.set()

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        attempted.append(messages)
        raise ProviderAttemptError("Provider HTTP 429", retryable=True)

    monkeypatch.setattr("tasks.stochopia_p1.request_log.sleep", pause_on_backoff)
    ledger = Ledger(tmp_path)
    adapter = TwoStepProposer(transport, lambda _source: None, lambda _usage: None, count_tokens=lambda _: 11)
    with ledger.lock():
        calls = TrialCalls(ledger, ("trial",), adapter, stop_requested=stop.is_set)
        with pytest.raises(ProviderOutageError, match="paused"):
            calls.transport([], response_json=True)
        assert len(attempted) == 1
        assert ledger.read("trial", "requests", "hypothesis-01-result.json")["retryable"]
        assert ledger.read("trial", "requests", "hypothesis-01-http-02.json") is None


@pytest.mark.parametrize("outage_stage", ["hypothesis", "synthesis"])
@pytest.mark.parametrize("process_interruption", [False, True])
def test_development_trial_resumes_same_logical_plan_after_provider_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outage_stage: str, process_interruption: bool,
) -> None:
    monkeypatch.setattr("tasks.stochopia_p1.request_log.sleep", lambda _: None)
    online = False
    sent: list[str] = []
    evaluated: list[str] = []
    recorded: list[CallUsage] = []
    plan_reply = {
        "hypothesis": {
            "observed_bottleneck": "weak impact realism",
            "operator_diagnosis": "fast forgetting",
            "structural_prescription": "retain context",
            "falsifier": "gain does not reach the forecast minimum",
        },
        "predictions": [
            {"role": "primary", "metric": "lob_impact_score", "direction": "increase", "delta_interval": [1, 3]},
            {"role": "guardrail", "metric": "val_nll_total", "direction": "equivalent", "equivalence_band": [-0.05, 0.05]},
        ],
    }

    def transport(_messages: Messages, *, response_json: bool) -> ProviderReply:
        stage = "hypothesis" if response_json else "synthesis"
        sent.append(stage)
        if not online and stage == outage_stage:
            if process_interruption:
                raise RuntimeError("process stopped before receiving a response")
            raise ProviderAttemptError("Provider transport failure", retryable=True)
        text = json.dumps(plan_reply) if response_json else "```python\n# fake source\n```"
        return ProviderReply(text, 10, 2)

    def evaluate(request: EvaluationRequest) -> PairedEvaluation:
        evaluated.append(request.trial_id)
        assert request.analysis_version == "m5"
        raise CandidateEvaluationFailed("offline fixture evaluation failure")

    plan = StudyPlan(
        seed_model=ModelRef("seed", "# fixture parent"),
        seed_evidence={},
        train_dates=(1, 2),
        dev_dates=(3, 4),
        metric_rules=(MetricRule("lob_impact_score", "increase", 0.50, (0.50, 10.0)),),
        rounds=2,
        groups=("metrics_ds",),
        analysis_version="m5",
    )
    adapter = TwoStepProposer(
        transport, lambda _source: None, recorded.append, count_tokens=lambda _: 10,
    )
    loop = DevelopmentLoop(tmp_path, plan, proposers={"metrics_ds": adapter}, gate=lambda _source: None, evaluate=evaluate)
    path = ("metrics_ds", "round-001")
    expected_error = RuntimeError if process_interruption else ProviderOutageError
    with pytest.raises(expected_error):
        loop.run_round()
    assert loop.ledger.read(*path, "program.json") is None
    assert loop.ledger.read(*path, "result.json") is None
    assert (loop.ledger.read(*path, "preregistration.json") is not None) == (outage_stage == "synthesis")
    assert evaluated == []
    assert loop.ledger.read("metrics_ds", "round-002", "result.json") is None

    online = True
    result = loop.run_round()["metrics_ds"]
    assert result["round"] == 1
    assert result["verdict"] == "evaluation_failed"
    assert evaluated == ["metrics_ds-r01"]
    assert sent.count(outage_stage) == (2 if process_interruption else 7)
    assert len(sent) == (3 if process_interruption else 8)
    assert [item.stage for item in recorded] == ["hypothesis", "code"]
    if process_interruption:
        interrupted = loop.ledger.read(*path, "requests", f"{outage_stage}-01-result.json")
        assert interrupted["interruption_without_result"] and interrupted["input_tokens"] is None


# --- Data-free startup handshake and interrupted-request accounting ---

@pytest.mark.parametrize(("response_text", "passed"), [('{"ok": true}', True), ('{"ok": 1}', False), ("not JSON", False)])
def test_data_free_provider_handshake_is_metered_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response_text: str, passed: bool,
) -> None:
    registry = tmp_path / "fixture.yaml"
    registry.write_text(
        "models:\n  tiny:\n    provider: openai-compatible\n    model: tiny\n"
        "    base_url: https://invalid.example/v1\n    api_key_env: P1_FIXTURE_ONLY\n"
        "    temperature: null\n    strict_model_match: true\n", encoding="utf-8",
    )
    monkeypatch.setenv("P1_FIXTURE_ONLY", "offline fixture")
    payloads: list[dict[str, object]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        payloads.append(kwargs["json"])
        return httpx.Response(200, json={
            "model": "tiny", "choices": [{"message": {"content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 23, "completion_tokens": 5},
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    if passed:
        assert provider_handshake(registry, "tiny") == ProviderReply(response_text, 23, 5)
    else:
        with pytest.raises(ProviderAttemptError, match="handshake") as failure:
            provider_handshake(registry, "tiny")
        assert failure.value.input_tokens == 23 and failure.value.output_tokens == 5
    assert len(payloads) == 1
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert "temperature" not in payloads[0]
    assert payloads[0]["messages"][-1]["content"] == "Connectivity check."


def test_interrupted_request_is_spent_and_has_no_search_resample(tmp_path: Path) -> None:
    calls = 0

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        nonlocal calls
        calls += 1
        raise RuntimeError("process stopped before receiving a response")

    adapter = TwoStepProposer(transport, lambda source: None, lambda usage: None, count_tokens=lambda _: 12)
    ledger = Ledger(tmp_path)
    with ledger.lock():
        with pytest.raises(RuntimeError, match="process stopped"):
            TrialCalls(ledger, ("trial",), adapter).transport([], response_json=False)
        with pytest.raises(ProposalBudgetExhausted, match="response unavailable"):
            TrialCalls(ledger, ("trial",), adapter).transport([], response_json=False)
    assert calls == 1


def test_m5_interrupted_request_retains_unknown_usage_but_recovers_its_logical_slot(tmp_path: Path) -> None:
    submitted: list[Messages] = []
    billed: list[CallUsage] = []
    request: Messages = [{"role": "user", "content": "fixture"}]

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert not response_json
        submitted.append(messages)
        if len(submitted) == 1:
            raise RuntimeError("process stopped before receiving a response")
        return ProviderReply("completed", 32, 2)

    adapter = TwoStepProposer(transport, lambda _source: None, billed.append, count_tokens=lambda _: 12)
    ledger = Ledger(tmp_path)
    with ledger.lock():
        calls = TrialCalls(ledger, ("trial",), adapter, analysis_version="m5")
        with pytest.raises(RuntimeError, match="process stopped"):
            calls.transport(request, response_json=False)
        reservation = ledger.file("trial", "requests", "synthesis-01.json")
        original_reservation = reservation.read_bytes()
        assert ledger.read("trial", "requests", "synthesis-01-result.json") is None

        resumed = TrialCalls(ledger, ("trial",), adapter, analysis_version="m5")
        assert resumed.transport(request, response_json=False) == ProviderReply("completed", 32, 2)
        resumed.record(CallUsage("code", 32, 2))
        missing = ledger.read("trial", "requests", "synthesis-01-result.json")
        assert missing["interruption_without_result"] and missing["retryable"]
        assert missing["input_tokens"] is None and missing["output_tokens"] is None and missing["seconds"] is None
        assert reservation.read_bytes() == original_reservation
        recovered = ledger.read("trial", "requests", "synthesis-01-http-02.json")
        assert recovered["attempt"] == 1 and recovered["http_attempt"] == 2
        assert TrialCalls(ledger, ("trial",), adapter, analysis_version="m5").transport(
            request, response_json=False,
        ).text == "completed"
    assert submitted == [request] * 2
    assert billed == [CallUsage("code", 32, 2)]


def test_repaired_plan_and_resumed_synthesis_have_stable_usage_slots(tmp_path: Path) -> None:
    attempt_modes: list[bool] = []
    billed: list[CallUsage] = []
    source = "class CandidateLOBWorldModel(LOBWorldModel):\n    pass\n"
    plan = {
        "hypothesis": {
            "observed_bottleneck": "weak impact realism", "operator_diagnosis": "fast forgetting",
            "structural_prescription": "retain context", "falsifier": "gain outside forecast interval",
        },
        "predictions": [
            {"role": "primary", "metric": "lob_impact_score", "direction": "increase", "delta_interval": [1, 3]},
            {"role": "guardrail", "metric": "val_nll_total", "direction": "equivalent", "equivalence_band": [-0.05, 0.05]},
        ],
    }

    def fake_provider(messages: Messages, *, response_json: bool) -> ProviderReply:
        attempt_modes.append(response_json)
        if not response_json:
            return ProviderReply(f"```python\n{source}```", 35, 22)
        if len(attempt_modes) == 1:
            return ProviderReply("{}", 32, 2)
        assert "Previous JSON rejected" in messages[-1]["content"]
        return ProviderReply(json.dumps(plan), 45, 23)

    def accepted(_source: str) -> None:
        return None

    def count(_messages: Messages) -> int:
        return 50

    ledger = Ledger(tmp_path)
    path = ("metrics_ds", "round-001")
    adapter = TwoStepProposer(fake_provider, accepted, billed.append, count_tokens=count)

    with ledger.lock():
        calls = TrialCalls(ledger, path, adapter)
        proposer = TwoStepProposer(calls.transport, accepted, calls.record, count_tokens=count)
        result = proposer.propose("metrics", "seed source", {}, (), lambda *_: None, trial_id="trial")
    assert result.source == source
    assert attempt_modes == [True, True, False]

    with ledger.lock():
        calls = TrialCalls(ledger, path, adapter)
        proposer = TwoStepProposer(calls.transport, accepted, calls.record, count_tokens=count)
        replayed = proposer.synthesize("metrics", "seed source", {}, (), result.hypothesis, result.predictions)
    assert replayed == result
    assert attempt_modes == [True, True, False]
    assert [item.stage for item in billed] == ["hypothesis", "hypothesis_repair", "code"]
    assert ledger.read(*path, "usage", "02.json")["stage"] == "hypothesis_repair"
    assert ledger.read(*path, "usage", "03.json")["stage"] == "code"
