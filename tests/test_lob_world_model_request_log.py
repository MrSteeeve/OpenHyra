"""Offline verification of metering, retry classification, and request replay."""

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from threading import Event

import httpx
import pytest

from tasks.lob_world_model.contracts import CallUsage, Messages, ProviderReply, Transport
from tasks.lob_world_model.provider import (
    DEFAULT_MODELS_PATH, ProviderAttemptError, ProviderOutageError, provider_handshake, registry_transport,
)
from tasks.lob_world_model.provider_client import BaseLLMClient, LLMTruncationError, ModelConfig, OpenAICompatClient
from tasks.lob_world_model.request_log import HTTP_RETRY_DELAYS, ProposalBudgetExhausted, TrialCalls
from tasks.lob_world_model.store import Ledger


@dataclass
class Adapter:
    transport: Transport
    record_usage: Callable[[CallUsage], None]


# --- Transport classification and attempt metering ---

@pytest.mark.parametrize(("outcome", "retryable"), [(429, True), (502, True), (400, False), (None, True)])
def test_transport_errors_are_classified_without_implicit_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: int | None, retryable: bool,
) -> None:
    registry = tmp_path / "fixture.yaml"
    registry.write_text(
        "models:\n  tiny:\n    provider: openai-compatible\n    model: tiny\n"
        "    base_url: https://invalid.example/v1\n    api_key_env: WORLD_MODEL_FIXTURE_ONLY\n", encoding="utf-8",
    )
    monkeypatch.setenv("WORLD_MODEL_FIXTURE_ONLY", "offline fixture")
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
    monkeypatch.setattr("tasks.lob_world_model.request_log.sleep", backoff.append)
    request: Messages = [{"role": "system", "content": "fixture"}]

    def transient(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        calls.append(messages)
        if len(calls) <= 3:
            raise ProviderAttemptError("Provider HTTP 502", retryable=True)
        return ProviderReply("{}", 45, 21)

    adapter = Adapter(transient, counted.append)
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
    monkeypatch.setattr("tasks.lob_world_model.request_log.sleep", delays.append)
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
    adapter = Adapter(transport, usage.append)
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

    monkeypatch.setattr("tasks.lob_world_model.request_log.sleep", pause_on_backoff)
    ledger = Ledger(tmp_path)
    adapter = Adapter(transport, lambda _usage: None)
    with ledger.lock():
        calls = TrialCalls(ledger, ("trial",), adapter, stop_requested=stop.is_set)
        with pytest.raises(ProviderOutageError, match="paused"):
            calls.transport([], response_json=True)
        assert len(attempted) == 1
        assert ledger.read("trial", "requests", "hypothesis-01-result.json")["retryable"]
        assert ledger.read("trial", "requests", "hypothesis-01-http-02.json") is None


@pytest.mark.parametrize("response_json", [True, False])
@pytest.mark.parametrize("process_interruption", [False, True])
def test_logical_request_resumes_after_provider_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response_json: bool, process_interruption: bool,
) -> None:
    monkeypatch.setattr("tasks.lob_world_model.request_log.sleep", lambda _: None)
    online = False
    attempts = []
    recorded = []
    request = [{"role": "user", "content": "fixed request"}]
    usage = CallUsage("hypothesis" if response_json else "code", 10, 2)

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        attempts.append(messages)
        if not online:
            if process_interruption:
                raise RuntimeError("process stopped before receiving a response")
            raise ProviderAttemptError("Provider transport failure", retryable=True)
        return ProviderReply("fixed reply", 10, 2)

    ledger = Ledger(tmp_path)
    adapter = Adapter(transport, recorded.append)
    expected_error = RuntimeError if process_interruption else ProviderOutageError
    with ledger.lock():
        calls = TrialCalls(ledger, ("trial",), adapter, analysis_version="m5")
        with pytest.raises(expected_error):
            calls.transport(request, response_json=response_json)
        assert not recorded
        online = True
        resumed = TrialCalls(ledger, ("trial",), adapter, analysis_version="m5")
        assert resumed.transport(request, response_json=response_json).text == "fixed reply"
        resumed.record(usage)
        assert recorded == [usage]
        assert attempts == [request] * (2 if process_interruption else 7)
        phase = "hypothesis" if response_json else "synthesis"
        if process_interruption:
            interrupted = ledger.read("trial", "requests", f"{phase}-01-result.json")
            assert interrupted["interruption_without_result"] and interrupted["input_tokens"] is None


# --- Data-free startup handshake and interrupted-request accounting ---

@pytest.mark.parametrize(("response_text", "passed"), [('{"ok": true}', True), ('{"ok": 1}', False), ("not JSON", False)])
def test_data_free_provider_handshake_is_metered_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response_text: str, passed: bool,
) -> None:
    registry = tmp_path / "fixture.yaml"
    registry.write_text(
        "models:\n  tiny:\n    provider: openai-compatible\n    model: tiny\n"
        "    base_url: https://invalid.example/v1\n    api_key_env: WORLD_MODEL_FIXTURE_ONLY\n"
        "    temperature: null\n    strict_model_match: true\n", encoding="utf-8",
    )
    monkeypatch.setenv("WORLD_MODEL_FIXTURE_ONLY", "offline fixture")
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

    adapter = Adapter(transport, lambda usage: None)
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

    adapter = Adapter(transport, billed.append)
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
    replies = [ProviderReply("{}", 32, 2), ProviderReply("valid plan", 45, 23),
               ProviderReply("valid source", 35, 22)]
    billed = []
    submitted = []
    requests = [[{"role": "user", "content": text}] for text in ("plan", "repair", "code")]
    modes = [True, True, False]
    stages = ["hypothesis", "hypothesis_repair", "code"]

    def provider(messages: Messages, *, response_json: bool) -> ProviderReply:
        submitted.append(response_json)
        return replies[len(submitted) - 1]

    ledger = Ledger(tmp_path)
    adapter = Adapter(provider, billed.append)
    path = ("trial",)
    with ledger.lock():
        for _ in range(2):
            calls = TrialCalls(ledger, path, adapter)
            for request, mode, stage, expected in zip(requests, modes, stages, replies, strict=True):
                reply = calls.transport(request, response_json=mode)
                assert reply == expected
                calls.record(CallUsage(stage, reply.input_tokens, reply.output_tokens))
    assert submitted == modes
    assert [item.stage for item in billed] == stages
    assert ledger.read(*path, "usage", "02.json")["stage"] == "hypothesis_repair"
    assert ledger.read(*path, "usage", "03.json")["stage"] == "code"


@pytest.mark.parametrize("model_name", ["gpt-6.2-sol", "deepseek-flash"])
def test_promoted_registry_preserves_provider_settings_and_metered_truncation(model_name: str) -> None:
    configs: list[ModelConfig] = []

    class FixtureClient(OpenAICompatClient):
        async def chat(self, messages: Messages, temperature: float | None = None,
                       max_tokens: int | None = None, seed: int | None = None) -> str:
            assert max_tokens == 8192 and temperature is None and seed is None
            assert self.RETRY_DELAYS == () and self.TRUNCATION_RETRIES == 0
            self.total_usage["prompt_tokens"] = 200
            if len(configs) == 2:
                self.total_usage["completion_tokens"] = 8192
                raise LLMTruncationError("fixture reached limit")
            self.total_usage["completion_tokens"] = 300
            return "fixture reply"

    def factory(name: str, registry: dict[str, ModelConfig]) -> BaseLLMClient:
        config = registry[name]
        assert name == config.model == model_name
        configs.append(config)
        return FixtureClient(config)

    transport = registry_transport(DEFAULT_MODELS_PATH, model_name, client_factory=factory)
    assert not configs
    assert transport([], response_json=True) == ProviderReply("fixture reply", 200, 300)
    truncated = transport([], response_json=False)
    assert truncated.text == "" and truncated.output_tokens == 8192 and truncated.input_tokens == 200
    assert "8192-token output cap" in truncated.error
    assert [config.response_json for config in configs] == [True, False]
    assert all(config.timeout == 180 and config.temperature is None for config in configs)
    assert all(config.max_tokens == 8192 and config.strict_model_match for config in configs)
    assert all(config.use_max_completion_tokens == (model_name == "gpt-6.2-sol") for config in configs)
