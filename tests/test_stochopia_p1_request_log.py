"""Offline verification of metering, retry classification, and request replay."""

from pathlib import Path

import httpx
import pytest

from tasks.stochopia_p1.protocol import CallUsage, Messages, ProviderReply, TwoStepProposer
from tasks.stochopia_p1.provider import ProviderAttemptError, registry_transport
from tasks.stochopia_p1.request_log import ProposalBudgetExhausted, TrialCalls
from tasks.stochopia_p1.store import Ledger


@pytest.mark.parametrize(("outcome", "retryable"), [(429, False), (502, True), (400, False), (None, True)])
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
    assert backoff == [1, 2, 4]
    failed = ledger.read(*path, "requests", "hypothesis-01-result.json")
    assert failed["input_tokens"] is None and failed["seconds"] >= 0
    success = ledger.read(*path, "requests", "hypothesis-01-http-04-result.json")
    assert success["input_tokens"] == 45 and success["output_tokens"] == 21
    with ledger.lock():
        resumed = TrialCalls(ledger, path, adapter)
        assert resumed.transport(request, response_json=True).text == "{}"
        resumed.record(usage)
    assert len(calls) == 4 and counted == [usage]


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
