"""Offline HTTP/proposal tests; no real credentials or provider requests are used."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import httpx
import pytest

from tasks.lob_world_model.call_budget import BudgetExceededError, CallBudgetGuard
from tasks.lob_world_model.contracts import Messages, ProviderReply
from tasks.lob_world_model.proposal import ProposalFormatError, parse_proposal
from tasks.lob_world_model.provider import ProviderAttemptError, provider_handshake, registry_transport


MESSAGE: Messages = [{"role": "system", "content": "registered task"}, {"role": "user", "content": "fixture"}]
PREDICTION = {"selected_card_ids": ["C03", "C06"], "target_metric": "dt_nll",
              "expected_direction": "decrease", "regime_hypothesis": "Persistent state can condition timing.",
              "unfold_ids": ["candidate-03"]}
SOURCE = "def create_model():\n    return None\n"


def _registry(path: Path, *, provider: str = "openai-compatible") -> Path:
    path.write_text(
        "models:\n  fixture:\n"
        f"    provider: {provider}\n"
        "    model: exact-fixture\n    base_url: https://invalid.example/v1\n"
        "    api_key_env: ONLY_OFFLINE_PROVIDER_FIXTURE\n    temperature: null\n"
        "    max_tokens: 8192\n    strict_model_match: true\n"
        "    use_max_completion_tokens: true\n    extra_body: {reasoning_effort: low}\n",
        encoding="utf-8",
    )
    return path


def _openai_response(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={
        "model": "exact-fixture",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 231, "completion_tokens": 87},
    })


def test_openai_payload_and_usage_then_parse_complete_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path / "models.yaml")
    monkeypatch.setenv("ONLY_OFFLINE_PROVIDER_FIXTURE", "offline-test-key")
    payloads: list[dict] = []
    text = f"```json\n{json.dumps(PREDICTION)}\n```\n\n```python\n{SOURCE}```"

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        assert url == "https://invalid.example/v1/chat/completions"
        payloads.append(kwargs["json"])
        return _openai_response(text)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    reply = registry_transport(registry, "fixture")(MESSAGE, response_json=False)
    proposal = parse_proposal(reply.text)
    assert proposal.source == SOURCE
    assert proposal.prediction.target_metric == "dt_nll"
    assert proposal.prediction.selected_card_ids == ("C03", "C06")
    assert proposal.prediction.expected_direction == "decrease"
    assert proposal.prediction.unfold_ids == ("candidate-03",)
    assert reply.input_tokens == 231 and reply.output_tokens == 87
    assert payloads == [{"model": "exact-fixture", "messages": MESSAGE, "max_completion_tokens": 8192,
                         "reasoning_effort": "low"}]


def test_anthropic_payload_and_multi_block_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path / "models.yaml", provider="anthropic")
    monkeypatch.setenv("ONLY_OFFLINE_PROVIDER_FIXTURE", "offline-test-key")
    payloads: list[dict] = []

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        assert url == "https://invalid.example/v1/messages"
        payloads.append(kwargs["json"])
        return httpx.Response(200, json={
            "model": "exact-fixture", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "ignore"},
                        {"type": "text", "text": f"```json\n{json.dumps(PREDICTION)}\n```"},
                        {"type": "text", "text": f"```python\n{SOURCE}```"}],
            "usage": {"input_tokens": 102, "output_tokens": 203},
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    reply = registry_transport(registry, "fixture")(MESSAGE, response_json=False)
    assert "ignore" not in reply.text
    assert parse_proposal(reply.text).source == SOURCE
    assert reply.input_tokens == 102 and reply.output_tokens == 203
    assert payloads[0]["messages"] == [MESSAGE[1]]
    assert payloads[0]["system"] == MESSAGE[0]["content"]
    assert payloads[0]["max_tokens"] == 8192
    assert "temperature" not in payloads[0]
    assert len(payloads) == 1


@pytest.mark.parametrize("invalid", [
    {"expected_direction": "equivalent"}, {"target_metric": "val_nll_total"},
    {"selected_card_ids": ["C99"]}, {"selected_card_ids": "C01"},
    {"selected_card_ids": ["C01", "C01"]}, {"unfold_ids": ["a", "b", "c"]},
    {"regime_hypothesis": ""}, {"regime_hypothesis": None}, {"extra": 1},
])
def test_prediction_contract_errors_are_explicit(invalid: dict) -> None:
    value = {**PREDICTION, **invalid}
    with pytest.raises(ProposalFormatError):
        parse_proposal(json.dumps({**value, "candidate_source": SOURCE}))


@pytest.mark.parametrize("text", ["", "not json", "```python\nfoo()\n```", '{}',
                                     '{"candidate_source": "x", "candidate_source": "y"}'])
def test_invalid_reply_has_no_synthetic_repair(text: str) -> None:
    with pytest.raises(ProposalFormatError):
        parse_proposal(text)


def test_combined_json_proposal_and_optional_unfold() -> None:
    value = {key: item for key, item in PREDICTION.items() if key != "unfold_ids"}
    proposal = parse_proposal(json.dumps({"candidate_source": SOURCE, "prediction": value}))
    assert proposal.prediction.unfold_ids == () and proposal.source == SOURCE


@pytest.mark.parametrize("failure", ["truncation", "malformed", "mismatched", "HTTP", "timeout"])
def test_failed_http_attempt_is_charged_without_retry_and_known_usage_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    registry = _registry(tmp_path / "models.yaml")
    monkeypatch.setenv("ONLY_OFFLINE_PROVIDER_FIXTURE", "offline-test-key")
    attempts = 0

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "truncation":
            return _openai_response("partial assistant", "length")
        if failure == "malformed":
            return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 231, "completion_tokens": 87}})
        if failure == "mismatched":
            data = _openai_response("assistant text").json()
            data["model"] = "different-model"
            return httpx.Response(200, json=data)
        if failure == "HTTP":
            return httpx.Response(503, json={"usage": {"prompt_tokens": 231, "completion_tokens": 87}})
        raise httpx.ConnectTimeout("offline fixture")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    guard = CallBudgetGuard(tmp_path / "api.jsonl", limit=1)
    transport = registry_transport(registry, "fixture")
    if failure == "truncation":
        reply = guard.call(transport, MESSAGE, call_id="slot")
        assert reply.error and reply.text == "partial assistant"
        assert guard.result("slot") == reply
    else:
        with pytest.raises(ProviderAttemptError):
            guard.call(transport, MESSAGE, call_id="slot")
        with pytest.raises(ProviderAttemptError):
            guard.result("slot")
    assert attempts == guard.count == 1
    completion = guard.entries()[-1]
    expected = (None, None) if failure == "timeout" else (231, 87)
    assert (completion["input_tokens"], completion["output_tokens"]) == expected
    with pytest.raises(BudgetExceededError):
        guard.call(transport, MESSAGE, call_id="another-slot")
    assert attempts == 1


def test_partial_or_missing_usage_is_not_falsely_reported_as_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path / "models.yaml")
    monkeypatch.setenv("ONLY_OFFLINE_PROVIDER_FIXTURE", "offline-test-key")

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        data = _openai_response("assistant text").json()
        data["usage"] = {"prompt_tokens": 200}
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(ProviderAttemptError) as failure:
        registry_transport(registry, "fixture")(MESSAGE, response_json=False)
    assert failure.value.raw_reply == "assistant text"
    assert failure.value.input_tokens == 200 and failure.value.output_tokens is None


def test_budget_is_globally_durable_across_parallel_trajectories(tmp_path: Path) -> None:
    path = tmp_path / "api.jsonl"

    def fake(messages: Messages, *, response_json: bool) -> ProviderReply:
        return ProviderReply("{}", 4, 5)

    def call(i: int) -> str:
        try:
            CallBudgetGuard(path).call(fake, MESSAGE, call_id=f"slot-{i}")
        except BudgetExceededError:
            return "rejected"
        return "completed"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(40)))
    guard = CallBudgetGuard(path)
    assert results.count("completed") == guard.count == 36
    assert results.count("rejected") == 4
    assert len(guard.entries()) == 72
    with pytest.raises(BudgetExceededError):
        guard.call(fake, MESSAGE, call_id="later")
    with pytest.raises(ValueError, match="registered limit"):
        CallBudgetGuard(path, limit=12).entries()


def test_duplicate_or_interrupted_slot_cannot_be_sampled_again(tmp_path: Path) -> None:
    guard = CallBudgetGuard(tmp_path / "api.jsonl", 3)
    guard._reserve("interrupted", MESSAGE, False)
    assert guard.count == 1
    with pytest.raises(BudgetExceededError, match="result unavailable"):
        guard.result("interrupted")
    called = []

    def fake(messages: Messages, *, response_json: bool) -> ProviderReply:
        called.append(messages)
        return ProviderReply("{}", 4, 5)

    with pytest.raises(BudgetExceededError, match="already consumed"):
        CallBudgetGuard(guard.path, 3).call(fake, MESSAGE, call_id="interrupted")
    assert not called


def test_handshake_tiny_completion_cap_and_one_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path / "models.yaml")
    monkeypatch.setenv("ONLY_OFFLINE_PROVIDER_FIXTURE", "offline-test-key")
    payloads = []

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        payloads.append(kwargs["json"])
        return _openai_response('{"ok": true}')

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    assert provider_handshake(registry, "fixture", completion_limit=128).text == '{"ok": true}'
    assert len(payloads) == 1
    assert payloads[0]["max_completion_tokens"] == 128
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_replayed_slot_requires_exact_registered_messages(tmp_path: Path) -> None:
    guard = CallBudgetGuard(tmp_path / "api.jsonl")
    reply = guard.call(lambda messages, *, response_json: ProviderReply("ok", 10, 2), MESSAGE, call_id="a")
    assert guard.result("a", messages=MESSAGE) == reply
    with pytest.raises(ValueError, match="differs"):
        guard.result("a", messages=[{"role": "user", "content": "modified"}])
    assert guard.count == 1
