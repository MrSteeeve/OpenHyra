"""Provider protocols, fixed call budgets and independent research groups."""
import json

import httpx
import pytest
import torch

from experiments.run_stochopia_model_comparison import comparison_plan, run_comparison
from tasks.stochopia_structuring.api_config import api_generator, load_api_models
from tasks.stochopia_structuring.generator import SYSTEM_PROMPT


PRESETS = load_api_models()["models"]


def answer(coefficient=.105):
    return json.dumps({"contract_sexpr": "(contract (meta (underlyings u) (maturity 3) "
        "(schedules (obs 1 2 3))) (funded 1 $issue_price (pay "
        f"(mul_const {coefficient} (clip (lin 1 (terminal (spot u)) -1) 0 0.10)) maturity)))",
        "notional": 1e6, "rationale": f"mock coefficient {coefficient}"})


def response_payload(name, text=None, finish=None):
    preset = PRESETS[name]
    if preset["api_format"] == "anthropic-messages":
        return {"id": "msg-test", "model": preset["model"], "content": [
            {"type": "thinking", "thinking": "private mock reasoning", "signature": "test"},
            {"type": "text", "text": text or answer()}], "stop_reason": finish or "end_turn",
            "usage": {"input_tokens": 20, "cache_creation_input_tokens": 3,
                      "cache_read_input_tokens": 7, "output_tokens": 40,
                      "output_tokens_details": {"thinking_tokens": 15}}}
    return {"id": "chat-test", "model": preset["model"], "provider": "mock-provider",
            "choices": [{"message": {"content": text or answer(), "reasoning_content": "mock reasoning"},
                         "finish_reason": finish or "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70,
                      "completion_tokens_details": {"reasoning_tokens": 15}}}


def stub_credentials(monkeypatch):
    for preset in PRESETS.values():
        monkeypatch.setenv(preset["api_key_env"], "unit-test-placeholder")


@pytest.mark.parametrize("name", PRESETS)
def test_native_provider_protocol_and_actual_usage(monkeypatch, name):
    stub_credentials(monkeypatch)
    sent = []
    payload = response_payload(name)

    def post(url, *, json, headers, timeout):
        sent.append((url, json, headers))
        return httpx.Response(200, json=payload, headers={"x-request-id": "test-request"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", post)
    generator = api_generator(name, max_tokens=1234)
    request = {"conditions": [], "demand": {"investment_budget_cny": 1e6}}
    result = generator.generate(request)
    assert result.error is None and len(sent) == generator.calls == 1
    url, body, headers = sent[0]
    preset = PRESETS[name]
    assert body["model"] == preset["model"] and body["max_tokens"] == 1234
    assert not {"temperature", "top_p", "top_k", "thinking", "reasoning", "reasoning_effort", "tools"} & set(body)
    if name == "claude-haiku-5.5":
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["anthropic-version"] == "2023-06-01" and "x-api-key" in headers
        assert body["system"] == SYSTEM_PROMPT and len(body["messages"]) == 1
    else:
        assert url == preset["base_url"] + "/chat/completions"
        assert "Authorization" in headers and body["messages"][0]["content"] == SYSTEM_PROMPT
    assert json.loads(body["messages"][-1]["content"]) == request
    if name == "nemotron-3-super":
        assert body["provider"] == {"allow_fallbacks": False}
    assert result.raw_response == payload
    assert result.raw_output == answer()
    assert result.usage["prompt_tokens"] == 30
    assert result.usage["completion_tokens"] == 40 and result.usage["total_tokens"] == 70
    assert result.usage["reasoning_tokens"] == 15
    assert result.usage["requested_model"] == result.usage["response_model"] == preset["model"]
    assert result.usage["provider_request_id"] == "test-request"
    assert result.usage["started_at_utc"]
    assert "unit-test-placeholder" not in json.dumps(generator.metadata())


@pytest.mark.parametrize("name,finish", [("claude-haiku-5.5", "max_tokens"),
    ("claude-haiku-5.5", "refusal"), ("deepseek-v4.1-flash", "length"),
    ("step-5-preview", "content_filter"), ("nemotron-3-super", "length")])
def test_truncated_or_refused_output_spends_one_call(monkeypatch, name, finish):
    stub_credentials(monkeypatch)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"]["model"])
        return httpx.Response(200, json=response_payload(name, finish=finish),
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", post)
    generator = api_generator(name, max_calls=1)
    result = generator.generate({})
    assert result.error and result.usage["finish_reason"] == finish
    assert result.raw_response and "generation_error" in result.artifact
    with pytest.raises(RuntimeError, match="budget"):
        generator.generate({})
    assert sent == [PRESETS[name]["model"]]


def test_network_failure_and_unknown_reasoning_usage(monkeypatch):
    stub_credentials(monkeypatch)
    generator = api_generator("step-5-preview", max_calls=2)
    sent = []

    def post(url, **kwargs):
        sent.append(url)
        raise httpx.ConnectError("mock connection failed")

    monkeypatch.setattr("httpx.post", post)
    result = generator.generate({})
    assert result.error and len(sent) == 1 and result.usage["api_calls"] == 1
    payload = response_payload("step-5-preview")
    del payload["usage"]["completion_tokens_details"]
    monkeypatch.setattr("httpx.post", lambda url, **kwargs: httpx.Response(
        200, json=payload, request=httpx.Request("POST", url)))
    result = generator.generate({})
    assert result.error is None and result.usage["reasoning_tokens"] is None
    assert result.usage["provider_usage"] == payload["usage"]


def test_plan_and_missing_credentials_make_no_model_calls(monkeypatch, tmp_path):
    stub_credentials(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr("httpx.post", lambda *a, **k: pytest.fail("unexpected API call"))
    plan = comparison_plan()
    assert plan["planned_max_api_calls"] == 64
    assert plan["planned_max_output_tokens"] is None
    assert comparison_plan(max_tokens=4000)["planned_max_output_tokens"] == 256000
    assert len(plan["groups"]) == 4 and plan["reasoning_policy"] == "provider-default"
    assert not plan["groups"][0]["credential_present"]
    assert "unit-test-placeholder" not in json.dumps(plan)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        run_comparison(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError, match="budget"):
        comparison_plan(rounds=3, per_round=2, max_calls=4)


@pytest.mark.parametrize("name", PRESETS)
def test_default_output_policy_does_not_apply_experiment_cap(name):
    generator = api_generator(name)
    body = generator.request_body({"conditions": []})
    assert generator.metadata()["max_tokens"] is None
    assert generator.metadata()["read_timeout_seconds"] is None
    if name == "claude-haiku-5.5":
        assert body["max_tokens"] == 128000
    else:
        assert "max_tokens" not in body
    assert not {"thinking", "reasoning", "reasoning_effort"} & set(body)


def test_model_searches_have_separate_feedback_banks_and_results(monkeypatch, tmp_path):
    stub_credentials(monkeypatch)
    requests = {p["model"]: [] for p in PRESETS.values()}
    coefficients = {name: .106 + i * .001 for i, name in enumerate(PRESETS)}

    def post(url, **kwargs):
        body = kwargs["json"]
        name = next(n for n, p in PRESETS.items() if p["model"] == body["model"])
        requests[body["model"]].append(json.loads(body["messages"][-1]["content"]))
        return httpx.Response(200, json=response_payload(name, answer(coefficients[name])),
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", post)
    previous_threads = torch.get_num_threads()
    try:
        index = run_comparison(tmp_path / "comparison", demands=["upside-protected", "upside-risk20"],
            rounds=2, per_round=1, max_calls=2, paths=128)
    finally:
        torch.set_num_threads(previous_threads)
    assert set(index["groups"]) == set(PRESETS) and "score" not in index
    first_requests = [requests[p["model"]][0] for p in PRESETS.values()]
    assert all(r == first_requests[0] for r in first_requests)
    for name, preset in PRESETS.items():
        rows = requests[preset["model"]]
        assert len(rows) == 4
        # New demand starts again with just its handwritten seed.
        assert len(rows[0]["feedback"]) == len(rows[2]["feedback"]) == 1
        for row in (rows[1], rows[3]):
            assert len(row["feedback"]) == 2
            assert row["feedback"][1]["artifact"]["contract"]["pitch"] == f"mock coefficient {coefficients[name]}"
            assert all(f["result"]["split"] == "development" for f in row["feedback"])
            assert "trajectory" not in json.dumps(row["feedback"])
        folder = tmp_path / "comparison" / name
        summary = json.loads((folder / "model_summary.json").read_text())
        assert summary["model_group"] == name and len(summary["demands"]) == 2
        for demand, run in summary["demands"].items():
            assert run["api_calls"] == 2 and run["model_group"] == name
            bank = [json.loads(line) for line in (folder / demand / "experience_bank/records.jsonl").read_text().splitlines()]
            assert len(bank) == 3 and all(r["metadata"]["model_group"] == name for r in bank)
            generations = [json.loads(line) for line in (folder / demand / "generation.jsonl").read_text().splitlines()]
            assert [g["generation"]["usage"]["call_index"] for g in generations] == [1, 2]
            assert all(g["generation"]["raw_response"] for g in generations)
