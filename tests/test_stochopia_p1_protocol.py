"""P1 protocol and native zero-data sandbox acceptance tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from stochopia import __file__ as STOCHOPIA_SOURCE
from stochopia.llm import BaseLLMClient, LLMTruncationError, ModelConfig, OpenAICompatClient
from stochopia.neural.primitives import seed_lob_source
from tasks.stochopia_p1.gate import CandidateGate
from tasks.stochopia_p1.protocol import (
    CallUsage,
    DynamicHypothesis,
    Messages,
    ProposalValidationError,
    ProviderReply,
    StructuredPrediction,
    TwoStepProposer,
)
from tasks.stochopia_p1.provider import registry_transport


PLAN = DynamicHypothesis(
    "depth_memory", None, "depth recovers too slowly", "short memory",
    "use a bank of decay scales", "depth recovery stays unchanged",
)
PREDICTIONS = (StructuredPrediction("recovery_ratio", "decrease", (-0.3, -0.1)),)
SEED = """from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    pass
"""
MULTISCALE = """import torch
from torch import nn
from stochopia.neural.primitives import SeedGRULOBModel, affine_scan_step, parallel_affine_scan, real_diagonal
class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        super().__init__()
        self.decay = nn.Parameter(torch.linspace(-2, 2, 48))

    def sequence_context(self, observation):
        inputs = self._features(observation)
        gates = real_diagonal(self.decay).expand_as(inputs)
        return parallel_affine_scan(gates, inputs)

    def step_context(self, observation, state):
        inputs = self._features(observation)
        if state is None:
            state = torch.zeros_like(inputs)
        current = affine_scan_step(real_diagonal(self.decay), inputs, state)
        return current, current
"""
BAD_MATRIX = """from torch import nn
from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        super().__init__()
        self.bad = nn.Linear(2, 4)

    def sequence_context(self, observation):
        return self.bad(super().sequence_context(observation))
"""
FUTURE = """from torch import nn
from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        super().__init__(state_dim=32)
        self.future = nn.Conv1d(32, 32, kernel_size=3, padding=1)

    def sequence_context(self, observation):
        inputs = super().sequence_context(observation)
        return self.future(inputs.transpose(1, 2)).transpose(1, 2)
"""
OVER_BUDGET = """import torch
from torch import nn
from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        super().__init__()
        self.extra = nn.Parameter(torch.zeros(28001))
"""
NONFINITE_GRADIENT = """import torch
from torch import nn
from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        super().__init__()
        self.extra = nn.Parameter(torch.zeros(()))

    def head_logits(self, context, stage, prior):
        logits = super().head_logits(context, stage, prior)
        return logits + self.extra.sqrt()
"""


def words(messages: Messages) -> int:
    """Deterministic fixture tokenizer; provider-specific counts are injected in use."""
    return sum(len(message["content"].split()) for message in messages)


def plan_json() -> str:
    return json.dumps({"hypothesis": asdict(PLAN), "predictions": [asdict(p) for p in PREDICTIONS]})


def fence(source: str) -> str:
    return "```python\n" + source.rstrip() + "\n```"


def test_actual_compiler_repair_records_seal_before_synthesis(tmp_path: Path) -> None:
    gate = CandidateGate()
    usage: list[CallUsage] = []
    modes: list[bool] = []
    seal = tmp_path / "presynthesis.json"

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        modes.append(response_json)
        message = messages[-1]["content"]
        if len(modes) == 1:
            assert not seal.exists()
            return ProviderReply(plan_json(), 257, 72)
        assert json.loads(seal.read_text()) == json.loads(plan_json())
        if len(modes) == 2:
            assert "Sealed proposal" in message
            return ProviderReply(fence(BAD_MATRIX), 340, 90)
        assert "mat1 and mat2 shapes cannot be multiplied" in message
        assert "<sandbox>" in message
        return ProviderReply(fence(MULTISCALE), 513, 220)

    def save(hypothesis: DynamicHypothesis, predictions: tuple[StructuredPrediction, ...]) -> None:
        content = {"hypothesis": asdict(hypothesis), "predictions": [asdict(p) for p in predictions]}
        seal.write_text(json.dumps(content), encoding="utf-8")

    proposer = TwoStepProposer(transport, gate, usage.append, count_tokens=words)
    result = proposer.propose("evidence", SEED, {"parameters": 21_932, "state_dim": 48}, (), save)
    assert modes == [True, False, False]
    assert result.source == MULTISCALE
    assert result.hypothesis == PLAN and result.predictions == PREDICTIONS
    assert not result.invalid_program
    assert usage == [CallUsage("hypothesis", 257, 72), CallUsage("code", 340, 90), CallUsage("repair", 513, 220)]


@pytest.mark.parametrize("model_name", ["gpt-6.2-sol", "deepseek-flash"])
def test_registered_provider_binding_preserves_settings_and_counts_truncation(model_name: str) -> None:
    configs: list[ModelConfig] = []
    usage: list[CallUsage] = []

    class FixtureClient(OpenAICompatClient):
        async def chat(self, messages: Messages, temperature: float | None = None,
                       max_tokens: int | None = None, seed: int | None = None) -> str:
            assert max_tokens == 8192 and temperature is None and seed is None
            assert self.RETRY_DELAYS == () and self.TRUNCATION_RETRIES == 0
            self.total_usage["prompt_tokens"] = 200
            if len(configs) == 1:
                self.total_usage["completion_tokens"] = 300
                return plan_json()
            if len(configs) == 2:
                self.total_usage["completion_tokens"] = 8192
                raise LLMTruncationError("fixture reached limit")
            assert "8192-token output cap" in messages[-1]["content"]
            self.total_usage["completion_tokens"] = 400
            return fence(SEED)

    def factory(name: str, registry: dict[str, ModelConfig]) -> BaseLLMClient:
        config = registry[name]
        assert name == config.model == model_name
        configs.append(config)
        return FixtureClient(config)

    models_path = Path(STOCHOPIA_SOURCE).resolve().parents[1] / "config" / "p1-lob-models.yaml"
    transport = registry_transport(models_path, model_name, client_factory=factory)
    assert not configs
    proposer = TwoStepProposer(transport, lambda source: None, usage.append, count_tokens=words)
    result = proposer.propose("metrics", SEED, {}, (), lambda hypothesis, predictions: None)
    assert result.source == SEED
    assert [config.response_json for config in configs] == [True, False, False]
    assert all(config.timeout == 180 and config.temperature is None for config in configs)
    assert all(config.max_tokens == 8192 and config.strict_model_match for config in configs)
    assert all(config.use_max_completion_tokens == (model_name == "gpt-6.2-sol") for config in configs)
    assert usage == [CallUsage("hypothesis", 200, 300), CallUsage("code", 200, 8192), CallUsage("repair", 200, 400)]


@pytest.mark.parametrize(("source", "error"), [
    (FUTURE, "Tensor-likes are not close"),
    (OVER_BUDGET, "model exceeds max_params"),
])
def test_actual_sandbox_rejects_noncausal_or_oversized_model(source: str, error: str) -> None:
    assert error in (CandidateGate()(source) or "")


def test_actual_sandbox_rejects_singular_backward_with_finite_forward() -> None:
    assert "missing or nonfinite training gradients" in (CandidateGate()(NONFINITE_GRADIENT) or "")


def test_actual_sandbox_denies_host_read_and_write(tmp_path: Path) -> None:
    secret = tmp_path / "synthetic-private.txt"
    secret.write_text("local fixture", encoding="utf-8")
    outside = tmp_path / "unapproved.txt"
    gate = CandidateGate()
    read = f"open({str(secret)!r}, 'r').read()\n"
    write = f"open({str(outside)!r}, 'w').write('blocked')\n"
    assert "PermissionError" in (gate(read + SEED) or "")
    assert "PermissionError" in (gate(write + SEED) or "")
    assert not outside.exists()


def test_actual_sandbox_accepts_the_exact_seed_source() -> None:
    assert CandidateGate()(seed_lob_source()) is None


@pytest.mark.parametrize("module", ["socket", "torch.storage"])
def test_candidate_imports_are_checked_at_runtime(module: str) -> None:
    assert f"Candidate cannot import {module!r}" in (CandidateGate()(f"import {module}\n" + SEED) or "")


@pytest.mark.parametrize("group", ["evidence", "metrics"])
def test_context_projects_only_group_allowed_fields_and_three_attempts(group: str) -> None:
    seen: list[str] = []
    evidence = {
        "parameters": 100, "state_dim": 16, "val_nll_total": 7.8, "lob_impact_score": 20.5,
        "nll_dt": 2.17, "raw_cache": "raw-forbidden", "experimental_holdout": "holdout-forbidden",
    }
    recent = [
        {"hypothesis": {**asdict(PLAN), "hypothesis_id": f"older-{idx}", "raw": "raw-history"},
         "verdict": "REFUTED"}
        for idx in range(4)
    ]

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        seen.append(messages[-1]["content"])
        return ProviderReply(plan_json() if response_json else fence(SEED), 100, 10)

    proposer = TwoStepProposer(
        transport, lambda source: None, lambda usage: None, count_tokens=words, core_metrics=("recovery_ratio",),
    )
    proposer.propose(group, SEED, evidence, recent, lambda hypothesis, predictions: None)
    assert len(seen) == 2
    assert 'Use one registered metric for the first forecast: ["recovery_ratio"]' in seen[0]
    for prompt in seen:
        assert SEED in prompt
        assert '"val_nll_total":7.8' in prompt and '"lob_impact_score":20.5' in prompt
        assert "raw-forbidden" not in prompt and "holdout-forbidden" not in prompt
        assert "raw-history" not in prompt and "older-0" not in prompt
        if group == "evidence":
            assert '"nll_dt":2.17' in prompt and "older-3" in prompt
        else:
            assert "nll_dt" not in prompt and "older-3" not in prompt


def test_exhausted_repair_resumes_a_sealed_proposal_without_call_one() -> None:
    usage: list[CallUsage] = []

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert not response_json
        return ProviderReply(fence(BAD_MATRIX), 100, 12)

    proposer = TwoStepProposer(transport, lambda source: "matrix incompatible", usage.append,
                               count_tokens=words)
    result = proposer.synthesize("metrics", SEED, {}, (), PLAN, PREDICTIONS)
    assert [item.stage for item in usage] == ["code", "repair", "repair"]
    assert result.invalid_program and result.source is None
    assert result.error == "matrix incompatible"


def test_repair_overflow_keeps_complete_parent_and_exact_terminal_error() -> None:
    usage: list[CallUsage] = []
    long_trace = "trace-context " * 3501 + "\nRuntimeError: exact matrix error"

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        if not usage:
            return ProviderReply(fence(BAD_MATRIX), 100, 12)
        prompt = messages[-1]["content"]
        assert SEED in prompt
        assert "RuntimeError: exact matrix error" in prompt
        assert "trace-context" not in prompt
        assert words(messages) <= 3500
        return ProviderReply(fence(SEED), 100, 12)

    def gate(source: str) -> str | None:
        if source == BAD_MATRIX:
            return long_trace
        return None

    proposer = TwoStepProposer(transport, gate, usage.append, count_tokens=words)
    result = proposer.synthesize("metrics", SEED, {}, (), PLAN, PREDICTIONS)
    assert result.source == SEED
    assert [item.stage for item in usage] == ["code", "repair"]


def test_context_limit_rejects_required_input_without_truncating_parent() -> None:
    def forbidden_transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        pytest.fail("over-budget context reached transport")

    proposer = TwoStepProposer(forbidden_transport, lambda source: None, lambda usage: None,
                               count_tokens=words)
    with pytest.raises(ValueError, match="3500"):
        proposer.propose("metrics", "source " * 3501, {}, (), lambda hypothesis, predictions: None)


@pytest.mark.parametrize("reply", [ProviderReply("{", 10, 5), ProviderReply("", 10, 8192, "truncated")])
def test_invalid_call_one_is_typed_and_never_sealed(reply: ProviderReply) -> None:
    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        assert response_json
        return reply

    def forbidden_seal(hypothesis: DynamicHypothesis, predictions: tuple[StructuredPrediction, ...]) -> None:
        pytest.fail("invalid Call 1 reply was sealed")

    usage: list[CallUsage] = []
    proposer = TwoStepProposer(transport, lambda source: None, usage.append, count_tokens=words)
    with pytest.raises(ProposalValidationError):
        proposer.propose("metrics", SEED, {}, (), forbidden_seal)
    assert usage == [CallUsage("hypothesis", reply.input_tokens, reply.output_tokens)]
