"""Two-step preregistration and sandboxed synthesis for P1 candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol


MAX_INPUT_TOKENS = 3500
MAX_OUTPUT_TOKENS = 8192
MAX_REPAIRS = 2

# These fields are the only experiment readouts allowed into the proposer.
METRIC_CONTEXT = ("parameters", "state_dim", "val_nll_total", "lob_impact_score")
EVIDENCE_CONTEXT = (
    "nll_dt", "nll_action", "nll_rel_ticks", "nll_volume",
    "event_count_delta", "zero_wait_rate_delta", "sign_acf_delta", "hurst_delta",
    "top_depth_delta", "idle_abs_drift", "impact_peak", "blind_ratio", "impact_y",
    "impact_delta", "recovery_ratio",
)
HYPOTHESIS_FIELDS = (
    "hypothesis_id", "parent_hypothesis_id", "observed_bottleneck",
    "operator_diagnosis", "structural_prescription", "falsifier",
)
CONTRACT = """PyTorch LOB contract (stochopia.neural.lob_interface):
StepObservation(book_ticks: Tensor, book_shares: Tensor, previous_event: Tensor,
is_own: Tensor, has_deep: Tensor | None = None). Pre-event book shape [...,2,10],
previous tokens [...,4], own flag has one value per prefix.
LOBBatch(observation: StepObservation, targets: Tensor); targets [B,T,4], bins (16,6,26,12).
StepSampler(model: LOBWorldModel, context: Tensor, observation: StepObservation).
Methods: conditioned(targets: Tensor) -> FiveTupleLogits; sample(generator: Generator | None = None) -> Tensor.
LOBWorldModel methods to implement:
sequence_context(observation: StepObservation) -> Tensor,
step_context(observation: StepObservation, state: State) -> tuple[Tensor, State],
head_logits(context: Tensor, stage: int, prior: tuple[Tensor,...]) -> Tensor.
Keep inherited forward_sequence(batch: LOBBatch) -> FiveTupleLogits and
forward_step(step_obs: StepObservation, hidden_state: State = None) -> tuple[StepSampler, State].
State is None, a Tensor, or a tuple of Tensors. Use at most 28000 trainable params and
128 real state scalars (complex counts twice). No cross-time access to the future.
From stochopia.neural.primitives:
parallel_affine_scan(gates: Tensor, inputs: Tensor, h0: Tensor | None = None, chunk_size: int = 32) -> Tensor,
affine_scan_step(gate_t: Tensor, input_t: Tensor, h_prev: Tensor) -> Tensor,
real_diagonal(decay_logit: Tensor) -> Tensor, complex_diagonal(decay_logit: Tensor, phase: Tensor) -> Tensor."""
PLAN_INSTRUCTION = """Return one JSON object and no code. Keys: hypothesis and predictions.
hypothesis fields: hypothesis_id, parent_hypothesis_id (null only for a root),
observed_bottleneck, operator_diagnosis, structural_prescription, falsifier.
predictions is a nonempty list. Each item has metric, direction (increase,
decrease or equivalent), interval [low,high], and optional equivalence_band
[low,high] (required for equivalent). Predict candidate minus parent.
Explicitly revise or open a competing hypothesis when feedback refutes it."""
CODE_INSTRUCTION = """Return exactly one fenced python code block. Define one self-contained
class CandidateLOBWorldModel(LOBWorldModel), constructible with no arguments.
Only import standard math libraries, torch, torch.nn, torch.nn.functional,
and stochopia.neural.lob_interface or stochopia.neural.primitives.
Implement the sealed prescription. Keep evaluation and data out of the code.
Keep source compact so the entire file can be included as a later parent."""

Message = dict[str, str]
Messages = list[Message]
Group = Literal["evidence", "metrics"]
Direction = Literal["increase", "decrease", "equivalent"]
Stage = Literal["hypothesis", "code", "repair"]


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")


def _bounds(values: Sequence[float], field: str) -> tuple[float, float]:
    if len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f"{field} must contain two finite numbers")
    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        raise ValueError(f"{field} must be ordered")
    return lower, upper


@dataclass(frozen=True)
class DynamicHypothesis:
    hypothesis_id: str
    parent_hypothesis_id: str | None
    observed_bottleneck: str
    operator_diagnosis: str
    structural_prescription: str
    falsifier: str

    def __post_init__(self) -> None:
        for field in HYPOTHESIS_FIELDS:
            value = getattr(self, field)
            if field == "parent_hypothesis_id" and value is None:
                continue
            _text(value, field)
        if self.hypothesis_id == self.parent_hypothesis_id:
            raise ValueError("a hypothesis cannot be its own parent")


@dataclass(frozen=True)
class StructuredPrediction:
    """An ex ante interval for candidate minus parent on one named readout."""

    metric: str
    direction: Direction
    interval: tuple[float, float]
    equivalence_band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _text(self.metric, "metric")
        lower, upper = _bounds(self.interval, "interval")
        object.__setattr__(self, "interval", (lower, upper))
        if self.equivalence_band is not None:
            band = _bounds(self.equivalence_band, "equivalence_band")
            if not band[0] < 0 < band[1]:
                raise ValueError("equivalence_band must straddle zero")
            object.__setattr__(self, "equivalence_band", band)
        if self.direction == "equivalent":
            band = self.equivalence_band
            if band is None or not band[0] <= lower <= upper <= band[1]:
                raise ValueError("equivalent predictions require an interval inside equivalence_band")
        elif self.direction == "increase":
            if lower <= 0:
                raise ValueError("increase requires a strictly positive interval")
        elif self.direction == "decrease":
            if upper >= 0:
                raise ValueError("decrease requires a strictly negative interval")
        else:
            raise ValueError("unknown prediction direction")


@dataclass(frozen=True)
class ProviderReply:
    text: str
    input_tokens: int
    output_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class CallUsage:
    stage: Stage
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SynthesisResult:
    hypothesis: DynamicHypothesis
    predictions: tuple[StructuredPrediction, ...]
    source: str | None
    invalid_program: bool
    error: str | None = None


class Transport(Protocol):
    def __call__(self, messages: Messages, *, response_json: bool) -> ProviderReply: ...


Seal = Callable[[DynamicHypothesis, tuple[StructuredPrediction, ...]], None]
Gate = Callable[[str], str | None]
History = Sequence[Mapping[str, object]]


class ContextLimitExceeded(ValueError):
    """Required prompt contents cannot fit within the registered input cap."""


class ProposalValidationError(ValueError):
    """Call 1 did not produce a valid hypothesis and prediction package."""


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _parse_plan(text: str) -> tuple[DynamicHypothesis, tuple[StructuredPrediction, ...]]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"hypothesis", "predictions"}:
        raise ValueError("Call 1 requires exactly hypothesis and predictions")
    if not isinstance(payload["predictions"], list) or not payload["predictions"]:
        raise ValueError("predictions must be a nonempty list")
    try:
        hypothesis = DynamicHypothesis(**payload["hypothesis"])
        predictions = tuple(StructuredPrediction(**item) for item in payload["predictions"])
    except TypeError as exc:
        raise ValueError(f"invalid prediction or hypothesis fields: {exc}") from exc
    if len({item.metric for item in predictions}) != len(predictions):
        raise ValueError("predict each metric at most once")
    return hypothesis, predictions


def extract_source(text: str) -> str:
    """Consume exactly one Python code fence without any surrounding prose."""
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0] != "```python" or lines[-1] != "```":
        raise ValueError("Call 2 requires one fenced python code block")
    body = lines[1:-1]
    if any(line.strip().startswith("```") for line in body) or not any(line.strip() for line in body):
        raise ValueError("Call 2 requires one nonempty Python code block")
    return "\n".join(body) + "\n"


def _context(group: Group, source: str, evidence: Mapping[str, object],
             recent: History) -> tuple[str, list[dict[str, object]]]:
    if group not in ("evidence", "metrics"):
        raise ValueError(f"unsupported proposal group {group!r}")
    keys = METRIC_CONTEXT
    if group == "evidence":
        keys += EVIDENCE_CONTEXT
    row = {}
    for key in keys:
        value = evidence.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            row[key] = value
        else:
            row[key] = None
    feedback = []
    if group == "evidence":
        for item in recent[-3:]:
            hypothesis = item["hypothesis"]
            if not isinstance(hypothesis, Mapping):
                raise ValueError("recent hypotheses must be mappings")
            feedback.append({
                "hypothesis": {key: hypothesis.get(key) for key in HYPOTHESIS_FIELDS},
                "verdict": item["verdict"],
            })
    return f"Parent Python source:\n{source}\nDevelopment summary:\n{_dump(row)}", feedback


def _messages(instruction: str, context: str, recent: list[dict[str, object]],
              suffix: str = "") -> Messages:
    content = context
    if recent:
        content += f"\nRecent group history:\n{_dump(recent)}"
    if suffix:
        content += f"\n{suffix}"
    return [{"role": "system", "content": f"{CONTRACT}\n\n{instruction}"},
            {"role": "user", "content": content}]


class TwoStepProposer:
    """Inject provider, tokenizer, usage recorder and the zero-data gate."""

    def __init__(self, transport: Transport, gate: Gate, record_usage: Callable[[CallUsage], None],
                 *, count_tokens: Callable[[Messages], int], core_metrics: tuple[str, ...] = ()) -> None:
        self.transport = transport
        self.gate = gate
        self.record_usage = record_usage
        self.count_tokens = count_tokens
        self.core_metrics = core_metrics

    def _bounded(self, instruction: str, context: str, recent: list[dict[str, object]],
                 suffix: str = "") -> Messages:
        for removed in range(len(recent) + 1):
            packet = _messages(instruction, context, recent[removed:], suffix)
            if self.count_tokens(packet) <= MAX_INPUT_TOKENS:
                return packet
        raise ContextLimitExceeded(
            f"complete parent and required context exceed {MAX_INPUT_TOKENS} input tokens"
        )

    def _repair_packet(self, context: str, recent: list[dict[str, object]], sealed: str,
                       diagnostic: str, failed_draft: str) -> Messages:
        """Keep complete parent and plan, retaining the exact error before draft and history."""
        lines = diagnostic.splitlines()
        for start in range(len(lines)):
            retained = "\n".join(lines[start:])
            suffix = f"{sealed}\nCompiler failure:\n{retained}"
            try:
                packet = self._bounded(CODE_INSTRUCTION, context, recent, suffix)
            except ContextLimitExceeded:
                continue
            if failed_draft:
                with_draft = _messages(
                    CODE_INSTRUCTION, context, [], f"{suffix}\nFailed draft:\n{failed_draft}"
                )
                if self.count_tokens(with_draft) <= MAX_INPUT_TOKENS:
                    return with_draft
            return packet
        raise ContextLimitExceeded(
            f"complete parent, sealed proposal and compiler diagnostic exceed "
            f"{MAX_INPUT_TOKENS} input tokens"
        )

    def _call(self, messages: Messages, stage: Stage) -> ProviderReply:
        reply = self.transport(messages, response_json=stage == "hypothesis")
        self.record_usage(CallUsage(stage, reply.input_tokens, reply.output_tokens))
        if reply.input_tokens > MAX_INPUT_TOKENS or reply.output_tokens > MAX_OUTPUT_TOKENS:
            raise ValueError("provider reported token use above the registered call limit")
        return reply

    def propose(self, group: Group, parent_source: str, parent_evidence: Mapping[str, object],
                recent: History, seal_callback: Seal) -> SynthesisResult:
        context, history = _context(group, parent_source, parent_evidence, recent)
        targets = ""
        if self.core_metrics:
            targets = f"Use one registered metric for the first forecast: {_dump(self.core_metrics)}"
        packet = self._bounded(PLAN_INSTRUCTION, context, history, targets)
        reply = self._call(packet, "hypothesis")
        if reply.error is not None:
            raise ProposalValidationError(f"Call 1 failed before preregistration: {reply.error}")
        try:
            hypothesis, predictions = _parse_plan(reply.text)
        except ValueError as exc:
            raise ProposalValidationError(str(exc)) from exc
        seal_callback(hypothesis, predictions)
        return self.synthesize(group, parent_source, parent_evidence, recent, hypothesis, predictions)

    def synthesize(self, group: Group, parent_source: str, parent_evidence: Mapping[str, object],
                   recent: History, hypothesis: DynamicHypothesis,
                   predictions: tuple[StructuredPrediction, ...]) -> SynthesisResult:
        """Continue from an existing seal without sending Call 1 twice."""
        context, history = _context(group, parent_source, parent_evidence, recent)
        sealed = "Sealed proposal:\n" + _dump({
            "hypothesis": asdict(hypothesis), "predictions": [asdict(item) for item in predictions],
        })
        error = ""
        failed_draft = ""
        for attempt in range(MAX_REPAIRS + 1):
            stage = "repair" if attempt else "code"
            if attempt:
                try:
                    packet = self._repair_packet(context, history, sealed, error, failed_draft)
                except ContextLimitExceeded as exc:
                    return SynthesisResult(hypothesis, predictions, None, True, f"{error}\n{exc}")
            else:
                packet = self._bounded(CODE_INSTRUCTION, context, history, sealed)
            reply = self._call(packet, stage)
            if reply.error is not None:
                error = reply.error
                failed_draft = ""
                continue
            failed_draft = reply.text
            try:
                source = extract_source(reply.text)
            except ValueError as exc:
                error = str(exc)
                continue
            diagnostic = self.gate(source)
            if diagnostic is None:
                return SynthesisResult(hypothesis, predictions, source, False)
            error = diagnostic
        return SynthesisResult(hypothesis, predictions, None, True, error)
