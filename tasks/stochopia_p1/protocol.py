"""Two-step preregistration and sandboxed synthesis for P1 candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from .prompt import (
    CODE_INSTRUCTION, HYPOTHESIS_FIELDS, HYPOTHESIS_TEXT_FIELDS, M5_MIN_EFFECTS, PLAN_INSTRUCTION,
    context_packet, dump_prompt, forecast_example, forecast_metrics, forecast_targets, message_packet,
)


MAX_INPUT_TOKENS = 3500
MAX_OUTPUT_TOKENS = 8192
MAX_REPAIRS = 2
MAX_HYPOTHESIS_REPAIRS = 1

Message = dict[str, str]
Messages = list[Message]
Group = Literal["evidence", "metrics"]
Direction = Literal["increase", "decrease", "equivalent"]
Stage = Literal["hypothesis", "hypothesis_repair", "code", "repair"]
PredictionRole = Literal["primary", "guardrail"]


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
    role: PredictionRole = field(kw_only=True)
    equivalence_band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _text(self.metric, "metric")
        if self.role not in ("primary", "guardrail"):
            raise ValueError("prediction role must be primary or guardrail")
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
            if not 0 <= lower < upper:
                raise ValueError("increase requires 0 <= low < high for signed delta_interval")
        elif self.direction == "decrease":
            if not lower < upper <= 0:
                raise ValueError("decrease requires low < high <= 0 for signed delta_interval")
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
class CompileAttempt:
    """One tried code response, including sources that fail the zero-data gate."""

    attempt: int
    stage: Stage
    source: str | None
    diagnostic: str | None


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


def validate_forecasts(
    group: str, predictions: tuple[StructuredPrediction, ...], registered: tuple[str, ...] = (),
    *, analysis_version: str = "m4",
) -> None:
    """Require an ordered mechanism forecast and a distinct non-degradation forecast."""
    if len(predictions) != 2:
        raise ProposalValidationError("predictions must contain exactly two items")
    primary, guardrail = predictions
    if primary.role != "primary" or guardrail.role != "guardrail":
        raise ProposalValidationError("prediction roles must be primary then guardrail")
    primary_metrics, guardrail_metrics = forecast_metrics(group, registered)
    if primary.metric not in primary_metrics:
        raise ProposalValidationError("primary forecast must use an allowed registered core metric")
    if guardrail.metric not in guardrail_metrics or primary.metric == guardrail.metric:
        raise ProposalValidationError("guardrail forecast must use a distinct allowed readout")
    if analysis_version == "m5":
        for prediction in predictions:
            if prediction.direction == "equivalent":
                continue
            minimum = M5_MIN_EFFECTS.get(prediction.metric)
            if minimum is None:
                raise ProposalValidationError(f"no M5 directional effect threshold for {prediction.metric}")
            bound = prediction.interval[0] if prediction.direction == "increase" else prediction.interval[1]
            if abs(bound) < minimum:
                raise ProposalValidationError(
                    f"{prediction.metric} requires a near-zero directional bound magnitude >= {minimum:g}"
                )


def _parse_prediction(item: object) -> StructuredPrediction:
    if not isinstance(item, dict):
        raise ValueError("each prediction must be a JSON object")
    delta = item.get("delta_interval", item.get("interval"))
    direction = item.get("direction")
    band = item.get("equivalence_band") if direction == "equivalent" else None
    if delta is None and direction == "equivalent":
        delta = band
    if delta is None:
        raise ValueError("provide delta_interval, or equivalence_band for an equivalent forecast")
    return StructuredPrediction(item.get("metric"), direction, delta, role=item.get("role"), equivalence_band=band)


def _parse_plan(
    text: str, trial_id: str, active_parent_hypothesis_id: str | None,
) -> tuple[DynamicHypothesis, tuple[StructuredPrediction, ...]]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("hypothesis"), dict):
        raise ValueError("Call 1 requires a hypothesis object and a predictions list")
    entries = payload.get("predictions")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("predictions must contain exactly two items")
    try:
        content = {name: payload["hypothesis"].get(name) for name in HYPOTHESIS_TEXT_FIELDS}
        hypothesis = DynamicHypothesis(f"{trial_id}-hyp", active_parent_hypothesis_id, **content)
        predictions = tuple(_parse_prediction(item) for item in entries)
    except TypeError as exc:
        raise ValueError(f"invalid prediction or hypothesis fields: {exc}") from exc
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


class TwoStepProposer:
    """Inject provider, tokenizer, usage recorder and the zero-data gate."""

    def __init__(self, transport: Transport, gate: Gate, record_usage: Callable[[CallUsage], None],
                 *, count_tokens: Callable[[Messages], int], core_metrics: tuple[str, ...] = (),
                 record_compile: Callable[[CompileAttempt], None] | None = None,
                 read_compile: Callable[[int], CompileAttempt | None] | None = None,
                 search_seed: int | None = None, search_round: int | None = None,
                 analysis_version: str = "m4") -> None:
        self.transport = transport
        self.gate = gate
        self.record_usage = record_usage
        self.count_tokens = count_tokens
        self.core_metrics = core_metrics
        self.record_compile = record_compile
        self.read_compile = read_compile
        self.search_seed = search_seed
        self.search_round = search_round
        self.analysis_version = analysis_version

    def _bounded(self, instruction: str, context: str, recent: list[dict[str, object]],
                 suffix: str = "", *, include_contract: bool = True) -> Messages:
        for removed in range(len(recent) + 1):
            packet = message_packet(instruction, context, recent[removed:], suffix, include_contract=include_contract)
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
                with_draft = message_packet(
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
        reply = self.transport(messages, response_json=stage in ("hypothesis", "hypothesis_repair"))
        self.record_usage(CallUsage(stage, reply.input_tokens, reply.output_tokens))
        if reply.input_tokens > MAX_INPUT_TOKENS or reply.output_tokens > MAX_OUTPUT_TOKENS:
            raise ProposalValidationError("provider reported token use above the registered call limit")
        return reply

    def propose(self, group: Group, parent_source: str, parent_evidence: Mapping[str, object],
                recent: History, seal_callback: Seal, *, trial_id: str,
                active_parent_hypothesis_id: str | None = None) -> SynthesisResult:
        """Repair one invalid JSON plan before sealing and synthesizing its candidate."""
        _text(trial_id, "trial_id")
        context, history = context_packet(group, parent_source, parent_evidence, recent)
        targets = forecast_targets(group, self.core_metrics, analysis_version=self.analysis_version)
        if self.search_seed is not None:
            targets += f" Independent search replicate seed: {self.search_seed}."
        if self.search_round is not None:
            targets += f" Search round {self.search_round}."
        example = forecast_example(group, self.core_metrics)
        diagnostic = ""
        for attempt in range(MAX_HYPOTHESIS_REPAIRS + 1):
            suffix = f"{targets}\n{diagnostic}{example}"
            packet = self._bounded(PLAN_INSTRUCTION, context, history, suffix, include_contract=False)
            stage = "hypothesis_repair" if attempt else "hypothesis"
            reply = self._call(packet, stage)
            try:
                if reply.error is not None:
                    raise ValueError(reply.error)
                hypothesis, predictions = _parse_plan(reply.text, trial_id, active_parent_hypothesis_id)
                validate_forecasts(group, predictions, self.core_metrics, analysis_version=self.analysis_version)
            except ValueError as exc:
                diagnostic = f"Previous JSON rejected: {exc}. Return a corrected JSON object.\n"
                continue
            seal_callback(hypothesis, predictions)
            return self.synthesize(group, parent_source, parent_evidence, recent, hypothesis, predictions)
        raise ProposalValidationError(diagnostic.strip())

    def synthesize(self, group: Group, parent_source: str, parent_evidence: Mapping[str, object],
                   recent: History, hypothesis: DynamicHypothesis,
                   predictions: tuple[StructuredPrediction, ...]) -> SynthesisResult:
        """Continue from an existing seal without sending Call 1 twice."""
        validate_forecasts(group, predictions, self.core_metrics, analysis_version=self.analysis_version)
        context, history = context_packet(group, parent_source, parent_evidence, recent)
        sealed = "Sealed proposal:\n" + dump_prompt({
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
            failed_draft = reply.text if reply.error is None else ""
            compiled = self.read_compile(attempt + 1) if self.read_compile is not None else None
            if compiled is None:
                compiled = self._compile(reply, attempt + 1, stage)
                if self.record_compile is not None:
                    self.record_compile(compiled)
            if compiled.diagnostic is None:
                return SynthesisResult(hypothesis, predictions, compiled.source, False)
            error = compiled.diagnostic
        return SynthesisResult(hypothesis, predictions, None, True, error)

    def _compile(self, reply: ProviderReply, attempt: int, stage: Stage) -> CompileAttempt:
        if reply.error is not None:
            return CompileAttempt(attempt, stage, None, reply.error)
        try:
            source = extract_source(reply.text)
        except ValueError as exc:
            return CompileAttempt(attempt, stage, None, str(exc))
        return CompileAttempt(attempt, stage, source, self.gate(source))
