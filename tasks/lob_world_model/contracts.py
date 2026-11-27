"""Data and transport contracts shared by the world-model research harness."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


MAX_OUTPUT_TOKENS = 8192
MAX_REPAIRS = 2
MAX_HYPOTHESIS_REPAIRS = 1

Message = dict[str, str]
Messages = list[Message]
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


class Transport(Protocol):
    def __call__(self, messages: Messages, *, response_json: bool) -> ProviderReply: ...


class RequestAdapter(Protocol):
    """Provider transport and durable metered-usage sink, independent of proposal design."""

    def transport(self, messages: Messages, *, response_json: bool) -> ProviderReply: ...

    def record_usage(self, usage: CallUsage) -> None: ...
