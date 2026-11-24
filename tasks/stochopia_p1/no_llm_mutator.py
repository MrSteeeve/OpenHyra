"""Generate deterministic operator variants for the no-LLM search arm."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from textwrap import dedent, indent

from .protocol import StructuredPrediction


@dataclass(frozen=True)
class Mutation:
    family: str
    width: int
    source: str


_HEADER = """\
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stochopia.neural.lob_interface import State, StepObservation, decode_events
from stochopia.neural.primitives import (
    SeedGRULOBModel,
    affine_scan_step,
    complex_diagonal,
    parallel_affine_scan,
    real_diagonal,
)

WIDTH = {width}
SCALE = {scale}


class CandidateLOBWorldModel(SeedGRULOBModel):
"""


# Scan operators retain the seed encoder and conditional heads, replacing its GRU.
_LRU = """
    def __init__(self) -> None:
        super().__init__(state_dim=WIDTH)
        del self.gru
        self.drive = nn.Linear(WIDTH, WIDTH)
        self.decay_logit = nn.Parameter(torch.linspace(0.3, 4.0, WIDTH) * SCALE)

    def _transition(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        decay = real_diagonal(self.decay_logit)
        drive = self.drive(self._features(observation))
        drive = drive * (1 - decay.square()).clamp_min(1e-6).sqrt()
        return decay.expand_as(drive), drive

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gate, drive = self._transition(observation)
        return parallel_affine_scan(gate, drive)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        gate, drive = self._transition(observation)
        prior = torch.zeros_like(drive) if state is None else state
        hidden = affine_scan_step(gate, drive, prior)
        return hidden, hidden
"""


_S5 = """
    def __init__(self) -> None:
        super().__init__(state_dim=WIDTH)
        del self.gru
        self.drive = nn.Linear(WIDTH, 2 * WIDTH)
        self.decay_logit = nn.Parameter(torch.linspace(0.3, 4.0, WIDTH))
        self.phase = nn.Parameter(torch.linspace(0.02, 0.9, WIDTH) * SCALE)
        self.readout = nn.Linear(2 * WIDTH, WIDTH)

    def _transition(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        real, imag = self.drive(self._features(observation)).chunk(2, dim=-1)
        drive = torch.complex(real, imag)
        gate = complex_diagonal(self.decay_logit, self.phase)
        return gate.expand_as(drive), drive

    def _readout(self, states: Tensor) -> Tensor:
        return self.readout(torch.cat((states.real, states.imag), dim=-1))

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gate, drive = self._transition(observation)
        return self._readout(parallel_affine_scan(gate, drive))

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        gate, drive = self._transition(observation)
        prior = torch.zeros_like(drive) if state is None else state
        hidden = affine_scan_step(gate, drive, prior)
        return self._readout(hidden), hidden
"""


_SELECTIVE_SCAN = """
    def __init__(self) -> None:
        super().__init__(state_dim=WIDTH)
        del self.gru
        self.drive = nn.Linear(WIDTH, WIDTH)
        self.step_size = nn.Linear(WIDTH, WIDTH)
        self.decay_logit = nn.Parameter(torch.linspace(-3.5, -0.5, WIDTH))

    def _transition(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        features = self._features(observation)
        elapsed_ms = decode_events(observation.previous_event)[..., :1]
        elapsed_ms = elapsed_ms.to(features.dtype)
        gap = torch.log1p(elapsed_ms / 10)
        step_size = F.softplus(self.step_size(features) + SCALE * gap)
        gate = torch.exp(-step_size * F.softplus(self.decay_logit))
        drive = (1 - gate) * self.drive(features)
        return gate, drive

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gate, drive = self._transition(observation)
        return parallel_affine_scan(gate, drive)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        gate, drive = self._transition(observation)
        prior = torch.zeros_like(drive) if state is None else state
        hidden = affine_scan_step(gate, drive, prior)
        return hidden, hidden
"""


_IMBALANCE_HEAD = """
    def __init__(self) -> None:
        super().__init__(state_dim=WIDTH)
        self.imbalance_gate = nn.Linear(1, WIDTH)

    def _imbalance(self, observation: StepObservation) -> Tensor:
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        bid, ask = shares.sum(dim=-1).unbind(dim=-1)
        return ((bid - ask) / (bid + ask).clamp_min(1)).unsqueeze(-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        hidden = super().sequence_context(observation)
        return torch.cat((hidden, self._imbalance(observation)), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        hidden, carry = super().step_context(observation, state)
        return torch.cat((hidden, self._imbalance(observation)), dim=-1), carry

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        hidden = context[..., :WIDTH]
        if stage == 1:
            imbalance = SCALE * context[..., WIDTH:]
            hidden = hidden * (2 * self.imbalance_gate(imbalance).sigmoid())
        return super().head_logits(hidden, stage, prior)
"""


_DUAL_STREAM = """
    def __init__(self) -> None:
        super().__init__(state_dim=WIDTH)
        del self.gru
        half = WIDTH // 2
        self.fast_drive = nn.Linear(WIDTH, half)
        self.slow_drive = nn.Linear(2, half)
        self.fast_decay_logit = nn.Parameter(torch.linspace(-1.3, 0.3, half))
        self.slow_decay_logit = nn.Parameter(torch.linspace(2.3, 4.0, half) * SCALE)
        self.merge = nn.Linear(WIDTH, WIDTH)

    def _transition(self, observation: StepObservation) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        fast_gate = real_diagonal(self.fast_decay_logit)
        fast_drive = (1 - fast_gate) * self.fast_drive(self._features(observation))

        shares = observation.book_shares.to(fast_drive.dtype)
        depth = torch.log1p(shares.sum(dim=-1) / 100)
        slow_gate = real_diagonal(self.slow_decay_logit)
        slow_drive = (1 - slow_gate) * self.slow_drive(depth)
        return fast_gate.expand_as(fast_drive), fast_drive, slow_gate.expand_as(slow_drive), slow_drive

    def sequence_context(self, observation: StepObservation) -> Tensor:
        fast_gate, fast_drive, slow_gate, slow_drive = self._transition(observation)
        fast = parallel_affine_scan(fast_gate, fast_drive)
        slow = parallel_affine_scan(slow_gate, slow_drive)
        return self.merge(torch.cat((fast, slow), dim=-1))

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        fast_gate, fast_drive, slow_gate, slow_drive = self._transition(observation)
        if state is None:
            fast_previous = torch.zeros_like(fast_drive)
            slow_previous = torch.zeros_like(slow_drive)
        else:
            fast_previous, slow_previous = state
        fast = affine_scan_step(fast_gate, fast_drive, fast_previous)
        slow = affine_scan_step(slow_gate, slow_drive, slow_previous)
        return self.merge(torch.cat((fast, slow), dim=-1)), (fast, slow)
"""


# Each family has a small, medium, and near-budget width. A complex S5 channel
# consumes two real state scalars, so its largest variant has 64 channels.
_OPERATORS = {
    "lru": (_LRU, (48, 80, 108)),
    "s5": (_S5, (48, 56, 64)),
    "selective_scan": (_SELECTIVE_SCAN, (48, 64, 84)),
    "imbalance_head": (_IMBALANCE_HEAD, (40, 48, 55)),
    "dual_stream": (_DUAL_STREAM, (48, 72, 94)),
}
FAMILIES = tuple(_OPERATORS)


def placeholder_forecasts() -> tuple[StructuredPrediction, StructuredPrediction]:
    """Record the common two-slot schema without making it a promotion condition."""
    primary = StructuredPrediction("lob_impact_score", "increase", (0.50, 100.0), role="primary")
    guardrail = StructuredPrediction(
        "val_nll_total", "equivalent", (-0.05, 0.05), role="guardrail", equivalence_band=(-0.05, 0.05),
    )
    return primary, guardrail


def mutate(seed: int, *, family: str | None = None) -> Mutation:
    """Select a legal operator, width, and scale using the registered attempt seed."""
    rng = Random(seed)
    family = rng.choice(FAMILIES) if family is None else family
    if family not in _OPERATORS:
        raise ValueError(f"Unknown operator family: {family}")
    operator, widths = _OPERATORS[family]
    width = rng.choice(widths)
    scale = rng.uniform(0.8, 1.2)
    header = _HEADER.format(width=width, scale=f"{scale:.6f}")
    source = header + indent(dedent(operator).lstrip("\n"), "    ")
    return Mutation(family, width, source)
