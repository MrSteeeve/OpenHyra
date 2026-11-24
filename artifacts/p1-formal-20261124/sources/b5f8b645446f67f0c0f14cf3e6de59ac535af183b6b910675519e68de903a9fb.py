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

WIDTH = 55
SCALE = 1.016789


class CandidateLOBWorldModel(SeedGRULOBModel):
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
