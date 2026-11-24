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

WIDTH = 72
SCALE = 1.044401


class CandidateLOBWorldModel(SeedGRULOBModel):
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
