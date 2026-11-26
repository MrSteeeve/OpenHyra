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

WIDTH = 108
SCALE = 0.848275


class CandidateLOBWorldModel(SeedGRULOBModel):
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
