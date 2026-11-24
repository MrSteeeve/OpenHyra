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

WIDTH = 56
SCALE = 1.069604


class CandidateLOBWorldModel(SeedGRULOBModel):
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
