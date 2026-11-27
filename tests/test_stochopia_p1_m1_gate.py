"""The P2 gate can admit real composite states without changing the P1 cap."""

from tasks.stochopia_p1.gate import CandidateGate


COMPOSITE = """
import torch
from torch import nn
from stochopia.neural.lob_interface import BUCKET_COUNTS, LOBWorldModel
from stochopia.neural.primitives import parallel_affine_scan

class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(16, count) for count in BUCKET_COUNTS)

    def _drive(self, observation):
        x = observation.book_shares[..., 0, :].float() / 10000
        return x[..., None].expand(*x.shape, 16)

    def sequence_context(self, observation):
        x = self._drive(observation)
        return parallel_affine_scan(x.new_full((10, 16), 0.2), x).mean(dim=-2)

    def step_context(self, observation, state):
        drive = self._drive(observation)
        h = drive if state is None else state['book'] * 0.2 + drive
        return h.mean(dim=-2), {'book': h, 'summary': h[:, 0]}

    def head_logits(self, context, stage, prior):
        return self.heads[stage](context)
"""


def test_gate_preserves_p1_cap_and_supports_explicit_p2_budget() -> None:
    assert "max_state_dim" in (CandidateGate()(COMPOSITE) or "")
    assert CandidateGate(max_state_dim=512)(COMPOSITE) is None
