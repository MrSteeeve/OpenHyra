"""The P2 gate can admit real composite states without changing the P1 cap."""

import pytest

from stochopia.neural.primitives import seed_lob_source
from tasks.lob_world_model.gate import CandidateGate


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


def test_gate_preserves_default_cap_and_supports_explicit_budget() -> None:
    assert "max_state_dim" in (CandidateGate()(COMPOSITE) or "")
    assert CandidateGate(max_state_dim=512)(COMPOSITE) is None


SEED = """from stochopia.neural.primitives import SeedGRULOBModel
class CandidateLOBWorldModel(SeedGRULOBModel):
    pass
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


@pytest.mark.parametrize(("source", "error"), [
    (FUTURE, "Tensor-likes are not close"),
    (OVER_BUDGET, "model exceeds max_params"),
])
def test_actual_gate_rejects_noncausal_or_oversized_model(source: str, error: str) -> None:
    assert error in (CandidateGate()(source) or "")


def test_actual_gate_rejects_singular_backward_with_finite_forward() -> None:
    assert "missing or nonfinite training gradients" in (CandidateGate()(NONFINITE_GRADIENT) or "")


def test_actual_gate_accepts_the_exact_seed_source() -> None:
    assert CandidateGate()(seed_lob_source()) is None


def test_gate_allows_standard_scientific_imports() -> None:
    source = """from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Optional
import torch
import torch.fft
import torch.linalg
from stochopia.neural.primitives import SeedGRULOBModel

@dataclass(frozen=True)
class ModelShape:
    state_dim: Optional[int] = 48

class CandidateLOBWorldModel(SeedGRULOBModel):
    def __init__(self):
        shape = ModelShape()
        values = torch.tensor(list(repeat(1.0, 2)))
        spectrum = partial(torch.fft.rfft, dim=-1)(values)
        assert torch.linalg.vector_norm(spectrum).item() == 2.0
        super().__init__(state_dim=shape.state_dim)
"""
    assert CandidateGate()(source) is None


def test_gate_requires_completed_tensor_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.lob_world_model import gate as module

    def completed_without_marker(command, *, timeout_s, work_dir, extra_env):
        assert command[1:3] == ["-s", "-B"]
        assert [path.name for path in work_dir.iterdir()] == ["candidate.py"]
        assert extra_env["PYTHONPATH"] == module._runtime_pythonpath()
        return {"returncode": 0, "timed_out": False, "cancelled": False,
                "elapsed_s": 0.1, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "run_worker_subprocess", completed_without_marker)
    assert "before completing its tensor checks" in (CandidateGate()(SEED) or "")
