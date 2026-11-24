"""Trusted child entry for a random, zero-data forward/backward probe."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stochopia.neural.lob_interface import LOBWorldModel


IMPORTS = frozenset({
    "__future__", "math", "cmath", "decimal", "fractions", "statistics",
    "torch", "torch.nn", "torch.nn.functional",
    "stochopia.neural.lob_interface", "stochopia.neural.primitives",
})


def _candidate_import(name: str, globals: Mapping[str, object] | None = None,
                      locals: Mapping[str, object] | None = None,
                      fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
    # The OS sandbox remains the security boundary. This is the source API.
    if level != 0 or name not in IMPORTS:
        raise ImportError(f"Candidate cannot import {name!r}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_candidate(candidate: Path) -> type[LOBWorldModel]:
    """Load from an isolated child after the caller stages its trusted imports."""
    import torch
    from stochopia.neural.lob_interface import LOBWorldModel

    # PyTorch lazily imports torch.storage from the first tensor caller's
    # builtins. Initialize it in trusted code before applying candidate imports.
    torch.tensor(0)
    namespace = {
        "__builtins__": {**vars(builtins), "__import__": _candidate_import},
        "__name__": "candidate",
        "__file__": str(candidate),
    }
    exec(compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec"), namespace)
    candidate_type = namespace.get("CandidateLOBWorldModel")
    if not isinstance(candidate_type, type) or not issubclass(candidate_type, LOBWorldModel):
        raise TypeError("CandidateLOBWorldModel must subclass LOBWorldModel")
    return candidate_type


def main(candidate: Path, completion: Path) -> None:
    sys.path.insert(0, str(candidate.parent))

    import torch
    from stochopia.neural.lob_interface import _probe, verify_world_model_contract, world_model_loss

    torch.manual_seed(291)
    model = load_candidate(candidate)()
    verify_world_model_contract(model, max_params=28_000, max_state_dim=128)

    model.train()
    batch = _probe(torch.device("cpu"), 417)
    loss = world_model_loss(model.forward_sequence(batch), batch.targets).total
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite training loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
        raise ValueError("missing or nonfinite training gradients")
    completion.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
