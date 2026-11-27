"""Trusted child entry for a random, zero-data forward/backward probe."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stochopia.neural.lob_interface import LOBWorldModel


def load_candidate(candidate: Path) -> type[LOBWorldModel]:
    """Load trusted candidate source as a standard Python module in the child."""
    from stochopia.neural.lob_interface import LOBWorldModel

    spec = spec_from_file_location("candidate", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load research candidate from {candidate}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate_type = getattr(module, "CandidateLOBWorldModel", None)
    if not isinstance(candidate_type, type) or not issubclass(candidate_type, LOBWorldModel):
        raise TypeError("CandidateLOBWorldModel must subclass LOBWorldModel")
    return candidate_type


def main(candidate: Path, completion: Path, max_state_dim: int = 128) -> None:
    import torch

    # Configure both thread pools and numeric precision before loading the model.
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from stochopia.neural.lob_interface import _probe, verify_world_model_contract, world_model_loss

    torch.manual_seed(291)
    model = load_candidate(candidate)()
    verify_world_model_contract(model, max_params=28_000, max_state_dim=max_state_dim)

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
    max_state_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    main(Path(sys.argv[1]), Path(sys.argv[2]), max_state_dim)
