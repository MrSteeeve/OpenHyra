"""Run the trusted P1 fitting or measurement protocol in one isolated child."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path


MODEL_FAILURE_EXIT = 65
NONFINITE_LOSS_EXIT = 66


class ModelFailure(RuntimeError):
    """The candidate model failed at a known fitting or inference boundary."""


def normalize_device(device: str) -> str:
    """Canonicalize CPU and process-visible CUDA indices without initializing CUDA."""
    import torch

    if not isinstance(device, str):
        raise ValueError("device must be a CPU or CUDA device string")
    try:
        selected = torch.device(device.strip().lower())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc
    if selected.type == "cpu" and selected.index in (None, 0):
        return "cpu"
    if selected.type == "cuda":
        return f"cuda:{0 if selected.index is None else selected.index}"
    raise ValueError(f"unsupported device {device!r}; expected cpu or cuda[:index]")


def device_cache_identity(device: str) -> str:
    """Identify the visible accelerator and numeric runtime for a cache key.

    A CUDA ordinal alone could identify different GPUs in different workers.
    Prefer the GPU UUID; where a runtime does not supply it, conservatively
    include the device characteristics and use a host-local cache root.
    """
    import torch

    canonical = normalize_device(device)
    if canonical == "cpu":
        return canonical
    index = int(canonical.split(":")[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"requested worker device is not available: {canonical}")
    props = torch.cuda.get_device_properties(index)
    fingerprint = {
        "uuid": str(getattr(props, "uuid", "")),
        "name": props.name,
        "capability": [props.major, props.minor],
        "total_memory": props.total_memory,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "tf32": False,
    }
    encoded = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return f"cuda-{index}-{sha256(encoded).hexdigest()[:24]}"


def main(
    mode: str, candidate: Path, input_dir: Path, output_dir: Path, seed: int | None = None,
    *, device: str = "cpu", device_id: str | None = None,
) -> None:
    """Fit from train-only caches or measure three fixed checkpoints on dev."""
    import torch

    # Configure both thread pools before constructing any model or tensor.
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = normalize_device(device)
    actual_device_id = device_cache_identity(device)
    if device_id is not None and device_id != actual_device_id:
        raise ValueError("worker device identity differs from the registered cache identity")
    if device != "cpu":
        torch.cuda.set_device(torch.device(device))

    from experiments.p1_m2_baseline import _load_days, evaluate_symbol
    from stochopia.common import atomic_write_json
    from stochopia.neural.lob_training import P1_TRAIN_SEEDS, NonfiniteModelLoss, fit_lob_world_model
    from tasks.stochopia_p1.gate_worker import load_candidate

    plan = json.loads((input_dir / "plan.json").read_text(encoding="utf-8"))
    try:
        factory = load_candidate(candidate)
    except (ImportError, SyntaxError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelFailure("Model could not be loaded") from exc
    symbol = plan["symbol"]
    train_dates = plan["train_dates"]
    if mode == "fit" and seed in P1_TRAIN_SEEDS:
        days = _load_days(input_dir, symbol, train_dates)
        try:
            fitted = fit_lob_world_model(factory, days, seed, device=device)
        except NonfiniteModelLoss as exc:
            raise ModelLossFailure(str(exc)) from exc
        except (TimeoutError, TypeError, ValueError, RuntimeError, AssertionError) as exc:
            raise ModelFailure("Model failed on the registered training windows") from exc
        # Checkpoints are portable serialized tensors, not permission to
        # share fit/evaluation cache entries between devices.
        torch.save(fitted.model.cpu().state_dict(), output_dir / "weights.pt")
        atomic_write_json(output_dir / "fit.json", {"seed": seed, "seconds": fitted.seconds,
                                                     "last_batch_nll": fitted.last_nll})
        return
    if mode != "evaluate" or seed is not None:
        raise ValueError("expected a registered fit seed or an evaluate-only request")

    directory = output_dir / "profile"
    checkpoints = {seed: input_dir / "weights" / f"{seed}.pt" for seed in P1_TRAIN_SEEDS}
    torch.manual_seed(plan["rollout_seed"])
    try:
        evaluate_symbol(
            input_dir, symbol, train_dates, plan["eval_dates"], directory,
            factory=factory, rollout_seed=plan["rollout_seed"], checkpoints=checkpoints, save_traces=False,
            device=device,
        )
    except NonfiniteModelLoss as exc:
        raise ModelLossFailure(str(exc)) from exc
    except RuntimeError as exc:
        raise ModelFailure("Model could not finish the registered rollouts") from exc
    profile = json.loads((directory / "residuals.json").read_text(encoding="utf-8"))
    atomic_write_json(output_dir / "result.json", {
        "day_metrics": profile["day_metrics"], "evidence": profile["evidence"],
        "market_bootstrap": profile["market_bootstrap"], "evaluation_seconds": profile["seconds"],
    })


class ModelLossFailure(ModelFailure):
    """A loss guardrail was violated during value or gradient calculation."""


def cli() -> None:
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("candidate", type=Path)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-id")
    args = parser.parse_intermixed_args()
    try:
        main(args.mode, args.candidate, args.input_dir, args.output_dir, args.seed,
             device=args.device, device_id=args.device_id)
    except ModelLossFailure:
        traceback.print_exc()
        sys.exit(NONFINITE_LOSS_EXIT)
    except ModelFailure:
        traceback.print_exc()
        sys.exit(MODEL_FAILURE_EXIT)


if __name__ == "__main__":
    cli()
