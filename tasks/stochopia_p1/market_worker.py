"""Run the trusted P1 fitting or measurement protocol in one isolated child."""

from __future__ import annotations

import json
import sys
from pathlib import Path


MODEL_FAILURE_EXIT = 65
NONFINITE_LOSS_EXIT = 66


class ModelFailure(RuntimeError):
    """The candidate model failed at a known fitting or inference boundary."""


def main(mode: str, input_dir: Path, output_dir: Path, seed: int | None = None) -> None:
    """Fit from train-only caches or measure three fixed checkpoints on dev."""
    source_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_dir))

    import torch
    from experiments.p1_m2_baseline import _load_days, evaluate_symbol
    from gate_worker import load_candidate
    from stochopia.common import atomic_write_json
    from stochopia.neural.lob_training import P1_TRAIN_SEEDS, NonfiniteModelLoss, fit_lob_world_model

    torch.set_num_threads(1)
    plan = json.loads((input_dir / "plan.json").read_text(encoding="utf-8"))
    try:
        factory = load_candidate(source_dir / "candidate.py")
    except (ImportError, SyntaxError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelFailure("Model could not be loaded") from exc
    symbol = plan["symbol"]
    train_dates = plan["train_dates"]
    if mode == "fit" and seed in P1_TRAIN_SEEDS:
        days = _load_days(input_dir, symbol, train_dates)
        try:
            fitted = fit_lob_world_model(factory, days, seed)
        except NonfiniteModelLoss as exc:
            raise ModelLossFailure(str(exc)) from exc
        except (TimeoutError, TypeError, ValueError, RuntimeError) as exc:
            raise ModelFailure("Model failed on the registered training windows") from exc
        torch.save(fitted.model.state_dict(), output_dir / "weights.pt")
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


if __name__ == "__main__":
    import traceback

    registered_seed = int(sys.argv[4]) if len(sys.argv) == 5 else None
    try:
        main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), registered_seed)
    except ModelLossFailure:
        traceback.print_exc()
        sys.exit(NONFINITE_LOSS_EXIT)
    except ModelFailure:
        traceback.print_exc()
        sys.exit(MODEL_FAILURE_EXIT)
