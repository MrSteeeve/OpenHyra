"""Trusted child processes for train-only fitting and split-only frozen scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from time import perf_counter


def main(mode: str, source: Path, inputs: Path, output: Path) -> None:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from experiments.synthetic_tau_z.design import FIT_PROTOCOL, SEEDS
    from experiments.synthetic_tau_z.discovery_metrics import evaluate_discovery_model
    from stochopia.common import atomic_write_json
    from stochopia.harness.lob_rollout_batch import NonfiniteRolloutProbability
    from stochopia.neural.lob_interface import verify_world_model_contract
    from stochopia.neural.lob_training import NonfiniteModelLoss, encode_day, fit_lob_world_model
    from tasks.lob_world_model.gate_worker import load_candidate
    from tasks.lob_world_model.market_worker import MODEL_FAILURE_EXIT, NONFINITE_LOSS_EXIT
    from tasks.lob_world_model.synthetic_stats import pool_training_seeds, summarize_levels

    # Deserialize only the orchestrator-owned data staged for this phase. No
    # shared cache path, generator state, other split or API key is supplied.
    data_file = "train.pkl" if mode == "fit" else "panel.pkl"
    with (inputs / data_file).open("rb") as stream:
        data = pickle.load(stream)
    output.mkdir()
    started = perf_counter()
    try:
        factory = load_candidate(source)
        if mode == "fit":
            days = [encode_day(day) for day in data]
            fits = []
            sizes = set()
            for seed in SEEDS:
                fitted = fit_lob_world_model(factory, days, seed, protocol=FIT_PROTOCOL)
                sizes.add(verify_world_model_contract(fitted.model, max_params=28_000, max_state_dim=60))
                torch.save(fitted.model.state_dict(), output / f"{seed}.pt")
                fits.append({"seed": seed, "fit_seconds": fitted.seconds, "last_nll": fitted.last_nll})
            if len(sizes) != 1:
                raise ValueError("trained candidate capacities differ across registered seeds")
            params, state = next(iter(sizes))
            atomic_write_json(output / "fit.json", {"params": params, "state": state, "fits": fits})
            return
        per_seed = []
        sizes = set()
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = factory().eval()
            weights = torch.load(inputs / f"{seed}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(weights)
            sizes.add(verify_world_model_contract(model, max_params=28_000, max_state_dim=60))
            per_seed.append(evaluate_discovery_model(model, data))
        if len(sizes) != 1:
            raise ValueError("evaluated candidate capacities differ across registered seeds")
        params, state = next(iter(sizes))
        daily = pool_training_seeds(per_seed)
        aggregate, support = summarize_levels(daily)
        atomic_write_json(output / "result.json", {
            "params": params, "state": state, "daily": daily, "aggregate": aggregate, "support": support,
            "seed_daily": dict(zip(map(str, SEEDS), per_seed, strict=True)),
            "evaluation_seconds": perf_counter() - started,
        })
    except (NonfiniteModelLoss, NonfiniteRolloutProbability) as exc:
        print(f"candidate has nonfinite loss/probabilities: {exc}", file=sys.stderr)
        raise SystemExit(NONFINITE_LOSS_EXIT) from exc
    except (ImportError, AttributeError, AssertionError, IndexError, KeyError, SyntaxError, TypeError, ValueError,
            RuntimeError, TimeoutError) as exc:
        print(f"candidate tensor/training/evaluation failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(MODEL_FAILURE_EXIT) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "measure"))
    parser.add_argument("source", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.mode, args.source, args.inputs, args.output)
