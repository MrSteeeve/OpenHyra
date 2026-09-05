#!/usr/bin/env python3
"""Bootstrap-stabilized backward exercise-boundary search.

The program learns a direct lower-bound policy.  At each exercise date it
chooses a threshold on the evaluator's discounted immediate payoff that
maximizes the realized improvement over the already learned future policy.
Bootstrap medians make the small (1024 path) empirical policy search less
sensitive to individual paths.  This representation encodes the monotone
exercise regions of puts and calls without estimating noisy continuation
levels far away from the stopping boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _best_threshold(payoff: np.ndarray, future: np.ndarray) -> float:
    """Return the payoff cutoff with largest in-sample incremental value."""
    payoff = np.asarray(payoff, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    positive = np.flatnonzero(payoff > 1e-14)
    if positive.size < 6:
        return float("inf")

    # Exercising the first k observations after descending payoff sort is a
    # complete finite search over all payoff-threshold policies.
    order = positive[np.argsort(-payoff[positive], kind="stable")]
    gains = np.cumsum(payoff[order] - future[order])
    # Very tiny leaves are unstable and cannot materially help the mean.
    first = min(5, gains.size - 1)
    k = first + int(np.argmax(gains[first:]))
    if gains[k] <= 0.0:
        return float("inf")

    chosen = payoff[order[k]]
    if k + 1 < order.size:
        rejected = payoff[order[k + 1]]
        return float(0.5 * (chosen + rejected))
    return float(max(np.nextafter(0.0, 1.0), 0.5 * chosen))


def _bagged_threshold(
    payoff: np.ndarray, future: np.ndarray, rng: np.random.Generator
) -> float:
    """Median of full-sample and bootstrap boundary estimates."""
    n = payoff.shape[0]
    estimates = [_best_threshold(payoff, future)]
    for _ in range(24):
        sample = rng.integers(0, n, size=n)
        estimates.append(_best_threshold(payoff[sample], future[sample]))
    finite = np.asarray([x for x in estimates if np.isfinite(x)], dtype=float)
    # A majority voting for no exercise is represented exactly by infinity.
    if finite.size < (len(estimates) + 1) // 2:
        return float("inf")
    return float(np.median(finite))


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.asarray(
        np.load(input_dir / "payoffs.npy", allow_pickle=False), dtype=np.float64
    )
    # Loading and validating the instance makes the fitted model self-describing;
    # policy training itself uses only pathwise, evaluator-supplied quantities.
    instance = json.loads((input_dir / "instance.json").read_text(encoding="utf-8"))
    if payoffs.ndim != 2 or paths.shape[:2] != payoffs.shape:
        raise ValueError("training paths and payoffs have incompatible shapes")

    rng = np.random.default_rng(int(seed))
    steps = payoffs.shape[1] - 1
    thresholds = np.full(steps, np.inf, dtype=np.float64)
    cash = payoffs[:, -1].copy()
    exercise_rates = np.zeros(steps, dtype=np.float64)

    for time_index in range(steps - 1, -1, -1):
        immediate = payoffs[:, time_index]
        threshold = _bagged_threshold(immediate, cash, rng)
        exercise = (immediate > 1e-14) & (immediate >= threshold)
        cash = np.where(exercise, immediate, cash)
        thresholds[time_index] = threshold
        exercise_rates[time_index] = float(np.mean(exercise))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        thresholds=thresholds,
        exercise_rates=exercise_rates,
        training_value=np.asarray(np.mean(cash)),
        seed=np.asarray(int(seed), dtype=np.int64),
    )
    (output_dir / "model_info.json").write_text(
        json.dumps(
            {
                "algorithm": "bootstrap_payoff_boundary_v1",
                "instance_id": instance.get("instance_id", instance.get("id", "unknown")),
                "exercise_dates": int(steps),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    time_index = int(request["time_index"])
    # Read states/history to validate the causal query's batch contract.  The
    # learned sufficient statistic is the current discounted immediate payoff.
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    history = np.load(input_dir / "history.npy", allow_pickle=False)
    immediate = np.asarray(
        np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False),
        dtype=np.float64,
    )
    if immediate.shape[0] != states.shape[0] or history.shape[0] != states.shape[0]:
        raise ValueError("query arrays have incompatible batch dimensions")
    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        threshold = float(model["thresholds"][time_index])
    decisions = (immediate > 1e-14) & (immediate >= threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", decisions, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
