#!/usr/bin/env python3
"""Small deterministic MLP continuation-policy matched control.

The trusted training bridge invokes this file once for one contract and one
fit-path sample.  It receives only the four files in ``--input`` and writes a
frozen, data-only artifact to ``--output``.  Financial simulation, stopping,
pricing, and auditing remain evaluator-owned.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


HIDDEN_WIDTHS = (16, 16)
OUTPUT_CLIP = (-1_000_000.0, 1_000_000.0)
NORMALIZATION_EPSILON = 1e-10
TRAINING_EPOCHS = 140


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def _load_array(root: Path, name: str, *, ndim: int) -> np.ndarray:
    path = root / name
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D NumPy array")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _load_inputs(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = _load_array(root, "training_paths.npy", ndim=3)
    payoffs = _load_array(root, "payoffs.npy", ndim=2)
    discounts = _load_array(root, "discount_factors.npy", ndim=1)
    if paths.shape[:2] != payoffs.shape:
        raise ValueError("payoffs shape must match training_paths first dimensions")
    if discounts.shape != (paths.shape[1],):
        raise ValueError("discount_factors length must match exercise times")
    if paths.shape[0] < 1 or paths.shape[2] < 1:
        raise ValueError("training_paths has an empty dimension")
    if np.any(paths <= 0.0):
        raise ValueError("training_paths must contain positive states")
    instance_path = root / "instance.json"
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    if not isinstance(instance, dict):
        raise ValueError("instance.json must contain an object")
    return paths, payoffs, discounts, instance


def _step_seed(seed: int, time_index: int) -> int:
    # SeedSequence accepts bounded Python integers and keeps the per-step
    # streams independent while remaining reproducible across processes.
    low = seed & 0xFFFFFFFF
    high = (seed >> 32) & 0xFFFFFFFF
    return int(np.random.SeedSequence([low, high, time_index]).generate_state(1)[0])


def _normalization(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(states, axis=0, dtype=np.float64)
    scale = np.std(states, axis=0, dtype=np.float64)
    scale = np.where(scale > NORMALIZATION_EPSILON, scale, 1.0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("state normalization is non-finite")
    return mean, scale


def _zero_network(input_dim: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    widths = (input_dim, *HIDDEN_WIDTHS, 1)
    weights = [
        np.zeros((out_dim, in_dim), dtype=np.float64)
        for in_dim, out_dim in zip(widths, widths[1:])
    ]
    biases = [np.zeros(out_dim, dtype=np.float64) for out_dim in widths[1:]]
    return weights, biases


def _forward(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preactivations: list[np.ndarray] = []
    current = x
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        pre = current @ weight.T + bias
        pre = np.clip(pre, -40.0, 40.0)
        preactivations.append(pre)
        if index + 1 < len(weights):
            current = np.tanh(pre)
        else:
            current = pre
        activations.append(current)
    return activations, preactivations


def _fit_network(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fit a tiny tanh MLP with deterministic, uniformly weighted MSE.

    This candidate is the matched control for ``boundary_weighted_loss_v1``:
    every retained training example contributes the same loss weight.  The
    architecture, initialization, optimizer, and update count are unchanged
    so any paired difference isolates the absence of focal weighting.
    """
    input_dim = states.shape[1]
    weights, biases = _zero_network(input_dim)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if states.shape[0] == 0 or y.shape[0] != states.shape[0]:
        raise ValueError("network training arrays are misaligned")
    y = np.clip(y, OUTPUT_CLIP[0], OUTPUT_CLIP[1])
    # A zero target occurs frequently in early backward steps.  Emitting the
    # exact zero network avoids spurious exercise caused by random weights.
    if float(np.max(np.abs(y))) <= 1e-12:
        return weights, biases

    rng = np.random.default_rng(seed)
    widths = (input_dim, *HIDDEN_WIDTHS, 1)
    for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:])):
        # Xavier-scale initialization keeps tanh in its useful region.  The
        # final bias starts at the target mean, making the seed sensible even
        # when a short cell receives little variation.
        scale = math.sqrt(2.0 / float(in_dim + out_dim))
        weights[index][:] = rng.normal(0.0, scale, size=(out_dim, in_dim))
        biases[index].fill(0.0)
    biases[-1][0] = float(np.mean(y))

    count = float(states.shape[0])
    for epoch in range(TRAINING_EPOCHS):
        activations, preactivations = _forward(states, weights, biases)
        prediction = activations[-1][:, 0]
        if not np.all(np.isfinite(prediction)):
            return _zero_network(input_dim)
        # Uniform-loss control: deliberately no payoff/target-derived focal
        # weights are applied.  ``count`` stays fixed for the full-batch MSE.
        gradient = (2.0 / count) * (prediction - y)[:, None]
        grad_weights: list[np.ndarray] = [
            np.zeros_like(weight) for weight in weights
        ]
        grad_biases: list[np.ndarray] = [
            np.zeros_like(bias) for bias in biases
        ]
        for index in range(len(weights) - 1, -1, -1):
            grad_weights[index] = gradient.T @ activations[index]
            grad_biases[index] = np.sum(gradient, axis=0)
            if index:
                hidden = np.tanh(preactivations[index - 1])
                gradient = (gradient @ weights[index]) * (1.0 - hidden * hidden)

        # A global norm cap makes the artifact finite for adversarially scaled
        # public inputs while retaining deterministic updates.
        norm_sq = sum(
            float(np.sum(np.square(value)))
            for value in (*grad_weights, *grad_biases)
        )
        norm = math.sqrt(norm_sq) if math.isfinite(norm_sq) else math.inf
        multiplier = 1.0 if norm <= 5.0 else 5.0 / norm
        learning_rate = 0.035 / (1.0 + epoch / 70.0)
        for index in range(len(weights)):
            weights[index] -= learning_rate * multiplier * grad_weights[index]
            biases[index] -= learning_rate * multiplier * grad_biases[index]
            weights[index][:] = np.clip(weights[index], -25.0, 25.0)
            biases[index][:] = np.clip(biases[index], -25.0, 25.0)
        if not all(
            np.all(np.isfinite(value)) for value in (*weights, *biases)
        ):
            return _zero_network(input_dim)
    return weights, biases


def _flatten_network(
    weights: list[np.ndarray], biases: list[np.ndarray],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for weight, bias in zip(weights, biases):
        chunks.extend((np.asarray(weight, dtype=np.float64).reshape(-1),
                       np.asarray(bias, dtype=np.float64).reshape(-1)))
    flat = np.concatenate(chunks).astype(np.float64, copy=False)
    if flat.ndim != 1 or not np.all(np.isfinite(flat)):
        raise ValueError("network parameters are non-finite")
    return np.ascontiguousarray(flat)


def _write_artifact(
    output: Path,
    paths: np.ndarray,
    payoffs: np.ndarray,
    discounts: np.ndarray,
    seed: int,
) -> None:
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("output directory must be an existing empty directory")
    n_paths, n_times, input_dim = paths.shape
    del n_paths, discounts  # the discounting is already reflected in payoffs
    steps = n_times - 1
    if steps < 1:
        raise ValueError("exercise grid must contain a non-terminal step")

    # Backward induction mirrors the evaluator's fixed stopping convention.
    # The candidate only sees fit paths; the evaluator later freezes and
    # applies the exported runner to independent pricing paths.
    cashflow = np.asarray(payoffs[:, -1], dtype=np.float64).copy()
    normalizations: list[dict[str, list[float]]] = []
    flat_steps: list[np.ndarray] = [
        np.zeros(0, dtype=np.float64) for _ in range(steps)
    ]
    for time_index in range(steps - 1, -1, -1):
        states = np.asarray(paths[:, time_index, :], dtype=np.float64)
        mean, scale = _normalization(states)
        standardized = (states - mean) / scale
        immediate = np.asarray(payoffs[:, time_index], dtype=np.float64)
        eligible = immediate > 0.0
        fit_mask = (
            eligible
            if int(np.sum(eligible)) >= 2
            else np.ones(len(states), dtype=bool)
        )
        weights, biases = _fit_network(
            standardized[fit_mask],
            cashflow[fit_mask],
            seed=_step_seed(seed, time_index),
        )
        flat_steps[time_index] = _flatten_network(weights, biases)
        all_activations, _ = _forward(standardized, weights, biases)
        continuation = all_activations[-1][:, 0]
        continuation = np.clip(continuation, OUTPUT_CLIP[0], OUTPUT_CLIP[1])
        exercise = eligible & (immediate >= continuation)
        cashflow[exercise] = immediate[exercise]
        normalizations.append({
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        })

    # ``normalizations`` was collected in backward order; the runner indexes
    # files chronologically, so reverse it before serialization.
    normalizations.reverse()
    for index, flat in enumerate(flat_steps):
        np.save(
            output / f"step_{index:03d}.npy",
            np.ascontiguousarray(flat, dtype=np.float64),
            allow_pickle=False,
        )
    (output / "normalization.json").write_text(
        json.dumps(
            {"steps": normalizations},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = _arguments()
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    input_root = Path(args.input)
    output_root = Path(args.output)
    paths, payoffs, discounts, _instance = _load_inputs(input_root)
    _write_artifact(output_root, paths, payoffs, discounts, args.seed)


if __name__ == "__main__":
    main()
