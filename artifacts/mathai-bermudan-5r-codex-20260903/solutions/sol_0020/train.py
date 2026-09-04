#!/usr/bin/env python3
"""Data-only affine continuation-policy trainer.

The affine rule is selected independently at each exercise date.  A small
deterministic holdout gate keeps the affine inductive bias only when it beats
a constant continuation estimate; otherwise the constant is emitted as the
linear runner's zero-slope special case.  All targets and normalization are
computed from the supplied fit bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPS = 1.0e-10
RIDGE = 1.0e-3
CLIP = (-1_000_000.0, 1_000_000.0)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def _load(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.asarray(np.load(root / "training_paths.npy", allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(root / "payoffs.npy", allow_pickle=False), dtype=np.float64)
    discounts = np.asarray(np.load(root / "discount_factors.npy", allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2 or discounts.ndim != 1:
        raise ValueError("invalid training dimensions")
    if paths.shape[:2] != payoffs.shape or discounts.shape != (paths.shape[1],):
        raise ValueError("training arrays are misaligned")
    if paths.shape[0] < 1 or paths.shape[2] < 1 or np.any(paths <= 0.0):
        raise ValueError("invalid states")
    if not all(np.all(np.isfinite(x)) for x in (paths, payoffs, discounts)):
        raise ValueError("non-finite training input")
    return paths, payoffs, discounts


def _normalization(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(states, axis=0, dtype=np.float64)
    scale = np.std(states, axis=0, dtype=np.float64)
    scale = np.where(scale > EPS, scale, 1.0)
    return mean, scale


def _affine_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge fit of [x, 1] -> y, with an unpenalized intercept."""
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * RIDGE
    penalty[-1, -1] = 0.0
    rhs = design.T @ np.clip(y, CLIP[0], CLIP[1])
    try:
        theta = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(design, y, rcond=None)[0]
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (x.shape[1] + 1,) or not np.all(np.isfinite(theta)):
        return np.zeros(x.shape[1] + 1, dtype=np.float64)
    return np.clip(theta, CLIP[0], CLIP[1])


def _select_affine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Use a deterministic even/odd holdout to gate the affine model."""
    y = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), CLIP[0], CLIP[1])
    if y.size == 0:
        raise ValueError("empty target")
    constant = float(np.mean(y))
    theta = _affine_fit(x, y)
    if y.size < 8:
        return theta
    train = (np.arange(y.size) % 2) == 0
    test = ~train
    fitted = _affine_fit(x[train], y[train])
    design_test = np.column_stack((x[test], np.ones(int(np.sum(test)))))
    affine_pred = np.clip(design_test @ fitted, CLIP[0], CLIP[1])
    constant_pred = np.full(affine_pred.shape, constant, dtype=np.float64)
    affine_mse = float(np.mean(np.square(affine_pred - y[test])))
    constant_mse = float(np.mean(np.square(constant_pred - y[test])))
    # Require a material out-of-fold improvement, avoiding noisy slope fitting.
    if constant_mse <= EPS or affine_mse > 0.99 * constant_mse:
        return np.concatenate((np.zeros(x.shape[1], dtype=np.float64), [constant]))
    return theta


def _write(output: Path, paths: np.ndarray, payoffs: np.ndarray, discounts: np.ndarray) -> None:
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("output directory must be an existing empty directory")
    del discounts  # payoffs are already discounted to time zero.
    n_times = paths.shape[1]
    steps = n_times - 1
    if steps < 1:
        raise ValueError("exercise grid must contain a non-terminal step")

    cashflow = np.asarray(payoffs[:, -1], dtype=np.float64).copy()
    normalizations: list[dict[str, list[float]]] = []
    coefficients: list[np.ndarray] = [np.zeros(paths.shape[2] + 1, dtype=np.float64) for _ in range(steps)]

    for i in range(steps - 1, -1, -1):
        states = paths[:, i, :]
        mean, scale = _normalization(states)
        x = (states - mean) / scale
        immediate = np.asarray(payoffs[:, i], dtype=np.float64)
        eligible = immediate > 0.0
        fit_mask = eligible if int(np.sum(eligible)) >= 2 else np.ones(states.shape[0], dtype=bool)
        coefficients[i] = _select_affine(x[fit_mask], cashflow[fit_mask])
        prediction = np.clip(x @ coefficients[i][:-1] + coefficients[i][-1], CLIP[0], CLIP[1])
        exercise = eligible & (immediate >= prediction)
        cashflow[exercise] = immediate[exercise]
        normalizations.append({"mean": mean.tolist(), "scale": scale.tolist()})

    normalizations.reverse()
    for i, theta in enumerate(coefficients):
        np.save(output / f"step_{i:03d}.npy", np.ascontiguousarray(theta, dtype=np.float64), allow_pickle=False)
    (output / "normalization.json").write_text(
        json.dumps({"steps": normalizations}, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = _args()
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    paths, payoffs, discounts = _load(Path(args.input))
    _write(Path(args.output), paths, payoffs, discounts)


if __name__ == "__main__":
    main()
