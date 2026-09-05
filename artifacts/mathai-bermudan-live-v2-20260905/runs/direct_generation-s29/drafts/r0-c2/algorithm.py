"""Cross-fitted spline LSMC candidate for the open program track.

The inherited normalized linear LSMC is retained as the numerical backbone,
but its backward state transition is mutated to use out-of-fold continuation
estimates.  A small hinge basis lets the stopping boundary bend without a
large or stochastic model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _base_features(states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    """A bounded, product-agnostic polynomial summary of the current state."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    z = np.log(np.maximum(s, 1e-12))
    p = np.maximum(np.asarray(immediate, dtype=np.float64).reshape(-1), 0.0)
    cols = [np.ones(len(s)), p, np.sqrt(p + 1e-12), p * p]
    for j in range(z.shape[1]):
        cols.extend((z[:, j], z[:, j] ** 2, z[:, j] ** 3))
    if z.shape[1] > 1:
        cols.extend((z.mean(1), z.min(1), z.max(1), z.std(1)))
        for j in range(z.shape[1]):
            for k in range(j + 1, z.shape[1]):
                cols.append(z[:, j] * z[:, k])
    return np.column_stack(cols)


def features(states: np.ndarray, immediate: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Retain the parent basis and add payoff splines near exercise boundaries."""
    base = _base_features(states, immediate)
    p = np.maximum(np.asarray(immediate, dtype=np.float64).reshape(-1), 0.0)
    hinges = [np.maximum(p - float(k), 0.0) for k in np.asarray(knots).reshape(-1)]
    return np.column_stack([base, *hinges]) if hinges else base


def _normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    mean[0] = 0.0
    scale[scale < 1e-10] = 1.0
    return (x - mean) / scale, mean, scale


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float = 3e-3) -> np.ndarray:
    """Stable primal/dual ridge solve; the intercept is effectively unpenalized."""
    n, d = x.shape
    if n == 0:
        return np.zeros(d)
    if n >= d:
        gram = x.T @ x
        penalty = alpha * max(n, 1)
        gram.flat[:: d + 1] += penalty
        gram[0, 0] -= penalty
        try:
            return np.linalg.solve(gram, x.T @ y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram, x.T @ y, rcond=1e-10)[0]
    return x.T @ np.linalg.solve(x @ x.T + alpha * max(n, 1) * np.eye(n), y)


def _fit_on_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on economically relevant paths, falling back safely for tiny samples."""
    use = np.asarray(mask, dtype=bool)
    if use.sum() < max(20, x.shape[1] + 3):
        use = np.ones(len(x), dtype=bool)
    xn, mean, scale = _normalizer(x[use])
    return _ridge(xn, y[use]), mean, scale


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.asarray(np.load(input_dir / "payoffs.npy", allow_pickle=False), dtype=np.float64)
    steps = paths.shape[1] - 1
    # Fixed fold assignment makes replay deterministic while allowing the
    # evaluator seed to vary the bias-control partition across repetitions.
    rng = np.random.default_rng(int(seed))
    folds = rng.permutation(len(paths)) % 5
    cash = payoffs[:, -1].copy()
    deploy = [None] * steps

    for t in range(steps - 1, -1, -1):
        immediate = payoffs[:, t]
        positive = immediate > 1e-14
        pos_values = immediate[positive]
        if len(pos_values) >= 20:
            knots = np.unique(np.quantile(pos_values, [0.18, 0.36, 0.55, 0.72, 0.86]))
        else:
            knots = np.zeros(5, dtype=np.float64)
        if len(knots) < 5:
            knots = np.pad(knots, (0, 5 - len(knots)), mode="edge") if len(knots) else np.zeros(5)
        x = features(paths[:, t, :], immediate, knots)
        target = cash.copy()

        # Out-of-fold predictions drive the stopping transition.  This is the
        # AST-level mutation: a path can no longer train its own exercise rule.
        continuation = np.empty(len(paths), dtype=np.float64)
        for fold in range(5):
            train = (folds != fold) & positive
            coef, mean, scale = _fit_on_mask(x, target, train)
            test = folds == fold
            continuation[test] = ((x[test] - mean) / scale) @ coef
        continuation = np.maximum(continuation, 1e-10)
        exercise = positive & (immediate >= continuation)
        cash = np.where(exercise, immediate, target)

        # The deployable model sees all relevant paths, but its target is the
        # cash flow entering (not resulting from) this exercise decision.
        coef, mean, scale = _fit_on_mask(x, target, payoffs[:, t] > 1e-14)
        deploy[t] = (coef, mean, scale, knots)

    coefficients = np.stack([v[0] for v in deploy])
    means = np.stack([v[1] for v in deploy])
    scales = np.stack([v[2] for v in deploy])
    knots = np.stack([v[3] for v in deploy])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", coefficients=coefficients, means=means,
             scales=scales, knots=knots)


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        x = features(states, immediate, model["knots"][t])
        prediction = ((x - model["means"][t]) / model["scales"][t]) @ model["coefficients"][t]
    # A strictly positive floor prevents zero-payoff states from exercising on
    # harmless negative regression extrapolation.
    prediction = np.nan_to_num(prediction, nan=1e-10, posinf=1e6, neginf=1e-10)
    prediction = np.maximum(prediction, 1e-10)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", prediction.astype(np.float64), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--input", required=True); train.add_argument("--output", required=True)
    train.add_argument("--seed", required=True, type=int)
    query = commands.add_parser("predict")
    query.add_argument("--model", required=True); query.add_argument("--input", required=True)
    query.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
