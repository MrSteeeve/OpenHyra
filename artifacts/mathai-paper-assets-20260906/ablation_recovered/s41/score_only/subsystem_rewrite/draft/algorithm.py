"""Complete fit/predict seed for the open Python-program track.

This is deliberately just one ordinary starting program. Search candidates
may replace every function, model representation, objective, and update rule.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    logged = np.log(np.maximum(states, 1e-12))
    columns = [np.ones(states.shape[0], dtype=np.float64), *[logged[:, index] for index in range(logged.shape[1])], *[logged[:, index] ** 2 for index in range(logged.shape[1])], np.mean(logged, axis=1), np.min(logged, axis=1), np.max(logged, axis=1), np.asarray(immediate, dtype=np.float64)]
    return np.column_stack(columns)

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:

    def ridge_gcv(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected full-sample coefficients and its LOO fits."""
        n_rows, n_columns = design.shape
        gram = design.T @ design
        rhs = design.T @ target
        penalty_mask = np.ones(n_columns, dtype=np.float64)
        penalty_mask[0] = 0.0
        lambdas = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
        best = None
        for ridge in lambdas:
            system = gram + ridge * np.diag(penalty_mask)
            try:
                inverse = np.linalg.solve(system, np.eye(n_columns, dtype=np.float64))
                coefficients = inverse @ rhs
            except np.linalg.LinAlgError:
                continue
            fitted = design @ coefficients
            residual = target - fitted
            leverage = np.einsum('ij,jk,ik->i', design, inverse, design)
            effective_df = float(np.sum(leverage))
            denominator = max(float(n_rows) - effective_df, 1e-06)
            score = float(np.dot(residual, residual)) / (denominator * denominator)
            if np.isfinite(score) and np.all(np.isfinite(coefficients)) and np.all(np.isfinite(fitted)) and np.all(np.isfinite(leverage)):
                candidate = (score, coefficients, fitted, residual, leverage)
                if best is None or score < best[0]:
                    best = candidate
        if best is None:
            ridge = 1.0
            augmented_x = np.vstack((design, np.diag(np.sqrt(ridge * penalty_mask))))
            augmented_y = np.concatenate((target, np.zeros(n_columns, dtype=np.float64)))
            coefficients = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
            fitted = design @ coefficients
            residual = target - fitted
            system = gram + ridge * np.diag(penalty_mask)
            inverse = np.linalg.pinv(system, rcond=1e-12)
            leverage = np.einsum('ij,jk,ik->i', design, inverse, design)
        else:
            _, coefficients, fitted, residual, leverage = best
        safe_one_minus_h = np.maximum(1.0 - leverage, 0.001)
        loo = fitted - leverage * residual / safe_one_minus_h
        loo = np.where(np.isfinite(loo), loo, fitted)
        return (coefficients, loo)
    paths = np.asarray(np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('expected paths (n_paths,n_times,n_assets) and payoffs (n_paths,n_times)')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times):
        raise ValueError('training path and payoff grids do not match')
    initial_design = features(paths[:, 0, :], payoffs[:, 0])
    n_features = initial_design.shape[1]
    coefficients = np.zeros((n_times, n_features), dtype=np.float64)
    means = np.zeros((n_times, n_features), dtype=np.float64)
    scales = np.ones((n_times, n_features), dtype=np.float64)
    downstream = payoffs[:, -1].copy()
    for time_index in range(n_times - 2, -1, -1):
        design = features(paths[:, time_index, :], payoffs[:, time_index])
        column_means = np.mean(design, axis=0)
        column_scales = np.std(design, axis=0)
        column_scales = np.where(np.isfinite(column_scales) & (column_scales > 1e-12), column_scales, 1.0)
        column_means[0] = 0.0
        column_scales[0] = 1.0
        normalized = (design - column_means) / column_scales
        step_coefficients, loo_continuation = ridge_gcv(normalized, downstream)
        coefficients[time_index] = step_coefficients
        means[time_index] = column_means
        scales[time_index] = column_scales
        immediate = payoffs[:, time_index]
        exercise = (immediate > 0.0) & (immediate >= loo_continuation)
        downstream = np.where(exercise, immediate, downstream)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=coefficients, means=means, scales=scales)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / 'request.json').read_text(encoding='utf-8'))
    time_index = int(request['time_index'])
    states = np.load(input_dir / 'states.npy', allow_pickle=False)
    immediate = np.load(input_dir / 'immediate_payoffs.npy', allow_pickle=False)
    with np.load(model_dir / 'model.npz', allow_pickle=False) as model:
        design = features(states, immediate)
        normalized = (design - model['means'][time_index]) / model['scales'][time_index]
        predictions = normalized @ model['coefficients'][time_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predictions.npy', np.asarray(predictions, dtype=np.float64), allow_pickle=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    fit_parser = subparsers.add_parser('fit')
    fit_parser.add_argument('--input', required=True)
    fit_parser.add_argument('--output', required=True)
    fit_parser.add_argument('--seed', required=True, type=int)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--model', required=True)
    predict_parser.add_argument('--input', required=True)
    predict_parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))
if __name__ == '__main__':
    main()
