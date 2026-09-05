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
    paths = np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('unexpected training array shape')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times):
        raise ValueError('paths and payoffs disagree')
    n_features = features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    means = np.zeros((n_times, n_features), dtype=np.float64)
    scales = np.ones((n_times, n_features), dtype=np.float64)
    coefficients = np.zeros((n_times, n_features), dtype=np.float64)
    rng = np.random.default_rng(29000088)
    n_folds = min(5, max(2, n_paths))
    fold_id = np.empty(n_paths, dtype=np.int64)
    fold_id[rng.permutation(n_paths)] = np.arange(n_paths) % n_folds
    cashflow = payoffs[:, -1].copy()
    shrinkage_grid = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
    for time_index in range(n_times - 2, -1, -1):
        design = features(paths[:, time_index, :], payoffs[:, time_index])
        mean = np.mean(design, axis=0)
        scale = np.std(design, axis=0)
        mean[0] = 0.0
        scale[0] = 1.0
        scale[~np.isfinite(scale) | (scale < 1e-10)] = 1.0
        normalized = (design - mean) / scale
        means[time_index] = mean
        scales[time_index] = scale
        in_money = payoffs[:, time_index] > 0.0

        def ridge(rows: np.ndarray) -> np.ndarray:
            chosen = rows & in_money
            if np.count_nonzero(chosen) < n_features + 2:
                chosen = rows
            x = normalized[chosen]
            y = cashflow[chosen]
            if x.shape[0] == 0:
                return np.zeros(n_features, dtype=np.float64)
            gram = x.T @ x
            penalty = 1e-06 * max(1, x.shape[0])
            diagonal = np.full(n_features, penalty, dtype=np.float64)
            diagonal[0] = 0.0
            gram.flat[::n_features + 1] += diagonal
            try:
                return np.linalg.solve(gram, x.T @ y)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(gram, x.T @ y, rcond=1e-12)[0]
        full_coef = ridge(np.ones(n_paths, dtype=bool))
        fold_coefs = np.empty((n_folds, n_features), dtype=np.float64)
        oof_continuation = np.empty(n_paths, dtype=np.float64)
        for fold in range(n_folds):
            training_rows = fold_id != fold
            fold_coefs[fold] = ridge(training_rows)
            validation_rows = ~training_rows
            oof_continuation[validation_rows] = normalized[validation_rows] @ fold_coefs[fold]
        aggregate_coef = np.mean(fold_coefs, axis=0)
        full_continuation = normalized @ full_coef
        immediate = payoffs[:, time_index]
        values = np.empty(shrinkage_grid.size, dtype=np.float64)
        for grid_index, weight in enumerate(shrinkage_grid):
            heldout_prediction = (1.0 - weight) * full_continuation + weight * oof_continuation
            exercise = in_money & (immediate >= heldout_prediction)
            values[grid_index] = np.mean(np.where(exercise, immediate, cashflow))
        best_index = int(np.argmax(values))
        weight = float(shrinkage_grid[best_index])
        coefficients[time_index] = (1.0 - weight) * full_coef + weight * aggregate_coef
        selected_continuation = (1.0 - weight) * full_continuation + weight * oof_continuation
        exercise = in_money & (immediate >= selected_continuation)
        cashflow = np.where(exercise, immediate, cashflow)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', means=means, scales=scales, coefficients=coefficients)

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
