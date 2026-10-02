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

    def standardized_design(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = np.asarray(design, dtype=np.float64)
        mean = np.mean(design, axis=0)
        scale = np.std(design, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        if design.shape[1]:
            mean[0] = 0.0
            scale[0] = 1.0
        normalized = (design - mean) / scale
        return (normalized, mean, scale)

    def ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized, mean, scale = standardized_design(design)
        target = np.asarray(target, dtype=np.float64)
        count = max(1, normalized.shape[0])
        gram = normalized.T @ normalized / count
        rhs = normalized.T @ target / count
        penalty = np.eye(normalized.shape[1], dtype=np.float64) * float(ridge)
        if penalty.shape[0]:
            penalty[0, 0] = 0.0
        try:
            coefficient = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        coefficient = np.nan_to_num(coefficient, nan=0.0, posinf=0.0, neginf=0.0)
        return (coefficient, mean, scale)
    paths = np.asarray(np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2 or paths.shape[:2] != payoffs.shape:
        raise ValueError('training_paths and payoffs have incompatible shapes')
    n_paths, n_times, _ = paths.shape
    if n_paths < 1 or n_times < 1:
        raise ValueError('empty training data')
    designs = [features(paths[:, time_index, :], payoffs[:, time_index]) for time_index in range(n_times)]
    feature_count = designs[0].shape[1]
    minimum_positive = max(20, 2 * feature_count)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_paths)
    fold_ids = np.empty(n_paths, dtype=np.int64)
    fold_ids[permutation] = np.arange(n_paths, dtype=np.int64) % 4
    ridge_values = (1e-06, 0.0001, 0.01, 0.1)

    def cross_fitted_pass(ridge: float, keep_targets: bool=False):
        cash_flow = payoffs[:, -1].copy()
        downstream_targets = [None] * n_times if keep_targets else None
        for time_index in range(n_times - 2, -1, -1):
            target = cash_flow.copy()
            if keep_targets:
                downstream_targets[time_index] = target
            positive = payoffs[:, time_index] > 0.0
            use_positive = int(np.count_nonzero(positive)) >= minimum_positive
            continuation = np.zeros(n_paths, dtype=np.float64)
            for fold in range(4):
                held_out = fold_ids == fold
                if not np.any(held_out):
                    continue
                training = ~held_out
                if use_positive:
                    training &= positive
                coefficient, mean, scale = ridge_fit(designs[time_index][training], target[training], ridge)
                held_design = (designs[time_index][held_out] - mean) / scale
                continuation[held_out] = held_design @ coefficient
            exercise = positive & (payoffs[:, time_index] >= continuation)
            cash_flow[exercise] = payoffs[exercise, time_index]
        return (cash_flow, downstream_targets)
    candidate_results = []
    for ridge in ridge_values:
        realized, _ = cross_fitted_pass(ridge)
        fold_means = np.asarray([np.mean(realized[fold_ids == fold]) if np.any(fold_ids == fold) else np.mean(realized) for fold in range(4)])
        candidate_results.append((ridge, realized, float(np.mean(realized)), fold_means))
    best = max(candidate_results, key=lambda item: item[2])
    if best[3].size > 1:
        standard_error = float(np.std(best[3], ddof=1) / np.sqrt(best[3].size))
    else:
        standard_error = 0.0
    cutoff = best[2] - standard_error
    selected_ridge = max((item[0] for item in candidate_results if item[2] >= cutoff))
    _, selected_targets = cross_fitted_pass(selected_ridge, keep_targets=True)
    coefficients = np.zeros((n_times, feature_count), dtype=np.float64)
    means = np.zeros((n_times, feature_count), dtype=np.float64)
    scales = np.ones((n_times, feature_count), dtype=np.float64)
    for time_index in range(n_times - 2, -1, -1):
        positive = payoffs[:, time_index] > 0.0
        rows = positive if int(np.count_nonzero(positive)) >= minimum_positive else np.ones(n_paths, dtype=bool)
        coefficient, mean, scale = ridge_fit(designs[time_index][rows], selected_targets[time_index][rows], selected_ridge)
        coefficients[time_index] = coefficient
        means[time_index] = mean
        scales[time_index] = scale
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
