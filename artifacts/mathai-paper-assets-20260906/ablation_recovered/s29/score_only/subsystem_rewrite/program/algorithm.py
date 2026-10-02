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

    def normalization(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(design, axis=0)
        scale = np.std(design, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        mean[0] = 0.0
        scale[0] = 1.0
        return ((design - mean) / scale, mean, scale)

    def ridge(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
        gram = design.T @ design
        rhs = design.T @ target
        diagonal = np.full(design.shape[1], penalty, dtype=np.float64)
        diagonal[0] = 0.0
        gram.flat[::gram.shape[0] + 1] += diagonal
        jitter = np.finfo(np.float64).eps * max(1.0, float(np.max(np.diag(gram))))
        for _ in range(5):
            try:
                factor = np.linalg.cholesky(gram)
                return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
            except np.linalg.LinAlgError:
                gram.flat[::gram.shape[0] + 1] += jitter
                jitter *= 10.0
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]
    paths = np.asarray(np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('training paths and payoffs have invalid dimensions')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times) or n_paths == 0 or n_times == 0:
        raise ValueError('training paths and payoffs have incompatible shapes')
    feature_count = features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    coefficients = np.zeros((n_times, feature_count), dtype=np.float64)
    means = np.zeros_like(coefficients)
    scales = np.ones_like(coefficients)
    fold_count = min(8, n_paths)
    permutation = np.random.default_rng(int(seed)).permutation(n_paths)
    fold_id = np.empty(n_paths, dtype=np.int64)
    fold_id[permutation] = np.arange(n_paths, dtype=np.int64) % fold_count
    penalties = np.asarray((0.001, 0.01, 0.1, 1.0, 10.0), dtype=np.float64)
    cash_flow = payoffs[:, -1].copy()
    for time_index in range(n_times - 2, -1, -1):
        design = features(paths[:, time_index, :], payoffs[:, time_index])
        normalized, mean, scale = normalization(design)
        target = cash_flow.copy()
        predictions_by_penalty = np.empty((penalties.size, n_paths), dtype=np.float64)
        for penalty_index, penalty in enumerate(penalties):
            out_of_fold = predictions_by_penalty[penalty_index]
            for held_out in range(fold_count):
                validation = fold_id == held_out
                training = ~validation
                beta = ridge(normalized[training], target[training], float(penalty))
                out_of_fold[validation] = normalized[validation] @ beta
        errors = np.mean((predictions_by_penalty - target[np.newaxis, :]) ** 2, axis=1)
        best_error = float(np.min(errors))
        tolerance = 1e-12 + 1e-06 * max(1.0, abs(best_error))
        eligible = np.flatnonzero(errors <= best_error + tolerance)
        selected = int(eligible[-1])
        continuation = predictions_by_penalty[selected]
        exercise = payoffs[:, time_index] >= continuation
        cash_flow = np.where(exercise, payoffs[:, time_index], target)
        coefficients[time_index] = ridge(normalized, target, float(penalties[selected]))
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
