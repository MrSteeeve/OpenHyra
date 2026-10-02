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
    paths = np.asarray(np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('training_paths and payoffs must be rank 3 and 2')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times) or n_paths < 4 or n_times < 1:
        raise ValueError('incompatible or insufficient training data')
    alphas = np.asarray((0.0001, 0.001, 0.01, 0.1, 1.0), dtype=np.float64)
    n_features = features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    coefficients = np.zeros((n_times, n_features), dtype=np.float64)
    means = np.zeros((n_times, n_features), dtype=np.float64)
    scales = np.ones((n_times, n_features), dtype=np.float64)

    def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.mean(x, axis=0)
        spread = np.std(x, axis=0)
        center[0] = 0.0
        spread[0] = 1.0
        spread[~np.isfinite(spread) | (spread < 1e-12)] = 1.0
        return ((x - center) / spread, center, spread)

    def select_and_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u, singular, vt = np.linalg.svd(x, full_matrices=False)
        projected = u.T @ y
        y_energy = float(y @ y)
        captured = float(projected @ projected)
        orthogonal = max(0.0, y_energy - captured)
        best_score = np.inf
        best_coef = np.zeros(x.shape[1], dtype=np.float64)
        singular_sq = singular * singular
        for alpha in alphas:
            shrink = singular_sq / (singular_sq + alpha)
            residual_projection = (1.0 - shrink) * projected
            rss = orthogonal + float(residual_projection @ residual_projection)
            denominator = max(1e-12, 1.0 - float(np.sum(shrink)) / x.shape[0])
            score = rss / x.shape[0] / (denominator * denominator)
            if score < best_score:
                best_score = score
                best_coef = vt.T @ (singular / (singular_sq + alpha) * projected)
        return best_coef
    permutation = np.random.default_rng(seed).permutation(n_paths)
    fold_id = np.empty(n_paths, dtype=np.int8)
    fold_id[permutation] = np.arange(n_paths, dtype=np.int64) % 4
    cash_flow = payoffs[:, -1].copy()
    for time_index in range(n_times - 2, -1, -1):
        design = features(paths[:, time_index, :], payoffs[:, time_index])
        continuation_target = cash_flow.copy()
        held_out_continuation = np.empty(n_paths, dtype=np.float64)
        for fold in range(4):
            held_out = fold_id == fold
            training = ~held_out
            x_training, fold_means, fold_scales = normalize(design[training])
            fold_coef = select_and_fit(x_training, continuation_target[training])
            held_out_continuation[held_out] = (design[held_out] - fold_means) / fold_scales @ fold_coef
        immediate = payoffs[:, time_index]
        exercise = (immediate > 0.0) & (immediate >= held_out_continuation)
        cash_flow[exercise] = immediate[exercise]
        x_all, all_means, all_scales = normalize(design)
        all_coef = select_and_fit(x_all, continuation_target)
        fitted_continuation = x_all @ all_coef
        zero_payoff = immediate <= 0.0
        if np.any(zero_payoff):
            q01 = float(np.percentile(fitted_continuation[zero_payoff], 1.0))
            all_coef[0] += max(0.0, -q01)
        coefficients[time_index] = all_coef
        means[time_index] = all_means
        scales[time_index] = all_scales
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
