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
    import numpy as _np
    paths = _np.asarray(_np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=_np.float64)
    payoffs = _np.asarray(_np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=_np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('unexpected training array rank')
    if paths.shape[:2] != payoffs.shape:
        raise ValueError('paths and payoffs have incompatible shapes')
    n_paths, n_times, _ = paths.shape
    if n_paths == 0 or n_times < 2:
        raise ValueError('training set must contain paths and exercise times')

    def _design(state, immediate):
        logged = _np.log(_np.maximum(state, 1e-12))
        cols = [_np.ones(state.shape[0], dtype=_np.float64)]
        cols.extend((logged[:, j] for j in range(logged.shape[1])))
        cols.extend((logged[:, j] ** 2 for j in range(logged.shape[1])))
        cols.extend((_np.mean(logged, axis=1), _np.min(logged, axis=1), _np.max(logged, axis=1), immediate))
        return _np.column_stack(cols)
    sample = _design(paths[:, 0, :], payoffs[:, 0])
    n_features = sample.shape[1]
    coefficients = _np.zeros((n_times, n_features), dtype=_np.float64)
    means = _np.zeros_like(coefficients)
    scales = _np.ones_like(coefficients)
    cash = payoffs[:, -1].copy()
    tiny = 1e-12

    def _ridge(x, y, lam):
        gram = x.T @ x
        penalty = _np.eye(x.shape[1], dtype=_np.float64) * lam
        penalty[0, 0] = 0.0
        rhs = x.T @ y
        try:
            return _np.linalg.solve(gram + penalty, rhs)
        except _np.linalg.LinAlgError:
            return _np.linalg.lstsq(gram + penalty, rhs, rcond=1e-10)[0]
    rng = _np.random.default_rng(int(seed))
    fold = rng.integers(0, 4, size=n_paths)
    lambdas = (1e-08, 1e-06, 0.0001, 0.01, 0.1, 1.0)
    for time_index in range(n_times - 2, -1, -1):
        immediate = payoffs[:, time_index]
        raw = _design(paths[:, time_index, :], immediate)
        mu = _np.mean(raw, axis=0)
        sigma = _np.std(raw, axis=0)
        mu[0] = 0.0
        sigma[0] = 1.0
        sigma[~_np.isfinite(sigma) | (sigma < 1e-10)] = 1.0
        x_all = (raw - mu) / sigma
        itm = immediate > tiny
        useful = _np.flatnonzero(itm)
        minimum = min(n_paths, max(32, 4 * n_features))
        if useful.size < minimum:
            order = _np.argsort(immediate, kind='stable')
            useful = order[-minimum:]
        x = x_all[useful]
        y = cash[useful]
        best_lambda = lambdas[2]
        best_error = _np.inf
        if useful.size >= max(16, 2 * n_features):
            for lam in lambdas:
                error_sum = 0.0
                error_count = 0
                for k in range(4):
                    valid = fold[useful] == k
                    if not _np.any(valid) or _np.sum(~valid) < n_features:
                        continue
                    beta_cv = _ridge(x[~valid], y[~valid], lam)
                    residual = y[valid] - x[valid] @ beta_cv
                    error_sum += float(residual @ residual)
                    error_count += int(valid.sum())
                if error_count and error_sum / error_count < best_error:
                    best_error = error_sum / error_count
                    best_lambda = lam
        beta = _ridge(x, y, best_lambda)
        beta[~_np.isfinite(beta)] = 0.0
        continuation = x_all @ beta
        exercise = itm & (immediate > continuation)
        cash[exercise] = immediate[exercise]
        coefficients[time_index] = beta
        means[time_index] = mu
        scales[time_index] = sigma
    output_dir.mkdir(parents=True, exist_ok=True)
    _np.savez(output_dir / 'model.npz', coefficients=coefficients, means=means, scales=scales)

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
