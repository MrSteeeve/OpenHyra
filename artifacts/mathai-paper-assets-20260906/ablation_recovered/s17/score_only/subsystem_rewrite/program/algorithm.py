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
        raise ValueError('training paths and payoffs must be rank 3 and rank 2')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times) or n_paths == 0 or n_times == 0:
        raise ValueError('training paths and payoffs have incompatible shapes')
    lambdas = np.asarray((0.0001, 0.001, 0.01, 0.1, 1.0, 10.0))
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_paths)
    fold = np.empty(n_paths, dtype=np.int64)
    fold[permutation] = np.arange(n_paths, dtype=np.int64) % 4
    example = features(paths[:, 0, :], payoffs[:, 0])
    n_features = example.shape[1]
    means = np.zeros((n_times, n_features), dtype=np.float64)
    scales = np.ones((n_times, n_features), dtype=np.float64)
    coefficients = np.zeros((n_times, n_features), dtype=np.float64)

    def ridge_model(x, y, lam):
        """Return normalization and coefficients in predict's model format."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        sigma[~np.isfinite(sigma) | (sigma < 1e-10)] = 1.0
        mu[0] = 0.0
        sigma[0] = 1.0
        z = (x - mu) / sigma
        gram = z.T @ z
        penalty = np.eye(z.shape[1], dtype=np.float64) * float(lam)
        penalty[0, 0] = 0.0
        rhs = z.T @ y
        try:
            beta = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return (mu, sigma, beta)

    def constant_model(value):
        mu = np.zeros(n_features, dtype=np.float64)
        sigma = np.ones(n_features, dtype=np.float64)
        beta = np.zeros(n_features, dtype=np.float64)
        beta[0] = float(value) if np.isfinite(value) else 0.0
        return (mu, sigma, beta)

    def apply_model(x, model):
        mu, sigma, beta = model
        return (x - mu) / sigma @ beta

    def select_lambda(x, y, labels):
        """Fold CV followed by the conservative one-standard-error rule."""
        unique = np.unique(labels)
        losses = [[] for _ in lambdas]
        for validation_fold in unique:
            validation = labels == validation_fold
            training = ~validation
            if np.count_nonzero(validation) == 0 or np.count_nonzero(training) < 12:
                continue
            for k, lam in enumerate(lambdas):
                model = ridge_model(x[training], y[training], lam)
                error = apply_model(x[validation], model) - y[validation]
                losses[k].extend(np.square(error).tolist())
        if not any(losses):
            return float(lambdas[-1])
        average = np.asarray([np.mean(v) if v else np.inf for v in losses], dtype=np.float64)
        best = int(np.argmin(average))
        best_losses = np.asarray(losses[best], dtype=np.float64)
        if best_losses.size > 1:
            standard_error = np.std(best_losses, ddof=1) / np.sqrt(best_losses.size)
        else:
            standard_error = 0.0
        eligible = np.flatnonzero(average <= average[best] + standard_error)
        return float(lambdas[eligible[-1]])
    cash_flow = payoffs[:, -1].copy()
    for time_index in range(n_times - 2, -1, -1):
        immediate = payoffs[:, time_index]
        design = features(paths[:, time_index, :], immediate)
        future_targets = cash_flow.copy()
        in_money = immediate > 0.0
        continuation_oof = np.empty(n_paths, dtype=np.float64)
        for held_out_fold in range(4):
            held_out = fold == held_out_fold
            training = ~held_out & in_money
            count = int(np.count_nonzero(training))
            if count < 12:
                fallback = np.mean(future_targets[training]) if count else np.mean(future_targets)
                model = constant_model(fallback)
            else:
                lam = select_lambda(design[training], future_targets[training], fold[training])
                model = ridge_model(design[training], future_targets[training], lam)
            continuation_oof[held_out] = apply_model(design[held_out], model)
        exercise = in_money & (immediate > continuation_oof)
        cash_flow[exercise] = immediate[exercise]
        deployment = np.flatnonzero(in_money)
        if deployment.size < 12:
            value = np.mean(future_targets[deployment]) if deployment.size else np.mean(future_targets)
            model = constant_model(value)
        else:
            lam = select_lambda(design[deployment], future_targets[deployment], fold[deployment])
            model = ridge_model(design[deployment], future_targets[deployment], lam)
        means[time_index], scales[time_index], coefficients[time_index] = model
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
