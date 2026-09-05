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
    payoffs = np.load(input_dir / 'payoffs.npy', allow_pickle=False)
    steps = paths.shape[1] - 1
    x0 = features(paths[:, 0, :], payoffs[:, 0])
    lo_coef = np.zeros((steps, x0.shape[1]))
    hi_coef = np.zeros_like(lo_coef)
    means = np.zeros_like(lo_coef)
    scales = np.ones_like(lo_coef)
    thresholds = np.zeros(steps)
    cash = np.asarray(payoffs[:, -1], float).copy()
    for t in range(steps - 1, -1, -1):
        x = features(paths[:, t, :], payoffs[:, t])
        gate = np.mean(np.log(np.maximum(paths[:, t, :], 1e-12)), 1)
        thresholds[t] = np.median(gate)
        lo, hi = (gate <= thresholds[t], gate > thresholds[t])
        mean, scale = (x.mean(0), x.std(0))
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
        xn = (x - mean) / scale
        reg = np.eye(x.shape[1]) * 1e-06

        def solve(mask):
            if int(mask.sum()) < x.shape[1]:
                mask = np.ones(len(x), bool)
            return np.linalg.solve(xn[mask].T @ xn[mask] + reg, xn[mask].T @ cash[mask])
        lo_coef[t], hi_coef[t] = (solve(lo), solve(hi))
        pred = np.where(lo, xn @ lo_coef[t], xn @ hi_coef[t])
        cash = np.where(payoffs[:, t] >= pred, payoffs[:, t], cash)
        means[t], scales[t] = (mean, scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=(lo_coef + hi_coef) * 0.5, coefficients_lo=lo_coef, coefficients_hi=hi_coef, means=means, scales=scales, thresholds=thresholds)

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
