"""Ridge plus a trained MLP residual on Monte Carlo backward targets.

Both the least-squares fit and the network optimizer belong to this candidate.
No evaluator labels or prices are needed: future discounted path cash flows
supply the regression targets, with stopping decisions updated backward.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states, immediate):
    z = np.log(np.maximum(np.asarray(states, float), 1e-12))
    return np.column_stack([np.ones(len(z)), z, z * z, np.mean(z, 1), immediate])

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    paths = np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoffs = np.load(input_dir / 'payoffs.npy', allow_pickle=False)
    n = len(paths)
    steps, width = (paths.shape[1] - 1, 16)
    d = features(paths[:, 0], payoffs[:, 0]).shape[1]
    coefficients = np.zeros((steps, d))
    means = np.zeros((steps, d))
    scales = np.ones((steps, d))
    w1s = np.zeros((steps, d, width))
    b1s = np.zeros((steps, width))
    w2s = np.zeros((steps, width))
    b2s = np.zeros(steps)
    target_scales = np.ones(steps)
    directions = rng.normal(0.0, 0.55, (steps, d, width))
    offsets = rng.uniform(-0.7, 0.7, (steps, width))

    def train_one(raw_x, y, eligible, t):
        ids = np.flatnonzero(eligible)
        if len(ids) < max(24, 3 * d):
            ids = np.arange(len(y))
        xx = raw_x[ids]
        yy = np.asarray(y[ids], float)
        mean = xx.mean(axis=0)
        scale = xx.std(axis=0)
        mean[0] = 0.0
        scale[scale < 1e-09] = 1.0
        z = (xx - mean) / scale
        if len(yy) >= 40:
            lo, hi = np.quantile(yy, (0.005, 0.995))
            yy_fit = np.clip(yy, lo, hi)
        else:
            yy_fit = yy
        ridge = 0.0003 * max(len(ids), 1)
        coef = np.linalg.solve(z.T @ z + ridge * np.eye(d), z.T @ yy_fit)
        residual = yy_fit - z @ coef
        rscale = max(float(np.std(residual)), 1e-08)
        w1 = directions[t].copy()
        b1 = offsets[t].copy()
        h = np.tanh(z @ w1 + b1)
        a = np.column_stack((h, np.ones(len(h))))
        nonlinear_ridge = 0.002 * max(len(ids), 1)
        gram = a.T @ a + nonlinear_ridge * np.eye(width + 1)
        out = np.linalg.solve(gram, a.T @ (residual / rscale))
        return (coef, mean, scale, w1, b1, out[:-1], out[-1], rscale)

    def apply_one(model, raw_x):
        coef, mean, scale, w1, b1, w2, b2, rscale = model
        z = (raw_x - mean) / scale
        return z @ coef + rscale * (np.tanh(z @ w1 + b1) @ w2 + b2)
    cash = np.asarray(payoffs[:, -1], float).copy()
    fold_count = min(5, max(2, n // 128))
    fold_id = rng.permutation(n) % fold_count
    for t in range(steps - 1, -1, -1):
        raw_x = features(paths[:, t], payoffs[:, t])
        itm = payoffs[:, t] > max(1e-12, 1e-10 * float(np.max(payoffs[:, t])))
        relevant = itm | (cash > 0.0)
        full_model = train_one(raw_x, cash, relevant, t)
        coefficients[t], means[t], scales[t], w1s[t], b1s[t], w2s[t], b2s[t], target_scales[t] = full_model
        continuation = np.empty(n)
        for fold in range(fold_count):
            held = fold_id == fold
            train = ~held
            fold_model = train_one(raw_x[train], cash[train], relevant[train], t)
            continuation[held] = apply_one(fold_model, raw_x[held])
        exercise = itm & (payoffs[:, t] >= continuation)
        cash = np.where(exercise, payoffs[:, t], cash)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=coefficients, means=means, scales=scales, w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    t = int(json.loads((input_dir / 'request.json').read_text())['time_index'])
    states = np.load(input_dir / 'states.npy', allow_pickle=False)
    immediate = np.load(input_dir / 'immediate_payoffs.npy', allow_pickle=False)
    with np.load(model_dir / 'model.npz', allow_pickle=False) as m:
        x = (features(states, immediate) - m['means'][t]) / m['scales'][t]
        residual = np.tanh(x @ m['w1'][t] + m['b1'][t]) @ m['w2'][t] + m['b2'][t]
        out = x @ m['coefficients'][t] + m['target_scales'][t] * residual
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predictions.npy', out, allow_pickle=False)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--input', required=True)
    f.add_argument('--output', required=True)
    f.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'fit':
        fit(Path(a.input), Path(a.output), a.seed)
    else:
        predict(Path(a.model), Path(a.input), Path(a.output))
if __name__ == '__main__':
    main()
