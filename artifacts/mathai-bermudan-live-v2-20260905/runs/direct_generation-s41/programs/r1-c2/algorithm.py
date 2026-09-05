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
    import numpy as _np
    paths = _np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoff = _np.load(input_dir / 'payoffs.npy', allow_pickle=False).astype(float)
    if paths.ndim != 3 or payoff.ndim != 2 or paths.shape[:2] != payoff.shape:
        raise ValueError('incompatible training_paths.npy and payoffs.npy shapes')
    n, nt, _ = paths.shape
    if n < 2 or nt < 2:
        raise ValueError('at least two paths and two exercise dates are required')
    p = features(paths[:, 0], payoff[:, 0]).shape[1]
    width = 24
    means = _np.zeros((nt, p))
    scales = _np.ones((nt, p))
    coefficients = _np.zeros((nt, p))
    target_scales = _np.ones(nt)
    w1 = _np.zeros((nt, p, width))
    b1 = _np.zeros((nt, width))
    w2 = _np.zeros((nt, width))
    b2 = _np.zeros(nt)
    rng = _np.random.default_rng(int(seed))

    def _design(t, rows, mean=None, scale=None):
        raw = features(paths[rows, t], payoff[rows, t])
        if mean is None:
            mean = _np.mean(raw, axis=0)
            scale = _np.std(raw, axis=0)
            scale = _np.where(scale > 1e-08, scale, 1.0)
            mean[0], scale[0] = (0.0, 1.0)
        return ((raw - mean) / scale, mean, scale)

    def _solve(a, y, lam):
        gram = a.T @ a
        penalty = _np.eye(a.shape[1]) * lam
        penalty[0, 0] = lam * 0.001
        try:
            return _np.linalg.solve(gram + penalty, a.T @ y)
        except _np.linalg.LinAlgError:
            return _np.linalg.lstsq(gram + penalty, a.T @ y, rcond=1e-10)[0]

    def _train(t, train_rows, eval_rows, full=False):
        x, mu, sd = _design(t, train_rows)
        xe, _, _ = _design(t, eval_rows, mu, sd)
        y = cash[train_rows]
        beta = _solve(x, y, max(1e-07, 2e-05 * len(train_rows)))
        resid = y - x @ beta
        ys = max(float(_np.std(y)), 1e-08)
        rw = rng.normal(0.0, 0.55 / _np.sqrt(max(p - 1, 1)), (p, width))
        rw[0] = 0.0
        rb = rng.uniform(-0.7, 0.7, width)
        h = _np.tanh(x @ rw + rb)
        he = _np.tanh(xe @ rw + rb)
        aug = _np.column_stack((_np.ones(len(h)), h))
        out = _solve(aug, resid / ys, max(1e-06, 0.002 * len(train_rows)))
        pred = xe @ beta + ys * (_np.column_stack((_np.ones(len(he)), he)) @ out)
        if full:
            means[t], scales[t], coefficients[t] = (mu, sd, beta)
            target_scales[t], w1[t], b1[t] = (ys, rw, rb)
            b2[t], w2[t] = (out[0], out[1:])
        return pred
    cash = payoff[:, -1].copy()
    folds = (_np.arange(n) * 1103515245 + (int(seed) & 2147483647)) % 3
    for t in range(nt - 2, -1, -1):
        itm = _np.flatnonzero(payoff[:, t] > 0.0)
        minimum = min(n, max(p + 8, 32))
        if len(itm) < minimum:
            itm = _np.argsort(payoff[:, t])[-minimum:]
        continuation = _np.empty(n)
        for fold in range(3):
            train = itm[folds[itm] != fold]
            test = _np.flatnonzero(folds == fold)
            if len(train) < p + 2:
                train = itm
            continuation[test] = _train(t, train, test)
        _train(t, itm, itm, full=True)
        exercise = (payoff[:, t] > 0.0) & (payoff[:, t] >= continuation)
        cash[exercise] = payoff[exercise, t]
    output_dir.mkdir(parents=True, exist_ok=True)
    _np.savez(output_dir / 'model.npz', means=means, scales=scales, coefficients=coefficients, target_scales=target_scales, w1=w1, b1=b1, w2=w2, b2=b2)

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
