
import argparse as _openhyra_argparse
from pathlib import Path as _OpenHyraPath
import shutil as _openhyra_shutil
import sys as _openhyra_sys
import types as _openhyra_types
import numpy as _openhyra_numpy

_PARENT_A_SOURCE = '#!/usr/bin/env python3\n"""Ridge plus a trained MLP residual on Monte Carlo backward targets.\n\nBoth the least-squares fit and the network optimizer belong to this candidate.\nNo evaluator labels or prices are needed: future discounted path cash flows\nsupply the regression targets, with stopping decisions updated backward.\n"""\nfrom __future__ import annotations\nimport argparse\nimport json\nfrom pathlib import Path\nimport numpy as np\n\n\ndef features(states, immediate):\n    z = np.log(np.maximum(np.asarray(states, float), 1e-12))\n    return np.column_stack([np.ones(len(z)), z, z * z, np.mean(z, 1), immediate])\n\n\ndef fit(input_dir: Path, output_dir: Path, seed: int) -> None:\n    rng = np.random.default_rng(seed)\n    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)\n    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)\n    steps, width = paths.shape[1] - 1, 16\n    d = features(paths[:, 0], payoffs[:, 0]).shape[1]\n    coefs = np.zeros((steps, d)); means = np.zeros_like(coefs); scales = np.ones_like(coefs)\n    w1s = np.zeros((steps, d, width)); b1s = np.zeros((steps, width))\n    w2s = np.zeros((steps, width)); b2s = np.zeros(steps); target_scales = np.ones(steps)\n    targets = np.zeros((steps, len(paths))); losses = np.zeros((steps, 2)); updates = np.zeros(steps)\n    cash = payoffs[:, -1].copy()\n    for t in range(steps - 1, -1, -1):\n        targets[t] = cash\n        x = features(paths[:, t], payoffs[:, t])\n        mean, scale = x.mean(0), x.std(0); mean[0] = 0.; scale[scale < 1e-10] = 1.\n        x = (x - mean) / scale\n        coef = np.linalg.solve(x.T @ x + np.eye(d) * 1e-5, x.T @ cash)\n        residual = cash - x @ coef\n        target_scale = max(float(residual.std()), 1e-5)\n        y = residual / target_scale\n        w1 = rng.normal(0., .15, (d, width)); initial = w1.copy()\n        b1 = np.zeros(width); w2 = rng.normal(0., .1, width); b2 = 0.\n        losses[t, 0] = np.mean((np.tanh(x @ w1 + b1) @ w2 + b2 - y) ** 2)\n        for _ in range(64):\n            h = np.tanh(x @ w1 + b1); err = h @ w2 + b2 - y\n            g = 2. * err / len(x); hidden_g = (g[:, None] * w2) * (1. - h * h)\n            gw1, gb1, gw2, gb2 = x.T @ hidden_g, hidden_g.sum(0), h.T @ g, g.sum()\n            w1 -= .03 * gw1; b1 -= .03 * gb1; w2 -= .03 * gw2; b2 -= .03 * gb2\n        residual_pred = np.tanh(x @ w1 + b1) @ w2 + b2\n        losses[t, 1] = np.mean((residual_pred - y) ** 2)\n        updates[t] = np.linalg.norm(w1 - initial)\n        continuation = x @ coef + target_scale * residual_pred\n        cash = np.where(payoffs[:, t] >= continuation, payoffs[:, t], cash)\n        coefs[t], means[t], scales[t] = coef, mean, scale\n        w1s[t], b1s[t], w2s[t], b2s[t], target_scales[t] = w1, b1, w2, b2, target_scale\n    output_dir.mkdir(parents=True, exist_ok=True)\n    np.savez(output_dir / "model.npz", coefficients=coefs, means=means, scales=scales,\n             w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)\n    np.savez(output_dir / "training_trace.npz", backward_targets=targets,\n             loss_before_after=losses, first_layer_update_norm=updates, seed=np.array(seed))\n\n\ndef predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:\n    t = int(json.loads((input_dir / "request.json").read_text())["time_index"])\n    states = np.load(input_dir / "states.npy", allow_pickle=False)\n    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)\n    with np.load(model_dir / "model.npz", allow_pickle=False) as m:\n        x = (features(states, immediate) - m["means"][t]) / m["scales"][t]\n        residual = np.tanh(x @ m["w1"][t] + m["b1"][t]) @ m["w2"][t] + m["b2"][t]\n        out = x @ m["coefficients"][t] + m["target_scales"][t] * residual\n    output_dir.mkdir(parents=True, exist_ok=True)\n    np.save(output_dir / "predictions.npy", out, allow_pickle=False)\n\n\ndef main():\n    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)\n    f = sub.add_parser("fit"); f.add_argument("--input", required=True); f.add_argument("--output", required=True); f.add_argument("--seed", type=int, required=True)\n    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True)\n    a = parser.parse_args()\n    if a.command == "fit": fit(Path(a.input), Path(a.output), a.seed)\n    else: predict(Path(a.model), Path(a.input), Path(a.output))\n\nif __name__ == "__main__": main()\n'
_PARENT_B_SOURCE = '#!/usr/bin/env python3\n"""Small one-hidden-layer MLP continuation program trained in NumPy."""\nfrom __future__ import annotations\nimport argparse\nimport json\nfrom pathlib import Path\nimport numpy as np\n\ndef features(states, immediate):\n    z=np.log(np.maximum(np.asarray(states,float),1e-12))\n    return np.column_stack([np.ones(len(z)),z,z*z,np.mean(z,1),np.asarray(immediate,float)])\n\ndef fit(input_dir:Path,output_dir:Path,seed:int)->None:\n    rng=np.random.default_rng(int(seed)); paths=np.load(input_dir/"training_paths.npy",allow_pickle=False)\n    payoffs=np.load(input_dir/"payoffs.npy",allow_pickle=False); steps=paths.shape[1]-1\n    d=features(paths[:,0,:],payoffs[:,0]).shape[1]; hidden=min(24,max(8,2*d))\n    w1s=[]; b1s=[]; w2s=[]; b2s=[]; cash=np.asarray(payoffs[:,-1],float).copy()\n    targets=np.zeros((steps,len(paths))); losses=np.zeros((steps,2)); updates=np.zeros(steps)\n    for t in range(steps-1,-1,-1):\n        targets[t]=cash.copy()\n        x=features(paths[:,t,:],payoffs[:,t]); mean=x.mean(0); scale=x.std(0); mean[0]=0.; scale[scale<1e-10]=1.; x=(x-mean)/scale\n        w1=rng.normal(0,0.15,(d,hidden)); b1=np.zeros(hidden); w2=rng.normal(0,0.15,hidden); b2=float(cash.mean())\n        initial=w1.copy(); losses[t,0]=np.mean((np.tanh(x@w1+b1)@w2+b2-cash)**2)\n        for _ in range(36):\n            h=np.tanh(x@w1+b1); pred=h@w2+b2; err=pred-cash\n            grad=2.0*err/len(x); gw2=h.T@grad; gb2=grad.sum(); gh=(grad[:,None]*w2)*(1-h*h)\n            w1-=0.04*(x.T@gh); b1-=0.04*gh.sum(0); w2-=0.04*gw2; b2-=0.04*gb2\n        losses[t,1]=np.mean((np.tanh(x@w1+b1)@w2+b2-cash)**2); updates[t]=np.linalg.norm(w1-initial)\n        pred=np.tanh(x@w1+b1)@w2+b2; cash=np.where(payoffs[:,t]>=pred,payoffs[:,t],cash)\n        w1s.append(w1); b1s.append(b1); w2s.append(w2); b2s.append(b2)\n    w1s=w1s[::-1]; b1s=b1s[::-1]; w2s=w2s[::-1]; b2s=b2s[::-1]\n    # Recompute and store normalizers in a deterministic second pass.\n    means=[]; scales=[]\n    for t in range(steps):\n        x=features(paths[:,t,:],payoffs[:,t]); m=x.mean(0); s=x.std(0); m[0]=0.; s[s<1e-10]=1.; means.append(m); scales.append(s)\n    output_dir.mkdir(parents=True,exist_ok=True)\n    np.savez(output_dir/"model.npz",w1=np.asarray(w1s),b1=np.asarray(b1s),w2=np.asarray(w2s),b2=np.asarray(b2s),means=np.asarray(means),scales=np.asarray(scales))\n\n    np.savez(output_dir/"training_trace.npz",backward_targets=targets,loss_before_after=losses,first_layer_update_norm=updates,seed=np.array(seed))\n\ndef predict(model_dir:Path,input_dir:Path,output_dir:Path)->None:\n    req=json.loads((input_dir/"request.json").read_text()); t=int(req["time_index"])\n    states=np.load(input_dir/"states.npy",allow_pickle=False); immediate=np.load(input_dir/"immediate_payoffs.npy",allow_pickle=False)\n    with np.load(model_dir/"model.npz",allow_pickle=False) as m:\n        x=features(states,immediate); x=(x-m["means"][t])/m["scales"][t]; out=np.tanh(x@m["w1"][t]+m["b1"][t])@m["w2"][t]+m["b2"][t]\n    output_dir.mkdir(parents=True,exist_ok=True); np.save(output_dir/"predictions.npy",np.asarray(out,float),allow_pickle=False)\n\ndef main():\n    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)\n    f=s.add_parser("fit"); f.add_argument("--input",required=True); f.add_argument("--output",required=True); f.add_argument("--seed",type=int,required=True)\n    q=s.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True)\n    a=p.parse_args()\n    if a.command=="fit": fit(Path(a.input),Path(a.output),a.seed)\n    else: predict(Path(a.model),Path(a.input),Path(a.output))\nif __name__=="__main__": main()\n'

def _load_parent_module(name, source):
    module = _openhyra_types.ModuleType(name)
    module.__file__ = name + ".py"
    module.__package__ = name.rpartition(".")[0]
    _openhyra_sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module

_PARENT_A_MODULE = "_openhyra_parent_a"
_PARENT_B_MODULE = "_openhyra_parent_b"
_parent_namespace = globals().get("__name__", "_openhyra_composite")
_parent_a = _load_parent_module(
    _parent_namespace + "." + _PARENT_A_MODULE, _PARENT_A_SOURCE
)
_parent_b = _load_parent_module(
    _parent_namespace + "." + _PARENT_B_MODULE, _PARENT_B_SOURCE
)
_parent_a_fit = _parent_a.fit
_parent_b_fit = _parent_b.fit
_parent_a_predict = _parent_a.predict
_parent_b_predict = _parent_b.predict

def _combine_predictions(left, right):
    if left.dtype == _openhyra_numpy.bool_ and right.dtype == _openhyra_numpy.bool_:
        return _openhyra_numpy.logical_or(left, right)
    return _openhyra_numpy.asarray(left, dtype=_openhyra_numpy.float64) / 2.0 + _openhyra_numpy.asarray(right, dtype=_openhyra_numpy.float64) / 2.0

def fit(input_dir, output_dir, seed):
    output_root = _OpenHyraPath(output_dir)
    left_model = output_root / "parent_a"
    right_model = output_root / "parent_b"
    left_model.mkdir(parents=True, exist_ok=True)
    right_model.mkdir(parents=True, exist_ok=True)
    _parent_a_fit(_OpenHyraPath(input_dir), left_model, seed)
    _parent_b_fit(_OpenHyraPath(input_dir), right_model, seed)

def predict(model_dir, input_dir, output_dir):
    model_root = _OpenHyraPath(model_dir)
    output_root = _OpenHyraPath(output_dir)
    scratch = output_root / "_parent_predictions"
    left_output = scratch / "parent_a"
    right_output = scratch / "parent_b"
    left_output.mkdir(parents=True, exist_ok=True)
    right_output.mkdir(parents=True, exist_ok=True)
    _parent_a_predict(model_root / "parent_a", _OpenHyraPath(input_dir), left_output)
    _parent_b_predict(model_root / "parent_b", _OpenHyraPath(input_dir), right_output)
    left = _openhyra_numpy.load(left_output / "predictions.npy", allow_pickle=False)
    right = _openhyra_numpy.load(right_output / "predictions.npy", allow_pickle=False)
    if left.shape != right.shape:
        raise ValueError("parent prediction shapes differ")
    combined = _combine_predictions(left, right)
    _openhyra_shutil.rmtree(scratch)
    _openhyra_numpy.save(
        output_root / "predictions.npy", combined, allow_pickle=False
    )

def main():
    parser = _openhyra_argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(_OpenHyraPath(args.input), _OpenHyraPath(args.output), args.seed)
    else:
        predict(
            _OpenHyraPath(args.model),
            _OpenHyraPath(args.input),
            _OpenHyraPath(args.output),
        )

if __name__ == "__main__":
    main()
