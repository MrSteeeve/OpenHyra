
import argparse as _openhyra_argparse
from pathlib import Path as _OpenHyraPath
import shutil as _openhyra_shutil
import sys as _openhyra_sys
import types as _openhyra_types
import numpy as _openhyra_numpy

_PARENT_A_SOURCE = '#!/usr/bin/env python3\n"""Complete fit/predict seed for the open Python-program track.\n\nThis is deliberately just one ordinary starting program. Search candidates\nmay replace every function, model representation, objective, and update rule.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nfrom pathlib import Path\n\nimport numpy as np\n\n\ndef features(states: np.ndarray, immediate: np.ndarray) -> np.ndarray:\n    states = np.asarray(states, dtype=np.float64)\n    logged = np.log(np.maximum(states, 1e-12))\n    columns = [\n        np.ones(states.shape[0], dtype=np.float64),\n        *[logged[:, index] for index in range(logged.shape[1])],\n        *[logged[:, index] ** 2 for index in range(logged.shape[1])],\n        np.mean(logged, axis=1),\n        np.min(logged, axis=1),\n        np.max(logged, axis=1),\n        np.asarray(immediate, dtype=np.float64),\n    ]\n    return np.column_stack(columns)\n\n\ndef fit(input_dir: Path, output_dir: Path, seed: int) -> None:\n    del seed\n    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)\n    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)\n    n_steps = paths.shape[1] - 1\n    feature_count = features(paths[:, 0, :], payoffs[:, 0]).shape[1]\n    coefficients = np.zeros((n_steps, feature_count), dtype=np.float64)\n    means = np.zeros_like(coefficients)\n    scales = np.ones_like(coefficients)\n    cash_flow = np.asarray(payoffs[:, -1], dtype=np.float64).copy()\n\n    for time_index in range(n_steps - 1, -1, -1):\n        design = features(paths[:, time_index, :], payoffs[:, time_index])\n        mean = np.mean(design, axis=0)\n        scale = np.std(design, axis=0)\n        mean[0] = 0.0\n        scale[scale < 1e-10] = 1.0\n        normalized = (design - mean) / scale\n        gram = normalized.T @ normalized\n        gram.flat[:: feature_count + 1] += 1e-6\n        coefficient = np.linalg.solve(gram, normalized.T @ cash_flow)\n        continuation = normalized @ coefficient\n        exercise = payoffs[:, time_index] >= continuation\n        cash_flow = np.where(exercise, payoffs[:, time_index], cash_flow)\n        coefficients[time_index] = coefficient\n        means[time_index] = mean\n        scales[time_index] = scale\n\n    output_dir.mkdir(parents=True, exist_ok=True)\n    np.savez(\n        output_dir / "model.npz",\n        coefficients=coefficients,\n        means=means,\n        scales=scales,\n    )\n\n\ndef predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:\n    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))\n    time_index = int(request["time_index"])\n    states = np.load(input_dir / "states.npy", allow_pickle=False)\n    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)\n    with np.load(model_dir / "model.npz", allow_pickle=False) as model:\n        design = features(states, immediate)\n        normalized = (\n            design - model["means"][time_index]\n        ) / model["scales"][time_index]\n        predictions = normalized @ model["coefficients"][time_index]\n    output_dir.mkdir(parents=True, exist_ok=True)\n    np.save(\n        output_dir / "predictions.npy",\n        np.asarray(predictions, dtype=np.float64),\n        allow_pickle=False,\n    )\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    subparsers = parser.add_subparsers(dest="command", required=True)\n    fit_parser = subparsers.add_parser("fit")\n    fit_parser.add_argument("--input", required=True)\n    fit_parser.add_argument("--output", required=True)\n    fit_parser.add_argument("--seed", required=True, type=int)\n    predict_parser = subparsers.add_parser("predict")\n    predict_parser.add_argument("--model", required=True)\n    predict_parser.add_argument("--input", required=True)\n    predict_parser.add_argument("--output", required=True)\n    args = parser.parse_args()\n\n    if args.command == "fit":\n        fit(Path(args.input), Path(args.output), args.seed)\n    else:\n        predict(Path(args.model), Path(args.input), Path(args.output))\n\n\nif __name__ == "__main__":\n    main()\n'
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
    import numpy as np
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype == np.bool_ and right_array.dtype == np.bool_:
        return np.logical_or(left_array, right_array)
    anchor = np.asarray(left_array, dtype=np.float64)
    nonlinear = np.asarray(right_array, dtype=np.float64)
    anchor = np.where(np.isfinite(anchor), anchor, 0.0)
    nonlinear = np.where(np.isfinite(nonlinear), nonlinear, anchor)
    disagreement = np.abs(nonlinear - anchor)
    local_scale = 1.0 + np.abs(anchor)
    agreement = 1.0 / (1.0 + (disagreement / local_scale) ** 2)
    nonlinear_weight = 0.35 * agreement
    combined = anchor + nonlinear_weight * (nonlinear - anchor)
    return np.maximum(combined, 0.0)

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
