#!/usr/bin/env python3
"""Small one-hidden-layer MLP continuation program trained in NumPy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states, immediate):
    z=np.log(np.maximum(np.asarray(states,float),1e-12))
    return np.column_stack([np.ones(len(z)),z,z*z,np.mean(z,1),np.asarray(immediate,float)])

def fit(input_dir:Path,output_dir:Path,seed:int)->None:
    rng=np.random.default_rng(int(seed)); paths=np.load(input_dir/"training_paths.npy",allow_pickle=False)
    payoffs=np.load(input_dir/"payoffs.npy",allow_pickle=False); steps=paths.shape[1]-1
    d=features(paths[:,0,:],payoffs[:,0]).shape[1]; hidden=min(24,max(8,2*d))
    w1s=[]; b1s=[]; w2s=[]; b2s=[]; cash=np.asarray(payoffs[:,-1],float).copy()
    for t in range(steps-1,-1,-1):
        x=features(paths[:,t,:],payoffs[:,t]); mean=x.mean(0); scale=x.std(0); mean[0]=0.; scale[scale<1e-10]=1.; x=(x-mean)/scale
        w1=rng.normal(0,0.15,(d,hidden)); b1=np.zeros(hidden); w2=rng.normal(0,0.15,hidden); b2=float(cash.mean())
        for _ in range(36):
            h=np.tanh(x@w1+b1); pred=h@w2+b2; err=pred-cash
            grad=2.0*err/len(x); gw2=h.T@grad; gb2=grad.sum(); gh=(grad[:,None]*w2)*(1-h*h)
            w1-=0.04*(x.T@gh); b1-=0.04*gh.sum(0); w2-=0.04*gw2; b2-=0.04*gb2
        pred=np.tanh(x@w1+b1)@w2+b2; cash=np.where(payoffs[:,t]>=pred,payoffs[:,t],cash)
        w1s.append(w1); b1s.append(b1); w2s.append(w2); b2s.append(b2)
    w1s=w1s[::-1]; b1s=b1s[::-1]; w2s=w2s[::-1]; b2s=b2s[::-1]
    # Recompute and store normalizers in a deterministic second pass.
    means=[]; scales=[]
    for t in range(steps):
        x=features(paths[:,t,:],payoffs[:,t]); m=x.mean(0); s=x.std(0); m[0]=0.; s[s<1e-10]=1.; means.append(m); scales.append(s)
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez(output_dir/"model.npz",w1=np.asarray(w1s),b1=np.asarray(b1s),w2=np.asarray(w2s),b2=np.asarray(b2s),means=np.asarray(means),scales=np.asarray(scales))

def predict(model_dir:Path,input_dir:Path,output_dir:Path)->None:
    req=json.loads((input_dir/"request.json").read_text()); t=int(req["time_index"])
    states=np.load(input_dir/"states.npy",allow_pickle=False); immediate=np.load(input_dir/"immediate_payoffs.npy",allow_pickle=False)
    with np.load(model_dir/"model.npz",allow_pickle=False) as m:
        x=features(states,immediate); x=(x-m["means"][t])/m["scales"][t]; out=np.tanh(x@m["w1"][t]+m["b1"][t])@m["w2"][t]+m["b2"][t]
    output_dir.mkdir(parents=True,exist_ok=True); np.save(output_dir/"predictions.npy",np.asarray(out,float),allow_pickle=False)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    f=s.add_parser("fit"); f.add_argument("--input",required=True); f.add_argument("--output",required=True); f.add_argument("--seed",type=int,required=True)
    q=s.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="fit": fit(Path(a.input),Path(a.output),a.seed)
    else: predict(Path(a.model),Path(a.input),Path(a.output))
if __name__=="__main__": main()
