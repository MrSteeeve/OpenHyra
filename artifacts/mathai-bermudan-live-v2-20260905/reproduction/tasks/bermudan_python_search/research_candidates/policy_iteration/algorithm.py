#!/usr/bin/env python3
"""Repeated backward Bellman policy-iteration continuation program."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states, immediate):
    z=np.log(np.maximum(np.asarray(states,float),1e-12))
    return np.column_stack([np.ones(len(z)),z,z*z,np.mean(z,1),np.asarray(immediate,float)])

def ridge(x,y,alpha):
    m=x.mean(0); s=x.std(0); m[0]=0.; s[s<1e-10]=1.; xn=(x-m)/s
    c=np.linalg.solve(xn.T@xn+np.eye(x.shape[1])*alpha,xn.T@y)
    return c,m,s

def fit(input_dir:Path,output_dir:Path,seed:int)->None:
    paths=np.load(input_dir/"training_paths.npy",allow_pickle=False)
    payoffs=np.load(input_dir/"payoffs.npy",allow_pickle=False)
    steps=paths.shape[1]-1; x0=features(paths[:,0,:],payoffs[:,0]); d=x0.shape[1]
    coefficients=np.zeros((steps,d)); means=np.zeros_like(coefficients); scales=np.ones_like(coefficients)
    # Two policy-evaluation/improvement sweeps make the update rule explicit.
    cash=np.asarray(payoffs[:,-1],float).copy()
    for sweep in range(2):
        cash=np.asarray(payoffs[:,-1],float).copy()
        for t in range(steps-1,-1,-1):
            x=features(paths[:,t,:],payoffs[:,t]); y=cash.copy()
            c,m,s=ridge(x,y,1e-6*(sweep+1)); coefficients[t],means[t],scales[t]=c,m,s
            continuation=((x-m)/s)@c
            cash=np.where(payoffs[:,t]>=continuation,payoffs[:,t],cash)
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez(output_dir/"model.npz",coefficients=coefficients,means=means,scales=scales,sweeps=np.array([2]))

def predict(model_dir:Path,input_dir:Path,output_dir:Path)->None:
    req=json.loads((input_dir/"request.json").read_text()); t=int(req["time_index"])
    states=np.load(input_dir/"states.npy",allow_pickle=False); immediate=np.load(input_dir/"immediate_payoffs.npy",allow_pickle=False)
    with np.load(model_dir/"model.npz",allow_pickle=False) as m:
        x=features(states,immediate); out=((x-m["means"][t])/m["scales"][t])@m["coefficients"][t]
    output_dir.mkdir(parents=True,exist_ok=True); np.save(output_dir/"predictions.npy",np.asarray(out,float),allow_pickle=False)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    f=s.add_parser("fit"); f.add_argument("--input",required=True); f.add_argument("--output",required=True); f.add_argument("--seed",type=int,required=True)
    q=s.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="fit": fit(Path(a.input),Path(a.output),a.seed)
    else: predict(Path(a.model),Path(a.input),Path(a.output))
if __name__=="__main__": main()
