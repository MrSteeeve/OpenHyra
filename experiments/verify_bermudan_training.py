#!/usr/bin/env python3
"""Export inspectable training/probe artifacts for the MLP and residual control."""
import argparse
import io
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping.training_pipeline import run_per_instance_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    instance=evaluator.public_suite()[0];paths=evaluator.simulate_paths(instance,256,71)
    rows=[]
    for family in ('mlp','residual_hybrid'):
        destination=a.output/family
        if destination.exists(): raise RuntimeError('use a fresh validation directory')
        result=run_per_instance_training(instance=instance,training_paths=paths,
            candidate_source_dir=ROOT/'tasks/bermudan_python_search/research_candidates'/family,
            cell_dir=destination,train_seed=71,runtime_roots=evaluator._training_runtime_roots())
        if result.status!='ok': raise RuntimeError(result.log_tail)
        files=dict(result.runner.artifact.files)
        with np.load(io.BytesIO(files['training_trace.npz']),allow_pickle=False) as trace:
            updated=bool(np.all(trace['first_layer_update_norm']>0))
            improved=bool(np.all(trace['loss_before_after'][:,1]<trace['loss_before_after'][:,0]))
            target_valid=bool(np.array_equal(trace['backward_targets'][-1],evaluator.discounted_rewards(paths,instance)[:,-1]))
        rewards=evaluator.discounted_rewards(paths,instance)
        prediction=result.runner.continuation(1,paths[:,1,:],history=paths[:,:2,:],immediate_payoffs=rewards[:,1])
        np.save(destination/'verified_predictions.npy',prediction,allow_pickle=False)
        rows.append({'family':family,'gradient_updates_observed':updated,'training_loss_decreased':improved,
                     'terminal_backward_target_matches_mc_payoff':target_valid,
                     'finite_predictions':bool(np.isfinite(prediction).all()),
                     'train_seed':result.train_seed,'model_sha256':result.policy_artifact_sha256,
                     'input_sha256':result.input_bundle_sha256,'model_file_sha256':result.policy_file_sha256,
                     'fit_wall_seconds':result.wall_seconds,'prediction_wall_seconds':result.runner.prediction_wall_seconds,
                     'isolation':result.isolation,'research_fallback':result.research_fallback})
    (a.output/'training_validation.json').write_text(json.dumps({'schema':'openhyra-bermudan-training-validation.v1','rows':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
