import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bermudan_research import paired_slice_effects
from harness import Task, _mechanism_slots, _materialize_prediction_table
from context_agent import _prediction_table
from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping.training_pipeline import run_per_instance_training


def cell(values, path_hash="same"):
    return {"summaries": [{"instance_id": "public-put-atm", "repeat": 0,
        "paired_pathwise_improvements": values, "pricing_paths_sha256": path_hash,
        "slice_labels": ["payoff:put"]}]}


def test_pathwise_pair_uses_covariance_and_preregistered_slice():
    result = paired_slice_effects(cell([1., 4., 7.]), cell([0., 3., 6.]), "payoff:put")
    assert result["effect"] == 1.
    assert result["standard_error"] == 0.
    assert result["prediction_verdict"] == "supported"
    assert paired_slice_effects(cell([1., 2.]), cell([0., 0.], "other"))["status"] == "invalid_control"
    assert paired_slice_effects(cell([1., 2.]), cell([0., 0.]), "payoff:call")["status"] == "not_observed"


def test_python_slots_canonicalize_task_aliases_and_preserve_context_priority():
    task = Task("bermudan_python_search", "operator-regression")
    for iteration in range(2):
        mechanisms = [{"id": "rewrite", "mechanism": "replace fit", "intervention_scope": "fit", "intervention_operator": "replace"},
                      {"id": "fresh", "mechanism": "restart the program", "intervention_operator": "restart"}]
        slots = _mechanism_slots(task, {"mechanism_candidates": mechanisms}, iteration, {"id":"parent"}, 4)
        assert [s["mechanism"]["intervention_operator"] for s in slots] == ["subsystem_rewrite"]*2 + ["whole_program_restart"]*2
        assert slots[0]["matched_seed"] == slots[1]["matched_seed"]


def test_next_context_table_joins_target_effect_instead_of_global_baseline(tmp_path):
    research=tmp_path/"research";research.mkdir()
    ledger=research/"prediction_ledger.jsonl"
    ledger.write_text(json.dumps({"record_id":"guided", "iteration":0, "prediction_verdict":"supported", "evaluator":{"effect":4.}})+"\n")
    (research/"matched_controls.jsonl").write_text(json.dumps({"pair":{"guided_record_id":"guided"},
        "prediction_test":{"effect":-.2,"standard_error":.01,"prediction_verdict":"refuted","next_action":"restart"}})+"\n")
    _materialize_prediction_table(ledger)
    table,metadata=_prediction_table(SimpleNamespace(run_dir=tmp_path))
    row=json.loads(table)["rows"][0]
    assert metadata["consumed"]
    assert row["prediction_verdict"] == "refuted"
    assert row["next_action"] == "restart"
    assert row["matched_observation"]["effect"] == -.2


@pytest.mark.parametrize("family", ["mlp", "residual_hybrid"])
def test_real_candidate_gradient_fit_and_sandbox_predict(tmp_path, family):
    if sys.platform != "darwin":
        pytest.skip("native Seatbelt integration test; needs an explicit isolated Linux runner")
    instance=evaluator.public_suite()[0]
    paths=evaluator.simulate_paths(instance,256,71)
    source=Path(__file__).resolve().parents[1]/"tasks/bermudan_python_search/research_candidates"/family
    result=run_per_instance_training(instance=instance,training_paths=paths,candidate_source_dir=source,
                                    cell_dir=tmp_path/"cell",train_seed=71,runtime_roots=evaluator._training_runtime_roots())
    assert result.status == "ok", result.log_tail
    assert result.isolation == "seatbelt" and not result.research_fallback
    files=dict(result.runner.artifact.files)
    with np.load(io.BytesIO(files["training_trace.npz"]),allow_pickle=False) as trace:
        assert np.all(trace["first_layer_update_norm"]>0)
        assert np.all(trace["loss_before_after"][:,1] < trace["loss_before_after"][:,0])
        np.testing.assert_array_equal(trace["backward_targets"][-1], evaluator.discounted_rewards(paths,instance)[:,-1])
    rewards=evaluator.discounted_rewards(paths,instance)
    prediction=result.runner.continuation(1,paths[:,1,:],history=paths[:,:2,:],immediate_payoffs=rewards[:,1])
    assert prediction.shape == (256,) and np.isfinite(prediction).all()
    assert {name for name,_ in result.policy_file_sha256} == {"model.npz","training_trace.npz"}


def test_denied_seatbelt_launch_never_runs_candidate_unisolated(tmp_path, monkeypatch):
    import sandbox
    source, inputs, output, scratch, runtime = [tmp_path/name for name in ("source","input","output","scratch","runtime")]
    for p in (source, inputs, output, scratch, runtime): p.mkdir()
    marker=output/"candidate_ran"
    candidate=[sys.executable,"-c","from pathlib import Path; import sys; Path(sys.argv[1]).touch()",str(marker)]
    denied=[sys.executable,"-c","print('sandbox_apply: Operation not permitted'); raise SystemExit(71)"]
    monkeypatch.setattr(sandbox,"_training_sandboxed_cmd",lambda *a,**k:(denied,"seatbelt"))
    result=sandbox.run_training_sandbox(candidate,source_dir=source,input_dir=inputs,output_dir=output,tmp_dir=scratch,runtime_roots=[runtime])
    assert result["status"] == "crash"
    assert not result["research_fallback"]
    assert not marker.exists()
