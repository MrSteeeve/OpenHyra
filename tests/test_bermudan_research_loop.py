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


def test_matched_control_reports_realized_calls_without_filtering_on_them(tmp_path):
    from harness import _finalize_matched_controls

    def entry(arm, attempts):
        records = [{"id": f"{arm}-{index}", "status": "ok", "score": 0.1, "metrics": {}}
                   for index in range(attempts)]
        item = {"matched_pair_id": "pair-1", "matched_arm": arm, "matched_seed": 3,
                "parent": {"id": "parent"}, "mechanism": {"id": "m1", "mechanism": "replace fit"}}
        return {"item": item, "records": records}

    eb = SimpleNamespace(root=tmp_path / "eb", records=lambda: [{"id": "parent", "score": 0.0}])
    task = SimpleNamespace(run_dir=tmp_path, metric="paired_lower_bound_lcb",
                           direction="max", candidate_repair_attempts=1)
    # The guided arm needed its one allotted repair, the control none.  The
    # realized counts are reported but must not drop the pair (that would
    # select pairs on outcome).
    [payload] = _finalize_matched_controls(task, 0, [entry("guided", 2), entry("control", 1)], eb)
    control = payload["hypothesis"]["matched_control"]
    assert (control["guided_llm_calls"], control["control_llm_calls"]) == (2, 1)
    assert control["allotted_llm_calls"] == 2
    assert control["same_compute_budget"] is True
    assert payload["result"]["verdict"] != "invalid_control"



def test_control_packet_does_not_enter_focal_mechanism_belief():
    from feedback import BeliefReducer
    from harness import _annotate_feedback_lineage

    def packet(effect, candidate):
        rows = [{"instance_id": "public-put-atm", "repeat": r, "paired_normalized_improvement": effect,
                 "paired_normalized_standard_error": 1e-4, "slice_labels": []} for r in range(2)]
        return evaluator._build_domain_feedback_packet(
            stage="search", task="bermudan_python_search", suite_id="s",
            request={"seed": 1, "candidate": candidate}, score=0.0, summaries=rows,
            confidence_level=0.95).to_dict()

    mechanism = {"id": "mech_X"}
    guided = _annotate_feedback_lineage({"feedback_packet": packet(0.01, "g")},
                                        {"mechanism": mechanism, "matched_arm": "guided"})
    control = _annotate_feedback_lineage({"feedback_packet": packet(-0.02, "c")},
                                         {"mechanism": mechanism, "matched_arm": "control"})
    assert "feedback_lineage" not in control
    state = BeliefReducer().rebuild([guided["feedback_packet"], control["feedback_packet"]])
    focal = [cell for key, cell in state.cells.items() if key.startswith("mech_X::")]
    assert [cell.n for cell in focal] == [2]
    assert focal[0].mean == pytest.approx(0.01)

def test_python_seed_never_exercises_zero_payoff_paths_in_fit(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seed_algorithm",
        Path(__file__).resolve().parents[1] / "tasks/bermudan_python_search/seed_solution/algorithm.py",
    )
    seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed)
    n = 64
    x = np.linspace(-1.0, 1.0, n)
    paths = np.exp(np.repeat(x[:, None, None], 3, axis=1))
    payoffs = np.zeros((n, 3))
    payoffs[:, 0] = 0.05
    # Tail-heavy targets make the quadratic continuation negative mid-range;
    # with all date-1 payoffs zero, nothing may be exercised there.
    payoffs[:, 2] = np.where(np.abs(x) > 0.8, 1.0, 0.0)
    payoffs[n // 2, 2] = 0.01

    def coefficients(p, q, name):
        (tmp_path / name).mkdir()
        np.save(tmp_path / name / "training_paths.npy", p)
        np.save(tmp_path / name / "payoffs.npy", q)
        seed.fit(tmp_path / name, tmp_path / f"{name}-model", 0)
        return np.load(tmp_path / f"{name}-model" / "model.npz")["coefficients"]

    full = coefficients(paths, payoffs, "full")
    skipped = coefficients(paths[:, [0, 2]], payoffs[:, [0, 2]], "skipped")
    np.testing.assert_allclose(full[0], skipped[0])


def test_bundle_replay_passes_the_recorded_pricing_salt(tmp_path, monkeypatch):
    import importlib.util
    bundle = tmp_path / "bundle"
    runner_dir = bundle / "reproduction" / "experiments"
    runner_dir.mkdir(parents=True)
    # Stand-in for the frozen workshop runner: replays only match with the salt.
    (runner_dir / "run_bermudan_live_workshop.py").write_text(
        "import json\n"
        "def save(path, value):\n"
        "    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))\n"
        "def evaluate(source, req, path):\n"
        "    salted = req['config'].get('pricing_salt') == 123\n"
        "    return {'status': 'ok', 'score': 1.0 if salted else 2.0, 'source_digest': 'd',\n"
        "            'source_unchanged': True, 'wall_seconds': 0.1,\n"
        "            'metrics': {'training_cells': [{'model_file_sha256': 'm'}], 'summaries': []}}\n"
    )
    old = {"status": "ok", "score": 1.0, "source_digest": "d", "request": {"config": {}},
           "metrics": {"pricing_salt": 123, "training_cells": [{"model_file_sha256": "m"}], "summaries": []}}
    row = {"mode": "m", "seed": 1, "candidate_id": "c", "program_path": "p", "result": old}
    (bundle / "candidate_ledger.jsonl").write_text(json.dumps(row) + "\n")
    (bundle / "summary.json").write_text(json.dumps({"private_audits": []}))
    spec = importlib.util.spec_from_file_location(
        "replay_bundle", Path(__file__).resolve().parents[1] / "experiments/replay_bermudan_bundle.py",
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    monkeypatch.setattr(sys, "argv", ["replay", "--bundle", str(bundle), "--output", str(tmp_path / "out")])
    replay.main()
    report = json.loads((tmp_path / "out" / "replay_verification.json").read_text())
    assert report["all_equal"] is True
