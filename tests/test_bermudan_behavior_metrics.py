"""Small behavior descriptors emitted by the Bermudan evaluator."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tasks.bermudan_optimal_stopping import evaluator


def _search_request(*, repeats: int = 2) -> dict:
    return {
        "schema": evaluator.REQUEST_SCHEMA,
        "stage": "search",
        "task": evaluator.TASK_NAME,
        "protocol": evaluator.TASK_PROTOCOL,
        "seed": 20260903,
        "suite_id": "behavior-metrics-smoke",
        "config": {
            "instance_count": 1,
            "repeats": repeats,
            "training_paths": 64,
            "pricing_paths": 128,
            "ridge_alpha": 1e-6,
        },
    }


def test_policy_behavior_descriptor_uses_early_exercise_rate() -> None:
    descriptor = evaluator._policy_behavior_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0, 1, 3, 3], dtype=np.int64),
        4,
    )

    # Two of four paths stop before maturity; terminal settlement is not an
    # exercise event for the scalar rate.
    assert descriptor["exercise_rate"] == 0.5
    assert descriptor["exercise_rate_by_time"] == [0.25, 0.25, 0.0, 0.5]
    assert math.isclose(descriptor["stop_time_mean"], 7.0 / 12.0)
    assert math.isclose(descriptor["stop_time_std"], 0.4330127019, rel_tol=1e-9)
    assert descriptor["finite"] is True


def test_search_emits_per_instance_behavior_and_repeat_stability() -> None:
    _score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
        evaluator.BASELINE_PROGRAM,
        _search_request(repeats=2),
    )

    instance_id = "public-put-atm"
    assert set(metrics["per_instance_scores"]) == {instance_id}
    assert set(metrics["baseline_scores"]) == {instance_id}
    assert set(metrics["per_instance_exercise_rates"]) == {instance_id}
    assert set(metrics["per_instance_exercise_rate_std"]) == {instance_id}
    assert metrics["per_instance_finite_rates"][instance_id] == 1.0
    assert metrics["behavior_finite"] is True
    assert 0.0 <= metrics["per_instance_exercise_rates"][instance_id] <= 1.0
    assert metrics["per_instance_exercise_rate_std"][instance_id] >= 0.0

    rows = metrics["summaries"]
    assert len(rows) == 2
    for row in rows:
        assert row["candidate_finite"] is True
        assert row["baseline_finite"] is True
        assert 0.0 <= row["candidate_exercise_rate"] <= 1.0
        assert len(row["candidate_exercise_rate_by_time"]) == 6
        assert math.isclose(
            sum(row["candidate_exercise_rate_by_time"]), 1.0,
            rel_tol=1e-12,
        )


def test_invalid_stopping_indices_raise_instead_of_placeholder_metrics() -> None:
    with pytest.raises(ValueError, match="stopping indices"):
        evaluator._policy_behavior_metrics(
            np.asarray([1.0, 2.0]),
            np.asarray([0.0, 2.0]),
            2,
        )


def test_search_failure_rate_is_not_reported_as_a_measurement() -> None:
    _score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
        evaluator.BASELINE_PROGRAM,
        _search_request(repeats=1),
    )
    assert metrics["failure_rate"] is None
    assert metrics["failure_rate_observed"] is False


def test_search_pricing_paths_use_a_private_salt() -> None:
    request = _search_request(repeats=1)
    request["config"]["public_pathwise_samples"] = True
    # Standalone call: a fresh salt is drawn per evaluation and recorded.
    _s1, first, _n1, _e1 = evaluator.evaluate_submission(evaluator.BASELINE_PROGRAM, request)
    _s2, second, _n2, _e2 = evaluator.evaluate_submission(evaluator.BASELINE_PROGRAM, request)
    assert first["pricing_salt"] != second["pricing_salt"]
    instance = evaluator.public_suite()[0]
    pricing = request["config"]["pricing_paths"]
    public_only = evaluator._derive_seed(request["seed"], request["suite_id"], 0, 0, "pricing")
    salted = evaluator._derive_seed(
        request["seed"], request["suite_id"], 0, 0, "pricing", first["pricing_salt"],
    )
    recorded = first["summaries"][0]["pricing_paths_sha256"]
    assert evaluator._sha256_array(evaluator.simulate_paths(instance, pricing, public_only)) != recorded
    assert evaluator._sha256_array(evaluator.simulate_paths(instance, pricing, salted)) == recorded


def test_run_salt_gives_common_pricing_paths_across_candidates() -> None:
    import math
    from bermudan_research import paired_slice_effects

    request = _search_request(repeats=2)
    request["config"].update(public_pathwise_samples=True, pricing_salt=987654321)
    other = {**evaluator.BASELINE_PROGRAM, "features": evaluator.BASELINE_PROGRAM["features"][:2]}
    _sg, guided, _ng, _eg = evaluator.evaluate_submission(evaluator.BASELINE_PROGRAM, request)
    _sc, control, _nc, _ec = evaluator.evaluate_submission(other, request)
    # A supplied (run-private) salt never reaches the emitted metrics.
    assert "pricing_salt" not in guided and "pricing_salt" not in control
    stats = paired_slice_effects(guided, control)
    assert stats["status"] == "observed"
    assert all(c["uncertainty_method"] == "common_random_numbers_pathwise" for c in stats["cells"])
    # Same independent recomputation as experiments/audit_bermudan_bundle.py.
    moments = []
    for a, b in zip(guided["summaries"], control["summaries"]):
        diffs = [x - y for x, y in zip(a["paired_pathwise_improvements"], b["paired_pathwise_improvements"])]
        mean = math.fsum(diffs) / len(diffs)
        moments.append((mean, math.fsum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)))
    effect = math.fsum(m[0] for m in moments) / len(moments)
    se = math.sqrt(math.fsum(m[1] for m in moments)) / len(moments)
    assert math.isclose(effect, stats["effect"], abs_tol=1e-13)
    assert math.isclose(se, stats["standard_error"], abs_tol=1e-13)


def test_audit_dual_and_pricing_streams_use_the_salt() -> None:
    request = {
        "schema": evaluator.REQUEST_SCHEMA, "stage": "audit", "task": evaluator.TASK_NAME,
        "protocol": evaluator.TASK_PROTOCOL, "seed": 9017, "suite_id": "salted-audit",
        "config": {"instance_count": 1, "repeats": 1, "training_paths": 128,
                   "pricing_paths": 128, "outer_paths": 64, "inner_paths": 2},
    }
    results = []
    for salt in (1, 1, 2):
        salted = {**request, "config": {**request["config"], "pricing_salt": salt}}
        results.append(evaluator.evaluate_submission(evaluator.BASELINE_PROGRAM, salted))
    assert results[0][0] == results[1][0] and results[0][0] != results[2][0]
    assert "pricing_salt" not in results[0][1]
    _score, drawn, _n, _e = evaluator.evaluate_submission(evaluator.BASELINE_PROGRAM, request)
    assert isinstance(drawn["pricing_salt"], int)


def test_harness_run_salt_is_private_stable_and_bermudan_only(tmp_path) -> None:
    import os
    from types import SimpleNamespace
    from provenance import with_run_pricing_salt

    task = SimpleNamespace(run_dir=tmp_path)
    request = {"protocol": evaluator.TASK_PROTOCOL, "config": {"repeats": 1}}
    first = with_run_pricing_salt(task, request)
    second = with_run_pricing_salt(task, request)
    assert first["config"]["pricing_salt"] == second["config"]["pricing_salt"]
    assert "pricing_salt" not in request["config"]  # manifest copy untouched
    assert os.stat(tmp_path / "private_pricing_salt").st_mode & 0o777 == 0o600
    other = {"protocol": "stochopia-structuring-search.v1", "config": {}}
    assert with_run_pricing_salt(task, other) is other


def test_algorithm_bundle_search_exposes_the_same_behavior_projection() -> None:
    """The open Python path gets descriptors from the trusted evaluator too."""
    manifest = {
        "schema": "continuation-expression.v1",
        "runner_type": "expression",
        "inference_config": {
            "input_dim": "n_assets",
            "output_dim": 1,
            "output_clip": [-1_000_000.0, 1_000_000.0],
        },
        "output_semantics": "discounted_continuation_value_t0",
        "normalization": "none",
        "weight_pattern": "step_{:03d}.json",
    }
    train_source = """\
import argparse, json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', required=True)
args = parser.parse_args()
steps = np.load(Path(args.input) / 'training_paths.npy', allow_pickle=False).shape[1] - 1
for index in range(steps):
    (Path(args.output) / f'step_{index:03d}.json').write_text(
        json.dumps({'op': 'constant', 'value': 0.0})
    )
"""
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "candidate"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(manifest))
        (source / "train.py").write_text(train_source)
        _score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
            manifest,
            _search_request(repeats=1),
            candidate_source_dir=source,
        )

    assert metrics["candidate_kind"] == "algorithm_bundle"
    assert metrics["runner_type"] == "expression"
    assert metrics["training_cell_count"] == 1
    assert metrics["per_instance_exercise_rates"]
    assert metrics["per_instance_finite_rates"] == {"public-put-atm": 1.0}


class _ConstantContinuation:
    policy_interface = "continuation"
    runner_type = "python_program"
    policy_artifact_sha256 = "same-model"

    def __init__(self, instance, bump_time=None):
        self.instance = instance
        self.bump_time = bump_time

    def continuation(self, time_index, states, *, history, immediate_payoffs):
        bump = 1.0 if time_index == self.bump_time else 0.0
        return np.full(np.asarray(states).shape[0], 0.1 + bump)


def test_replay_prediction_check_covers_every_decision_date(tmp_path, monkeypatch) -> None:
    instance = evaluator.public_suite()[0]
    # Same model digest; predictions differ only at t=2, which a t=0-only
    # comparison could not see.
    monkeypatch.setattr(
        evaluator, "_fit_candidate_policy",
        lambda *args, **kwargs: (_ConstantContinuation(instance, bump_time=2), {}),
    )
    result = evaluator._independent_validation(
        policy=_ConstantContinuation(instance), source_dir=tmp_path, manifest=None,
        instance=instance, training_paths=np.zeros((1, 1, 1)), train_seed=5,
        training_root=tmp_path, config={},
    )
    replay = result["deterministic_replay"]
    assert replay["model_equal"] is True
    assert replay["prediction_equal"] is False
    assert replay["status"] == "failed"
    # The renamed probe is a determinism check, not a look-ahead claim.
    assert "lookahead_probe" not in result
    assert result["repeat_query_probe"]["status"] == "passed"


def test_external_isolation_is_opt_in(tmp_path, monkeypatch) -> None:
    from tasks.bermudan_optimal_stopping import training_pipeline

    seen = []

    def fake_training(**kwargs):
        seen.append(kwargs["externally_isolated"])
        return type("Result", (), {"status": "crash", "runner": None, "log_tail": "stub"})()

    monkeypatch.setattr(training_pipeline, "run_per_instance_training", fake_training)
    monkeypatch.setattr(evaluator, "_training_result_metrics", lambda result: {})
    instance = evaluator.public_suite()[0]
    for env_value, expected in ((None, False), ("1", True)):
        if env_value is None:
            monkeypatch.delenv("OPENHYRA_ALLOW_UNSANDBOXED", raising=False)
        else:
            monkeypatch.setenv("OPENHYRA_ALLOW_UNSANDBOXED", env_value)
        with pytest.raises(ValueError, match="candidate training failed"):
            evaluator._fit_algorithm_policy(
                tmp_path, None, instance, np.zeros((4, 7, 1)), train_seed=1,
                cell_dir=tmp_path / "cell", config={},
            )
        assert seen[-1] is expected
