"""Behavior-level tests for executable Bermudan policy programs."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest import mock

from tasks.bermudan_python_search import evaluator


PROGRAM = r"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
commands = parser.add_subparsers(dest="command", required=True)
fit_parser = commands.add_parser("fit")
fit_parser.add_argument("--input", required=True)
fit_parser.add_argument("--output", required=True)
fit_parser.add_argument("--seed", required=True)
predict_parser = commands.add_parser("predict")
predict_parser.add_argument("--model", required=True)
predict_parser.add_argument("--input", required=True)
predict_parser.add_argument("--output", required=True)
args = parser.parse_args()

if args.command == "fit":
    np.save(Path(args.output) / "state.npy", np.asarray([DECISION], dtype=np.bool_))
else:
    states = np.load(Path(args.input) / "states.npy", allow_pickle=False)
    decision = bool(np.load(Path(args.model) / "state.npy", allow_pickle=False)[0])
    np.save(
        Path(args.output) / "predictions.npy",
        np.full(states.shape[:-1], decision, dtype=np.bool_),
        allow_pickle=False,
    )
"""


def _candidate(root: Path, *, decision: bool) -> tuple[Path, dict]:
    source = root / ("exercise" if decision else "wait")
    source.mkdir()
    manifest = {
        "schema": "openhyra-python-program.v1",
        "interface": "decision",
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "algorithm.py").write_text(
        PROGRAM.replace("DECISION", "True" if decision else "False"),
        encoding="utf-8",
    )
    return source, manifest


def _request(stage: str) -> dict:
    config = {
        "instance_count": 1,
        "repeats": 1,
        "training_paths": 64,
        "pricing_paths": 64,
        "training_timeout_s": 10,
        "prediction_timeout_s": 10,
    }
    if stage == "audit":
        config.update({"outer_paths": 64, "inner_paths": 2})
    return {
        "schema": evaluator.REQUEST_SCHEMA,
        "stage": stage,
        "task": "bermudan_python_search",
        "protocol": "bermudan-python-program-search.v1",
        "seed": 20260904,
        "suite_id": f"python-program-{stage}",
        "config": config,
    }


def test_direct_decision_program_controls_stopping_without_registered_runner() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        exercise_source, manifest = _candidate(root, decision=True)
        wait_source, _ = _candidate(root, decision=False)

        with mock.patch(
            "tasks.bermudan_optimal_stopping.training_pipeline.load_continuation_runner",
            side_effect=AssertionError("legacy runner path must not execute"),
        ):
            exercise_score, exercise_metrics, normalized, _ = evaluator.evaluate_submission(
                manifest,
                _request("search"),
                candidate_source_dir=exercise_source,
            )
            wait_score, wait_metrics, _, _ = evaluator.evaluate_submission(
                manifest,
                _request("search"),
                candidate_source_dir=wait_source,
            )

        assert normalized == manifest
        assert exercise_metrics["candidate_kind"] == "python_program"
        assert exercise_metrics["policy_interface"] == "decision"
        exercise_stop = exercise_metrics["summaries"][0]["candidate_stop_time_mean"]
        wait_stop = wait_metrics["summaries"][0]["candidate_stop_time_mean"]
        assert exercise_stop < wait_stop
        assert exercise_score != wait_score


def test_direct_decision_hidden_audit_uses_an_independent_dual_verifier() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source, manifest = _candidate(Path(temporary), decision=False)
        score, metrics, _normalized, evidence = evaluator.evaluate_submission(
            manifest,
            _request("audit"),
            candidate_source_dir=source,
        )

    assert math.isfinite(score)
    assert metrics["policy_interface"] == "decision"
    assert metrics["dual_verifier"] == "evaluator_owned_ridge_lsmc"
    assert evidence["audit"]["dual_verifier"] == "evaluator_owned_ridge_lsmc"
