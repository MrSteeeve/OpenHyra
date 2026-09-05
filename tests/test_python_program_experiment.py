from __future__ import annotations

import json
from pathlib import Path

from algorithm_discovery import AlgorithmSpec, EvaluationResult
from program_search import PythonProgramSearchSpace
from python_program_experiment import (
    EXPERIMENT_SCHEMA,
    PythonProgramExperimentConfig,
    PythonProgramExperimentRunner,
    ast_lineage_sha256,
    source_tree_sha256,
)


def _config(tmp_path: Path) -> PythonProgramExperimentConfig:
    return PythonProgramExperimentConfig(
        root=tmp_path,
        output_dir=tmp_path / "run",
        trial_seed=20260905,
        search_request={"seed": 17, "suite_id": "cheap-test"},
        rounds=1,
        candidates_per_round=2,
    )


def test_source_and_ast_digests_are_order_independent() -> None:
    source = {"algorithm.py": "def solve(x):\n    return x + 1\n", "notes.txt": "idea"}
    reordered = {"notes.txt": "idea", "algorithm.py": source["algorithm.py"]}
    assert source_tree_sha256(source) == source_tree_sha256(reordered)
    assert ast_lineage_sha256(source) == ast_lineage_sha256(reordered)


def test_runner_persists_frozen_manifest_source_lineage_and_cost(tmp_path: Path) -> None:
    source = {
        "algorithm.py": "def solve(x):\n    return x + 1\n",
        "manifest.json": json.dumps({"schema": "openhyra-python-program.v1", "interface": "continuation"}),
    }
    space = PythonProgramSearchSpace(seeds=[source])

    def evaluate(candidate: AlgorithmSpec) -> EvaluationResult:
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            status="ok",
            score=0.25,
            split="development",
            seed=17,
            cost={"wall_seconds": 0.001, "budget_id": "matched-0"},
        )

    records = PythonProgramExperimentRunner(_config(tmp_path), space, evaluate=evaluate).run()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    record = json.loads((tmp_path / "run" / "records.jsonl").read_text().splitlines()[0])
    assert manifest["schema"] == EXPERIMENT_SCHEMA
    assert manifest["task"] == "bermudan_python_search"
    assert manifest["artifact_protocol"] == "openhyra-python-program.v1"
    assert manifest["matched_controls"]["same_seed"] is True
    assert records[0]["source_sha256"] == source_tree_sha256(records[0]["source"])
    assert records[0]["ast_sha256"] == ast_lineage_sha256(records[0]["source"])
    assert records[0]["cost"]["budget_id"] == "matched-0"
    assert records[0]["failure"] is None
    assert (tmp_path / "run" / "discovery_events.jsonl").is_file()


def test_runner_records_evaluator_failures_without_crashing(tmp_path: Path) -> None:
    source = {
        "algorithm.py": "def solve(x):\n    return x\n",
        "manifest.json": json.dumps({"schema": "openhyra-python-program.v1", "interface": "continuation"}),
    }
    space = PythonProgramSearchSpace(seeds=[source])

    def evaluate(candidate: AlgorithmSpec) -> EvaluationResult:
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            status="error",
            score=None,
            split="development",
            seed=17,
            metrics={"failure": "timeout"},
            cost={"wall_seconds": 1.5},
        )

    records = PythonProgramExperimentRunner(_config(tmp_path), space, evaluate=evaluate).run()
    assert records[0]["failure"] == {"status": "error", "reason": "timeout"}


def test_runner_converts_evaluator_exception_to_failure_record(tmp_path: Path) -> None:
    source = {
        "algorithm.py": "def solve(x):\n    return x\n",
        "manifest.json": json.dumps({"schema": "openhyra-python-program.v1", "interface": "continuation"}),
    }
    space = PythonProgramSearchSpace(seeds=[source])

    def evaluate(_candidate: AlgorithmSpec) -> EvaluationResult:
        raise TimeoutError("sandbox timeout")

    records = PythonProgramExperimentRunner(_config(tmp_path), space, evaluate=evaluate).run()
    assert records[0]["failure"]["status"] == "error"
    assert "sandbox timeout" in records[0]["failure"]["reason"]
