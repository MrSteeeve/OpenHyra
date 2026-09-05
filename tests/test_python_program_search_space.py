"""Functional tests for the concrete whole-program search space."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from algorithm_discovery import (
    AlgorithmDiscoveryLoop,
    AlgorithmSpec,
    EvaluationResult,
    SearchSpace,
    make_python_program_search_space,
)
from program_search import ProgramGenerationRequest, PythonProgramSearchSpace


def _run(source: str, value: float) -> object:
    namespace: dict[str, object] = {}
    exec(compile(source, "algorithm.py", "exec"), namespace)
    solve = namespace["solve"]
    assert callable(solve)
    return solve(value)


def _result(candidate: AlgorithmSpec, score: float) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        status="ok",
        score=score,
        split="development",
    )


def test_empty_population_calls_whole_program_generator_and_materializes_all_files(tmp_path: Path):
    requests: list[ProgramGenerationRequest] = []

    def generate(request: ProgramGenerationRequest):
        requests.append(request)
        return {
            "source": {
                "algorithm.py": "def solve(x):\n    return x * x\n",
                "notes/design.txt": "square the input with a direct program",
            },
            "mechanism_id": "direct_square",
            "prediction": {"effect": "positive"},
            "falsifier": {"condition": "score does not improve"},
            "metadata": {"idea": "whole-program generation"},
        }

    space = PythonProgramSearchSpace(generator=generate)
    assert isinstance(space, SearchSpace)
    candidate = space.propose({"task": "discover a transformation"}, slot=0)

    assert candidate.operator == "llm_generate"
    assert candidate.parent_ids == ()
    assert candidate.family == "python_program"
    assert candidate.mechanism_id == "direct_square"
    assert candidate.metadata["idea"] == "whole-program generation"
    assert set(candidate.implementation["source"]) == {
        "algorithm.py",
        "notes/design.txt",
    }
    assert requests[0].parents == ()
    assert requests[0].context["task"] == "discover a transformation"
    assert _run(candidate.implementation["source"]["algorithm.py"], 3) == 9

    written = space.materialize(candidate, tmp_path / "candidate")
    assert {path.relative_to(tmp_path / "candidate").as_posix() for path in written} == {
        "algorithm.py",
        "notes/design.txt",
    }
    assert (tmp_path / "candidate" / "notes" / "design.txt").read_text() == (
        "square the input with a direct program"
    )


def test_public_discovery_module_constructs_the_concrete_search_space():
    space = make_python_program_search_space(
        seeds=[{"algorithm.py": "def solve(x):\n    return x\n"}],
    )
    assert isinstance(space, PythonProgramSearchSpace)
    assert isinstance(space, SearchSpace)


def test_ast_mutation_changes_the_executable_program_structure_and_behavior():
    space = PythonProgramSearchSpace(
        seeds=[{"algorithm.py": "def solve(x):\n    return x + 3\n"}],
    )
    parent = space.candidates[0]
    child = space.propose({"operator": "ast_mutation"}, slot=0)

    parent_source = parent.implementation["source"]["algorithm.py"]
    child_source = child.implementation["source"]["algorithm.py"]
    assert child.operator == "ast_mutation"
    assert child.parent_ids == (parent.candidate_id,)
    assert child.metadata["mutation"] == "binary_operator"
    assert child_source != parent_source
    assert _run(parent_source, 5) == 8
    assert _run(child_source, 5) == 2


def test_ast_mutation_can_expand_a_branch_and_change_its_behavior():
    source = """
def solve(x):
    if x > 0:
        return x
    return 0
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    parent = space.candidates[0]

    # With no arithmetic site available, structural branch expansion is the
    # first mechanical operator in the normal slot rotation.
    child = space.mutate(parent, slot=0)
    child_source = child.implementation["source"]["algorithm.py"]

    assert child.metadata["mutation"] == "branch_expansion"
    assert "else:" in child_source
    assert _run(source, -3) == 0
    assert _run(child_source, -3) == -3


def test_ast_mutation_can_repeat_a_loop_update_and_change_dynamics():
    source = """
def solve(n):
    total = 0
    for index in range(n):
        total = total + index
    return total
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    parent = space.candidates[0]

    # Slot zero mutates the arithmetic expression; slot one moves to the next
    # available strategy and inserts a second update inside the loop.
    child = space.mutate(parent, slot=1)
    child_source = child.implementation["source"]["algorithm.py"]

    assert child.metadata["mutation"] == "loop_update_repeat"
    assert child_source.count("total = total + index") == 2
    assert _run(source, 4) == 6
    assert _run(child_source, 4) == 12


def test_ast_mutation_can_insert_a_second_state_update():
    source = """
def solve(x):
    total = 1
    total = total + x
    return total
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    parent = space.candidates[0]

    child = space.mutate(
        parent,
        context={"mutation_kind": "state_update_repeat"},
    )
    child_source = child.implementation["source"]["algorithm.py"]

    assert child.metadata["mutation"] == "state_update_repeat"
    assert child_source.count("total = total + x") == 2
    assert _run(source, 3) == 4
    assert _run(child_source, 3) == 7


def test_ast_subtree_swap_preserves_typed_expression_bindings():
    source = """
def solve(x):
    left = x + 1
    right = x + 2
    return left + right
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    child = space.mutate(
        space.candidates[0], context={"mutation_kind": "subtree_swap"}, slot=0
    )
    child_source = child.implementation["source"]["algorithm.py"]
    assert child.metadata["mutation"] == "subtree_swap"
    compile(child_source, "algorithm.py", "exec")
    assert _run(child_source, 3) == _run(source, 3)


def test_ast_function_extract_adds_callable_helper_and_digest():
    source = """
def solve(x):
    return (x + 2) * 3
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    child = space.mutate(
        space.candidates[0], context={"mutation_kind": "function_extract"}, slot=0
    )
    child_source = child.implementation["source"]["algorithm.py"]
    assert child.metadata["mutation"] == "function_extract"
    assert "_openhyra_extracted_0" in child_source
    assert _run(child_source, 4) == _run(source, 4)
    assert len(space.equivalent_candidates(child)) == 1
    assert len(child.metadata["source_digest"]) == 64


def test_ast_control_flow_guard_is_compile_safe_and_slot_bounded():
    source = """
def solve(x):
    total = x + 1
    return total
"""
    space = PythonProgramSearchSpace(seeds=[{"algorithm.py": source}])
    child = space.mutate(
        space.candidates[0], context={"mutation_kind": "control_flow_guard"}, slot=1
    )
    child_source = child.implementation["source"]["algorithm.py"]
    assert child.metadata["mutation"] == "control_flow_guard"
    compile(child_source, "algorithm.py", "exec")


def test_ast_mutation_preserves_fit_predict_cli_dispatch(tmp_path: Path):
    source = """
import argparse
from pathlib import Path

def fit(input_dir, output_dir, seed):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "model.txt").write_text(str(seed + 1))

def predict(model_dir, input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "predictions.txt").write_text("ready")

def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.input, args.output, args.seed)

if __name__ == "__main__":
    main()
"""
    space = PythonProgramSearchSpace(
        seeds=[{"algorithm.py": source}],
        required_symbol="predict",
    )
    child = space.mutate(space.candidates[0], slot=0)
    algorithm = tmp_path / "algorithm.py"
    algorithm.write_text(
        child.implementation["source"]["algorithm.py"], encoding="utf-8"
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(algorithm),
            "fit",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "model.txt").is_file()


def test_two_parent_ast_crossover_keeps_both_function_graphs_and_composes_them():
    left_source = """
BASE = 2

def left_feature(x):
    return x + BASE

def solve(x):
    return left_feature(x)
"""
    right_source = """
SCALE = 4

def right_feature(x):
    return x * SCALE

def solve(x):
    return right_feature(x)
"""
    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": left_source},
            {"algorithm.py": right_source},
        ]
    )
    left, right = space.candidates
    space.observe(_result(left, 0.4))
    space.observe(_result(right, 0.9))

    child = space.propose({"operator": "ast_crossover"}, slot=0)
    child_source = child.implementation["source"]["algorithm.py"]

    assert child.operator == "ast_crossover"
    assert child.parent_ids == (right.candidate_id, left.candidate_id)
    assert "_build_parent_a" in child_source
    assert "_build_parent_b" in child_source
    assert "def right_feature" in child_source
    assert "def left_feature" in child_source
    assert _run(right_source, 3) == 12
    assert _run(left_source, 3) == 5
    assert _run(child_source, 3) == 8.5


def test_fit_predict_crossover_produces_a_runnable_two_model_program(tmp_path: Path):
    def parent_source(value: int) -> str:
        return f"""
from pathlib import Path
import numpy as np

def fit(input_dir, output_dir, seed):
    np.save(Path(output_dir) / "value.npy", np.asarray([{value}], dtype=np.float64))

def predict(model_dir, input_dir, output_dir):
    states = np.load(Path(input_dir) / "states.npy", allow_pickle=False)
    stored = np.load(Path(model_dir) / "value.npy", allow_pickle=False)[0]
    np.save(
        Path(output_dir) / "predictions.npy",
        np.full(states.shape[:-1], stored, dtype=np.float64),
        allow_pickle=False,
    )

def main():
    return None
"""

    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": parent_source(2)},
            {"algorithm.py": parent_source(4)},
        ],
        required_symbol="predict",
    )
    child = space.crossover(space.candidates[0], space.candidates[1])
    source = child.implementation["source"]["algorithm.py"]
    namespace: dict[str, object] = {}
    exec(compile(source, "algorithm.py", "exec"), namespace)

    training = tmp_path / "training"
    model = tmp_path / "model"
    query = tmp_path / "query"
    output = tmp_path / "output"
    for directory in (training, model, query, output):
        directory.mkdir()
    np.save(query / "states.npy", np.ones((5, 1)), allow_pickle=False)
    namespace["fit"](training, model, 17)
    namespace["predict"](model, query, output)
    predictions = np.load(output / "predictions.npy", allow_pickle=False)

    assert child.metadata["crossover"] == "fit_predict_program_composition"
    assert '"_openhyra_parent_a"' in source
    assert '"_openhyra_parent_b"' in source
    assert np.array_equal(predictions, np.full(5, 3.0))


def test_fit_predict_crossover_preserves_direct_decision_interface(tmp_path: Path):
    def parent_source(decision: int) -> str:
        return f"""
from pathlib import Path
import numpy as np

def fit(input_dir, output_dir, seed):
    np.save(Path(output_dir) / "decision.npy", np.asarray([{decision}], dtype=np.int64))

def predict(model_dir, input_dir, output_dir):
    states = np.load(Path(input_dir) / "states.npy", allow_pickle=False)
    stored = np.load(Path(model_dir) / "decision.npy", allow_pickle=False)[0]
    np.save(
        Path(output_dir) / "predictions.npy",
        np.full(states.shape[:-1], stored, dtype=np.int64),
        allow_pickle=False,
    )
"""

    manifest = '{"schema":"openhyra-python-program.v1","interface":"decision"}'
    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": parent_source(0), "manifest.json": manifest},
            {"algorithm.py": parent_source(1), "manifest.json": manifest},
        ],
        required_symbol="predict",
    )
    child = space.crossover(space.candidates[0], space.candidates[1])
    namespace: dict[str, object] = {}
    exec(
        compile(child.implementation["source"]["algorithm.py"], "algorithm.py", "exec"),
        namespace,
    )

    training = tmp_path / "training"
    model = tmp_path / "model"
    query = tmp_path / "query"
    output = tmp_path / "output"
    for directory in (training, model, query, output):
        directory.mkdir()
    np.save(query / "states.npy", np.ones((4, 1)), allow_pickle=False)
    namespace["fit"](training, model, 17)
    namespace["predict"](model, query, output)
    predictions = np.load(output / "predictions.npy", allow_pickle=False)

    assert child.metadata["prediction_interface"] == "decision"
    assert predictions.dtype == np.dtype(np.bool_)
    assert np.array_equal(predictions, np.ones(4, dtype=np.bool_))


def test_fit_predict_crossover_preserves_module_level_pickle_models(tmp_path: Path):
    def parent_source(value: int) -> str:
        return f"""
from pathlib import Path
import argparse
import pickle
import numpy as np

class Model:
    def __init__(self, value):
        self.value = value

def fit(input_dir, output_dir, seed):
    with (Path(output_dir) / "model.pkl").open("wb") as stream:
        pickle.dump(Model({value}), stream)

def predict(model_dir, input_dir, output_dir):
    with (Path(model_dir) / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    states = np.load(Path(input_dir) / "states.npy", allow_pickle=False)
    np.save(
        Path(output_dir) / "predictions.npy",
        np.full(states.shape[:-1], model.value, dtype=np.float64),
        allow_pickle=False,
    )
"""

    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": parent_source(2)},
            {"algorithm.py": parent_source(6)},
            {"algorithm.py": parent_source(10)},
        ],
        required_symbol="predict",
    )
    pair = space.crossover(space.candidates[0], space.candidates[1])
    child = space.crossover(pair, space.candidates[2])
    algorithm = tmp_path / "algorithm.py"
    algorithm.write_text(
        child.implementation["source"]["algorithm.py"], encoding="utf-8"
    )
    training = tmp_path / "training"
    model = tmp_path / "model"
    query = tmp_path / "query"
    output = tmp_path / "output"
    for directory in (training, model, query, output):
        directory.mkdir()
    np.save(query / "states.npy", np.ones((3, 1)), allow_pickle=False)

    subprocess.run(
        [
            sys.executable, str(algorithm), "fit", "--input", str(training),
            "--output", str(model), "--seed", "7",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(algorithm), "predict", "--model", str(model),
            "--input", str(query), "--output", str(output),
        ],
        check=True,
    )

    predictions = np.load(output / "predictions.npy", allow_pickle=False)
    assert np.array_equal(predictions, np.full(3, 7.0))


def test_fit_predict_crossover_rejects_async_cli_functions():
    source = """
async def fit(input_dir, output_dir, seed):
    return None

async def predict(model_dir, input_dir, output_dir):
    return None
"""
    space = PythonProgramSearchSpace(
        seeds=[{"algorithm.py": source}, {"algorithm.py": source}],
        required_symbol="predict",
    )

    with pytest.raises(ValueError, match="requires synchronous CLI functions"):
        space.crossover(space.candidates[0], space.candidates[1])


def test_fit_predict_crossover_allows_linear_growth_but_refuses_balanced_embedding():
    source = """
def fit(input_dir, output_dir, seed):
    return None

def predict(model_dir, input_dir, output_dir):
    return None
"""
    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": source},
            {"algorithm.py": source + "\nVALUE = 2\n"},
            {"algorithm.py": source + "\nVALUE = 3\n"},
            {"algorithm.py": source + "\nVALUE = 4\n"},
        ],
        required_symbol="predict",
    )
    first = space.crossover(space.candidates[0], space.candidates[1])
    linear = space.crossover(first, space.candidates[2])
    second = space.crossover(space.candidates[2], space.candidates[3])

    assert linear.parent_ids == (
        first.candidate_id,
        space.candidates[2].candidate_id,
    )
    assert len(linear.implementation["source"]["algorithm.py"]) < (
        2 * len(first.implementation["source"]["algorithm.py"])
    )
    with pytest.raises(ValueError, match="will not embed two existing"):
        space.crossover(first, second)


def test_crossover_rejects_conflicting_same_path_helpers():
    base = "def solve(x):\n    return x\n"
    space = PythonProgramSearchSpace(
        seeds=[
            {"algorithm.py": base, "helper.py": "VALUE = 1\n"},
            {"algorithm.py": base, "helper.py": "VALUE = 2\n"},
        ],
    )

    with pytest.raises(ValueError, match="conflicting source file: helper.py"):
        space.crossover(space.candidates[0], space.candidates[1])


def test_crossover_accepts_semantically_equal_manifest_formatting():
    source = "def solve(x):\n    return x\n"
    space = PythonProgramSearchSpace(
        seeds=[
            {
                "algorithm.py": source,
                "manifest.json": '{"schema":"p","interface":"decision"}',
            },
            {
                "algorithm.py": source,
                "manifest.json": '{\n  "interface": "decision", "schema": "p"\n}',
            },
        ],
    )

    child = space.crossover(space.candidates[0], space.candidates[1])

    assert json.loads(child.implementation["source"]["manifest.json"]) == {
        "schema": "p",
        "interface": "decision",
    }


def test_observed_scores_choose_the_best_parent_for_mutation_and_llm_rewrite():
    requests: list[ProgramGenerationRequest] = []

    def rewrite(request: ProgramGenerationRequest):
        requests.append(request)
        return {"algorithm.py": "def solve(x):\n    return x ** 2 + 1\n"}

    space = PythonProgramSearchSpace(
        generator=rewrite,
        seeds=[
            {"algorithm.py": "def solve(x):\n    return x + 1\n"},
            {"algorithm.py": "def solve(x):\n    return x + 2\n"},
            {"algorithm.py": "def solve(x):\n    return x + 3\n"},
        ],
    )
    low, winner, middle = space.candidates
    space.observe(_result(low, 0.1))
    space.observe(_result(winner, 0.9))
    space.observe(_result(middle, 0.5))

    mutation = space.propose({"operator": "ast_mutation"}, slot=0)
    rewrite_child = space.propose({"operator": "llm_rewrite"}, slot=1)

    assert mutation.parent_ids == (winner.candidate_id,)
    assert rewrite_child.parent_ids == (winner.candidate_id,)
    assert requests[0].parents[0].candidate_id == winner.candidate_id
    assert {item.candidate_id for item in requests[0].observations} == {
        low.candidate_id,
        winner.candidate_id,
        middle.candidate_id,
    }
    assert rewrite_child.implementation["source"]["algorithm.py"].startswith("def solve")


def test_validation_checks_program_contract_without_restricting_algorithm_family():
    space = PythonProgramSearchSpace()
    missing_entrypoint = AlgorithmSpec(
        candidate_id="missing",
        family="python_program",
        operator="llm_generate",
        implementation={
            "entrypoint": "algorithm.py",
            "required_symbol": "solve",
            "source": {"helper.py": "def helper(x):\n    return x\n"},
        },
    )
    missing_symbol = AlgorithmSpec(
        candidate_id="missing-symbol",
        family="python_program",
        operator="llm_generate",
        implementation={
            "entrypoint": "algorithm.py",
            "required_symbol": "solve",
            "source": {"algorithm.py": "class EntirelyNewAlgorithm:\n    pass\n"},
        },
    )

    with pytest.raises(ValueError, match="missing entrypoint"):
        space.validate(missing_entrypoint)
    with pytest.raises(ValueError, match=r"define top-level solve\(\)"):
        space.validate(missing_symbol)


def test_recursive_loop_feeds_the_scored_winner_into_the_next_generation():
    generated = [
        "def solve(x):\n    return x + 1\n",
        "def solve(x):\n    return x + 5\n",
    ]
    requests: list[ProgramGenerationRequest] = []

    def generator(request: ProgramGenerationRequest):
        requests.append(request)
        if request.operator == "llm_generate":
            return {"algorithm.py": generated[len(requests) - 1]}
        return {"algorithm.py": "def solve(x):\n    return x + 6\n"}

    space = PythonProgramSearchSpace(generator=generator)
    loop = AlgorithmDiscoveryLoop()

    def context(round_index, _state):
        return {"operator": "llm_generate" if round_index == 0 else "llm_rewrite"}

    def evaluate(candidate: AlgorithmSpec):
        score = float(_run(candidate.implementation["source"]["algorithm.py"], 0))
        return _result(candidate, score)

    events = loop.run_search(
        space,
        evaluate,
        rounds=2,
        candidates_per_round=2,
        context=context,
    )

    first_round = events[:2]
    winner = max(first_round, key=lambda event: event.result.score or float("-inf"))
    second_round = events[2:]
    assert len(second_round) == 2
    assert all(event.candidate.parent_ids == (winner.candidate.candidate_id,) for event in second_round)
    assert requests[2].parents[0].candidate_id == winner.candidate.candidate_id


def test_recursive_loop_freezes_parents_until_the_round_is_scored():
    requests: list[ProgramGenerationRequest] = []

    def generator(request: ProgramGenerationRequest):
        requests.append(request)
        return {
            "algorithm.py": (
                "def solve(x):\n"
                f"    return x + {request.slot + 1}\n"
            )
        }

    space = PythonProgramSearchSpace(generator=generator)
    loop = AlgorithmDiscoveryLoop()
    loop.run_search(
        space,
        lambda candidate: _result(candidate, float(candidate.metadata["slot"])),
        rounds=1,
        candidates_per_round=3,
    )

    assert [request.operator for request in requests] == [
        "llm_generate", "llm_generate", "llm_generate",
    ]
    assert all(request.parents == () for request in requests)


def test_operator_rotation_uses_only_scored_round_parents():
    def generator(request: ProgramGenerationRequest):
        return {
            "algorithm.py": (
                "def solve(x):\n"
                f"    return x + {request.slot + 1}\n"
            )
        }

    space = PythonProgramSearchSpace(generator=generator)
    loop = AlgorithmDiscoveryLoop()
    events = loop.run_search(
        space,
        lambda candidate: _result(candidate, float(candidate.metadata["slot"])),
        rounds=2,
        candidates_per_round=3,
        select=1,
    )

    assert len(events) == 2
    assert events[1].candidate.parent_ids == (events[0].candidate.candidate_id,)


def test_failed_observation_is_not_selected_over_a_valid_seed():
    space = PythonProgramSearchSpace(
        seeds=[{"algorithm.py": "def solve(x):\n    return x + 1\n"}],
    )
    seed = space.candidates[0]
    failed = space.mutate(seed, slot=0)
    space.observe(
        EvaluationResult(
            candidate_id=failed.candidate_id,
            status="crash",
            score=None,
        )
    )

    child = space.propose({"operator": "ast_mutation"}, slot=0)

    assert child.parent_ids == (seed.candidate_id,)
