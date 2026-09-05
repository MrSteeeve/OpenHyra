"""Prompt and task-contract gates for the open Python program search track.

These tests deliberately check the candidate-facing research surface rather
than any particular seed algorithm.  The old continuation runners remain
useful baselines, but they must not define the Proposal Agent's search space.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from context_agent import _candidate_contract_block
from harness import Task, _candidate_preflight_issues, _secondary_program_parent
from proposal_agent import _protocol_prompt_block, propose


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "bermudan_python_search"
PROGRAM_SCHEMA = "openhyra-python-program.v1"
OLD_CLOSED_PROTOCOLS = (
    "openhyra-policy-spec.v1",
    "continuation-linear.v1",
    "continuation-expression.v1",
)
OLD_FAMILY_MENU = ("MLP", "Linear", "Expression")


def _program_task() -> SimpleNamespace:
    return SimpleNamespace(
        candidate_mode="python_program",
        candidate_source_files=("algorithm.py", "manifest.json"),
        candidate_entrypoint="algorithm.py",
        artifact_protocol=PROGRAM_SCHEMA,
        artifact_protocols=(PROGRAM_SCHEMA,),
    )


def _assert_open_program_contract(prompt: str) -> None:
    """One prompt must describe behavior, not prescribe an algorithm family."""
    lowered = prompt.lower()
    assert "python" in lowered
    assert "program" in lowered
    assert "fit" in lowered
    assert "predict" in lowered
    assert "continuation" in lowered
    assert "decision" in lowered
    assert PROGRAM_SCHEMA in prompt

    for protocol in OLD_CLOSED_PROTOCOLS:
        assert protocol not in prompt
    for family in OLD_FAMILY_MENU:
        assert family not in prompt


def test_context_prompt_exposes_the_whole_program_lifecycle() -> None:
    prompt = _candidate_contract_block(_program_task())
    _assert_open_program_contract(prompt)
    assert "algorithm.py" in prompt
    assert "manifest.json" in prompt


def test_proposal_prompt_exposes_the_same_open_program_contract() -> None:
    prompt = _protocol_prompt_block(
        "python_program",
        entrypoint="algorithm.py",
        artifact_protocol=PROGRAM_SCHEMA,
        source_files=("algorithm.py", "manifest.json"),
    )
    _assert_open_program_contract(prompt)
    # A direct stopping policy is a first-class output interface; the prompt
    # must not force every discovered algorithm back through continuation
    # regression merely because the historical baseline used it.
    assert "direct" in prompt.lower()


def test_bermudan_python_task_selects_the_open_program_schema() -> None:
    config = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))

    assert config["candidate_mode"] == "python_program"
    assert config["artifact_protocol"] == PROGRAM_SCHEMA
    assert config["artifact_protocols"] == [PROGRAM_SCHEMA]
    assert config["entrypoint"] == "algorithm.py"
    assert "algorithm.py" in config["source_files"]
    assert "manifest.json" in config["source_files"]
    subsystem = next(
        item
        for item in config["mechanism_design"]["directions"]
        if item["id"] == "subsystem_rewrite"
    )
    assert subsystem["intervention_scope"] == "fit"
    assert subsystem["intervention_operator"] == "replace"

    rendered = json.dumps(config, ensure_ascii=False)
    for protocol in OLD_CLOSED_PROTOCOLS:
        assert protocol not in rendered
    for family in OLD_FAMILY_MENU:
        assert family not in rendered

    instructions = "\n".join(config.get("candidate_instructions", ())).lower()
    assert "fit" in instructions
    assert "predict" in instructions
    assert "decision" in instructions

    # Loading through the real harness proves the JSON contract is not merely
    # documentary: the candidate mode and schema reach the prompt plumbing.
    task = Task("bermudan_python_search", "program-contract-test")
    assert task.candidate_mode == "python_program"
    assert task.artifact_protocol == PROGRAM_SCHEMA
    assert task.candidate_entrypoint == "algorithm.py"


def test_open_program_space_is_not_rejected_by_legacy_solver_lints(
    tmp_path: Path,
) -> None:
    (tmp_path / "algorithm.py").write_text(
        "import random\n\n"
        "def fit(progress, start, stop):\n"
        "    shaped = (1.0 - progress) ** 0.5\n"
        "    return shaped, random.randrange(start, stop)\n\n"
        "def predict(*args):\n    return None\n",
        encoding="utf-8",
    )
    task = SimpleNamespace(
        candidate_mode="python_program",
        editable_files=("algorithm.py",),
    )

    assert _candidate_preflight_issues(task, tmp_path) == []


def test_live_proposal_path_applies_a_requested_ast_mutation(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    draft = tmp_path / "draft"
    parent.mkdir()
    parent_source = (
        "def fit(*args):\n"
        "    return None\n\n"
        "def predict(x):\n"
        "    return x + 1\n\n"
        "def main():\n"
        "    return None\n"
    )
    (parent / "algorithm.py").write_text(parent_source, encoding="utf-8")
    (parent / "manifest.json").write_text(
        json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"}),
        encoding="utf-8",
    )
    response = subprocess.CompletedProcess(
        args=["codex"], returncode=0, stdout="", stderr="",
    )

    with patch("proposal_agent.run_agent", return_value=response):
        ok, description = propose(
            parent,
            draft,
            "mutate the executable structure",
            ["algorithm.py", "manifest.json"],
            backend="codex",
            candidate_mode="python_program",
            entrypoint="algorithm.py",
            artifact_protocol=PROGRAM_SCHEMA,
            source_files=("algorithm.py", "manifest.json"),
            intervention={
                "intervention_operator": "ast_mutation",
                "matched_arm": "guided",
                "slot": 0,
            },
        )

    assert ok, description
    mutated_source = (draft / "algorithm.py").read_text(encoding="utf-8")
    assert mutated_source != parent_source
    before: dict[str, object] = {}
    after: dict[str, object] = {}
    exec(parent_source, before)
    exec(mutated_source, after)
    assert before["predict"](5) == 6
    assert after["predict"](5) == 4
    assert description.startswith("binary_operator:")


def test_live_proposal_path_materializes_two_parent_program_crossover(
    tmp_path: Path,
) -> None:
    def make_parent(path: Path, value: int) -> None:
        path.mkdir()
        (path / "algorithm.py").write_text(
            f"""
from pathlib import Path
import numpy as np

def fit(input_dir, output_dir, seed):
    np.save(Path(output_dir) / "value.npy", np.asarray([{value}], dtype=np.float64))

def predict(model_dir, input_dir, output_dir):
    states = np.load(Path(input_dir) / "states.npy", allow_pickle=False)
    stored = np.load(Path(model_dir) / "value.npy", allow_pickle=False)[0]
    np.save(Path(output_dir) / "predictions.npy", np.full(states.shape[:-1], stored))

def main():
    return None
""",
            encoding="utf-8",
        )
        (path / "manifest.json").write_text(
            json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"}),
            encoding="utf-8",
        )

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    draft = tmp_path / "crossover"
    make_parent(primary, 2)
    make_parent(secondary, 6)
    response = subprocess.CompletedProcess(
        args=["codex"], returncode=0, stdout="", stderr="",
    )
    with patch("proposal_agent.run_agent", return_value=response):
        ok, description = propose(
            primary,
            draft,
            "compose both live programs",
            ["algorithm.py", "manifest.json"],
            backend="codex",
            candidate_mode="python_program",
            entrypoint="algorithm.py",
            artifact_protocol=PROGRAM_SCHEMA,
            source_files=("algorithm.py", "manifest.json"),
            intervention={
                "intervention_operator": "ast_crossover",
                "matched_arm": "guided",
                "secondary_parent_path": str(secondary),
            },
        )

    assert ok, description
    namespace: dict[str, object] = {}
    exec((draft / "algorithm.py").read_text(encoding="utf-8"), namespace)
    training = tmp_path / "training"
    model = tmp_path / "model"
    query = tmp_path / "query"
    output = tmp_path / "output"
    for directory in (training, model, query, output):
        directory.mkdir()
    np.save(query / "states.npy", np.ones((3, 1)), allow_pickle=False)
    namespace["fit"](training, model, 11)
    namespace["predict"](model, query, output)
    result = np.load(output / "predictions.npy", allow_pickle=False)

    assert np.array_equal(result, np.full(3, 4.0))
    assert description.startswith("fit_predict_program_composition:")


def test_crossover_rejects_a_rewrite_that_removes_the_combination_hook(
    tmp_path: Path,
) -> None:
    def make_parent(path: Path, value: int) -> None:
        path.mkdir()
        (path / "algorithm.py").write_text(
            "from pathlib import Path\nimport numpy as np\n\n"
            "def fit(input_dir, output_dir, seed):\n"
            f"    np.save(Path(output_dir) / 'value.npy', np.asarray([{value}]))\n\n"
            "def predict(model_dir, input_dir, output_dir):\n"
            "    states = np.load(Path(input_dir) / 'states.npy')\n"
            "    np.save(Path(output_dir) / 'predictions.npy', states[:, 0])\n",
            encoding="utf-8",
        )
        (path / "manifest.json").write_text(
            json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"})
        )

    primary = tmp_path / "lineage-primary"
    secondary = tmp_path / "lineage-secondary"
    make_parent(primary, 1)
    make_parent(secondary, 2)

    def erase_crossover(_prompt, *, cwd, **_kwargs):
        (Path(cwd) / "algorithm.py").write_text(
            "def fit(*args):\n    return None\n\n"
            "def predict(*args):\n    return None\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=["codex"], returncode=0, stdout="", stderr="",
        )

    with patch("proposal_agent.run_agent", side_effect=erase_crossover):
        ok, description = propose(
            primary,
            tmp_path / "lineage-draft",
            "compose both programs",
            ["algorithm.py", "manifest.json"],
            candidate_mode="python_program",
            entrypoint="algorithm.py",
            source_files=("algorithm.py", "manifest.json"),
            intervention={
                "intervention_operator": "ast_crossover",
                "matched_arm": "guided",
                "secondary_parent_path": str(secondary),
            },
        )

    assert not ok
    assert "invalid crossover combination hook" in description


def test_crossover_rebuilds_dispatcher_and_keeps_the_evolved_combiner(
    tmp_path: Path,
) -> None:
    def make_parent(path: Path, value: int) -> None:
        path.mkdir()
        (path / "algorithm.py").write_text(
            "from pathlib import Path\nimport numpy as np\n\n"
            "def fit(input_dir, output_dir, seed):\n"
            f"    np.save(Path(output_dir) / 'value.npy', np.asarray([{value}]))\n\n"
            "def predict(model_dir, input_dir, output_dir):\n"
            "    states = np.load(Path(input_dir) / 'states.npy')\n"
            "    value = np.load(Path(model_dir) / 'value.npy')[0]\n"
            "    np.save(Path(output_dir) / 'predictions.npy', "
            "np.full(states.shape[:-1], value))\n",
            encoding="utf-8",
        )
        (path / "manifest.json").write_text(
            json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"})
        )

    primary = tmp_path / "rebuild-primary"
    secondary = tmp_path / "rebuild-secondary"
    draft = tmp_path / "rebuild-draft"
    make_parent(primary, 2)
    make_parent(secondary, 6)

    def edit_combiner_and_tamper_dispatcher(_prompt, *, cwd, **_kwargs):
        program = Path(cwd) / "algorithm.py"
        tree = ast.parse(program.read_text(encoding="utf-8"))
        combiner = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_combine_predictions"
        )
        combiner.body = [ast.Return(value=ast.Name(id="left", ctx=ast.Load()))]
        ast.fix_missing_locations(tree)
        rewritten = ast.unparse(tree) + "\n"
        rewritten = rewritten.replace(
            "def _load_parent_module(name, source):\n",
            "def _load_parent_module(name, source):\n    source = _PARENT_A_SOURCE\n",
        )
        program.write_text(rewritten, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=["codex"], returncode=0, stdout="", stderr="",
        )

    with patch(
        "proposal_agent.run_agent", side_effect=edit_combiner_and_tamper_dispatcher
    ):
        ok, description = propose(
            primary,
            draft,
            "compose both programs",
            ["algorithm.py", "manifest.json"],
            candidate_mode="python_program",
            entrypoint="algorithm.py",
            source_files=("algorithm.py", "manifest.json"),
            intervention={
                "intervention_operator": "ast_crossover",
                "matched_arm": "guided",
                "secondary_parent_path": str(secondary),
            },
        )
    assert ok, description
    rebuilt = (draft / "algorithm.py").read_text(encoding="utf-8")
    assert "source = _PARENT_A_SOURCE" not in rebuilt
    assert "def _combine_predictions(left, right):\n    return left" in rebuilt
    assert "_parent_a_predict(" in rebuilt
    assert "_parent_b_predict(" in rebuilt

    namespace: dict[str, object] = {}
    exec(rebuilt, namespace)
    training = tmp_path / "rebuild-training"
    model = tmp_path / "rebuild-model"
    query = tmp_path / "rebuild-query"
    output = tmp_path / "rebuild-output"
    for directory in (training, model, query, output):
        directory.mkdir()
    np.save(query / "states.npy", np.ones((3, 1)), allow_pickle=False)
    namespace["fit"](training, model, 13)
    namespace["predict"](model, query, output)
    result = np.load(output / "predictions.npy", allow_pickle=False)
    assert np.array_equal(result, np.full(3, 2.0))


def test_requested_crossover_without_a_second_parent_fails_explicitly(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "algorithm.py").write_text(
        "def fit(*args):\n    return None\n\n"
        "def predict(*args):\n    return None\n",
        encoding="utf-8",
    )
    (parent / "manifest.json").write_text(
        json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"}),
        encoding="utf-8",
    )

    ok, description = propose(
        parent,
        tmp_path / "draft",
        "compose two programs",
        ["algorithm.py", "manifest.json"],
        candidate_mode="python_program",
        entrypoint="algorithm.py",
        source_files=("algorithm.py", "manifest.json"),
        intervention={"intervention_operator": "ast_crossover"},
    )

    assert not ok
    assert "requires a second program parent" in description


def test_whole_program_restart_removes_the_inherited_implementation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "restart-parent"
    parent.mkdir()
    (parent / "algorithm.py").write_text(
        "PARENT_ONLY_IDEA = 'legacy'\n\n"
        "def fit(*args):\n    return None\n\n"
        "def predict(*args):\n    return None\n",
        encoding="utf-8",
    )
    (parent / "manifest.json").write_text(
        json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"}),
        encoding="utf-8",
    )
    response = subprocess.CompletedProcess(
        args=["codex"], returncode=0, stdout="", stderr="",
    )

    with patch("proposal_agent.run_agent", return_value=response):
        ok, description = propose(
            parent,
            tmp_path / "restart-draft",
            "restart from a different principle",
            ["algorithm.py", "manifest.json"],
            candidate_mode="python_program",
            entrypoint="algorithm.py",
            source_files=("algorithm.py", "manifest.json"),
            intervention={
                "intervention_operator": "restart",
                "matched_arm": "guided",
            },
        )

    restarted = (tmp_path / "restart-draft" / "algorithm.py").read_text()
    assert ok, description
    assert "PARENT_ONLY_IDEA" not in restarted
    assert "Blank whole-program restart scaffold" in restarted
    assert description.startswith("whole_program_restart:")


def test_subsystem_rewrite_replaces_only_the_selected_fit_or_predict_body(
    tmp_path: Path,
) -> None:
    parent_source = """
PARENT_CONSTANT = "keep-parent-module"

def fit(input_dir, output_dir, seed):
    return "parent-fit", seed

def predict(model_dir, input_dir, output_dir):
    return "parent-predict"

def main():
    return "parent-cli"
"""
    original_manifest = json.dumps(
        {"schema": PROGRAM_SCHEMA, "interface": "continuation"},
        sort_keys=True,
    )

    cases = (
        (
            "fit",
            {"intervention_scope": "fit", "intervention_operator": "replace"},
        ),
        (
            "predict",
            {
                "intervention_scope": "subsystem",
                "intervention_operator": "replace",
                "target": "predict",
            },
        ),
    )
    for target, intervention in cases:
        parent = tmp_path / f"{target}-parent"
        draft = tmp_path / f"{target}-draft"
        parent.mkdir()
        (parent / "algorithm.py").write_text(parent_source, encoding="utf-8")
        (parent / "manifest.json").write_text(
            original_manifest, encoding="utf-8"
        )
        prompts = []

        def rewrite_selected_subsystem(prompt, *, cwd, **_kwargs):
            prompts.append(prompt)
            program = Path(cwd) / "algorithm.py"
            cleared = ast.parse(program.read_text(encoding="utf-8"))
            selected = next(
                node
                for node in cleared.body
                if isinstance(node, ast.FunctionDef) and node.name == target
            )
            assert isinstance(selected.body[0], ast.Raise)

            if target == "fit":
                replacement = """
PARENT_CONSTANT = "tampered-module"

def fit(any_signature):
    import math
    return "rewritten-fit", int(math.sqrt(49))

def predict(*args):
    return "tampered-predict"

def main():
    return "tampered-cli"
"""
            else:
                replacement = """
PARENT_CONSTANT = "tampered-module"

def fit(*args):
    return "tampered-fit"

def predict(any_signature):
    return "rewritten-predict"

def main():
    return "tampered-cli"
"""
            program.write_text(replacement, encoding="utf-8")
            (Path(cwd) / "manifest.json").write_text(
                json.dumps({"schema": "tampered", "interface": "decision"}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["codex"], returncode=0, stdout="", stderr="",
            )

        with patch(
            "proposal_agent.run_agent", side_effect=rewrite_selected_subsystem
        ):
            ok, description = propose(
                parent,
                draft,
                f"rewrite only {target}",
                ["algorithm.py", "manifest.json"],
                candidate_mode="python_program",
                entrypoint="algorithm.py",
                source_files=("algorithm.py", "manifest.json"),
                intervention={**intervention, "matched_arm": "guided"},
            )

        assert ok, description
        assert f"subsystem_rewrite:{target}:" in description
        assert f"Rewrite only the body of the top-level `{target}`" in prompts[0]
        assert (draft / "manifest.json").read_text() == original_manifest
        namespace: dict[str, object] = {}
        exec((draft / "algorithm.py").read_text(encoding="utf-8"), namespace)
        assert namespace["PARENT_CONSTANT"] == "keep-parent-module"
        assert namespace["main"]() == "parent-cli"
        if target == "fit":
            assert namespace["fit"](None, None, 99) == ("rewritten-fit", 7)
            assert namespace["predict"](None, None, None) == "parent-predict"
        else:
            assert namespace["fit"](None, None, 99) == ("parent-fit", 99)
            assert namespace["predict"](None, None, None) == "rewritten-predict"


def test_secondary_parent_prefers_behavioral_complement_and_custom_entrypoint(
    tmp_path: Path,
) -> None:
    def record(identifier, score, rate, candidate_hash, *, duplicate_of=""):
        path = tmp_path / identifier
        path.mkdir()
        (path / "solver.py").write_text("def fit(): pass\ndef predict(): pass\n")
        (path / "manifest.json").write_text(
            json.dumps({"schema": PROGRAM_SCHEMA, "interface": "continuation"})
        )
        return {
            "id": identifier,
            "path": str(path),
            "status": "ok",
            "score": score,
            "metrics": {
                "candidate_hash": candidate_hash,
                "per_instance_exercise_rates": {"case": rate},
            },
            "metadata": {"duplicate_of": duplicate_of},
        }

    primary = record("primary", 1.0, 0.2, "primary-hash")
    duplicate = record(
        "duplicate", 3.0, 0.9, "duplicate-hash", duplicate_of="primary"
    )
    similar = record("similar", 2.0, 0.25, "similar-hash")
    complement = record("complement", 1.5, 0.8, "complement-hash")
    bank = SimpleNamespace(records=lambda: [primary, duplicate, similar, complement])

    selected = _secondary_program_parent(
        bank,
        primary,
        "max",
        entrypoint="solver.py",
    )

    assert selected["id"] == "complement"
