from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness import Task, check_frozen
from sandbox import snapshot_algorithm_source
from tasks.bermudan_optimal_stopping import evaluator


def test_python_program_snapshot_and_digest_include_nested_helpers():
    task = Task("bermudan_python_search", "source-tree-focused")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "algorithm.py").write_text("from helpers.math import f\n")
        (source / "manifest.json").write_text(json.dumps({
            "schema": "openhyra-python-program.v1", "interface": "continuation",
        }))
        (source / "solve.sh").write_text("cp manifest.json solution.json\n")
        (source / "helpers").mkdir()
        (source / "helpers" / "math.py").write_text("def f(x): return x\n")
        sealed = snapshot_algorithm_source(source, root / "sealed", task, 1_000_000)
        assert (sealed / "helpers" / "math.py").is_file()
        assert not (sealed / "solve.sh").exists()
        validated, _manifest = evaluator._candidate_source_manifest(sealed)
        assert validated == sealed.resolve()


def test_python_program_source_tree_rejects_unsupported_files():
    task = Task("bermudan_python_search", "source-tree-invalid")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "algorithm.py").write_text("def fit(*a): pass\n")
        (source / "manifest.json").write_text(json.dumps({
            "schema": "openhyra-python-program.v1", "interface": "continuation",
        }))
        (source / "payload.bin").write_bytes(b"x")
        with pytest.raises(ValueError, match="unsupported extension"):
            evaluator._candidate_source_manifest(source)


def test_source_tree_freeze_allows_helpers_but_keeps_task_plumbing_frozen():
    task = Task("bermudan_python_search", "source-tree-freeze")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        parent = root / "parent"
        draft = root / "draft"
        parent.mkdir()
        draft.mkdir()
        for target in (parent, draft):
            (target / "algorithm.py").write_text("def fit(*a): pass\n")
            (target / "manifest.json").write_text("{}\n")
            (target / "solve.sh").write_text("frozen\n")
        (draft / "helpers").mkdir()
        (draft / "helpers" / "math.py").write_text("VALUE = 1\n")
        assert check_frozen(
            parent, draft, task.editable_files,
            allow_source_tree=task.candidate_source_tree,
        ) == []
        (draft / "solve.sh").write_text("changed\n")
        assert check_frozen(
            parent, draft, task.editable_files,
            allow_source_tree=task.candidate_source_tree,
        ) == ["solve.sh"]
