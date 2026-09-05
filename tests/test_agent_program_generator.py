"""Functional coverage for the concrete LLM whole-program adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from algorithm_discovery import (
    AlgorithmDiscoveryLoop,
    EvaluationResult,
    make_agent_python_program_search_space,
)


def _evaluate(candidate):
    namespace = {}
    source = candidate.implementation["source"]["algorithm.py"]
    exec(compile(source, "algorithm.py", "exec"), namespace)
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        status="ok",
        score=float(namespace["solve"](0)),
    )


def test_agent_generator_drives_recursive_whole_program_search(tmp_path: Path):
    calls = []

    def edit_program(prompt, *, cwd, **_kwargs):
        calls.append(prompt)
        value = 2 if len(calls) == 1 else 7
        (Path(cwd) / "algorithm.py").write_text(
            f"def solve(x):\n    return x + {value}\n",
            encoding="utf-8",
        )
        if len(calls) == 2:
            helper = Path(cwd) / "helpers" / "idea.py"
            helper.parent.mkdir()
            helper.write_text("MECHANISM = 'stateful rewrite'\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=["codex"], returncode=0, stdout="", stderr="",
        )

    with patch("agent_program_generator.run_agent", side_effect=edit_program):
        space = make_agent_python_program_search_space(
            tmp_path / "agent-workspaces",
            required_symbol="solve",
        )
        events = AlgorithmDiscoveryLoop().run_search(
            space,
            _evaluate,
            rounds=2,
            candidates_per_round=1,
            context=lambda round_index, _state: {
                "operator": "llm_generate" if round_index == 0 else "llm_rewrite",
                "research_question": "invent a better update",
            },
        )

    assert [event.result.score for event in events] == [2.0, 7.0]
    assert events[1].candidate.parent_ids == (
        events[0].candidate.candidate_id,
    )
    assert "helpers/idea.py" in events[1].candidate.implementation["source"]
    assert "Parent 1 file: algorithm.py" in calls[1]
