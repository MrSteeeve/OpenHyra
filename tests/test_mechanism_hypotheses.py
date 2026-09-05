import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_agent import _llm_context_analysis, build_inspiration
from eb import ExperienceBank
from feedback import BeliefReducer
from harness import Task, _mechanism_slots
from mechanism_hypotheses import (
    CONFIG_SCHEMA,
    HYPOTHESIS_SCHEMA,
    candidate_hypotheses,
    hypothesis_to_analogy,
    load_mechanism_design,
    matched_control_enabled,
    render_context_block,
    render_proposal_block,
)
from stopping import ContextDecision


ROOT = Path(__file__).resolve().parents[1]


def _candidate_payload():
    return {
        "id": "agent_boundary_1",
        "family": "optimization",
        "mechanism": "weight errors near the exercise boundary",
        "prediction": "improves the hidden boundary slice",
        "failure_condition": "no paired improvement after matched training",
        "matched_control": "same parent and update budget with uniform loss",
    }


def test_task_declares_open_mechanism_portfolio():
    task = SimpleNamespace(dir=ROOT / "tasks" / "bermudan_python_search")
    design = load_mechanism_design(task)
    assert design.active
    assert design.schema == CONFIG_SCHEMA
    assert len(design.directions) >= 4
    assert len({item.id for item in design.directions}) == len(design.directions)
    assert all(item.prediction and item.failure_condition for item in design.directions)
    assert matched_control_enabled(task)


def test_candidate_hypotheses_are_sloted_and_rotate_by_iteration():
    task = SimpleNamespace(
        dir=ROOT / "tasks" / "bermudan_python_search",
        candidates_per_context=3,
    )
    first = candidate_hypotheses(task, candidate_count=3, iteration=0)
    later = candidate_hypotheses(task, candidate_count=3, iteration=1)
    assert len(first) == len(later) == 3
    assert len({item["id"] for item in first}) == 3
    assert first[0]["id"] != later[0]["id"]
    assert all(
        item["prediction"] and item["failure_condition"] and item["matched_control"]
        for item in first
    )


def test_hypothesis_to_analogy_returns_valid_v5_projection():
    hypothesis = _candidate_payload()
    analogy = hypothesis_to_analogy(
        hypothesis,
        target_parent_id="parent_001",
        source_record_ids=["source_001"],
    )
    analogy.validate()
    assert analogy.id == "analogy_agent_boundary_1"
    assert analogy.target_parent_id == "parent_001"
    assert analogy.matched_control["mechanism_id"] == "agent_boundary_1"
    assert analogy.matched_control["family"] == "optimization"
    payload = hypothesis_to_analogy(
        hypothesis,
        target_parent_id="parent_001",
        as_dict=True,
    )
    assert payload["schema"] == "openhyra-analogy-hypothesis.v1"


def test_harness_slots_pair_guided_and_control_with_shared_seed():
    task = Task("bermudan_python_search", "mechanism-slot-test")
    slots = _mechanism_slots(
        task,
        {"trial_seed": 17, "mechanism_candidates": []},
        iteration=0,
        baseline={"id": "parent_001"},
        candidate_count=4,
    )
    assert [slot["matched_arm"] for slot in slots] == [
        "guided", "control", "guided", "control"
    ]
    assert slots[0]["matched_pair_id"] == slots[1]["matched_pair_id"]
    assert slots[2]["matched_pair_id"] == slots[3]["matched_pair_id"]
    assert slots[0]["matched_seed"] == slots[1]["matched_seed"]
    assert slots[2]["matched_seed"] == slots[3]["matched_seed"]
    assert slots[0]["hypothesis_id"]


def test_context_and_proposal_blocks_expose_structured_fields():
    task = SimpleNamespace(
        dir=ROOT / "tasks" / "bermudan_python_search",
        candidate_mode="algorithm_bundle",
        candidates_per_context=4,
    )
    context = render_context_block(task, candidate_count=4)
    proposal = render_proposal_block(
        task,
        candidate_count=4,
        context_hypotheses=[_candidate_payload()],
    )
    assert "Open algorithm-design portfolio" in context
    assert "mechanism_candidates" in context
    assert "whole_program_restart" in context
    assert "Open algorithm-design assignment" in proposal
    assert "matched_control" in proposal
    assert "agent_boundary_1" in proposal
    assert "PROPOSAL.md" in proposal


def test_context_decision_round_trips_mechanism_candidates():
    payload = {
        "action": "continue",
        "analysis": "Compare two distinct mechanisms.",
        "reason": "The boundary slice remains uncertain.",
        "expected_gain": 0.01,
        "confidence": 0.7,
        "phase": "numeric",
        "next": "run the selected mechanisms",
        "mechanism_candidates": [_candidate_payload(), _candidate_payload()],
    }
    decision = ContextDecision.from_payload(payload)
    assert len(decision.mechanism_candidates) == 1
    item = decision.mechanism_candidates[0]
    assert item["schema"] == HYPOTHESIS_SCHEMA
    assert item["failure_condition"]
    serialized = decision.to_dict()
    assert serialized["mechanism_candidates"][0]["id"] == "agent_boundary_1"
    restored = ContextDecision.from_payload(serialized)
    assert restored.mechanism_candidates == decision.mechanism_candidates


def test_context_can_omit_ascii_slug_for_a_new_mechanism():
    """A human-language idea should not disappear just because id is absent."""
    decision = ContextDecision.from_payload({
        "action": "continue",
        "analysis": "Try the new structure.",
        "reason": "The boundary is unresolved.",
        "expected_gain": 0.01,
        "confidence": 0.4,
        "phase": "numeric",
        "next": "run the new structure",
        "mechanism_candidates": [{
            "family": "表示学习",
            "mechanism": "按剩余期限重参数化 continuation surface",
            "prediction": "改善边界切片",
            "failure_condition": "held-out gain is absent",
            "matched_control": "same parent and compute budget",
        }],
    })
    assert len(decision.mechanism_candidates) == 1
    identifier = decision.mechanism_candidates[0]["id"]
    assert identifier[0].isalnum()
    assert all(char.isalnum() or char in "_.-" for char in identifier)
    assert len(identifier) <= 64


def test_build_inspiration_forwards_context_hypotheses_to_proposal_prompt(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('seed')\n")
    bank = ExperienceBank(tmp_path / "eb", direction="max")
    bank.commit(source, 0.1, "ok", "seed", None, "")
    task = SimpleNamespace(
        dir=ROOT / "tasks" / "bermudan_python_search",
        candidate_mode="algorithm_bundle",
        candidates_per_context=4,
        direction="max",
        metric="score",
        description="Search continuation policies.",
        editable_files=["train.py", "manifest.json"],
        candidate_source_files=["train.py", "manifest.json"],
        candidate_entrypoint="train.py",
        artifact_protocol="openhyra-policy-spec.v1",
        artifact_protocols=["openhyra-policy-spec.v1"],
        fallback_directions=["try a mechanism"],
        engineering_invariants=[],
        allowed_context_phases=["numeric"],
    )
    response = subprocess.CompletedProcess(
        args=["codex"],
        returncode=0,
        stdout=json.dumps({
            "action": "continue",
            "analysis": "Compare mechanisms.",
            "reason": "Several mechanisms remain plausible.",
            "expected_gain": 0.01,
            "confidence": 0.6,
            "phase": "numeric",
            "next": "try the portfolio",
            "state_version": 999,
            "state_hash": "model-supplied-wrong-hash",
            "mechanism_candidates": [_candidate_payload()],
        }),
        stderr="",
    )
    feedback_state = BeliefReducer().rebuild([], state_id="trusted-state")
    with patch("context_agent.run_agent", return_value=response):
        decision, _baseline, prompt, _direction, metadata = build_inspiration(
            task, bank, 0, backend="codex", feedback_state=feedback_state,
        )
    assert decision.mechanism_candidates
    assert decision.state_version == feedback_state.state_version
    assert decision.state_hash == feedback_state.state_hash
    assert metadata["state_hash"] == feedback_state.state_hash
    assert metadata["mechanism_candidates"][0]["id"] == "agent_boundary_1"
    assert "agent_boundary_1" in prompt
    assert "whole_program_restart" in prompt


def test_open_algorithm_task_gets_portfolio_output_slot_without_task_design(tmp_path):
    """An open task can ask Context for new families even without seed directions."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('seed')\n")
    bank = ExperienceBank(tmp_path / "eb", direction="max")
    record = bank.commit(source, 0.1, "ok", "seed", None, "")
    task = SimpleNamespace(
        dir=tmp_path / "task-without-design",
        direction="max",
        metric="score",
        description="Search an open algorithm bundle.",
        editable_files=["train.py", "manifest.json"],
        candidate_mode="algorithm_bundle",
        adaptive_feedback=False,
        candidates_per_context=3,
        fallback_directions=["probe a new family"],
        allowed_context_phases=["discover"],
    )
    response = subprocess.CompletedProcess(
        args=["codex"],
        returncode=0,
        stdout=json.dumps({
            "action": "continue",
            "analysis": "The open space needs several falsifiable families.",
            "reason": "No task seed portfolio was supplied.",
            "expected_gain": 0.01,
            "confidence": 0.4,
            "phase": "discover",
            "next": "propose a bounded family",
        }),
        stderr="",
    )
    with patch("context_agent.run_agent", return_value=response) as mocked:
        state = BeliefReducer().rebuild([], state_id="prompt-state")
        decision = _llm_context_analysis(
            task,
            bank,
            [record],
            record,
            "history",
            0,
            1,
            (),
            17,
            backend="codex",
            feedback_state=state,
        )
    prompt = mocked.call_args.args[0]
    assert '"mechanism_candidates"' in prompt
    assert state.state_hash in prompt
    assert decision is not None


def test_legacy_tasks_do_not_receive_mechanism_block():
    task = SimpleNamespace()
    assert render_context_block(task) == ""
    assert render_proposal_block(task) == ""


def test_context_only_hypothesis_reaches_proposal_without_seed_portfolio():
    task = SimpleNamespace()
    block = render_proposal_block(
        task,
        candidate_count=2,
        context_hypotheses=[_candidate_payload()],
    )
    assert "Open algorithm-design assignment" in block
    assert "agent_boundary_1" in block
