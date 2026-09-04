"""Focused checks for the open mechanism portfolio path."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_agent import ContextDecision
from eb import ExperienceBank
from feedback import DirectionalFeedback, FeedbackPacket
from harness import _build_experiment_plan, _mechanism_slots, run_pipeline
from harness_v5 import V5Bridge
from proposal_agent import prepare_draft, propose


def _design():
    return {
        "schema": "openhyra-mechanism-design.v1",
        "enabled": True,
        "matched_control": {"enabled": True},
        "directions": [
            {
                "id": "novel_boundary",
                "family": "representation",
                "mechanism": "condition the continuation representation on boundary distance",
                "prediction": "improves the boundary slice",
                "failure_condition": "no paired gain on held-out boundary cells",
                "matched_control": "same parent and budget without boundary conditioning",
            },
        ],
    }


def _task(root):
    return SimpleNamespace(
        run_dir=root / "run",
        eval_concurrency=1,
        candidates_per_context=2,
        candidate_repair_attempts=0,
        research_revision_attempts=0,
        editable_files=["solver.py"],
        direction="max",
        protocol="test-v1",
        run_id="open-loop",
        timeout_s=10,
        max_artifact_bytes=100_000,
        max_output_mb=1,
        description="A small open algorithm-design loop.",
        metric="score",
        fallback_directions=["try a mechanism"],
        engineering_invariants=[],
        allowed_context_phases=None,
        candidate_instructions="",
        mechanism_design=_design(),
        matched_control_enabled=True,
    )


def test_mechanism_slots_pair_seed_and_structure():
    task = _task(Path("/tmp"))
    decision_meta = {
        "trial_seed": 11,
        "mechanism_candidates": [
            {
                "id": "agent_invented",
                "family": "estimation",
                "mechanism": "cross-fit the backward targets",
                "prediction": "less target reuse",
                "failure_condition": "no held-out gain",
                "matched_control": "ordinary targets",
            }
        ],
    }
    slots = _mechanism_slots(task, decision_meta, 0, {"id": "parent"}, 2)
    assert [slot["matched_arm"] for slot in slots] == ["guided", "control"]
    assert slots[0]["hypothesis_id"] == slots[1]["hypothesis_id"]
    assert slots[0]["matched_seed"] == slots[1]["matched_seed"]
    assert slots[0]["mechanism_id"] == "agent_invented"


def test_odd_paired_budget_keeps_tail_unpaired():
    task = _task(Path("/tmp"))
    slots = _mechanism_slots(
        task,
        {
            "trial_seed": 11,
            "mechanism_candidates": [
                {
                    "id": "agent_invented",
                    "family": "estimation",
                    "mechanism": "cross-fit the backward targets",
                }
            ],
        },
        0,
        {"id": "parent"},
        3,
    )
    assert slots[-1]["matched_arm"] == "guided"
    assert slots[-1]["matched_pair_id"] == ""
    assert slots[-1]["matched_control_enabled"] is False


def test_mechanism_plan_names_hypothesis_and_operator():
    task = _task(Path("/tmp"))
    decision = SimpleNamespace(action="continue", success_criterion="score improves")
    plan = _build_experiment_plan(
        iteration=2,
        island_epoch_id="island_00_epoch_00",
        direction="try boundary representation",
        decision=decision,
        baseline={"id": "parent"},
        task=task,
        candidates_per_context=2,
        mechanism={
            "id": "novel_boundary",
            "family": "representation",
            "mechanism": "compose a boundary representation",
            "hypothesis_id": "analogy_0002_novel_boundary_x",
        },
        mechanism_hypotheses=[{"source_record_id": "parent"}],
    )
    assert plan.generation_operator == "composition"
    assert plan.analogy_hypothesis_id == "analogy_0002_novel_boundary_x"
    plan.validate()


def test_algorithm_bundle_control_can_be_unchanged_parent(tmp_path):
    """A no-op control must reach the evaluator as the counterfactual arm."""
    parent = tmp_path / "parent"
    draft = tmp_path / "draft"
    parent.mkdir()
    (parent / "solver.py").write_text("print('seed')\n")
    response = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout="wrote control note", stderr=""
    )
    with patch("proposal_agent.run_agent", return_value=response):
        ok, description = propose(
            parent,
            draft,
            "keep this matched control unchanged",
            ["solver.py"],
            backend="claude",
            candidate_mode="algorithm_bundle",
            allow_no_change=True,
        )
    assert ok is True
    assert "matched control" in description
    assert (draft / "solver.py").read_text() == (parent / "solver.py").read_text()
    assert (draft / "PROPOSAL.md").is_file()


def test_pipeline_records_paired_open_mechanism_result(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solver.py").write_text("print('seed')\n")
    eb = ExperienceBank(tmp_path / "eb", direction="max")
    seed = eb.commit(source, 1.0, "ok", "seed", None, "")
    task = _task(tmp_path)
    task.candidates_per_context = 4
    bridge = V5Bridge(task.run_dir, num_islands=2)
    bridge.record_seed(seed["id"], seed["score"], seed["metrics"])
    bridge.initialize([seed["id"]])
    decision = ContextDecision(
        action="continue",
        analysis="Compare an invented structure against its control.",
        reason="The boundary remains uncertain.",
        expected_gain=0.1,
        confidence=0.5,
        next_experiment="try the portfolio",
        mechanism_candidates=(
            {
                "id": "agent_invented",
                "family": "estimation",
                "mechanism": "cross-fit the backward targets",
                "prediction": "less target reuse",
                "failure_condition": "no held-out gain",
                "matched_control": "ordinary targets",
            },
            {
                "id": "boundary_composition",
                "family": "representation",
                "mechanism": "compose a boundary-distance representation",
                "prediction": "improves near-boundary exercise decisions",
                "failure_condition": "no held-out boundary gain",
                "matched_control": "same parent and budget without composition",
            },
        ),
    )

    def fake_context(*_args, **_kwargs):
        return (
            decision,
            seed,
            "Implement __OPENHYRA_CANDIDATE_SEED__",
            "try the portfolio",
            {
                "iteration": 0,
                "eb_version": 1,
                "visible_solution_ids": [seed["id"]],
                "trial_seed": 7,
                "direction": "try the portfolio",
                "context_decision": decision.to_dict(),
                "mechanism_candidates": list(decision.mechanism_candidates),
            },
        )

    prompts = []

    def fake_propose(parent, draft, prompt, editable_files, **_kwargs):
        prompts.append(prompt)
        prepare_draft(parent, draft)
        (draft / editable_files[0]).write_text("# candidate\nprint('ok')\n")
        return True, "proposal"

    def fake_evaluator(candidate, sandbox, _task):
        candidate_text = (Path(candidate) / "solver.py").read_text()
        score = 1.1 if "candidate" in candidate_text else 1.0
        Path(sandbox).mkdir(parents=True, exist_ok=True)
        return score, "ok", "evaluator", {"set_hash": str(score)}

    with (
        patch("harness.build_inspiration", side_effect=fake_context),
        patch("harness.propose", side_effect=fake_propose),
        patch("harness.run_solution", side_effect=fake_evaluator),
    ):
        outcome = run_pipeline(
            task,
            eb,
            iterations=1,
            workers=1,
            backend="codex",
            model="test",
            trial_seed=0,
            v5_bridge=bridge,
        )

    assert outcome["reason"] == "iteration_limit"
    candidates = eb.records()[1:]
    assert len(candidates) == 4
    assert [record["metadata"]["matched_arm"] for record in candidates].count("guided") == 2
    assert [record["metadata"]["matched_arm"] for record in candidates].count("control") == 2
    assert candidates[0]["metadata"]["candidate_seed"] == candidates[1]["metadata"]["candidate_seed"]
    assert candidates[2]["metadata"]["candidate_seed"] == candidates[3]["metadata"]["candidate_seed"]
    assert any("agent_invented" in prompt for prompt in prompts)
    assert any("boundary_composition" in prompt for prompt in prompts)
    pair_path = task.run_dir / "research" / "matched_controls.jsonl"
    payload = json.loads(pair_path.read_text().splitlines()[0])
    assert payload["result"]["guided_record_id"]
    assert payload["result"]["control_record_id"]
    assert len(bridge.hypotheses) == 2
    assert len(bridge.event_store.read_analogy_results()) == 2
    plans = bridge.event_store.read_plan_events()
    assert len(plans) == 2
    assert {plan.generation_operator for plan in plans} == {
        "local_mutation", "composition"
    }
    assert len({plan.id for plan in plans}) == 2
    candidate_events = [
        event for event in bridge.event_store.read_experiment_events()
        if event.record_id != seed["id"]
    ]
    assert candidate_events
    runtime_path = bridge.event_store.object_store.get_path(
        candidate_events[0].runtime_metrics_ref,
        "runtime_metrics.json",
    )
    runtime = json.loads(runtime_path.read_text())
    lineage = runtime["mechanism_lineage"]
    assert lineage["mechanism_id"] == "agent_invented"
    assert lineage["matched_arm"] in {"guided", "control"}
    assert lineage["generation_operator"] == "local_mutation"
    # The V5 Proposal packet for the second mechanism carries its own plan,
    # rather than the first slot's primary hypothesis/operator.
    assert any(
        "boundary_composition" in prompt
        and "generation_operator\": \"composition\"" in prompt
        for prompt in prompts
    )


def test_adaptive_loop_updates_context_state_without_v5(tmp_path):
    """Directional feedback remains recursive when islands are disabled."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "solver.py").write_text("print('seed')\n")
    eb = ExperienceBank(tmp_path / "eb", direction="max")
    seed = eb.commit(source, 1.0, "ok", "seed", None, "")
    task = _task(tmp_path)
    task.adaptive_feedback = True
    task.feedback_mode = "adaptive"
    task.context_barrier = True
    seen_state_versions = []

    def fake_context(_task, _bank, iteration, **kwargs):
        state = kwargs.get("feedback_state")
        seen_state_versions.append(None if state is None else state.state_version)
        decision = ContextDecision(
            action="continue",
            analysis="Use the completed round's directional evidence.",
            reason="The mechanism remains testable.",
            expected_gain=0.1,
            confidence=0.5,
            next_experiment="try the mechanism",
            mechanism_candidates=({
                "id": "adaptive_mechanism",
                "family": "representation",
                "mechanism": "change the representation",
                "prediction": "positive paired effect",
                "failure_condition": "effect is non-positive",
                "matched_control": "same parent without the change",
            },),
        )
        return (
            decision,
            seed,
            "Implement __OPENHYRA_CANDIDATE_SEED__",
            "try the mechanism",
            {
                "iteration": iteration,
                "eb_version": len(_bank.records()),
                "visible_solution_ids": [row["id"] for row in _bank.records()],
                "trial_seed": 10 + iteration,
                "direction": "try the mechanism",
                "context_decision": decision.to_dict(),
                "mechanism_candidates": list(decision.mechanism_candidates),
                "selected_mechanism_candidates": list(decision.mechanism_candidates),
            },
        )

    def fake_propose(parent, draft, _prompt, editable_files, **_kwargs):
        prepare_draft(parent, draft)
        (draft / editable_files[0]).write_text("# candidate\nprint('ok')\n")
        return True, "proposal"

    evaluation_index = 0

    def fake_evaluator(_candidate, sandbox, _task):
        nonlocal evaluation_index
        evaluation_index += 1
        Path(sandbox).mkdir(parents=True, exist_ok=True)
        packet = FeedbackPacket(
            packet_id=f"packet-{evaluation_index}",
            candidate_id=f"candidate-{evaluation_index}",
            mechanism_id="candidate_vs_baseline",
            directional=[DirectionalFeedback(
                id=f"effect-{evaluation_index}",
                candidate_id=f"candidate-{evaluation_index}",
                mechanism_id="candidate_vs_baseline",
                slice_key="slice:public",
                direction="positive",
                observed={"effect": 0.1 * evaluation_index},
                data={"split": "public"},
            )],
            observed={"aggregate_effect": 0.1 * evaluation_index},
            data={"split": "public"},
        )
        return 1.0 + 0.01 * evaluation_index, "ok", "evaluator", {
            "set_hash": f"set-{evaluation_index}",
            "feedback_packet": packet.to_dict(),
        }

    with (
        patch("harness.build_inspiration", side_effect=fake_context),
        patch("harness.propose", side_effect=fake_propose),
        patch("harness.run_solution", side_effect=fake_evaluator),
    ):
        outcome = run_pipeline(
            task,
            eb,
            iterations=2,
            workers=1,
            backend="codex",
            model="test",
            trial_seed=0,
            v5_bridge=None,
        )

    assert outcome["reason"] == "iteration_limit"
    assert seen_state_versions == [0, 2]
    feedback_path = task.run_dir / "v5" / "feedback_packets.jsonl"
    assert len(feedback_path.read_text().splitlines()) == 4
