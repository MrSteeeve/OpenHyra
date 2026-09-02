import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eb import ExperienceBank
from context_agent import (
    MAX_CONTEXT_PROMPT_CHARS,
    MAX_HISTORY_RECORDS,
    MAX_PROPOSAL_PROMPT_CHARS,
    _history_table,
    _parse_context_decision,
    _select_history_records,
    build_inspiration,
)
from harness import (
    _assemble_commit_snapshot,
    _copy_generated_file,
    _duplicate_of,
    _numeric_duplicate_of,
    _evaluate_candidate,
    _termination_payload,
    _known_solver_issues,
    _next_context_iteration,
    check_frozen,
    ensure_run_resumable,
    run_pipeline,
)
from proposal_agent import prepare_draft
from llm_backend import _run_cli
from provenance import (
    RunLock,
    build_run_manifest,
    load_run_manifest,
    validate_run_manifest,
    write_run_manifest,
)
from sandbox import (
    _snapshot_artifact,
    run_solution,
    source_tree_hash,
    trusted_artifact_dir,
)
from reporting import export_bundle
from stopping import (
    ContextDecision,
    StopController,
    StopPolicy,
    stopping_evidence,
    write_termination,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tasks" / "sums_diffs" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("sums_diffs_evaluator", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _context_result(parent, direction, metadata):
    decision = ContextDecision(
        action="continue",
        analysis="Continue testing the next concrete direction.",
        reason="Useful experiments remain.",
        expected_gain=0.001,
        confidence=0.8,
        next_experiment=direction,
    )
    return (
        decision, parent, "seed=__OPENHYRA_CANDIDATE_SEED__",
        direction, metadata,
    )


def _research_payload():
    return {
        "schema": "openhyra-research",
        "scope": "all_finite_integer_sets",
        "hypothesis": (
            "A base-12 digit gadget separates modular sums from differences."
        ),
        "construction": {
            "schema": "openhyra-digit-product",
            "base": 12,
            "digits": [0, 1, 2, 4, 5, 9],
            "levels": 2,
            "check_levels": [1, 2],
            "obligations": [{
                "id": "O1",
                "type": "level_counts",
                "level": 1,
                "expected_n": 6,
                "expected_sum_count": 15,
                "expected_diff_count": 17,
            }],
        },
        "claims": [{
            "id": "G1",
            "template": "supporting_lemma",
            "statement": (
                "W+W covers every residue modulo 12 and W-W has size 11."
            ),
            "depends_on": [],
            "obligation_ids": ["O1"],
        }],
        "falsification_plan": (
            "Recompute the modular sum and difference residue counts exactly."
        ),
        "proof_sketch": (
            "A digit product may amplify the one-residue modular gap."
        ),
        "certificates": [{
            "type": "modular_sum_diff",
            "modulus": 12,
            "residues": [0, 1, 2, 4, 5, 9],
            "expected_sum_count": 12,
            "expected_diff_count": 11,
        }],
    }


class EvaluatorTests(unittest.TestCase):
    def test_official_simpletes_seed(self):
        values = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
        score, metrics, normalized = EVALUATOR.evaluate_values(values)
        self.assertAlmostEqual(score, 1.0597930945472454, places=14)
        self.assertEqual(metrics["n"], 17)
        self.assertEqual(metrics["sums"], 59)
        self.assertEqual(metrics["diffs"], 55)
        self.assertEqual(normalized, values)

    def test_canonical_hash_removes_affine_symmetries(self):
        values = [0, 1, 3, 7]
        translated_scaled = [19 + 5 * value for value in values]
        reflected = [max(values) - value for value in values]
        expected = EVALUATOR.canonical_hash(values)
        self.assertEqual(expected, EVALUATOR.canonical_hash(translated_scaled))
        self.assertEqual(expected, EVALUATOR.canonical_hash(reflected))

    def test_simpletes_normalizes_integer_duplicates(self):
        score, metrics, normalized = EVALUATOR.evaluate_values(
            [3, 0, 1, 3, 0],
        )
        self.assertEqual(normalized, [0, 1, 3])
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["sums"], 6)
        self.assertEqual(metrics["diffs"], 7)
        self.assertGreater(score, 0)

    def test_candidate_values_require_actual_integers(self):
        for value in (1.0, True, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    EVALUATOR.evaluate_values([0, value])

    def test_accepts_more_than_512_elements(self):
        score, metrics, normalized = EVALUATOR.evaluate_values(list(range(513)))
        self.assertEqual(metrics["n"], 513)
        self.assertEqual(metrics["sums"], 1025)
        self.assertEqual(metrics["diffs"], 1025)
        self.assertEqual(normalized, list(range(513)))
        self.assertEqual(score, 1.0)

    def test_rejects_fewer_than_two_distinct_elements(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            EVALUATOR.evaluate_values([7, 7])

    def test_integer_range_remains_bounded(self):
        for values in ([-1_000_001, 0], [0, 1_000_001]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "elements must be in"):
                    EVALUATOR.evaluate_values(values)

    def test_research_certificate_is_bounded_evidence_not_a_proof(self):
        values = [0, 1, 2, 4, 5, 9]
        expected_score = EVALUATOR.evaluate_values(values)[0]
        score, metrics, normalized, evidence = EVALUATOR.evaluate_submission({
            "A": values,
            "research": _research_payload(),
        })
        self.assertEqual(score, expected_score)
        self.assertEqual(
            metrics["evidence_level"],
            "proposal_with_bounded_support",
        )
        self.assertEqual(metrics["research_claim_count"], 1)
        self.assertEqual(metrics["verified_certificate_count"], 1)
        self.assertEqual(evidence["numeric"]["status"], "exact")
        self.assertEqual(evidence["research"]["status"], "bounded_supported")
        self.assertEqual(
            evidence["research"]["certificate_summary"],
            "contains_bounded_certificate",
        )
        self.assertEqual(
            evidence["research"]["claims"][0]["status"],
            "unverified",
        )
        self.assertEqual(
            evidence["research"]["claims"][0][
                "linked_obligation_status"
            ],
            "bounded_checked",
        )
        self.assertEqual(
            evidence["research"]["certificates"][0]["status"],
            "bounded_checked",
        )
        self.assertEqual(
            evidence["research"]["formalization"]["status"],
            "not_submitted",
        )
        self.assertNotIn("verified", normalized["research"]["certificates"][0])

        repeated = EVALUATOR.evaluate_submission(normalized)
        self.assertEqual(repeated[2], normalized)
        self.assertEqual(
            repeated[1]["research_sha256"],
            metrics["research_sha256"],
        )

    def test_false_research_certificate_is_refuted_without_losing_score(self):
        research = _research_payload()
        research["certificates"][0]["expected_diff_count"] = 12
        score, metrics, _normalized, evidence = EVALUATOR.evaluate_submission({
            "A": [0, 1, 2, 4, 5, 9],
            "research": research,
        })
        self.assertIsNotNone(score)
        self.assertEqual(
            metrics["evidence_level"],
            "proposal_with_refutation",
        )
        self.assertEqual(metrics["verified_certificate_count"], 0)
        self.assertEqual(metrics["refuted_certificate_count"], 1)
        certificate = evidence["research"]["certificates"][0]
        self.assertEqual(certificate["status"], "refuted")
        self.assertEqual(certificate["expected_diff_count"], 12)
        self.assertEqual(certificate["diff_count"], 11)
        self.assertEqual(evidence["research"]["status"], "contains_refutation")
        self.assertEqual(
            evidence["research"]["certificate_summary"],
            "contains_refuted_certificate",
        )

    def test_refutation_outranks_a_submitted_formalization(self):
        research = _research_payload()
        research["construction"]["obligations"][0][
            "expected_sum_count"
        ] = 14
        research["claims"][0].update({
            "id": "U1",
            "template": "universal_upper_bound",
            "statement": "Every admissible exponent is below two.",
            "target": {"numerator": 2, "denominator": 1},
        })
        research["formalization"] = {
            "schema": "openhyra-lean4-request",
            "target": "lean4",
            "proofs": [{
                "claim_id": "U1",
                "term": "by exact proof",
            }],
        }

        _score, metrics, _normalized, evidence = (
            EVALUATOR.evaluate_submission({
                "A": [0, 1, 2, 4, 5, 9],
                "research": research,
            })
        )

        self.assertEqual(
            metrics["evidence_level"],
            "proposal_with_refutation",
        )
        self.assertEqual(metrics["formalization_status"], "submitted")
        self.assertEqual(evidence["research"]["status"], "contains_refutation")
        self.assertEqual(
            evidence["research"]["claims"][0]["status"],
            "unverified",
        )
        self.assertEqual(
            evidence["research"]["claims"][0][
                "linked_obligation_status"
            ],
            "contains_refutation",
        )

    def test_candidate_cannot_forge_research_evidence_status(self):
        research = _research_payload()
        research["status"] = "formal_checked"
        with self.assertRaisesRegex(ValueError, "unknown field"):
            EVALUATOR.evaluate_submission({
                "A": [0, 1, 2, 4, 5, 9],
                "research": research,
            })

    def test_research_claim_dependencies_must_be_acyclic(self):
        research = _research_payload()
        research["claims"] = [
            {
                "id": "C1",
                "template": "supporting_lemma",
                "statement": "First claim.",
                "depends_on": ["C2"],
                "obligation_ids": [],
            },
            {
                "id": "C2",
                "template": "supporting_lemma",
                "statement": "Second claim.",
                "depends_on": ["C1"],
                "obligation_ids": [],
            },
        ]
        with self.assertRaisesRegex(ValueError, "acyclic"):
            EVALUATOR.evaluate_submission({
                "A": [0, 1, 2, 4, 5, 9],
                "research": research,
            })

    def test_research_hash_is_canonical_but_scope_sensitive(self):
        first = _research_payload()
        first["claims"].append({
            "id": "G0",
            "template": "observation",
            "statement": "The digit set has six residues.",
            "depends_on": [],
            "obligation_ids": [],
        })
        first["claims"][0]["depends_on"] = ["G0"]
        second = {
            key: first[key]
            for key in reversed(list(first))
        }
        second["claims"] = list(reversed(second["claims"]))
        first_hash = EVALUATOR.evaluate_submission({
            "A": [0, 1, 2, 4, 5, 9],
            "research": first,
        })[1]["research_sha256"]
        second_hash = EVALUATOR.evaluate_submission({
            "A": [0, 1, 2, 4, 5, 9],
            "research": second,
        })[1]["research_sha256"]
        self.assertEqual(first_hash, second_hash)

        second["scope"] = "current_task"
        scoped_hash = EVALUATOR.evaluate_submission({
            "A": [0, 1, 2, 4, 5, 9],
            "research": second,
        })[1]["research_sha256"]
        self.assertNotEqual(first_hash, scoped_hash)

    def test_supremum_claim_never_becomes_finite_candidate_score(self):
        research = _research_payload()
        research["claims"][0].update({
            "id": "S1",
            "template": "supremum_eq",
            "statement": "The supremum of C(A) is 2.",
            "target": {"numerator": 2, "denominator": 1},
        })
        score, _metrics, _normalized, evidence = (
            EVALUATOR.evaluate_submission({
                "A": [0, 1, 2, 4, 5, 9],
                "research": research,
            })
        )
        self.assertLess(score, 2)
        self.assertEqual(
            evidence["research"]["claims"][0]["status"],
            "unverified",
        )
        self.assertEqual(evidence["research"]["status"], "bounded_supported")
        self.assertNotIn(
            "claim_id",
            evidence["research"]["certificates"][0],
        )

    def test_certificate_cannot_bind_itself_to_a_free_text_claim(self):
        research = _research_payload()
        research["claims"][0].update({
            "id": "S1",
            "template": "supremum_eq",
            "statement": "The supremum of C(A) is 2.",
            "target": {"numerator": 2, "denominator": 1},
        })
        research["certificates"][0]["claim_id"] = "S1"
        with self.assertRaisesRegex(ValueError, "unknown field"):
            EVALUATOR.evaluate_submission({
                "A": [0, 1, 2, 4, 5, 9],
                "research": research,
            })

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "solution.json"
            path.write_text('{"A":[0,1,3],"A":[0,1,4]}')
            result = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertIn("duplicate JSON key", result.stdout)


class ExperienceBankTests(unittest.TestCase):
    def test_concurrent_commits_are_complete_and_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("pass\n")
            bank = ExperienceBank(root / "eb", direction="max")

            def commit(index):
                bank.commit(
                    source, float(index), "ok", f"candidate {index}", None, "",
                    metrics={"artifact_sha256": hashlib.sha256(str(index).encode()).hexdigest()},
                )

            threads = [threading.Thread(target=commit, args=(index,)) for index in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            version, records = bank.snapshot()
            self.assertEqual(version, 20)
            self.assertEqual(len(records), 20)
            self.assertEqual({record["id"] for record in records}, {
                f"sol_{index:04d}" for index in range(20)
            })
            self.assertEqual(bank.best()["score"], 19.0)


class ProvenanceTests(unittest.TestCase):
    def test_manifest_round_trip_and_resume_drift_rejection(self):
        from harness import Task

        with tempfile.TemporaryDirectory() as temporary, \
                patch("provenance.command_version", return_value="test-cli 1.0"):
            task = Task("sums_diffs", "provenance-test")
            recorded = build_run_manifest(
                task, ROOT, backend="codex", model="test-model", workers=2,
                candidates_per_context=4, trial_seed=7,
                stopping_policy={"enabled": True, "stop_patience": 4},
            )
            path = Path(temporary) / "run_manifest.json"
            write_run_manifest(path, recorded)
            loaded = load_run_manifest(path)
            current = build_run_manifest(
                task, ROOT, backend="codex", model="test-model", workers=2,
                candidates_per_context=4, trial_seed=7,
                stopping_policy={"enabled": True, "stop_patience": 4},
            )
            self.assertEqual(
                validate_run_manifest(loaded, current)["manifest_sha256"],
                recorded["manifest_sha256"],
            )

            current["search"]["workers"] = 3
            with self.assertRaisesRegex(RuntimeError, "provenance drift"):
                validate_run_manifest(loaded, current)

            current["search"]["workers"] = 2
            current["environment"]["backend_cli"] = "test-cli 2.0"
            with self.assertRaisesRegex(RuntimeError, "environment"):
                validate_run_manifest(loaded, current)

            current["environment"]["backend_cli"] = "test-cli 1.0"
            current["stopping_policy"]["stop_patience"] = 5
            with self.assertRaisesRegex(RuntimeError, "stopping_policy"):
                validate_run_manifest(loaded, current)

    def test_v5_config_provenance_drift_rejected(self):
        from harness import Task

        with tempfile.TemporaryDirectory() as temporary, \
                patch("provenance.command_version", return_value="test-cli 1.0"):
            task = Task("sums_diffs", "provenance-v5-test")
            recorded = build_run_manifest(
                task, ROOT, backend="codex", model="test-model", workers=2,
                candidates_per_context=4, trial_seed=7,
                v5_config={"enabled": True, "num_islands": 4},
            )
            path = Path(temporary) / "run_manifest.json"
            write_run_manifest(path, recorded)
            loaded = load_run_manifest(path)
            self.assertEqual(loaded["v5"], {"enabled": True, "num_islands": 4})

            current = build_run_manifest(
                task, ROOT, backend="codex", model="test-model", workers=2,
                candidates_per_context=4, trial_seed=7,
                v5_config={"enabled": True, "num_islands": 8},
            )
            with self.assertRaisesRegex(RuntimeError, "v5"):
                validate_run_manifest(loaded, current)

            current_no_v5 = build_run_manifest(
                task, ROOT, backend="codex", model="test-model", workers=2,
                candidates_per_context=4, trial_seed=7,
            )
            with self.assertRaisesRegex(RuntimeError, "v5"):
                validate_run_manifest(loaded, current_no_v5)

    def test_manifest_checksum_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            payload = {"manifest_sha256": "wrong", "task": {"name": "x"}}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                load_run_manifest(path)

    def test_only_one_process_lock_can_own_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.lock"
            first, second = RunLock(path), RunLock(path)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    second.acquire()
            finally:
                first.release()


class AgentStoppingTests(unittest.TestCase):
    @staticmethod
    def _record(iteration, score, *, status="ok", candidate_index=0,
                candidate_count=1, duplicate_of=None):
        return {
            "id": f"record-{iteration}-{candidate_index}",
            "score": score,
            "status": status,
            "metadata": {
                "iteration": iteration,
                "candidate_index": candidate_index,
                "candidate_count": candidate_count,
                "duplicate_of": duplicate_of,
                "direction": f"direction-{iteration}",
            },
        }

    @staticmethod
    def _stop_decision():
        return ContextDecision(
            action="stop",
            analysis="Recent valid candidates have converged.",
            reason="Expected marginal gain is low.",
            expected_gain=0.00001,
            confidence=0.9,
            next_experiment=None,
        )

    def test_context_decision_parser_is_fail_safe(self):
        parsed = _parse_context_decision(json.dumps({
            "action": "stop",
            "analysis": "Search appears locally exhausted.",
            "reason": "No meaningful recent gain.",
            "expected_gain": 0.00001,
            "confidence": 0.91,
            "next": None,
        }))
        self.assertEqual(parsed.action, "stop")
        self.assertIsNone(_parse_context_decision("not JSON"))
        self.assertIsNone(_parse_context_decision(json.dumps({
            "action": "stop",
            "analysis": "Stop but also proposes more work.",
            "reason": "This is internally inconsistent.",
            "expected_gain": 0.0,
            "confidence": 0.9,
            "next": "another experiment",
        })))
        self.assertIsNone(_parse_context_decision(json.dumps({
            "action": "continue",
            "analysis": "Keep going.",
            "reason": "Missing a concrete next experiment.",
            "expected_gain": 0.1,
            "confidence": 0.5,
            "next": None,
        })))
        self.assertIsNone(_parse_context_decision(json.dumps({
            "action": "stop",
            "analysis": "Invalid expected gain.",
            "reason": "Negative gains are invalid input.",
            "expected_gain": -1,
            "confidence": 0.5,
            "next": None,
        })))

    def test_invalid_context_output_becomes_continue_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 1.0, "ok", "seed", None, "seed log")
            task = SimpleNamespace(
                direction="max",
                metric="score",
                description="Test task.",
                editable_files=["solver.py"],
                fallback_directions=["deterministic fallback"],
                engineering_invariants=[],
            )
            invalid = subprocess.CompletedProcess(
                args=["codex"], returncode=0, stdout="not JSON", stderr="",
            )
            with patch("context_agent.run_agent", return_value=invalid):
                decision, _baseline, _prompt, direction, metadata = build_inspiration(
                    task, bank, 0, backend="codex", model="test",
                    agent_stop_enabled=True,
                )
            self.assertEqual(decision.action, "continue")
            self.assertEqual(direction, "deterministic fallback")
            self.assertEqual(metadata["context_decision"]["action"], "continue")

    def test_stop_request_is_rejected_before_minimum_contexts(self):
        policy = StopPolicy(enabled=True, min_contexts_before_stop=6)
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            self._record(0, 1.01),
            self._record(1, 1.01),
        ]
        review = StopController(policy, "max").review(
            self._stop_decision(), records,
        )
        self.assertFalse(review.accepted)
        self.assertIn("minimum_contexts_not_met", review.reasons)

    def test_stop_request_is_rejected_when_recent_candidates_failed(self):
        policy = StopPolicy(
            enabled=True, min_contexts_before_stop=6, stop_patience=4,
            recent_window=4, min_successful_candidates=4,
        )
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            self._record(0, 1.01),
            self._record(1, 1.01),
            *(
                self._record(iteration, None, status="crash")
                for iteration in range(2, 6)
            ),
        ]
        review = StopController(policy, "max").review(
            self._stop_decision(), records,
        )
        self.assertFalse(review.accepted)
        self.assertIn("insufficient_successful_candidates", review.reasons)

    def test_stop_request_is_accepted_only_after_deterministic_guards(self):
        policy = StopPolicy(
            enabled=True, min_contexts_before_stop=6, stop_patience=4,
            meaningful_delta=0.0001, recent_window=4,
            min_successful_candidates=4,
        )
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            self._record(0, 1.01),
            *(self._record(iteration, 1.01) for iteration in range(1, 6)),
        ]
        review = StopController(policy, "max").review(
            self._stop_decision(), records,
        )
        self.assertTrue(review.accepted)
        self.assertEqual(
            review.evidence["contexts_since_meaningful_improvement"], 5,
        )
        self.assertEqual(review.evidence["recent_successful_candidates"], 4)

    def test_incomplete_context_does_not_satisfy_stop_guards(self):
        policy = StopPolicy(
            enabled=True, min_contexts_before_stop=1, stop_patience=0,
            min_successful_candidates=0,
        )
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            self._record(0, 1.0, candidate_index=0, candidate_count=4),
        ]
        review = StopController(policy, "max").review(
            self._stop_decision(), records,
        )
        self.assertFalse(review.accepted)
        self.assertEqual(review.evidence["completed_contexts"], 0)
        self.assertEqual(review.evidence["incomplete_contexts"], [0])

    def test_incomplete_context_rejects_stop_after_other_guards_pass(self):
        policy = StopPolicy(
            enabled=True, min_contexts_before_stop=6, stop_patience=4,
            recent_window=4, min_successful_candidates=4,
        )
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            *(self._record(iteration, 1.0) for iteration in range(6)),
            self._record(6, 1.0, candidate_index=0, candidate_count=4),
        ]
        review = StopController(policy, "max").review(
            self._stop_decision(), records,
        )
        self.assertFalse(review.accepted)
        self.assertIn("incomplete_contexts_exist", review.reasons)
        self.assertEqual(review.evidence["completed_contexts"], 6)

    def test_cumulative_small_gains_reset_stop_patience(self):
        policy = StopPolicy(
            enabled=True, meaningful_delta=0.0001, recent_window=4,
        )
        records = [
            {"id": "seed", "score": 1.0, "status": "ok", "metadata": {}},
            self._record(0, 1.00006),
            self._record(1, 1.00012),
            *(self._record(iteration, 1.00012) for iteration in range(2, 6)),
        ]
        evidence = stopping_evidence(records, direction="max", policy=policy)
        self.assertTrue(evidence["context_improvements"][1]["meaningful"])
        self.assertAlmostEqual(
            evidence["context_improvements"][1]["cumulative_gain"],
            0.00012,
        )
        self.assertEqual(
            evidence["contexts_since_meaningful_improvement"], 4,
        )

    def test_resume_rejects_terminal_and_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("pass\n")
            run_dir = root / "run"
            task = SimpleNamespace(run_dir=run_dir, run_id="resume-test")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 1.0, "ok", "seed", None, "")

            write_termination(run_dir / "termination.json", {
                "reason": "agent_converged", "terminal": True,
            })
            with self.assertRaisesRegex(RuntimeError, "already terminated"):
                ensure_run_resumable(task, bank)

            write_termination(run_dir / "termination.json", {
                "reason": "iteration_limit", "terminal": False,
            })
            ensure_run_resumable(task, bank)

            (run_dir / "termination.json").unlink()
            bank.commit(
                source, 1.0, "ok", "partial", "sol_0000", "",
                metadata={
                    "iteration": 0,
                    "candidate_index": 0,
                    "candidate_count": 4,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete Context"):
                ensure_run_resumable(task, bank)

    def test_context_history_is_representative_and_bounded(self):
        records = []
        for index in range(200):
            metrics = {
                "detail": "m" * 2000,
                "evidence_level": "proposal",
                "research_claim_count": 1,
            }
            if index == 3:
                metrics.update({
                    "evidence_level": "proposal_with_bounded_support",
                    "research_claim_count": 1,
                    "verified_certificate_count": 1,
                    "research_context": (
                        "hypothesis: a modular gadget may tensorize; "
                        "claim remains unverified"
                    ),
                    "certificate_context": (
                        "bounded_checked modular_sum_diff modulus=12 "
                        "trusted(sum=12,diff=11)"
                    ),
                })
            if index == 4:
                metrics.update({
                    "evidence_level": "proposal_with_refutation",
                    "verified_certificate_count": 0,
                    "refuted_certificate_count": 1,
                    "research_context": (
                        "hypothesis: a deliberately false modular count"
                    ),
                    "certificate_context": (
                        "refuted standalone modular_sum_diff modulus=12 "
                        "trusted(sum=12,diff=11)"
                    ),
                })
            records.append({
                "id": f"record-{index}",
                "score": 10.0 if index == 5 else index / 1000,
                "status": "crash" if index % 13 == 0 else "ok",
                "description": f"description-{index}-" + "x" * 2000,
                "metrics": metrics,
                "metadata": {
                    "iteration": index,
                    "direction": f"direction-{index % 30}",
                },
            })
        selected = _select_history_records(records, "max")
        self.assertEqual(len(selected), MAX_HISTORY_RECORDS)
        self.assertIn("record-5", {record["id"] for record in selected})
        self.assertIn("record-3", {record["id"] for record in selected})
        self.assertIn("record-4", {record["id"] for record in selected})
        self.assertIn("record-199", {record["id"] for record in selected})
        self.assertTrue(any(record["status"] == "crash" for record in selected))
        table = _history_table(records, "max")
        self.assertIn("Showing 80 representative records out of 200", table)
        self.assertIn(
            "Evidence-class highlights (reserved before table truncation)",
            table,
        )
        self.assertIn("record-3 [proposal_with_bounded_support]", table)
        self.assertIn("record-4 [proposal_with_refutation]", table)
        self.assertIn("UNVERIFIED research", table)
        self.assertIn("TRUSTED bounded verdicts", table)
        self.assertIn("bounded certificates=1", table)
        self.assertEqual(
            sum(line.startswith("| record-") for line in table.splitlines()),
            MAX_HISTORY_RECORDS,
        )

    def test_context_and_proposal_prompts_have_hard_character_caps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            for index in range(100):
                metrics = {
                    "detail": "m" * 2000,
                    "evidence_level": "proposal",
                    "research_claim_count": 1,
                    "research_context": "r" * 5000,
                    "research_hypothesis": "h" * 1000,
                }
                if index == 50:
                    metrics.update({
                        "evidence_level": "proposal_with_refutation",
                        "refuted_certificate_count": 1,
                        "certificate_context": (
                            "refuted standalone modular_sum_diff modulus=12"
                        ),
                    })
                if index == 51:
                    metrics.update({
                        "evidence_level": "proposal_with_bounded_support",
                        "verified_certificate_count": 1,
                        "certificate_context": (
                            "bounded_checked standalone modular_sum_diff "
                            "modulus=12"
                        ),
                    })
                bank.commit(
                    source, float(index), "crash" if index % 9 == 0 else "ok",
                    f"record-{index}-" + "d" * 2000,
                    f"sol_{index - 1:04d}" if index else None,
                    "l" * 10000,
                    metrics=metrics,
                    metadata={
                        "iteration": index,
                        "candidate_index": 0,
                        "candidate_count": 1,
                        "direction": f"direction-{index % 20}",
                    },
                )
            task = SimpleNamespace(
                direction="max",
                metric="score",
                description="T" * 50000,
                editable_files=["solver.py"],
                fallback_directions=["fallback"],
                engineering_invariants=[],
            )
            captured = []
            valid = subprocess.CompletedProcess(
                args=["codex"], returncode=0,
                stdout=json.dumps({
                    "action": "continue",
                    "analysis": "Bounded history is sufficient.",
                    "reason": "A concrete direction remains.",
                    "expected_gain": 0.001,
                    "confidence": 0.8,
                    "next": "try a bounded experiment",
                }),
                stderr="",
            )

            def capture(prompt, **_kwargs):
                captured.append(prompt)
                return valid

            with patch("context_agent.run_agent", side_effect=capture):
                _decision, _baseline, proposal_prompt, _direction, _metadata = (
                    build_inspiration(task, bank, 100, backend="codex")
                )
            self.assertLessEqual(len(captured[0]), MAX_CONTEXT_PROMPT_CHARS)
            self.assertIn(
                "sol_0050 [proposal_with_refutation]",
                captured[0],
            )
            self.assertIn(
                "sol_0051 [proposal_with_bounded_support]",
                captured[0],
            )
            self.assertLessEqual(len(proposal_prompt), MAX_PROPOSAL_PROMPT_CHARS)

    def test_pipeline_accepts_eligible_stop_without_creating_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('baseline')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 1.0, "ok", "seed", None, "")
            for iteration in range(6):
                bank.commit(
                    source, 1.01, "ok", f"context {iteration}", "sol_0000", "",
                    metadata={
                        "iteration": iteration,
                        "candidate_index": 0,
                        "candidate_count": 1,
                        "direction": f"direction-{iteration}",
                    },
                )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=2,
                candidates_per_context=4,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="guarded-stop",
            )
            decision = self._stop_decision()

            def fake_context(*_args, **_kwargs):
                parent = bank.best()
                metadata = {
                    "iteration": 6,
                    "eb_version": 7,
                    "visible_solution_ids": [record["id"] for record in bank.records()],
                    "trial_seed": 6,
                    "direction": "fallback",
                    "context_decision": decision.to_dict(),
                }
                return decision, parent, "unused", "fallback", metadata

            policy = StopPolicy(
                enabled=True, min_contexts_before_stop=6, stop_patience=4,
                recent_window=4, min_successful_candidates=4,
            )
            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose") as propose_mock):
                outcome = run_pipeline(
                    task, bank, iterations=3, workers=2, backend="codex",
                    model="test", trial_seed=0, stop_policy=policy,
                )

            self.assertEqual(outcome["reason"], "agent_converged")
            self.assertTrue(outcome["stop_review"]["accepted"])
            propose_mock.assert_not_called()
            self.assertEqual(len(bank.records()), 7)

            payload = _termination_payload(task, bank, policy, outcome, 3)
            termination = write_termination(root / "termination.json", payload)
            self.assertEqual(termination["accepted_by"], "stop_controller")
            self.assertEqual(termination["completed_contexts"], 6)
            self.assertEqual(termination["candidate_attempts"], 6)

    def test_pipeline_rejects_early_stop_and_runs_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(source, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="rejected-stop",
            )
            decision = self._stop_decision()

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 1,
                    "direction": "fallback",
                    "context_decision": decision.to_dict(),
                }
                return decision, parent, "fallback", "fallback", metadata

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text("print('candidate')\n")
                return True, "continued after rejected stop"

            def fake_run_solution(_draft, sandbox, _task):
                Path(sandbox).mkdir(parents=True)
                return 1.0, "ok", "ok", {"set_hash": "same"}

            policy = StopPolicy(enabled=True, min_contexts_before_stop=6)
            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                outcome = run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=0, stop_policy=policy,
                )

            self.assertEqual(outcome["reason"], "iteration_limit")
            candidate = bank.records()[-1]
            self.assertEqual(candidate["status"], "ok")
            review = candidate["metadata"]["stop_review"]
            self.assertFalse(review["accepted"])
            self.assertIn("minimum_contexts_not_met", review["reasons"])
            self.assertEqual(
                candidate["metadata"]["effective_context_decision"]["action"],
                "continue",
            )

    def test_agent_stop_mode_waits_for_prior_context_to_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="sequential-context",
            )
            observed_versions = []

            def fake_context(_task, _eb, iteration, **_kwargs):
                records = bank.records()
                observed_versions.append(len(records))
                parent = bank.best()
                metadata = {
                    "iteration": iteration,
                    "eb_version": len(records),
                    "visible_solution_ids": [record["id"] for record in records],
                    "trial_seed": iteration,
                    "direction": f"direction-{iteration}",
                }
                return _context_result(parent, f"direction-{iteration}", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text("print('candidate')\n")
                return True, "sequential candidate"

            def fake_run_solution(_draft, sandbox, _task):
                iteration = int(Path(sandbox).parent.name.split("_")[1])
                Path(sandbox).mkdir(parents=True)
                return 1.01 + iteration * 0.01, "ok", "ok", {
                    "set_hash": f"set-{iteration}",
                }

            policy = StopPolicy(enabled=True, min_contexts_before_stop=6)
            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=2, workers=2, backend="codex",
                    model="test", trial_seed=0, stop_policy=policy,
                )

            self.assertEqual(observed_versions, [1, 2])
            self.assertEqual(len(bank.records()), 3)
            self.assertEqual(
                [record["status"] for record in bank.records()[1:]],
                ["ok", "ok"],
            )


class ReportingTests(unittest.TestCase):
    def test_export_includes_hashed_termination_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            (source / "solution.json").write_text('{"A":[0,1,3]}')
            (source / "evidence.json").write_text(
                '{"schema":"openhyra-evidence","research":{"status":"proposed"}}'
            )
            evidence_hash = hashlib.sha256(
                (source / "evidence.json").read_bytes()
            ).hexdigest()
            artifact_hash = hashlib.sha256(
                (source / "solution.json").read_bytes()
            ).hexdigest()
            solver_hash = hashlib.sha256(
                (source / "solver.py").read_bytes()
            ).hexdigest()
            source_hash = source_tree_hash(
                source, 64 * 1024 * 1024,
            )[0]
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(
                source, 1.0, "ok", "seed", None, "",
                metrics={
                    "evidence_level": "proposal",
                    "research_claim_count": 1,
                    "verified_certificate_count": 0,
                    "research_sha256": "research-hash",
                    "evidence_sha256": evidence_hash,
                    "artifact_sha256": artifact_hash,
                    "source_snapshot_sha256": source_hash,
                },
                metadata={"editable_file_sha256": {"solver.py": solver_hash}},
            )
            run_dir = root / "run"
            termination = write_termination(run_dir / "termination.json", {
                "reason": "agent_converged",
                "terminal": True,
            })
            task = SimpleNamespace(
                name="test",
                protocol="test-v1",
                run_id="reporting",
                editable_files=["solver.py"],
                run_dir=run_dir,
            )
            destination = root / "bundle"
            export_bundle(
                task, bank, destination, root=ROOT,
                run_manifest={"manifest_sha256": "test-manifest"},
            )
            exported = json.loads((destination / "termination.json").read_text())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(exported, termination)
            self.assertTrue(
                (destination / "solutions" / "sol_0000" / "evidence.json").is_file()
            )
            summary = (destination / "summary.tsv").read_text()
            self.assertIn("evidence_level", summary.splitlines()[0])
            self.assertIn("proposal", summary)
            self.assertEqual(
                manifest["termination_sha256"],
                hashlib.sha256(
                    (destination / "termination.json").read_bytes()
                ).hexdigest(),
            )

    def test_export_rejects_candidate_forged_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            (source / "evidence.json").write_text(
                '{"research":{"status":"formal_checked"}}'
            )
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                name="test",
                protocol="sums-diffs",
                run_id="forged-evidence",
                editable_files=["solver.py"],
                run_dir=root / "run",
            )
            with self.assertRaisesRegex(RuntimeError, "without a trusted hash"):
                export_bundle(
                    task, bank, root / "bundle", root=ROOT,
                    run_manifest={"manifest_sha256": "test-manifest"},
                )

    def test_export_rejects_missing_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            solver_hash = hashlib.sha256(
                (source / "solver.py").read_bytes()
            ).hexdigest()
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(
                source, 1.0, "ok", "seed", None, "",
                metrics={"evidence_sha256": "0" * 64},
                metadata={"editable_file_sha256": {"solver.py": solver_hash}},
            )
            task = SimpleNamespace(
                name="test",
                protocol="sums-diffs",
                run_id="missing-evidence",
                editable_files=["solver.py"],
                run_dir=root / "run",
            )
            with self.assertRaisesRegex(RuntimeError, "is missing evidence.json"):
                export_bundle(
                    task, bank, root / "bundle", root=ROOT,
                    run_manifest={"manifest_sha256": "test-manifest"},
                )

    def test_export_rejects_tampered_solution_and_editable_files(self):
        for target_name in ("solution.json", "solver.py"):
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                source.mkdir()
                (source / "solver.py").write_text("print('tampered')\n")
                (source / "solution.json").write_text('{"A":[0,1,3]}')
                solution_hash = hashlib.sha256(
                    (source / "solution.json").read_bytes()
                ).hexdigest()
                solver_hash = hashlib.sha256(
                    (source / "solver.py").read_bytes()
                ).hexdigest()
                metrics = {"artifact_sha256": solution_hash}
                metadata = {
                    "editable_file_sha256": {"solver.py": solver_hash},
                }
                if target_name == "solution.json":
                    metrics["artifact_sha256"] = "0" * 64
                else:
                    metadata["editable_file_sha256"]["solver.py"] = "0" * 64
                bank = ExperienceBank(root / "eb", direction="max")
                bank.commit(
                    source, 1.0, "ok", "seed", None, "",
                    metrics=metrics, metadata=metadata,
                )
                task = SimpleNamespace(
                    name="test",
                    protocol="sums-diffs",
                    run_id="tampered-export",
                    editable_files=["solver.py"],
                    run_dir=root / "run",
                )
                with self.assertRaisesRegex(RuntimeError, "trusted hash"):
                    export_bundle(
                        task, bank, root / "bundle", root=ROOT,
                        run_manifest={"manifest_sha256": "test-manifest"},
                    )

    def test_export_rejects_tampered_noneditable_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('seed')\n")
            (source / "solve.sh").write_text("safe\n")
            source_hash = source_tree_hash(
                source, 64 * 1024 * 1024,
            )[0]
            solver_hash = hashlib.sha256(
                (source / "solver.py").read_bytes()
            ).hexdigest()
            (source / "solve.sh").write_text("tampered\n")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(
                source, None, "violation", "tampered source", None, "",
                metrics={"source_snapshot_sha256": source_hash},
                metadata={"editable_file_sha256": {"solver.py": solver_hash}},
            )
            task = SimpleNamespace(
                name="test",
                protocol="sums-diffs",
                run_id="tampered-source",
                editable_files=["solver.py"],
                run_dir=root / "run",
            )
            with self.assertRaisesRegex(
                RuntimeError, "source snapshot does not match",
            ):
                export_bundle(
                    task, bank, root / "bundle", root=ROOT,
                    run_manifest={"manifest_sha256": "test-manifest"},
                )


class DraftIsolationTests(unittest.TestCase):
    def test_draft_copies_code_without_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            draft = root / "draft"
            parent.mkdir()
            (parent / "solver.py").write_text("print('parent')\n")
            (parent / "solution.json").write_text('{"A":[0,1,3,7]}')
            (parent / "evidence.json").write_text(
                '{"schema":"openhyra-evidence"}'
            )

            prepare_draft(parent, draft)

            self.assertEqual((draft / "solver.py").read_text(), "print('parent')\n")
            self.assertFalse((draft / "solution.json").exists())
            self.assertFalse((draft / "evidence.json").exists())
            self.assertEqual(check_frozen(parent, draft, ["solver.py"]), [])

    def test_trusted_generated_copy_replaces_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trusted = root / "trusted-evidence.json"
            trusted.write_text('{"status":"bounded_checked"}')
            victim = root / "victim.txt"
            victim.write_text("do not overwrite")
            destination = root / "evidence.json"
            destination.symlink_to(victim)

            _copy_generated_file(trusted, destination)

            self.assertFalse(destination.is_symlink())
            self.assertEqual(
                destination.read_text(),
                trusted.read_text(),
            )
            self.assertEqual(victim.read_text(), "do not overwrite")

    def test_commit_snapshot_rejects_mismatched_trusted_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('candidate')\n")
            sandbox = root / "sandboxes" / "candidate"
            trusted = trusted_artifact_dir(sandbox)
            (trusted / "source").mkdir(parents=True)
            (trusted / "source" / "solver.py").write_text(
                "print('candidate')\n"
            )
            (trusted / "evidence.json").write_text(
                '{"research":{"status":"proposed"}}'
            )
            task = SimpleNamespace(
                editable_files=["solver.py"],
                max_output_mb=8,
                max_artifact_bytes=65536,
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _assemble_commit_snapshot(
                    source, sandbox, task,
                    {"evidence_sha256": "0" * 64},
                )

    def test_sealed_snapshot_rechecks_noneditable_files_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            draft = root / "draft"
            parent.mkdir()
            draft.mkdir()
            for directory in (parent, draft):
                (directory / "solver.py").write_text("print('candidate')\n")
                (directory / "solve.sh").write_text("safe\n")
            (draft / "solve.sh").write_text("late mutation\n")
            item = {
                "iteration": 0,
                "candidate_index": 0,
                "candidate_count": 1,
                "draft": draft,
                "parent": {"id": "sol_0000", "path": str(parent)},
                "failure": None,
                "failure_status": None,
            }
            task = SimpleNamespace(
                run_dir=root / "run",
                editable_files=["solver.py"],
                max_output_mb=8,
            )
            with patch("harness.run_solution") as run_mock:
                result = _evaluate_candidate(
                    item, task, threading.Lock(),
                )
            run_mock.assert_not_called()
            self.assertEqual(result["status"], "violation")
            self.assertIn("non-editable", result["log_tail"])
            self.assertTrue(result["item"]["commit_dir"].is_dir())


class ArtifactIntakeTests(unittest.TestCase):
    def test_regular_artifact_is_copied_to_trusted_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "solution.json"
            artifact.write_text('{"A":[0,1,3]}')
            snapshot, data = _snapshot_artifact(
                artifact, root / "trusted", max_bytes=1024,
            )
            self.assertEqual(data, artifact.read_bytes())
            self.assertEqual(snapshot.read_bytes(), data)
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)

    def test_symbolic_link_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.json"
            outside.write_text('{"A":[0,1,3]}')
            (root / "solution.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _snapshot_artifact(
                    root / "solution.json", root / "trusted", max_bytes=1024,
                )

    def test_non_regular_and_multiply_linked_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "solution.json"
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                _snapshot_artifact(directory, root / "trusted-dir", max_bytes=1024)

            directory.rmdir()
            artifact = root / "solution.json"
            artifact.write_text('{"A":[0,1,3]}')
            os.link(artifact, root / "second-link.json")
            with self.assertRaisesRegex(ValueError, "exactly one hard link"):
                _snapshot_artifact(artifact, root / "trusted-link", max_bytes=1024)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFOs")
    def test_fifo_artifact_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fifo = root / "solution.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular file"):
                _snapshot_artifact(fifo, root / "trusted", max_bytes=1024)

    def test_oversized_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "solution.json"
            artifact.write_bytes(b"x" * 17)
            with self.assertRaisesRegex(ValueError, "16-byte"):
                _snapshot_artifact(artifact, root / "trusted", max_bytes=16)


class PublishedArtifactTests(unittest.TestCase):
    def test_legacy_winner_bundle_verifies_and_matches_manifest(self):
        artifact = (
            ROOT / "artifacts" / "sums_diffs" /
            "openhyra-1.111814562869239-legacy"
        )
        result = subprocess.run(
            [sys.executable, str(artifact / "verify.py")],
            capture_output=True, text=True, check=True,
        )
        verdict = json.loads(result.stdout)
        self.assertEqual(verdict["n"], 405)
        self.assertEqual(verdict["sums"], 2395)
        self.assertEqual(verdict["diffs"], 2003)
        self.assertAlmostEqual(verdict["score"], 1.111814562869239, places=15)

        manifest = json.loads((artifact / "manifest.json").read_text())
        self.assertEqual(manifest["artifact_kind"], "legacy-winner-only")
        self.assertFalse(manifest["retention"]["current_harness_rerun"])
        for name, expected in manifest["files_sha256"].items():
            actual = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)


class CandidatePipelineTests(unittest.TestCase):
    def test_next_context_iteration_ignores_candidates_per_context(self):
        records = [
            {"metadata": {}},
            *({"metadata": {"iteration": 0}} for _ in range(4)),
            *({"metadata": {"iteration": 1}} for _ in range(4)),
        ]
        self.assertEqual(_next_context_iteration(records), 2)

    def test_research_progress_is_not_suppressed_by_numeric_duplicate(self):
        records = [{
            "id": "sol_0001",
            "status": "ok",
            "metrics": {
                "set_hash": "same-set",
                "candidate_hash": "first-research-artifact",
            },
        }]
        result = {
            "status": "ok",
            "metrics": {
                "set_hash": "same-set",
                "candidate_hash": "different-research-artifact",
            },
        }
        self.assertIsNone(_duplicate_of(result, records))
        self.assertEqual(_numeric_duplicate_of(result, records), "sol_0001")


    def test_preflight_detects_unclamped_fractional_progress_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            draft = Path(temporary)
            solver = draft / "solver.py"
            solver.write_text(
                "progress = elapsed / budget\n"
                "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
            )
            self.assertIn("without clamping", _known_solver_issues(draft, ["solver.py"])[0])

            solver.write_text(
                "progress = min(1.0, max(0.0, elapsed / budget))\n"
                "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
            )
            self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])

    def test_preflight_requires_nonempty_dynamic_randrange_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            draft = Path(temporary)
            solver = draft / "solver.py"
            solver.write_text("value = rng.randrange(width, new_width)\n")
            self.assertIn("proving stop > start", _known_solver_issues(draft, ["solver.py"])[0])

            solver.write_text(
                "if new_width > width:\n"
                "    value = rng.randrange(width, new_width)\n"
            )
            self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])

    def test_pipeline_commits_all_four_outcomes_for_one_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(
                parent_dir, 1.0, "ok", "seed", None, "", metrics={"n": 4},
            )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=2,
                candidates_per_context=4,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 9,
                    "direction": "vary",
                }
                return _context_result(parent, "vary", metadata)

            def fake_propose(parent_path, draft, prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                commented_prompt = "\n".join(f"# {line}" for line in prompt.splitlines())
                (draft / editable_files[0]).write_text(commented_prompt + "\n")
                return True, "candidate"

            scores = [1.01, None, 1.09, 1.03]

            def fake_run_solution(_draft, sandbox, _task):
                index = int(Path(sandbox).name.split("_")[1])
                Path(sandbox).mkdir(parents=True)
                if index == 1:
                    (Path(sandbox) / "run.log").write_text("candidate crashed\n")
                    return None, "crash", "candidate crashed", {"candidate": index}
                (Path(sandbox) / "run.log").write_text("ok\n")
                (Path(sandbox) / "evaluated_solution.json").write_text(
                    json.dumps({"A": [0, index + 2]}),
                )
                return scores[index], "ok", "ok", {"candidate": index}

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=2, backend="codex",
                    model="test", trial_seed=9,
                )

            records = bank.records()
            self.assertEqual(len(records), 5)
            candidates = [record for record in records if record["parent"] is not None]
            self.assertEqual(
                {record["metadata"]["candidate_index"] for record in candidates},
                {0, 1, 2, 3},
            )
            failed = next(
                record for record in candidates
                if record["metadata"]["candidate_index"] == 1
            )
            self.assertEqual(failed["status"], "crash")
            self.assertIsNone(failed["score"])
            self.assertIn("candidate crashed", failed["log_tail"])
            self.assertEqual(bank.best()["score"], 1.09)
            self.assertTrue(all(
                record["metadata"]["candidate_count"] == 4
                for record in candidates
            ))

    def test_pipeline_cancellation_commits_pending_candidates_then_joins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=4,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="cancel-test",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 0,
                    "direction": "cancel",
                }
                return _context_result(parent, "cancel", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files,
                             cancel_event=None, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text("print('candidate')\n")
                cancel_event.set()
                return True, "candidate interrupted during proposal"

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.run_solution") as solver_mock):
                outcome = run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=0, candidates_per_context=4,
                )

            self.assertEqual(outcome["reason"], "user_interrupt")
            solver_mock.assert_not_called()
            candidates = bank.records()[1:]
            self.assertEqual(len(candidates), 4)
            self.assertEqual({record["status"] for record in candidates}, {
                "cancelled",
            })
            self.assertFalse(any(
                thread.name.startswith(("context-producer", "proposal-", "evaluator-"))
                for thread in threading.enumerate()
            ))

    def test_single_candidate_can_repair_one_runtime_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(
                parent_dir, 1.0, "ok", "seed", None, "",
                metrics={"n": 4, "set_hash": "parent"},
            )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=1,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test-repair",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 3,
                    "direction": "repair",
                }
                return _context_result(parent, "repair", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text("print('broken')\n")
                return True, "candidate with a repairable bug"

            def fake_repair(source, draft, feedback, editable_files, **_kwargs):
                self.assertIn("TypeError", feedback)
                prepare_draft(source, draft)
                (Path(draft) / editable_files[0]).write_text("print('fixed')\n")
                return True, "clamped progress"

            evaluations = []

            def fake_run_solution(_draft, sandbox, _task):
                attempt = len(evaluations)
                evaluations.append(attempt)
                sandbox = Path(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)
                if attempt == 0:
                    (sandbox / "run.log").write_text("TypeError: complex temperature\n")
                    return None, "crash", "TypeError: complex temperature", {
                        "solver_seconds": 1.0,
                    }
                (sandbox / "run.log").write_text("fixed\n")
                (sandbox / "evaluated_solution.json").write_text('{"A":[0,1,4]}')
                return 1.2, "ok", "fixed", {
                    "set_hash": "novel", "solver_seconds": 2.0,
                }

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.repair_candidate", side_effect=fake_repair),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=3,
                    candidates_per_context=1,
                )

            records = bank.records()
            self.assertEqual(len(records), 3)
            failed, repaired = records[-2:]
            self.assertEqual(failed["status"], "crash")
            self.assertIn("TypeError", failed["log_tail"])
            self.assertEqual(repaired["status"], "ok")
            self.assertEqual(repaired["parent"], failed["id"])
            self.assertEqual(repaired["metadata"]["repair_of"], failed["id"])
            self.assertEqual(repaired["metadata"]["attempt_index"], 1)
            self.assertEqual(
                (Path(failed["path"]) / "solver.py").read_text(),
                "print('broken')\n",
            )
            self.assertEqual(
                (Path(repaired["path"]) / "solver.py").read_text(),
                "print('fixed')\n",
            )
            self.assertEqual(len(evaluations), 2)

    def test_preflight_repairs_known_progress_bug_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=1,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test-preflight",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 4,
                    "direction": "preflight",
                }
                return _context_result(parent, "preflight", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text(
                    "progress = elapsed / budget\n"
                    "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
                )
                return True, "unsafe annealing schedule"

            def fake_repair(source, draft, feedback, editable_files, **_kwargs):
                self.assertIn("engineering preflight failed", feedback)
                prepare_draft(source, draft)
                (Path(draft) / editable_files[0]).write_text(
                    "progress = min(1.0, max(0.0, elapsed / budget))\n"
                    "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
                )
                return True, "clamped progress"

            def fake_run_solution(draft, sandbox, _task):
                self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])
                sandbox = Path(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)
                (sandbox / "run.log").write_text("ok\n")
                (sandbox / "evaluated_solution.json").write_text('{"A":[0,1,4]}')
                return 1.2, "ok", "ok", {"set_hash": "safe"}

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.repair_candidate", side_effect=fake_repair),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=4,
                    candidates_per_context=1,
                )

            records = bank.records()
            self.assertEqual(len(records), 3)
            rejected, repaired = records[-2:]
            self.assertEqual(rejected["status"], "rejected")
            self.assertTrue(rejected["metadata"]["preflight_notes"])
            self.assertEqual(repaired["status"], "ok")
            self.assertEqual(repaired["metadata"]["repair_of"], rejected["id"])
            self.assertIn(
                "progress = elapsed / budget",
                (Path(rejected["path"]) / "solver.py").read_text(),
            )


class CancellationTests(unittest.TestCase):
    def test_llm_cli_process_group_is_cancelled_promptly(self):
        cancel_event = threading.Event()
        timer = threading.Timer(0.2, cancel_event.set)
        timer.start()
        started = time.monotonic()
        try:
            result = _run_cli(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=None, prompt_stdin=None, timeout_s=20,
                cancel_event=cancel_event,
            )
        finally:
            timer.cancel()
        self.assertEqual(result.returncode, 130)
        self.assertLess(time.monotonic() - started, 3)

    def test_successful_cli_exit_cleans_ordinary_background_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "late-write.txt"
            child = (
                "import pathlib,time; "
                "time.sleep(0.4); "
                f"pathlib.Path({str(marker)!r}).write_text('late')"
            )
            parent = (
                "import subprocess,sys; "
                "subprocess.Popen("
                f"[sys.executable,'-c',{child!r}],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL)"
            )
            result = _run_cli(
                [sys.executable, "-c", parent],
                cwd=None, prompt_stdin=None, timeout_s=5,
                cancel_event=None,
            )
            self.assertEqual(result.returncode, 0)
            time.sleep(0.7)
            self.assertFalse(marker.exists())


class ResearchArtifactPipelineTests(unittest.TestCase):
    def test_research_and_trusted_evidence_survive_the_snapshot_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            (source / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            payload = {
                "A": [0, 1, 2, 4, 5, 9],
                "research": _research_payload(),
            }
            (source / "solver.py").write_text(
                "from pathlib import Path\n"
                f"Path('solution.json').write_text({json.dumps(payload)!r})\n"
            )
            task = SimpleNamespace(
                evaluator=EVALUATOR_PATH,
                python_bin=sys.executable,
                timeout_s=10,
                max_memory_mb=512,
                max_output_mb=8,
                max_artifact_bytes=65536,
                evaluator_timeout_s=10,
                evaluator_max_memory_mb=512,
            )
            with patch.dict(
                os.environ,
                {"OPENHYRA_ALLOW_UNSANDBOXED": "1"},
            ):
                score, status, _tail, metrics = run_solution(
                    source, root / "sandbox", task,
                )

            self.assertEqual(status, "ok")
            self.assertIsNotNone(score)
            self.assertEqual(
                metrics["evidence_level"],
                "proposal_with_bounded_support",
            )
            trusted = trusted_artifact_dir(root / "sandbox")
            evaluated = json.loads(
                (trusted / "evaluated_solution.json").read_text()
            )
            evidence_path = trusted / "evidence.json"
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(
                evaluated["research"]["hypothesis"],
                payload["research"]["hypothesis"],
            )
            self.assertEqual(
                evidence["research"]["claims"][0]["status"],
                "unverified",
            )
            self.assertEqual(
                evidence["research"]["certificates"][0]["status"],
                "bounded_checked",
            )
            self.assertEqual(
                metrics["evidence_sha256"],
                hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            )

    def test_research_and_evidence_reach_the_experience_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            (parent_dir / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(
                parent_dir, 0.5, "ok", "seed", None, "",
                metrics={"set_hash": "seed"},
            )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="sums-diffs",
                run_id="research-round-trip",
                evaluator=EVALUATOR_PATH,
                python_bin=sys.executable,
                timeout_s=10,
                max_memory_mb=512,
                max_output_mb=8,
                max_artifact_bytes=65536,
                evaluator_timeout_s=10,
                evaluator_max_memory_mb=512,
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 12,
                    "direction": "certify a modular gadget",
                }
                return _context_result(
                    parent, "certify a modular gadget", metadata,
                )

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                payload = {
                    "A": [0, 1, 2, 4, 5, 9],
                    "research": _research_payload(),
                }
                (draft / editable_files[0]).write_text(
                    "from pathlib import Path\n"
                    f"Path('solution.json').write_text({json.dumps(payload)!r})\n"
                )
                return True, "certify the base-12 modular gadget"

            with (
                patch("harness.build_inspiration", side_effect=fake_context),
                patch("harness.propose", side_effect=fake_propose),
                patch.dict(
                    os.environ,
                    {"OPENHYRA_ALLOW_UNSANDBOXED": "1"},
                ),
            ):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=12,
                )

            record = bank.records()[-1]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                record["metrics"]["evidence_level"],
                "proposal_with_bounded_support",
            )
            stored = Path(record["path"])
            self.assertTrue((stored / "evidence.json").is_file())
            self.assertIn(
                "research",
                json.loads((stored / "solution.json").read_text()),
            )

    def test_detached_draft_writer_cannot_change_committed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 0.5, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="sums-diffs",
                run_id="detached-writer",
            )
            children = []

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 14,
                    "direction": "seal candidate source",
                }
                return _context_result(parent, "seal candidate source", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                target = draft / editable_files[0]
                target.write_text("print('sealed')\n")
                late_writer = (
                    "import pathlib,time; time.sleep(0.5); "
                    f"pathlib.Path({str(target)!r}).write_text(\"print('late')\\n\")"
                )
                children.append(subprocess.Popen(
                    [sys.executable, "-c", late_writer],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                ))
                return True, "candidate with a detached late writer"

            def fake_run_solution(_source, sandbox, _task):
                Path(sandbox).mkdir(parents=True)
                return 1.0, "ok", "ok", {"set_hash": "sealed"}

            with (
                patch("harness.build_inspiration", side_effect=fake_context),
                patch("harness.propose", side_effect=fake_propose),
                patch("harness.run_solution", side_effect=fake_run_solution),
            ):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=14,
                )
            for child in children:
                child.wait(timeout=3)

            record = bank.records()[-1]
            self.assertEqual(
                (Path(record["path"]) / "solver.py").read_text(),
                "print('sealed')\n",
            )
            draft = task.run_dir / "drafts" / "iter_0000" / "cand_00"
            self.assertEqual(
                (draft / "solver.py").read_text(),
                "print('late')\n",
            )

    def test_unsafe_source_intake_is_recorded_without_copying_candidate_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 0.5, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="sums-diffs",
                run_id="unsafe-intake",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 15,
                    "direction": "reject unsafe source",
                }
                return _context_result(parent, "reject unsafe source", metadata)

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                editable = draft / editable_files[0]
                editable.unlink()
                victim = root / "outside.py"
                victim.write_text("print('must not be copied')\n")
                editable.symlink_to(victim)
                return True, "candidate with an unsafe editable link"

            with (
                patch("harness.build_inspiration", side_effect=fake_context),
                patch("harness.propose", side_effect=fake_propose),
            ):
                outcome = run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=15,
                )

            self.assertEqual(outcome["reason"], "iteration_limit")
            record = bank.records()[-1]
            self.assertEqual(record["status"], "crash")
            self.assertIn("intake failed closed", record["log_tail"])
            self.assertFalse((Path(record["path"]) / "solver.py").exists())

    def test_failed_proposal_cannot_smuggle_an_evidence_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 0.5, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="sums-diffs",
                run_id="reject-forged-evidence",
            )

            def fake_context(*_args, **_kwargs):
                metadata = {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 13,
                    "direction": "invalid proposal",
                }
                return _context_result(parent, "invalid proposal", metadata)

            def fake_propose(parent_path, draft, _prompt, _editable, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / "evidence.json").write_text(
                    '{"research":{"status":"formal_checked"}}'
                )
                return False, "proposal failed after writing a reserved file"

            with (
                patch("harness.build_inspiration", side_effect=fake_context),
                patch("harness.propose", side_effect=fake_propose),
            ):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=13,
                )

            record = bank.records()[-1]
            self.assertEqual(record["status"], "crash")
            self.assertFalse((Path(record["path"]) / "evidence.json").exists())


@unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
class SandboxTests(unittest.TestCase):
    def test_cancel_event_kills_active_solver_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            (source / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            (source / "solver.py").write_text(
                "import time\ntime.sleep(10)\n"
            )
            cancel_event = threading.Event()
            task = SimpleNamespace(
                evaluator=EVALUATOR_PATH,
                python_bin=sys.executable,
                timeout_s=20,
                max_memory_mb=512,
                max_output_mb=8,
                cancel_event=cancel_event,
            )
            timer = threading.Timer(0.2, cancel_event.set)
            timer.start()
            started = time.monotonic()
            try:
                score, status, log_tail, _metrics = run_solution(
                    source, root / "sandbox", task,
                )
            finally:
                timer.cancel()
            self.assertIsNone(score)
            self.assertEqual(status, "cancelled")
            self.assertIn("cancelled solver process group", log_tail)
            self.assertLess(time.monotonic() - started, 3)

    def test_background_writer_cannot_change_scored_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            (source / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            (source / "solver.py").write_text(
                "import json, os, time\n"
                "seed=[0,1,2,4,5,9,12,13,14,16,17,21,24,25,26,28,29]\n"
                "with open('solution.json','w') as f: json.dump({'A':seed},f)\n"
                "pid=os.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(0.5)\n"
                "    with open('solution.json','w') as f: json.dump({'A':[0,1]},f)\n"
                "    os._exit(0)\n"
            )
            stale = source / "solution.json"
            stale.write_text('{"A":[0,1]}')
            stale.chmod(0o444)
            task = SimpleNamespace(
                evaluator=EVALUATOR_PATH,
                python_bin=sys.executable,
                timeout_s=10,
                max_memory_mb=512,
                max_output_mb=8,
            )
            score, status, _tail, metrics = run_solution(
                source, root / "sandbox", task,
            )
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(score, 1.0597930945472454, places=14)
            self.assertEqual(metrics["n"], 17)
            trusted = trusted_artifact_dir(root / "sandbox")
            snapshot = json.loads((trusted / "solution.snapshot.json").read_text())
            self.assertEqual(snapshot["A"], [0,1,2,4,5,9,12,13,14,16,17,21,24,25,26,28,29])
            evaluated = trusted / "evaluated_solution.json"
            self.assertEqual(
                hashlib.sha256(evaluated.read_bytes()).hexdigest(),
                metrics["artifact_sha256"],
            )
            self.assertEqual(
                hashlib.sha256((trusted / "solution.snapshot.json").read_bytes()).hexdigest(),
                metrics["candidate_artifact_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
