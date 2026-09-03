import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_agent import build_inspiration
from eb import ExperienceBank
from sandbox import _apply_formalization_verdict
from stopping import ContextDecision, StopController, StopPolicy


FORMAL_TEMPLATES = (
    "universal_upper_bound",
    "approximating_family",
    "supremum_eq",
    "nonattainment",
)
MATHLIB_REVISION = "3" * 40
RUNTIME_ATTESTATION = {
    "environment_sha256": "1" * 64,
    "lean_binary_sha256": "2" * 64,
    "toolchain": "leanprover/lean4:v4.26.0",
    "mathlib_revision": MATHLIB_REVISION,
    "mathlib_tree_sha256": "4" * 64,
}


def _formal_state(claim_ids=("U",)):
    template_by_id = {
        "U": "universal_upper_bound",
        "A": "approximating_family",
        "S": "supremum_eq",
        "N": "nonattainment",
    }
    target = {"numerator": 2, "denominator": 1}
    claims = [
        {
            "id": claim_id,
            "template": template_by_id[claim_id],
            "statement": f"formal claim {claim_id}",
            "depends_on": [],
            "obligation_ids": [],
            "target": target,
        }
        for claim_id in claim_ids
    ]
    request = {
        "schema": "openhyra-lean4-request",
        "target": "lean4",
        "proofs": [
            {
                "claim_id": claim_id,
                "term": "by exact proof",
            }
            for claim_id in claim_ids
        ],
    }
    normalized = {
        "A": [0, 1],
        "research": {
            "claims": claims,
            "formalization": request,
        },
    }
    evidence = {
        "research": {
            "status": "formalization_submitted",
            "claims": [
                {
                    "id": claim["id"],
                    "template": claim["template"],
                    "target": target,
                    "status": "unverified",
                }
                for claim in claims
            ],
            "formalization": {"status": "submitted", "target": "lean4"},
        },
    }
    metrics = {
        "research_rank": 30,
        "evidence_level": "formalization_submitted",
        "formally_checked_claim_count": 0,
        "refuted_claim_count": 0,
        "bounded_supported_claim_count": 0,
        "formalization_request_sha256": "a" * 64,
    }
    return normalized, evidence, metrics


class FormalPromotionTests(unittest.TestCase):
    def test_only_exact_verified_claim_set_is_promoted(self):
        normalized, evidence, metrics = _formal_state(("U",))
        proof_hash = hashlib.sha256(
            b"by exact proof"
        ).hexdigest()

        def verifier(_request, _claims, **_kwargs):
            return {
                "target": "lean4",
                "status": "verified",
                "reason": "all_checks_passed",
                "request_sha256": "a" * 64,
                "wrapper_sha256": "b" * 64,
                "proofs": [{
                    "claim_id": "U",
                    "proof_sha256": proof_hash,
                }],
                "verified_claim_ids": ["U"],
            }

        task = SimpleNamespace(
            verify_formalization=verifier,
            formalization={},
            formal_runner=object(),
            formal_spec_files={},
        )
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )

        self.assertEqual(metrics["formalization_status"], "verified")
        self.assertEqual(metrics["evidence_level"], "formal_checked")
        self.assertEqual(metrics["formally_checked_claim_count"], 1)
        self.assertEqual(
            evidence["research"]["claims"][0]["status"],
            "formal_checked",
        )
        self.assertEqual(
            metrics["formal_checked_targets"][0]["target"],
            {"numerator": 2, "denominator": 1},
        )

    def test_verifier_claim_set_mismatch_is_an_infrastructure_error(self):
        normalized, evidence, metrics = _formal_state(("U",))

        def verifier(_request, _claims, **_kwargs):
            return {
                "target": "lean4",
                "status": "verified",
                "reason": "all_checks_passed",
                "verified_claim_ids": [],
            }

        task = SimpleNamespace(
            verify_formalization=verifier,
            formalization={},
            formal_runner=object(),
            formal_spec_files={},
        )
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )

        self.assertEqual(
            metrics["formalization_status"],
            "infrastructure_error",
        )
        self.assertEqual(metrics["formally_checked_claim_count"], 0)
        self.assertEqual(
            evidence["research"]["formalization"]["reason"],
            "formal_verifier_claim_set_mismatch",
        )

    def test_proof_hash_mismatch_prevents_promotion(self):
        normalized, evidence, metrics = _formal_state(("U",))

        def verifier(_request, _claims, **_kwargs):
            return {
                "target": "lean4",
                "status": "verified",
                "reason": "all_checks_passed",
                "request_sha256": "a" * 64,
                "proofs": [{
                    "claim_id": "U",
                    "proof_sha256": "0" * 64,
                }],
                "verified_claim_ids": ["U"],
            }

        task = SimpleNamespace(
            verify_formalization=verifier,
            formalization={},
            formal_runner=object(),
            formal_spec_files={},
        )
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )

        self.assertEqual(
            metrics["formalization_status"],
            "infrastructure_error",
        )
        self.assertEqual(
            evidence["research"]["formalization"]["reason"],
            "formal_verifier_proof_hash_mismatch",
        )
        self.assertEqual(metrics["formally_checked_claim_count"], 0)
        self.assertEqual(
            evidence["research"]["claims"][0]["status"],
            "unverified",
        )

    def test_infrastructure_error_does_not_hide_existing_refutation(self):
        normalized, evidence, metrics = _formal_state(("U",))
        evidence["research"]["status"] = "contains_refutation"
        metrics["evidence_level"] = "proposal_with_refutation"
        metrics["research_rank"] = 5
        metrics["refuted_obligation_count"] = 1

        def verifier(_request, _claims, **_kwargs):
            return {
                "target": "lean4",
                "status": "verified",
                "reason": "all_checks_passed",
                "request_sha256": "a" * 64,
                "proofs": [{
                    "claim_id": "U",
                    "proof_sha256": "0" * 64,
                }],
                "verified_claim_ids": ["U"],
            }

        task = SimpleNamespace(
            verify_formalization=verifier,
            formalization={},
            formal_runner=object(),
            formal_spec_files={},
        )
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )

        self.assertEqual(
            metrics["formalization_status"],
            "infrastructure_error",
        )
        self.assertEqual(
            evidence["research"]["status"],
            "contains_refutation",
        )
        self.assertEqual(
            metrics["evidence_level"],
            "proposal_with_refutation",
        )
        self.assertEqual(metrics["research_rank"], 5)

    def test_refuted_claim_makes_multi_claim_promotion_atomic(self):
        normalized, evidence, metrics = _formal_state(("U", "A"))
        evidence["research"]["claims"][1]["status"] = "refuted"
        proof_hash = hashlib.sha256(
            b"by exact proof"
        ).hexdigest()

        def verifier(_request, _claims, **_kwargs):
            return {
                "target": "lean4",
                "status": "verified",
                "reason": "all_checks_passed",
                "request_sha256": "a" * 64,
                "proofs": [
                    {"claim_id": "U", "proof_sha256": proof_hash},
                    {"claim_id": "A", "proof_sha256": proof_hash},
                ],
                "verified_claim_ids": ["U", "A"],
            }

        task = SimpleNamespace(
            verify_formalization=verifier,
            formalization={},
            formal_runner=object(),
            formal_spec_files={},
        )
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )

        self.assertEqual(
            metrics["formalization_status"],
            "infrastructure_error",
        )
        self.assertEqual(
            evidence["research"]["formalization"]["reason"],
            "formal_proof_conflicts_with_trusted_refutation",
        )
        self.assertEqual(metrics["formally_checked_claim_count"], 0)
        self.assertEqual(
            [claim["status"] for claim in evidence["research"]["claims"]],
            ["unverified", "refuted"],
        )


class ProofCompletionTests(unittest.TestCase):
    def setUp(self):
        self.policy = StopPolicy(
            enabled=True,
            min_contexts_before_stop=0,
            stop_patience=0,
            meaningful_delta=0,
            recent_window=1,
            min_successful_candidates=0,
        )
        self.decision = ContextDecision(
            action="stop",
            analysis="All machine-checkable work is complete.",
            reason="The trusted proof gate is closed.",
            expected_gain=0,
            confidence=1,
            next_experiment=None,
        )

    def _record(self, targets):
        return {
            "id": "sol_formal",
            "score": 1.5,
            "status": "ok",
            "metrics": {
                "formalization_status": "verified",
                "formal_checked_targets": targets,
                "refuted_claim_count": 0,
                "refuted_obligation_count": 0,
                "refuted_certificate_count": 0,
            },
            "metadata": {},
        }

    def test_stop_is_rejected_when_templates_are_split_across_targets(self):
        targets = [
            {
                "claim_id": str(index),
                "template": template,
                "target": {
                    "numerator": 2 if index < 3 else 3,
                    "denominator": 1,
                },
            }
            for index, template in enumerate(FORMAL_TEMPLATES)
        ]
        review = StopController(
            self.policy,
            "max",
            required_formal_claims=FORMAL_TEMPLATES,
        ).review(self.decision, [self._record(targets)])

        self.assertFalse(review.accepted)
        self.assertIn(
            "required_formal_claims_not_complete",
            review.reasons,
        )

    def test_stop_accepts_all_templates_in_one_record_at_one_target(self):
        targets = [
            {
                "claim_id": str(index),
                "template": template,
                "target": {"numerator": 2, "denominator": 1},
            }
            for index, template in enumerate(FORMAL_TEMPLATES)
        ]
        review = StopController(
            self.policy,
            "max",
            required_formal_claims=FORMAL_TEMPLATES,
        ).review(self.decision, [self._record(targets)])

        self.assertTrue(review.accepted)
        self.assertTrue(review.evidence["proof_complete"])
        self.assertEqual(
            review.evidence["formal_complete_targets"]["sol_formal"],
            {"numerator": 2, "denominator": 1},
        )

    def test_stop_rejects_formal_record_with_trusted_refutation(self):
        targets = [
            {
                "claim_id": str(index),
                "template": template,
                "target": {"numerator": 2, "denominator": 1},
            }
            for index, template in enumerate(FORMAL_TEMPLATES)
        ]
        record = self._record(targets)
        record["metrics"]["refuted_obligation_count"] = 1
        review = StopController(
            self.policy,
            "max",
            required_formal_claims=FORMAL_TEMPLATES,
        ).review(self.decision, [record])

        self.assertFalse(review.accepted)
        self.assertFalse(review.evidence["proof_complete"])
        self.assertIn(
            "required_formal_claims_not_complete",
            review.reasons,
        )

    def test_stop_rejects_formal_record_with_missing_refutation_ledger(self):
        targets = [
            {
                "claim_id": str(index),
                "template": template,
                "target": {"numerator": 2, "denominator": 1},
            }
            for index, template in enumerate(FORMAL_TEMPLATES)
        ]
        record = self._record(targets)
        del record["metrics"]["refuted_certificate_count"]
        review = StopController(
            self.policy,
            "max",
            required_formal_claims=FORMAL_TEMPLATES,
        ).review(self.decision, [record])

        self.assertFalse(review.accepted)
        self.assertFalse(review.evidence["proof_complete"])


class ResearchFrontierTests(unittest.TestCase):
    def test_formal_phase_uses_research_frontier_not_numeric_frontier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("print('candidate')\n")
            bank = ExperienceBank(root / "eb", direction="max")
            numeric = bank.commit(
                source,
                2.0,
                "ok",
                "numeric best",
                None,
                "",
                metrics={
                    "evidence_level": "numeric",
                    "research_rank": 0,
                },
            )
            research = bank.commit(
                source,
                1.0,
                "ok",
                "proof work",
                numeric["id"],
                "",
                metrics={
                    "evidence_level": "formalization_submitted",
                    "research_rank": 30,
                    "formalization_status": "rejected",
                    "formally_checked_claim_count": 0,
                    "verified_obligation_count": 2,
                },
            )
            task = SimpleNamespace(
                direction="max",
                metric="score",
                description="Research task.",
                editable_files=["solver.py"],
                fallback_directions=["continue"],
                engineering_invariants=[],
            )
            decision = ContextDecision(
                action="continue",
                analysis="Repair the typed proof.",
                reason="The proof runner returned a concrete rejection.",
                expected_gain=0,
                confidence=0.9,
                next_experiment="repair the rejected proof",
                phase="repair_formalization",
                target_claim_id="U",
                success_criterion="formalization_status becomes verified",
            )
            with patch(
                "context_agent._llm_context_analysis",
                return_value=decision,
            ):
                _decision, baseline, _prompt, _direction, metadata = (
                    build_inspiration(
                        task,
                        bank,
                        0,
                        backend="codex",
                        trial_seed=1,
                    )
                )

        self.assertEqual(baseline["id"], research["id"])
        self.assertEqual(metadata["baseline_kind"], "research_frontier")
        self.assertEqual(metadata["numeric_frontier_id"], numeric["id"])
        self.assertEqual(metadata["research_frontier_id"], research["id"])


if __name__ == "__main__":
    unittest.main()
