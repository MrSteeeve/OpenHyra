import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_agent import (
    DEFAULT_CONTEXT_PHASES,
    _allowed_context_phases,
    _parse_context_decision,
    build_inspiration,
)
from eb import ExperienceBank
from reporting import SUMMARY_FIELDS, export_bundle


ROOT = Path(__file__).resolve().parents[1]


def _decision_payload(*, phase):
    return {
        "action": "continue",
        "analysis": "The validation evidence supports another focused trial.",
        "reason": "One bounded improvement remains untested.",
        "expected_gain": 0.01,
        "confidence": 0.8,
        "phase": phase,
        "target_claim_id": None,
        "success_criterion": "the trusted score improves",
        "next": "add one causal state feature",
    }


class GenericPromptTests(unittest.TestCase):
    def test_none_phase_configuration_preserves_legacy_defaults(self):
        task = SimpleNamespace(allowed_context_phases=None)
        self.assertEqual(_allowed_context_phases(task), DEFAULT_CONTEXT_PHASES)

    def test_task_controls_candidate_contract_and_context_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "features.py").write_text("FEATURES = []\n")
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(source, 0.1, "ok", "generic seed", None, "seed log")
            task = SimpleNamespace(
                direction="max",
                metric="validation_lcb",
                description=(
                    "Define causal feature expressions for a stopping policy. "
                    "The trusted evaluator owns fitting and scoring."
                ),
                candidate_instructions=(
                    "Emit only FEATURE_IR; do not fit a model or read evaluator data."
                ),
                allowed_context_phases=["search", "audit"],
                editable_files=["features.py"],
                fallback_directions=["try a simple feature"],
                engineering_invariants=[],
            )
            captured = []
            response = subprocess.CompletedProcess(
                args=["codex"],
                returncode=0,
                stdout=json.dumps(_decision_payload(phase="not-allowed")),
                stderr="",
            )

            def capture(prompt, **_kwargs):
                captured.append(prompt)
                return response

            with patch("context_agent.run_agent", side_effect=capture):
                decision, _baseline, proposal, _direction, metadata = (
                    build_inspiration(task, bank, 0, backend="codex")
                )

        self.assertEqual(decision.phase, "search")
        self.assertEqual(metadata["phase"], "search")
        self.assertIn("one of: search, audit", captured[0])
        self.assertIn("Emit only FEATURE_IR", proposal)
        self.assertIn("candidate output contract", proposal)
        self.assertNotIn("explicit finite `A`", proposal)
        self.assertNotIn("strict optional `research` object", proposal)
        self.assertNotIn("research frontier", proposal.lower())

    def test_custom_allowed_phase_is_accepted_without_loosening_other_fields(self):
        parsed = _parse_context_decision(
            json.dumps(_decision_payload(phase="audit")),
            ("search", "audit"),
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.phase, "audit")

        malformed = _decision_payload(phase="audit")
        malformed["confidence"] = 2
        self.assertIsNone(_parse_context_decision(
            json.dumps(malformed),
            ("search", "audit"),
        ))


class GenericReportingTests(unittest.TestCase):
    def test_summary_flattens_generic_metrics_and_metadata_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            bank = ExperienceBank(root / "eb", direction="max")
            bank.commit(
                source,
                0.42,
                "ok",
                "generic candidate",
                None,
                "",
                metrics={
                    "mean_lower_bound": 1.25,
                    "diagnostics": {
                        "q90_regret": 0.03,
                        "seed_blocks": ["a", "b"],
                    },
                    "n": 7,
                },
                metadata={
                    "iteration": 3,
                    "evaluation": {"stage": "search", "fold": 2},
                },
            )
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "final_audit.json").write_text(
                '{"schema":"openhyra-final-audit.v1","status":"ok"}\n'
            )
            task = SimpleNamespace(
                name="generic",
                protocol="generic-v1",
                run_id="dynamic-summary",
                editable_files=[],
                run_dir=run_dir,
            )
            destination = root / "bundle"
            export_bundle(
                task,
                bank,
                destination,
                root=ROOT,
                run_manifest={"manifest_sha256": "test-manifest"},
            )

            with open(destination / "summary.tsv", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                row = next(reader)
                fields = reader.fieldnames
            manifest = json.loads((destination / "manifest.json").read_text())

            self.assertEqual(row["score"], "0.42")
            self.assertEqual(row["n"], "7")
            self.assertEqual(row["metrics.mean_lower_bound"], "1.25")
            self.assertEqual(row["metrics.diagnostics.q90_regret"], "0.03")
            self.assertEqual(
                row["metrics.diagnostics.seed_blocks"],
                '["a","b"]',
            )
            self.assertEqual(row["metadata.evaluation.stage"], "search")
            self.assertEqual(row["metadata.evaluation.fold"], "2")
            self.assertEqual(fields[len(SUMMARY_FIELDS):], sorted(
                fields[len(SUMMARY_FIELDS):]
            ))
            exported_audit = destination / "final_audit.json"
            self.assertTrue(exported_audit.is_file())
            self.assertEqual(
                manifest["final_audit_sha256"],
                hashlib.sha256(exported_audit.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
