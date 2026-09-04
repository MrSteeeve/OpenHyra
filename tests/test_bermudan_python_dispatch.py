"""End-to-end checks for the open Python continuation-policy track."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from tasks.bermudan_python_search import evaluator as python_evaluator
from tasks.bermudan_optimal_stopping import evaluator


MLP_MANIFEST = {
    "schema": "openhyra-policy-spec.v1",
    "runner_type": "mlp",
    "inference_config": {
        "input_dim": "n_assets",
        "layers": [],
        "activation": "tanh",
        "output_dim": 1,
        "output_clip": [-1_000_000.0, 1_000_000.0],
    },
    "output_semantics": "discounted_continuation_value_t0",
    "normalization": "per_step",
    "weight_pattern": "step_{:03d}.npy",
}


EXPRESSION_MANIFEST = {
    "schema": "continuation-expression.v1",
    "runner_type": "expression",
    "inference_config": {
        "input_dim": "n_assets",
        "output_dim": 1,
        "output_clip": [-1_000_000.0, 1_000_000.0],
    },
    "output_semantics": "discounted_continuation_value_t0",
    "normalization": "none",
    "weight_pattern": "step_{:03d}.json",
}


LINEAR_MANIFEST = {
    "schema": "continuation-linear.v1",
    "runner_type": "linear",
    "inference_config": {
        "input_dim": "n_assets",
        "output_dim": 1,
        "output_clip": [-1_000_000.0, 1_000_000.0],
    },
    "output_semantics": "discounted_continuation_value_t0",
    "normalization": "per_step",
    "weight_pattern": "step_{:03d}.npy",
}


REQUEST = {
    "schema": evaluator.REQUEST_SCHEMA,
    "stage": "search",
    "task": evaluator.TASK_NAME,
    "protocol": evaluator.ALGORITHM_TASK_PROTOCOL,
    "seed": 1729,
    "suite_id": "python-dispatch-smoke",
    "config": {
        "instance_count": 1,
        "repeats": 1,
        "training_paths": 64,
        "pricing_paths": 64,
        "training_timeout_s": 10,
    },
}


MLP_TRAIN = r'''
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", required=True)
a = p.parse_args()
inp, out = Path(a.input), Path(a.output)
paths = np.load(inp / "training_paths.npy", allow_pickle=False)
d, steps = paths.shape[2], paths.shape[1] - 1
for i in range(steps):
    np.save(out / f"step_{i:03d}.npy", np.zeros(d + 1, dtype=np.float64))
(out / "normalization.json").write_text(json.dumps({
    "steps": [{"mean": [0.0] * d, "scale": [1.0] * d} for _ in range(steps)]
}))
'''


EXPRESSION_TRAIN = r'''
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", required=True)
a = p.parse_args()
inp, out = Path(a.input), Path(a.output)
steps = np.load(inp / "training_paths.npy", allow_pickle=False).shape[1] - 1
for i in range(steps):
    (out / f"step_{i:03d}.json").write_text(
        json.dumps({"op": "constant", "value": 0.0})
    )
'''


LINEAR_TRAIN = r'''
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", required=True)
a = p.parse_args()
inp, out = Path(a.input), Path(a.output)
paths = np.load(inp / "training_paths.npy", allow_pickle=False)
d, steps = paths.shape[2], paths.shape[1] - 1
for i in range(steps):
    np.save(out / f"step_{i:03d}.npy", np.zeros(d + 1, dtype=np.float64))
(out / "normalization.json").write_text(json.dumps({
    "steps": [{"mean": [0.0] * d, "scale": [1.0] * d} for _ in range(steps)]
}))
'''


class BermudanPythonDispatchTests(unittest.TestCase):
    def _candidate(self, root: Path, manifest: dict, train_source: str) -> Path:
        source = root / "candidate"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(manifest))
        (source / "train.py").write_text(train_source)
        return source

    def test_mlp_bundle_runs_per_instance_and_keeps_financial_scoring_owned(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            score, metrics, normalized, evidence = evaluator.evaluate_submission(
                MLP_MANIFEST,
                REQUEST,
                candidate_source_dir=source,
            )
            self.assertTrue(isinstance(score, float))
            self.assertEqual(metrics["candidate_kind"], "algorithm_bundle")
            self.assertEqual(metrics["runner_type"], "mlp")
            self.assertEqual(metrics["training_cell_count"], 1)
            self.assertEqual(normalized["schema"], MLP_MANIFEST["schema"])
            self.assertTrue(evidence["candidate_supplied_prices_ignored"])
            self.assertEqual(evidence["search"]["candidate_kind"], "algorithm_bundle")

    def test_expression_bundle_uses_the_same_stopping_and_score_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), EXPRESSION_MANIFEST, EXPRESSION_TRAIN)
            _score, metrics, normalized, _evidence = evaluator.evaluate_submission(
                EXPRESSION_MANIFEST,
                REQUEST,
                candidate_source_dir=source,
            )
            self.assertEqual(metrics["runner_type"], "expression")
            self.assertEqual(metrics["training_cell_count"], 1)
            self.assertEqual(normalized["runner_type"], "expression")

    def test_linear_bundle_dispatches_through_the_registered_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), LINEAR_MANIFEST, LINEAR_TRAIN)
            _score, metrics, normalized, _evidence = evaluator.evaluate_submission(
                LINEAR_MANIFEST,
                REQUEST,
                candidate_source_dir=source,
            )
            self.assertEqual(metrics["runner_type"], "linear")
            self.assertEqual(metrics["training_cells"][0]["status"], "ok")
            self.assertEqual(normalized["schema"], "continuation-linear.v1")

    def test_algorithm_bundle_runs_in_private_primal_dual_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), EXPRESSION_MANIFEST, EXPRESSION_TRAIN)
            audit_request = {
                **REQUEST,
                "stage": "audit",
                "suite_id": "python-dispatch-hidden-smoke",
                "config": {
                    "instance_count": 1,
                    "repeats": 1,
                    "training_paths": 64,
                    "pricing_paths": 64,
                    "outer_paths": 64,
                    "inner_paths": 2,
                    "training_timeout_s": 10,
                },
            }
            score, metrics, _normalized, evidence = evaluator.evaluate_submission(
                EXPRESSION_MANIFEST,
                audit_request,
                candidate_source_dir=source,
            )
            self.assertTrue(math.isfinite(score))
            self.assertEqual(metrics["training_cell_count"], 1)
            self.assertEqual(evidence["audit"]["candidate_kind"], "algorithm_bundle")

    def test_training_failure_reports_the_instance_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(
                Path(temporary), MLP_MANIFEST, "raise SystemExit(9)\n",
            )
            with self.assertRaisesRegex(
                ValueError, "candidate training failed for public-put-atm: crash",
            ):
                evaluator.evaluate_submission(
                    MLP_MANIFEST,
                    REQUEST,
                    candidate_source_dir=source,
                )

    def test_bundle_manifest_mismatch_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            mismatched = dict(MLP_MANIFEST)
            mismatched["inference_config"] = dict(MLP_MANIFEST["inference_config"])
            mismatched["inference_config"]["layers"] = [2]
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluator.evaluate_submission(
                    mismatched,
                    REQUEST,
                    candidate_source_dir=source,
                )

    def test_bundle_source_rejects_undeclared_helper_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            (source / "helper.py").write_text("VALUE = 1\n")
            with self.assertRaisesRegex(
                ValueError, "undeclared file.*helper.py",
            ):
                evaluator.evaluate_submission(
                    MLP_MANIFEST,
                    REQUEST,
                    candidate_source_dir=source,
                )

    def test_algorithm_bundle_envelope_binds_execution_declarations(self):
        """The envelope cannot advertise a different runner boundary."""
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            valid = {
                "schema": "openhyra-algorithm-bundle.v1",
                "entrypoint": "train.py",
                "artifact_protocol": MLP_MANIFEST["schema"],
                "source_files": ["train.py", "manifest.json"],
            }
            resolved, manifest = evaluator._candidate_source_manifest(
                source, valid,
            )
            self.assertEqual(resolved, source.resolve())
            self.assertEqual(manifest.schema, MLP_MANIFEST["schema"])

            invalid_declarations = (
                ({"entrypoint": "other.py"}, "entrypoint must be train.py"),
                (
                    {"source_files": ["train.py", "helper.py"]},
                    "source_files must contain exactly",
                ),
                (
                    {"artifact_protocol": "continuation-expression.v1"},
                    "artifact_protocol does not match manifest",
                ),
            )
            for overrides, message in invalid_declarations:
                with self.subTest(overrides=overrides):
                    envelope = {**valid, **overrides}
                    with self.assertRaisesRegex(ValueError, message):
                        evaluator._candidate_source_manifest(source, envelope)

    def test_algorithm_bundle_v1_envelope_requires_boundary_declarations(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            envelope = {"schema": "openhyra-algorithm-bundle.v1"}
            with self.assertRaisesRegex(
                ValueError, "missing required field.*entrypoint",
            ):
                evaluator._candidate_source_manifest(source, envelope)

    def test_legacy_algorithm_bundle_envelope_without_declarations_remains_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._candidate(Path(temporary), MLP_MANIFEST, MLP_TRAIN)
            resolved, manifest = evaluator._candidate_source_manifest(
                source,
                {"schema": "openhyra-candidate-algorithm-bundle.v1"},
            )
            self.assertEqual(resolved, source.resolve())
            self.assertEqual(manifest.schema, MLP_MANIFEST["schema"])

    def test_python_wrapper_default_request_uses_python_suite(self):
        request = python_evaluator.default_search_request()
        self.assertEqual(request["task"], "bermudan_python_search")
        self.assertEqual(
            request["protocol"], "bermudan-lsmc-algorithm-bundle.v1",
        )
        self.assertEqual(request["suite_id"], "bermudan-python-public-v1")


if __name__ == "__main__":
    unittest.main()
