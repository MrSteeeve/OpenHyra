import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping import training_pipeline as pipeline


def policy_manifest():
    return {
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


TRAIN_SCRIPT = r'''import argparse, json, os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()
input_dir = Path(args.input)
output_dir = Path(args.output)
paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
discounts = np.load(input_dir / "discount_factors.npy", allow_pickle=False)
instance = json.loads((input_dir / "instance.json").read_text())
assert payoffs.shape == paths.shape[:2]
assert discounts.shape == (paths.shape[1],)
assert instance["dimension"] == paths.shape[2]
tmp_marker = Path(os.environ["TMPDIR"]) / "fresh-cell.marker"
if tmp_marker.exists():
    raise SystemExit(42)
tmp_marker.write_text(str(args.seed))
parameters = np.zeros(paths.shape[2] + 1, dtype=np.float64)
for index in range(paths.shape[1] - 1):
    np.save(output_dir / f"step_{index:03d}.npy", parameters)
normalization = {"steps": [
    {"mean": [0.0] * paths.shape[2], "scale": [1.0] * paths.shape[2]}
    for _ in range(paths.shape[1] - 1)
]}
(output_dir / "normalization.json").write_text(json.dumps(normalization))
'''


def runtime_roots():
    candidates = [Path(sys.prefix)]
    if sys.platform == "darwin":
        candidates.extend((
            Path("/usr/lib"),
            Path("/System/Library"),
            Path("/Library/Apple"),
        ))
    roots = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if any(
            resolved == existing
            or resolved in existing.parents
            or existing in resolved.parents
            for existing in roots
        ):
            continue
        roots.append(resolved)
    return roots


def write_candidate(root, script=TRAIN_SCRIPT):
    source = root / "candidate"
    source.mkdir()
    (source / "train.py").write_text(script)
    (source / "manifest.json").write_text(json.dumps(policy_manifest()))
    return source


class BermudanTrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.instance = evaluator.public_suite()[0]
        self.paths = evaluator.simulate_paths(self.instance, 24, 20260829)

    def run_cell(self, source, cell, seed=7):
        return pipeline.run_per_instance_training(
            instance=self.instance,
            training_paths=self.paths,
            candidate_source_dir=source,
            cell_dir=cell,
            train_seed=seed,
            runtime_roots=runtime_roots(),
            timeout_s=10,
            cpu_seconds=10,
            memory_bytes=512 * 1024 * 1024,
            file_size_bytes=2 * 1024 * 1024,
            externally_isolated=sys.platform != "darwin",
        )

    def test_smoke_trains_from_exact_canonical_inputs_and_loads_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = write_candidate(root)
            result = self.run_cell(source, root / "cell")
            self.assertEqual(result.status, "ok", result.log_tail)
            self.assertEqual(
                result.isolation,
                "seatbelt" if sys.platform == "darwin" else "external",
            )
            self.assertIsNotNone(result.runner)
            self.assertEqual(len(result.input_bundle_sha256), 64)
            self.assertEqual(len(result.policy_artifact_sha256), 64)
            self.assertGreaterEqual(result.wall_seconds, 0.0)
            self.assertGreater(result.peak_memory_bytes, 0)
            self.assertEqual(
                result.output_entries,
                len(self.instance.exercise_times),
            )
            self.assertGreater(result.output_bytes, 0)

            input_dir = root / "cell" / "input"
            output_dir = root / "cell" / "output"
            self.assertEqual(
                {path.name for path in input_dir.iterdir()},
                pipeline.TRAINING_INPUT_FILES,
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "normalization.json",
                    *(f"step_{index:03d}.npy"
                      for index in range(len(self.instance.exercise_times) - 1)),
                },
            )
            instance_payload = json.loads((input_dir / "instance.json").read_text())
            self.assertEqual(instance_payload["schema"], pipeline.TRAINING_INSTANCE_SCHEMA)
            self.assertEqual(set(instance_payload), {
                "schema", "payoff_type", "dimension", "spots",
                "strike", "rate", "dividends", "volatilities", "correlation",
                "maturity", "exercise_times", "weights",
            })
            forbidden = {
                "instance_id", "evaluation_request", "pricing_paths",
                "outer_paths", "inner_paths",
            }
            self.assertTrue(forbidden.isdisjoint(instance_payload))
            expected_instance_bytes = json.dumps(
                pipeline.training_instance_payload(self.instance),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(
                (input_dir / "instance.json").read_bytes(), expected_instance_bytes,
            )
            saved_paths = np.load(
                input_dir / "training_paths.npy", allow_pickle=False,
            )
            saved_payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
            saved_discounts = np.load(
                input_dir / "discount_factors.npy", allow_pickle=False,
            )
            for array in (saved_paths, saved_payoffs, saved_discounts):
                self.assertEqual(array.dtype, np.dtype(np.float64))
                self.assertTrue(array.flags.c_contiguous)
            np.testing.assert_array_equal(saved_paths, self.paths)
            np.testing.assert_array_equal(
                saved_payoffs,
                evaluator.discounted_rewards(self.paths, self.instance),
            )
            np.testing.assert_array_equal(
                saved_discounts,
                np.exp(
                    -self.instance.rate
                    * np.asarray(self.instance.exercise_times, dtype=np.float64)
                ),
            )
            values = result.runner.continuation(
                0, np.array([[0.9], [1.1]], dtype=np.float64),
            )
            np.testing.assert_array_equal(values, [0.0, 0.0])

    def test_failed_training_returns_status_without_loading_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = write_candidate(root, "raise SystemExit(9)\n")
            result = self.run_cell(source, root / "failed-cell")
            self.assertEqual(result.status, "crash")
            self.assertEqual(result.returncode, 9)
            self.assertIsNone(result.runner)
            self.assertIsNone(result.policy_artifact_sha256)
            self.assertEqual(len(result.input_bundle_sha256), 64)

    def test_cells_use_fresh_output_and_tmp_and_do_not_share_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = write_candidate(root)
            first = self.run_cell(source, root / "cell-a", seed=11)
            second = self.run_cell(source, root / "cell-b", seed=12)
            self.assertEqual(first.status, "ok", first.log_tail)
            self.assertEqual(second.status, "ok", second.log_tail)
            self.assertEqual(first.input_bundle_sha256, second.input_bundle_sha256)
            self.assertEqual(first.policy_artifact_sha256, second.policy_artifact_sha256)
            self.assertEqual(
                (root / "cell-a" / "tmp" / "fresh-cell.marker").read_text(),
                "11",
            )
            self.assertEqual(
                (root / "cell-b" / "tmp" / "fresh-cell.marker").read_text(),
                "12",
            )
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                self.run_cell(source, root / "cell-a", seed=13)

    def test_training_meta_is_rejected_by_exact_output_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = write_candidate(
                root,
                TRAIN_SCRIPT + '\n(output_dir / "training_meta.json").write_text("{}")\n',
            )
            result = self.run_cell(source, root / "extra-output")
            self.assertEqual(result.status, "invalid_artifact")
            self.assertIn("unexpected file", result.log_tail)
            self.assertIsNone(result.runner)


if __name__ == "__main__":
    unittest.main()
