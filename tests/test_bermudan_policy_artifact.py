import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tasks" / "bermudan_optimal_stopping" / "policy_artifact.py"
)
SPEC = importlib.util.spec_from_file_location("bermudan_policy_artifact", MODULE_PATH)
POLICY = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


def manifest(*, input_dim=2, layers=None, activation="tanh"):
    return {
        "schema": POLICY.POLICY_SCHEMA,
        "runner_type": "mlp",
        "inference_config": {
            "input_dim": input_dim,
            "layers": [2] if layers is None else layers,
            "activation": activation,
            "output_dim": 1,
            "output_clip": [-1_000_000.0, 1_000_000.0],
        },
        "output_semantics": POLICY.OUTPUT_SEMANTICS,
        "normalization": POLICY.NORMALIZATION_MODE,
        "weight_pattern": POLICY.WEIGHT_PATTERN,
    }


def parameter_count(input_dim=2, layers=(2,)):
    widths = (input_dim, *layers, 1)
    return sum(
        output_width * input_width + output_width
        for input_width, output_width in zip(widths, widths[1:])
    )


def write_artifact(
    root,
    *,
    steps=2,
    input_dim=2,
    layers=(2,),
    weights=None,
    normalization=None,
):
    root = Path(root)
    count = parameter_count(input_dim, layers)
    values = np.arange(1, count + 1, dtype=np.float64) / 10.0
    if weights is not None:
        values = weights
    for index in range(steps):
        np.save(root / f"step_{index:03d}.npy", values)
    if normalization is None:
        normalization = {
            "steps": [
                {"mean": [0.0] * input_dim, "scale": [1.0] * input_dim}
                for _ in range(steps)
            ]
        }
    (root / "normalization.json").write_text(json.dumps(normalization))


class ManifestTests(unittest.TestCase):
    def test_manifest_is_strict_and_protocol_semantics_are_fixed(self):
        validated = POLICY.validate_policy_manifest(manifest(input_dim="n_assets"))
        self.assertEqual(validated.schema, "openhyra-policy-spec.v1")
        self.assertEqual(validated.inference_config.input_dim, "n_assets")
        self.assertEqual(validated.inference_config.output_clip, POLICY.PROTOCOL_OUTPUT_CLIP)

        cases = []
        unknown = manifest()
        unknown["candidate_note"] = "trust me"
        cases.append((unknown, "unknown field"))
        runner = manifest()
        runner["runner_type"] = "onnx"
        cases.append((runner, "runner_type"))
        semantics = manifest()
        semantics["output_semantics"] = "undiscounted"
        cases.append((semantics, "output_semantics"))
        clip = manifest()
        clip["inference_config"]["output_clip"] = [-1e300, 1e300]
        cases.append((clip, "protocol limit"))
        pattern = manifest()
        pattern["weight_pattern"] = "../step_{}.npy"
        cases.append((pattern, "weight_pattern"))
        for payload, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                POLICY.validate_policy_manifest(payload)

    def test_direct_policy_manifest_instances_are_fully_revalidated(self):
        valid = POLICY.validate_policy_manifest(manifest())
        forged_runner = POLICY.PolicyManifest(
            schema=valid.schema,
            runner_type="onnx",
            inference_config=valid.inference_config,
            output_semantics=valid.output_semantics,
            normalization=valid.normalization,
            weight_pattern=valid.weight_pattern,
        )
        with self.assertRaisesRegex(ValueError, "runner_type"):
            POLICY.validate_policy_manifest(forged_runner)

        forged_config = POLICY.PolicyManifest(
            schema=valid.schema,
            runner_type=valid.runner_type,
            inference_config=POLICY.MLPInferenceConfig(
                input_dim=0,
                layers=(2,),
                activation="tanh",
                output_dim=1,
                output_clip=POLICY.PROTOCOL_OUTPUT_CLIP,
            ),
            output_semantics=valid.output_semantics,
            normalization=valid.normalization,
            weight_pattern=valid.weight_pattern,
        )
        with self.assertRaisesRegex(ValueError, "input_dim"):
            POLICY.validate_policy_manifest(forged_config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            with self.assertRaisesRegex(ValueError, "runner_type"):
                POLICY.load_policy_artifact(
                    forged_runner, root, n_exercise_times=3,
                )

    def test_manifest_file_rejects_duplicate_keys_symlinks_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                '{"schema":"openhyra-policy-spec.v1","schema":"duplicate"}'
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                POLICY.load_policy_manifest(manifest_path)

            target = root / "target.json"
            target.write_text(json.dumps(manifest()))
            manifest_path.unlink()
            manifest_path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                POLICY.load_policy_manifest(manifest_path)
            manifest_path.unlink()
            os.link(target, manifest_path)
            with self.assertRaisesRegex(ValueError, "hard link"):
                POLICY.load_policy_manifest(manifest_path)


class ArtifactLoadingTests(unittest.TestCase):
    def test_manifest_entry_forms_share_one_canonical_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            write_artifact(artifact_dir)

            mapping = manifest()
            validated = POLICY.validate_policy_manifest(mapping)
            reordered = dict(reversed(list(mapping.items())))
            reordered["inference_config"] = dict(reversed(list(
                mapping["inference_config"].items()
            )))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(reordered, indent=7))

            from_mapping = POLICY.load_policy_artifact(
                mapping, artifact_dir, n_exercise_times=3,
            )
            from_validated = POLICY.load_policy_artifact(
                validated, artifact_dir, n_exercise_times=3,
            )
            from_path = POLICY.load_policy_artifact(
                manifest_path, artifact_dir, n_exercise_times=3,
            )

            self.assertEqual(from_mapping.file_sha256, from_validated.file_sha256)
            self.assertEqual(from_mapping.file_sha256, from_path.file_sha256)
            self.assertEqual(from_mapping.bundle_sha256, from_validated.bundle_sha256)
            self.assertEqual(from_mapping.bundle_sha256, from_path.bundle_sha256)

    def test_loads_splits_hashes_and_freezes_canonical_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            artifact = POLICY.load_policy_artifact(
                manifest(), root, n_exercise_times=3,
            )
            self.assertEqual(len(artifact.steps), 2)
            self.assertEqual(artifact.parameter_count_per_step, 9)
            first = artifact.steps[0]
            self.assertEqual(first.layers[0].weights.shape, (2, 2))
            self.assertEqual(first.layers[0].bias.shape, (2,))
            self.assertEqual(first.layers[1].weights.shape, (1, 2))
            self.assertEqual(first.layers[1].bias.shape, (1,))
            np.testing.assert_array_equal(
                first.layers[0].weights, [[0.1, 0.2], [0.3, 0.4]],
            )
            np.testing.assert_array_equal(first.layers[0].bias, [0.5, 0.6])
            with self.assertRaises(ValueError):
                first.layers[0].weights.setflags(write=True)
            with self.assertRaises(FrozenInstanceError):
                artifact.input_dim = 7

            hashes = dict(artifact.file_sha256)
            self.assertEqual(
                hashes["step_000.npy"],
                hashlib.sha256((root / "step_000.npy").read_bytes()).hexdigest(),
            )
            repeated = POLICY.load_policy_artifact(
                manifest(), root, n_exercise_times=3,
            )
            self.assertEqual(artifact.file_sha256, repeated.file_sha256)
            self.assertEqual(artifact.bundle_sha256, repeated.bundle_sha256)
            changed_weights = np.arange(9, dtype=np.float64)
            np.save(root / "step_001.npy", changed_weights)
            changed = POLICY.load_policy_artifact(
                manifest(), root, n_exercise_times=3,
            )
            self.assertNotEqual(artifact.bundle_sha256, changed.bundle_sha256)

    def test_rejects_wrong_file_set_symlink_hardlink_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            (root / "unexpected.txt").write_text("x")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)
            (root / "unexpected.txt").unlink()
            (root / "step_001.npy").unlink()
            with self.assertRaisesRegex(ValueError, "missing file"):
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            target = root.parent / f"{root.name}-target.npy"
            (root / "step_000.npy").replace(target)
            try:
                (root / "step_000.npy").symlink_to(target)
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)
            finally:
                if target.exists():
                    target.unlink()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            target = root / "step_000.npy"
            alias = root / "alias.npy"
            os.link(target, alias)
            alias.unlink()
            # A second link outside the artifact still makes the candidate file unsafe.
            outside = root.parent / f"{root.name}-hardlink.npy"
            try:
                os.link(target, outside)
                with self.assertRaisesRegex(ValueError, "hard link"):
                    POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)
            finally:
                if outside.exists():
                    outside.unlink()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            oversized = root / "step_000.npy"
            with oversized.open("wb") as stream:
                stream.truncate(POLICY.MAX_STEP_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "byte limit"):
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)

    def test_rejects_noncanonical_or_malicious_npy_inputs(self):
        cases = {
            "two-dimensional": np.zeros((3, 3), dtype=np.float64),
            "float32": np.zeros(9, dtype=np.float32),
            "object": np.array([object()] * 9, dtype=object),
            "nan": np.array([np.nan] + [0.0] * 8, dtype=np.float64),
            "infinity": np.array([np.inf] + [0.0] * 8, dtype=np.float64),
        }
        if np.little_endian:
            cases["non-native"] = np.zeros(9, dtype=">f8")
        else:
            cases["non-native"] = np.zeros(9, dtype="<f8")
        for name, array in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_artifact(root)
                np.save(root / "step_000.npy", array)
                with self.assertRaises(ValueError):
                    POLICY.load_policy_artifact(
                        manifest(), root, n_exercise_times=3,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            with (root / "step_000.npy").open("ab") as stream:
                stream.write(b"hidden trailing payload")
            with self.assertRaisesRegex(ValueError, "trailing data"):
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)

    def test_normalization_is_strict_bounded_and_finite(self):
        invalid = [
            ({"steps": [{"mean": [0.0], "scale": [1.0]}] * 2}, "mean length"),
            ({"steps": [{"mean": [0.0, 0.0], "scale": [1.0, 0.0]}] * 2}, "greater than"),
            ({"steps": [{"mean": [0.0, float("nan")], "scale": [1.0, 1.0]}] * 2}, "non-finite JSON"),
            ({"steps": [{"mean": [0.0, 0.0], "scale": [1.0, 1.0], "extra": 1}] * 2}, "unknown field"),
            ({"steps": []}, "exactly 2"),
        ]
        for payload, error in invalid:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_artifact(root, normalization=payload)
                with self.assertRaisesRegex(ValueError, error):
                    POLICY.load_policy_artifact(
                        manifest(), root, n_exercise_times=3,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            with (root / "normalization.json").open("wb") as stream:
                stream.truncate(POLICY.MAX_NORMALIZATION_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "byte limit"):
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)


class RunnerTests(unittest.TestCase):
    def test_runner_normalizes_per_step_and_matches_manual_mlp(self):
        # W1=[[1,2],[-1,.5]], b1=[.25,-.5], W2=[[3,-2]], b2=[.75]
        flat = np.array([1, 2, -1, 0.5, 0.25, -0.5, 3, -2, 0.75], dtype=np.float64)
        normalization = {
            "steps": [
                {"mean": [1.0, 2.0], "scale": [2.0, 4.0]},
                {"mean": [-1.0, 0.0], "scale": [1.0, 2.0]},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, weights=flat, normalization=normalization)
            artifact = POLICY.load_policy_artifact(
                manifest(), root, n_exercise_times=3,
            )
            runner = POLICY.MLPContinuationRunner(artifact)
            states = np.array([[[1.0, 2.0], [3.0, 6.0]], [[5.0, 10.0], [0.0, -2.0]]])
            actual = runner.continuation(0, states)
            normalized = (states - np.array([1.0, 2.0])) / np.array([2.0, 4.0])
            hidden = np.tanh(
                normalized @ np.array([[1.0, 2.0], [-1.0, 0.5]]).T
                + np.array([0.25, -0.5])
            )
            expected = hidden @ np.array([[3.0, -2.0]]).T + 0.75
            np.testing.assert_array_equal(actual, expected[..., 0])
            self.assertEqual(actual.shape, states.shape[:-1])
            np.testing.assert_array_equal(actual, runner.continuation(0, states))

    def test_runner_is_stateless_call_order_independent_and_protocol_clipped(self):
        flat = np.array([1e308, 0.0, 0.0], dtype=np.float64)
        linear_manifest = manifest(input_dim=1, layers=[])
        normalization = {
            "steps": [
                {"mean": [0.0], "scale": [1.0]},
                {"mean": [10.0], "scale": [2.0]},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(
                root,
                input_dim=1,
                layers=(),
                weights=flat[:2],
                normalization=normalization,
            )
            artifact = POLICY.load_policy_artifact(
                linear_manifest, root, n_exercise_times=3,
            )
            runner = POLICY.MLPContinuationRunner(artifact)
            states = np.array([[2.0], [-2.0]])
            first = runner.continuation(0, states)
            runner.continuation(1, np.array([[12.0]]))
            repeated = runner.continuation(0, states)
            np.testing.assert_array_equal(first, repeated)
            np.testing.assert_array_equal(first, [1e6, -1e6])

    def test_extreme_cancellation_is_bit_exact_across_batch_partitions(self):
        # Fixed left-to-right accumulation deliberately loses the unit term in
        # 1e16 + 1 - 1e16.  The scientific contract here is not a preferred
        # summation algorithm; it is identical bits regardless of batch shape.
        weights = np.array([
            1e16, 1.0, -1e16,
            -1e16, 1.0, 1e16,
            0.0, 0.0,
            3.0, -2.0,
            0.5,
        ], dtype=np.float64)
        cancellation_manifest = manifest(input_dim=3, layers=[2])
        states = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [0.5, 1.0, 0.5],
            [1.0, 3.0, np.nextafter(1.0, 0.0)],
        ], dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(
                root,
                input_dim=3,
                layers=(2,),
                weights=weights,
            )
            runner = POLICY.MLPContinuationRunner(
                POLICY.load_policy_artifact(
                    cancellation_manifest, root, n_exercise_times=3,
                )
            )
            whole = runner.continuation(0, states)
            partitioned = np.concatenate([
                runner.continuation(0, states[:1]),
                runner.continuation(0, states[1:4]),
                runner.continuation(0, states[4:]),
            ])
            singleton = np.asarray([
                runner.continuation(0, state) for state in states
            ], dtype=np.float64)
            reversed_order = np.asarray([
                runner.continuation(0, state) for state in states[::-1]
            ], dtype=np.float64)[::-1]

            expected_bits = whole.view(np.uint64)
            np.testing.assert_array_equal(partitioned.view(np.uint64), expected_bits)
            np.testing.assert_array_equal(singleton.view(np.uint64), expected_bits)
            np.testing.assert_array_equal(reversed_order.view(np.uint64), expected_bits)
            self.assertEqual(whole[0].view(np.uint64), np.float64(0.5).view(np.uint64))

    def test_runner_rejects_bad_time_shape_dtype_and_nonfinite_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            runner = POLICY.MLPContinuationRunner(
                POLICY.load_policy_artifact(manifest(), root, n_exercise_times=3)
            )
            invalid_calls = [
                (2, np.ones((1, 2)), "time_index"),
                (0, np.ones((1, 3)), "shape"),
                (0, np.array([["1", "2"]]), "numeric"),
                (0, np.array([[np.nan, 1.0]]), "finite"),
            ]
            for time_index, states, error in invalid_calls:
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    runner.continuation(time_index, states)


if __name__ == "__main__":
    unittest.main()
