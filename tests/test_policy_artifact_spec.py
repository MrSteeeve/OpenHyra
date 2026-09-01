import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tasks" / "bermudan_optimal_stopping" / "policy_artifact.py"
)
SPEC_PATH = (
    ROOT / "tasks" / "bermudan_optimal_stopping" / "POLICY_ARTIFACT_SPEC_V1.md"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bermudan_policy_artifact_spec_contract", MODULE_PATH,
)
POLICY = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = POLICY
MODULE_SPEC.loader.exec_module(POLICY)


def _documented_manifest(text):
    begin = "<!-- BEGIN CANONICAL MANIFEST EXAMPLE -->"
    end = "<!-- END CANONICAL MANIFEST EXAMPLE -->"
    block = text.split(begin, 1)[1].split(end, 1)[0]
    return json.loads(block.split("```json", 1)[1].split("```", 1)[0])


class PolicyArtifactSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SPEC_PATH.read_text(encoding="utf-8")

    def test_manifest_example_and_normative_constants_match_loader(self):
        raw = _documented_manifest(self.text)
        manifest = POLICY.validate_policy_manifest(raw)
        self.assertEqual(manifest.schema, POLICY.POLICY_SCHEMA)
        self.assertEqual(manifest.runner_type, "mlp")
        self.assertEqual(manifest.output_semantics, POLICY.OUTPUT_SEMANTICS)
        self.assertEqual(manifest.normalization, POLICY.NORMALIZATION_MODE)
        self.assertEqual(manifest.weight_pattern, POLICY.WEIGHT_PATTERN)
        self.assertEqual(
            manifest.inference_config.output_clip,
            POLICY.PROTOCOL_OUTPUT_CLIP,
        )
        self.assertEqual(manifest.inference_config.activation, "tanh")
        for activation in POLICY.SUPPORTED_ACTIVATIONS:
            self.assertIn(f'`"{activation}"`', self.text)

    def test_documented_limits_match_loader(self):
        expected = {
            "`manifest.json`": POLICY.MAX_MANIFEST_BYTES,
            "`normalization.json`": POLICY.MAX_NORMALIZATION_BYTES,
            "each `step_XXX.npy`": POLICY.MAX_STEP_FILE_BYTES,
            "manifest plus complete artifact bundle": (
                POLICY.MAX_ARTIFACT_BUNDLE_BYTES
            ),
        }
        for label, limit in expected.items():
            with self.subTest(label=label):
                row = f"| {label} | {limit:,} bytes |"
                self.assertIn(row, self.text)
        self.assertIn(
            f"| parameters per step | at most {POLICY.MAX_PARAMETERS_PER_STEP:,} |",
            self.text,
        )
        self.assertIn(
            f"Every scale value is strictly greater than "
            f"`{POLICY.NORMALIZATION_EPSILON:g}`.",
            self.text,
        )

    def test_documented_flattening_order_matches_trusted_splitter(self):
        raw = _documented_manifest(self.text)
        raw["inference_config"]["input_dim"] = 2
        raw["inference_config"]["layers"] = [2]
        manifest = POLICY.validate_policy_manifest(raw)
        flat = np.arange(1, 10, dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step_index in range(2):
                np.save(
                    root / f"step_{step_index:03d}.npy",
                    flat,
                    allow_pickle=False,
                )
            (root / "normalization.json").write_text(
                json.dumps({
                    "steps": [
                        {"mean": [0.0, 0.0], "scale": [1.0, 1.0]},
                        {"mean": [0.0, 0.0], "scale": [1.0, 1.0]},
                    ],
                }),
                encoding="utf-8",
            )
            artifact = POLICY.load_policy_artifact(
                manifest, root, n_exercise_times=3,
            )
            first = artifact.steps[0]
            np.testing.assert_array_equal(
                first.layers[0].weights, [[1.0, 2.0], [3.0, 4.0]],
            )
            np.testing.assert_array_equal(first.layers[0].bias, [5.0, 6.0])
            np.testing.assert_array_equal(
                first.layers[1].weights, [[7.0, 8.0]],
            )
            np.testing.assert_array_equal(first.layers[1].bias, [9.0])

    def test_v1_explicitly_rejects_training_meta_and_manifest_in_output(self):
        self.assertIn("`training_meta.json`: **NOT ACCEPTED in v1**", self.text)
        self.assertIn(
            "`manifest.json`: **NOT ACCEPTED inside the per-instance output directory**",
            self.text,
        )
        raw = _documented_manifest(self.text)
        raw["inference_config"]["input_dim"] = 1
        raw["inference_config"]["layers"] = []
        manifest = POLICY.validate_policy_manifest(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(
                root / "step_000.npy",
                np.array([1.0, 0.0], dtype=np.float64),
                allow_pickle=False,
            )
            (root / "normalization.json").write_text(
                '{"steps":[{"mean":[0.0],"scale":[1.0]}]}',
                encoding="utf-8",
            )
            POLICY.load_policy_artifact(
                manifest, root, n_exercise_times=2,
            )
            (root / "training_meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                POLICY.load_policy_artifact(
                    manifest, root, n_exercise_times=2,
                )


if __name__ == "__main__":
    unittest.main()
