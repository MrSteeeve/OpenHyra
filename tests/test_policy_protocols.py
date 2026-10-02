import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping import policy_protocols as protocols


def linear_manifest(input_dim=2):
    return {
        "schema": protocols.LINEAR_SCHEMA,
        "runner_type": "linear",
        "inference_config": {
            "input_dim": input_dim,
            "output_dim": 1,
            "output_clip": [-1_000_000.0, 1_000_000.0],
        },
        "output_semantics": protocols.OUTPUT_SEMANTICS,
        "normalization": "per_step",
        "weight_pattern": protocols.LINEAR_WEIGHT_PATTERN,
    }


def expression_manifest(input_dim=1, normalization="none"):
    return {
        "schema": protocols.EXPRESSION_SCHEMA,
        "runner_type": "expression",
        "inference_config": {
            "input_dim": input_dim,
            "output_dim": 1,
            "output_clip": [-1_000_000.0, 1_000_000.0],
        },
        "output_semantics": protocols.OUTPUT_SEMANTICS,
        "normalization": normalization,
        "weight_pattern": protocols.EXPRESSION_WEIGHT_PATTERN,
    }


class ProtocolManifestTests(unittest.TestCase):
    def test_linear_and_expression_manifests_are_explicit(self):
        linear = protocols.validate_continuation_manifest(linear_manifest())
        self.assertEqual(linear.runner_type, "linear")
        self.assertEqual(linear.inference_config.input_dim, 2)
        expression = protocols.validate_continuation_manifest(expression_manifest())
        self.assertEqual(expression.runner_type, "expression")
        self.assertEqual(expression.normalization, "none")

    def test_unknown_runner_and_direct_stop_are_not_registered(self):
        unknown = linear_manifest()
        unknown["runner_type"] = "direct_stop"
        with self.assertRaises(ValueError):
            protocols.validate_continuation_manifest(unknown)
        expression = {"op": "should_stop", "value": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step_000.json").write_text(json.dumps(expression))
            (root / "step_001.json").write_text(json.dumps(expression))
            with self.assertRaisesRegex(ValueError, "not supported"):
                protocols.load_expression_artifact(
                    expression_manifest(), root, n_exercise_times=3
                )


class LinearRunnerTests(unittest.TestCase):
    def test_export_accepts_a_two_element_coefficient_vector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_linear_artifact(
                root,
                [[2.0, -1.0], [1.0, 3.0]],
                [
                    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
                    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
                ],
            )
            artifact = protocols.load_linear_artifact(
                linear_manifest(), root, n_exercise_times=3, input_dim=2
            )
            np.testing.assert_array_equal(artifact.steps[0].coefficients, [2.0, -1.0])
            self.assertEqual(artifact.steps[0].bias, 0.0)

    def test_export_load_dispatch_and_fixed_affine_evaluation(self):
        manifest = linear_manifest(input_dim=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_linear_artifact(
                root,
                [
                    (np.array([2.0, -1.0]), 0.5),
                    (np.array([-1.0, 3.0]), -2.0),
                ],
                [
                    (np.array([1.0, 2.0]), np.array([2.0, 4.0])),
                    (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
                ],
            )
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=2
            )
            self.assertIsInstance(runner, protocols.LinearContinuationRunner)
            states = np.array([[3.0, 6.0], [1.0, 2.0]])
            # Step 0 normalized values are [[1,1],[0,0]].
            np.testing.assert_array_equal(runner.continuation(0, states), [1.5, 0.5])
            # Optional instance argument is accepted and ignored by linear.
            np.testing.assert_array_equal(
                runner.continuation(0, states, evaluator.public_suite()[0]), [1.5, 0.5]
            )

    def test_linear_output_is_clipped_and_batch_partition_invariant(self):
        manifest = linear_manifest(input_dim=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_linear_artifact(
                root, [(np.array([1e308]), 0.0), (np.array([0.0]), 0.0)],
                [(np.array([0.0]), np.array([1.0])), (np.array([0.0]), np.array([1.0]))],
            )
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=1
            )
            states = np.array([[2.0], [-2.0]])
            whole = runner.continuation(0, states)
            pieces = np.concatenate([runner.continuation(0, states[:1]), runner.continuation(0, states[1:])])
            np.testing.assert_array_equal(whole, [1e6, -1e6])
            np.testing.assert_array_equal(whole.view(np.uint64), pieces.view(np.uint64))


class ExpressionRunnerTests(unittest.TestCase):
    def test_expression_units_are_converted_at_nonzero_time_and_nonunit_strike(self):
        """Expression terminals align with the evaluator's t0 cash-flow units."""
        instance = evaluator.BSInstance(
            "expression-units", "put", (1.0,), 2.0, 0.10, (0.0,), (0.20,),
            ((1.0,),), 1.0, (0.0, 0.5, 1.0),
        )
        manifest = expression_manifest(input_dim=1, normalization="none")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_expression_artifact(
                root,
                [{"op": "intrinsic"}, {"op": "underlying"}],
            )
            runner = protocols.load_continuation_runner(
                manifest,
                root,
                n_exercise_times=len(instance.exercise_times),
                input_dim=instance.dimension,
                instance=instance,
            )
            discount = np.exp(-instance.rate * instance.exercise_times[1])
            intrinsic_expected = evaluator.payoff(
                np.array([[1.0]]), instance,
            )[0] * discount
            underlying_expected = 1.5 * discount
            np.testing.assert_allclose(
                runner.continuation(1, np.array([[1.0]])),
                [intrinsic_expected],
            )
            np.testing.assert_allclose(
                runner.continuation(1, np.array([[1.5]])),
                [underlying_expected],
            )

    def test_expression_uses_finance_terminals_and_returns_continuation(self):
        instance = evaluator.public_suite()[0]
        manifest = expression_manifest(input_dim=1)
        expression = {
            "op": "add",
            "left": {"op": "intrinsic"},
            "right": {
                "op": "multiply",
                "left": {"op": "underlying"},
                "right": {"op": "time_to_maturity"},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_expression_artifact(
                root, [expression] * (len(instance.exercise_times) - 1)
            )
            runner = protocols.load_continuation_runner(
                manifest,
                root,
                n_exercise_times=len(instance.exercise_times),
                input_dim=instance.dimension,
                instance=instance,
            )
            states = np.array([[0.8], [1.2]])
            actual = runner.continuation(0, states)
            # At t=0: intrinsic=[.2,0], underlying=[.8,1.2], time-to-maturity=1.
            np.testing.assert_allclose(actual, [1.0, 1.2])
            # A caller can override the default instance at each invocation.
            np.testing.assert_allclose(runner.continuation(0, states, instance), actual)

    def test_expression_can_use_optional_per_step_normalization(self):
        manifest = expression_manifest(input_dim=1, normalization="per_step")
        expression = {"op": "spot", "asset": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_expression_artifact(
                root,
                [expression, expression],
                normalizations=[
                    (np.array([1.0]), np.array([2.0])),
                    (np.array([0.0]), np.array([1.0])),
                ],
            )
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=1
            )
            # Normalization is applied before the finance-aware expression.
            np.testing.assert_allclose(runner.continuation(0, np.array([[3.0]])), [1.0])

    def test_expression_normalization_file_is_optional_even_in_per_step_mode(self):
        manifest = expression_manifest(input_dim=1, normalization="per_step")
        expression = {"op": "spot", "asset": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_expression_artifact(root, [expression, expression])
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=1
            )
            np.testing.assert_array_equal(runner.continuation(0, np.array([[2.0]])), [2.0])

    def test_expression_rejects_unavailable_asset(self):
        manifest = expression_manifest(input_dim=1)
        expression = {"op": "spot", "asset": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocols.export_expression_artifact(root, [expression, expression])
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=1
            )
            with self.assertRaisesRegex(ValueError, "unavailable asset"):
                runner.continuation(0, np.array([[1.0]]))


class MLPDispatchTests(unittest.TestCase):
    def test_existing_mlp_protocol_dispatches_through_common_surface(self):
        manifest = {
            "schema": "openhyra-policy-spec.v1",
            "runner_type": "mlp",
            "inference_config": {
                "input_dim": 1,
                "layers": [],
                "activation": "tanh",
                "output_dim": 1,
                "output_clip": [-1_000_000.0, 1_000_000.0],
            },
            "output_semantics": protocols.OUTPUT_SEMANTICS,
            "normalization": "per_step",
            "weight_pattern": "step_{:03d}.npy",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "step_000.npy", np.array([2.0, 1.0]))
            np.save(root / "step_001.npy", np.array([3.0, -1.0]))
            (root / "normalization.json").write_text(
                json.dumps({"steps": [
                    {"mean": [0.0], "scale": [1.0]},
                    {"mean": [0.0], "scale": [1.0]},
                ]})
            )
            runner = protocols.load_continuation_runner(
                manifest, root, n_exercise_times=3, input_dim=1
            )
            np.testing.assert_array_equal(
                runner.continuation(0, np.array([[2.0]]), evaluator.public_suite()[0]), [5.0]
            )


if __name__ == "__main__":
    unittest.main()
