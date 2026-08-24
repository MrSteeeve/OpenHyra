import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tasks" / "bermudan_optimal_stopping" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("bermudan_evaluator", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


def tiny_request(stage="search", seed=20260813):
    config = {
        "direction": "max",
        "metric": "paired_lower_bound_lcb",
        "confidence_level": 0.95,
        "instance_count": 2,
        "repeats": 1,
        "training_paths": 192,
        "pricing_paths": 384,
        "ridge_alpha": 1e-6,
    }
    if stage == "audit":
        config.update({
            "direction": "min",
            "metric": "normalized_primal_dual_confidence_gap",
            "instance_count": 2,
            "outer_paths": 128,
            "inner_paths": 8,
        })
    return {
        "schema": EVALUATOR.REQUEST_SCHEMA,
        "stage": stage,
        "task": EVALUATOR.TASK_NAME,
        "protocol": EVALUATOR.TASK_PROTOCOL,
        "seed": seed,
        "suite_id": f"test-{stage}-v1",
        "config": config,
    }


class FeatureIRTests(unittest.TestCase):
    def test_typed_feature_interpreter_is_vectorized(self):
        program = EVALUATOR.validate_feature_program({
            "schema": EVALUATOR.FEATURE_SCHEMA,
            "features": [
                {"op": "constant", "value": 1},
                {"op": "spot", "asset": 0},
                {"op": "multiply", "left": {"op": "intrinsic"}, "right": {"op": "time_to_maturity"}},
            ],
        })
        instance = EVALUATOR.public_suite()[0]
        states = np.array([[0.8], [1.2]])
        features = EVALUATOR.evaluate_features(program, 2, states, instance)
        self.assertEqual(features.shape, (2, 3))
        np.testing.assert_allclose(features[:, 0], 1.0)
        np.testing.assert_allclose(features[:, 1], [0.8, 1.2])
        self.assertTrue(np.all(np.isfinite(features)))

    def test_feature_ir_rejects_unknown_fields_depth_and_unavailable_asset(self):
        with self.assertRaisesRegex(ValueError, "unknown field"):
            EVALUATOR.validate_feature_program({
                "schema": EVALUATOR.FEATURE_SCHEMA,
                "features": [{"op": "time", "leak": "future"}],
            })
        nested = {"op": "underlying"}
        for _ in range(EVALUATOR.MAX_AST_DEPTH):
            nested = {"op": "square", "arg": nested}
        with self.assertRaisesRegex(ValueError, "depth"):
            EVALUATOR.validate_feature_program({"schema": EVALUATOR.FEATURE_SCHEMA, "features": [nested]})
        program = EVALUATOR.validate_feature_program({
            "schema": EVALUATOR.FEATURE_SCHEMA,
            "features": [{"op": "spot", "asset": 3}],
        })
        with self.assertRaisesRegex(ValueError, "unavailable asset"):
            EVALUATOR.evaluate_features(program, 0, np.ones((2, 1)), EVALUATOR.public_suite()[0])


class FinancialKernelTests(unittest.TestCase):
    def setUp(self):
        self.european_put = EVALUATOR.BSInstance(
            "euro", "put", (100.0,), 100.0, 0.05, (0.02,), (0.20,),
            ((1.0,),), 1.0, (0.0, 1.0),
        )

    def test_correlated_risk_neutral_simulation(self):
        instance = EVALUATOR.public_suite()[2]
        paths = EVALUATOR.simulate_paths(instance, 120_000, 44)
        terminal = paths[:, -1, :]
        expected = np.asarray(instance.spots) * np.exp((instance.rate - np.asarray(instance.dividends)) * instance.maturity)
        np.testing.assert_allclose(terminal.mean(axis=0), expected, rtol=0.006)
        standardized = (
            np.log(terminal / np.asarray(instance.spots))
            - (instance.rate - np.asarray(instance.dividends) - 0.5 * np.asarray(instance.volatilities) ** 2) * instance.maturity
        ) / (np.asarray(instance.volatilities) * math.sqrt(instance.maturity))
        self.assertAlmostEqual(float(np.corrcoef(standardized.T)[0, 1]), 0.30, delta=0.012)

    def test_european_bs_mc_and_crr_cross_validation(self):
        analytic = EVALUATOR.black_scholes_european_price(100, 100, 0.05, 0.02, 0.20, 1.0, "put")
        crr = EVALUATOR.crr_price(self.european_put, 1000, exercise="european")
        paths = EVALUATOR.simulate_paths(self.european_put, 300_000, 7)
        samples = EVALUATOR.discounted_rewards(paths, self.european_put)[:, -1]
        mc, se = EVALUATOR._mean_se(samples)
        self.assertAlmostEqual(crr, analytic, delta=0.01)
        self.assertLess(abs(mc - analytic), 4.0 * se + 0.005)

    def test_european_leq_bermudan_leq_american_tree(self):
        bermudan = EVALUATOR.BSInstance(
            "berm", "put", (100.0,), 100.0, 0.06, (0.0,), (0.25,),
            ((1.0,),), 1.0, tuple(np.linspace(0.0, 1.0, 5)),
        )
        european = EVALUATOR.crr_price(bermudan, 800, exercise="european")
        bermudan_value = EVALUATOR.crr_price(bermudan, 800, exercise="bermudan")
        american = EVALUATOR.crr_price(bermudan, 800, exercise="american")
        self.assertLessEqual(european, bermudan_value + 1e-12)
        self.assertLessEqual(bermudan_value, american + 1e-12)
        self.assertGreater(american, european + 0.01)

    def test_all_payoff_types(self):
        put, max_call, basket = EVALUATOR.public_suite()[0], EVALUATOR.public_suite()[2], EVALUATOR.public_suite()[3]
        np.testing.assert_allclose(EVALUATOR.payoff(np.array([[0.8], [1.1]]), put), [0.2, 0.0])
        np.testing.assert_allclose(EVALUATOR.payoff(np.array([[0.8, 1.3], [0.5, 0.7]]), max_call), [0.3, 0.0])
        state = np.array([[0.5, 1.0, 1.5]])
        expected_basket = max(1.0 - (0.3 * 0.5 + 0.4 * 1.0 + 0.3 * 1.5), 0.0)
        np.testing.assert_allclose(EVALUATOR.payoff(state, basket), [expected_basket])


class LSMCAndDualTests(unittest.TestCase):
    def test_fit_freeze_and_oos_lower_bound_are_causal(self):
        instance = EVALUATOR.public_suite()[0]
        training = EVALUATOR.simulate_paths(instance, 512, 10)
        pricing = EVALUATOR.simulate_paths(instance, 1024, 11)
        policy = EVALUATOR.fit_lsmc(EVALUATOR.BASELINE_PROGRAM, instance, training)
        original_coefficients = [step.coefficients.copy() for step in policy.steps]
        samples, stops = EVALUATOR.apply_policy(policy, pricing)
        self.assertEqual(samples.shape, (1024,))
        self.assertTrue(np.all((0 <= stops) & (stops < len(instance.exercise_times))))
        for before, step in zip(original_coefficients, policy.steps):
            np.testing.assert_array_equal(before, step.coefficients)

    def test_dual_increment_conditional_mean_sanity(self):
        instance = EVALUATOR.public_suite()[0]
        training = EVALUATOR.simulate_paths(instance, 1024, 1)
        policy = EVALUATOR.fit_lsmc(EVALUATOR.BASELINE_PROGRAM, instance, training)
        previous = np.full((5000, 1), 0.9)
        dt = instance.exercise_times[1] - instance.exercise_times[0]
        rng = np.random.default_rng(99)
        paired = EVALUATOR.simulate_conditional_next(previous, dt, instance, 2, rng)
        outer_value = policy.approximate_value(1, paired[:, 0, :])
        inner_value = policy.approximate_value(1, paired[:, 1, :])
        differences = outer_value - inner_value
        mean, se = EVALUATOR._mean_se(differences)
        self.assertLess(abs(mean), 4.0 * se + 1e-4)

    def test_dual_samples_report_unclipped_bound_diagnostics(self):
        result = EVALUATOR.evaluate_submission(EVALUATOR.BASELINE_PROGRAM, tiny_request("audit", 31))
        score, metrics, normalized, evidence = result
        self.assertTrue(math.isfinite(score))
        self.assertEqual(normalized, EVALUATOR.BASELINE_PROGRAM)
        self.assertIn("raw_bound_order_all_ok", metrics)
        self.assertTrue(all("raw_primal_dual_gap" in row for row in metrics["summaries"]))
        self.assertFalse(evidence["audit"]["negative_raw_gaps_clipped"])
        self.assertEqual(evidence["audit"]["martingale"]["m0"], 0.0)
        self.assertEqual(metrics["confidence_construction"], "bonferroni_one_sided_components")
        expected_z = EVALUATOR.NormalDist().inv_cdf(1.0 - (1.0 - 0.95) / 2.0)
        self.assertAlmostEqual(metrics["confidence_component_z"], expected_z)


class EvaluationProtocolTests(unittest.TestCase):
    def test_fixed_seed_reproduces_search_and_audit(self):
        for stage in ("search", "audit"):
            first = EVALUATOR.evaluate_submission(EVALUATOR.BASELINE_PROGRAM, tiny_request(stage, 71))
            second = EVALUATOR.evaluate_submission(EVALUATOR.BASELINE_PROGRAM, tiny_request(stage, 71))
            self.assertEqual(first[0], second[0])
            first_metrics, second_metrics = dict(first[1]), dict(second[1])
            first_metrics.pop("runtime_seconds")
            second_metrics.pop("runtime_seconds")
            self.assertEqual(first_metrics, second_metrics)
            self.assertEqual(first[2:], second[2:])

    def test_baseline_search_has_exact_zero_paired_score(self):
        score, metrics, _, evidence = EVALUATOR.evaluate_submission(EVALUATOR.BASELINE_PROGRAM, tiny_request("search", 55))
        self.assertEqual(score, 0.0)
        self.assertEqual(metrics["mean_paired_normalized_improvement"], 0.0)
        self.assertEqual(metrics["paired_aggregate_standard_error"], 0.0)
        self.assertEqual(metrics["estimator_scope"], "fixed_public_suite_mean")
        self.assertTrue(evidence["search"]["common_random_numbers"])

    def test_fixed_suite_lcb_aggregates_path_level_cell_errors(self):
        means = [0.01, 0.03, -0.005]
        errors = [0.004, 0.003, 0.012]
        lcb, mean, aggregate_se = EVALUATOR._fixed_suite_lcb(means, errors, 0.95)
        expected_mean = sum(means) / len(means)
        expected_se = math.sqrt(sum(value * value for value in errors)) / len(errors)
        z = EVALUATOR.NormalDist().inv_cdf(0.975)
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(aggregate_se, expected_se)
        self.assertAlmostEqual(lcb, expected_mean - z * expected_se)

    def test_request_is_strict_and_hidden_suite_depends_on_seed(self):
        request = tiny_request()
        request["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown field"):
            EVALUATOR.validate_evaluation_request(request)
        first = EVALUATOR.derive_hidden_suite(1, 3)
        repeated = EVALUATOR.derive_hidden_suite(1, 3)
        different = EVALUATOR.derive_hidden_suite(2, 3)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_cli_accepts_optional_request_file_and_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "feature.json"
            request = root / "request.json"
            artifact.write_text(json.dumps(EVALUATOR.BASELINE_PROGRAM))
            request.write_text(json.dumps(tiny_request("search", 9)))
            result = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(artifact), str(request)],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["metrics"]["stage"], "search")
            self.assertEqual(payload["metrics"]["evaluation_request_sha256"], EVALUATOR._sha256_json(EVALUATOR.validate_evaluation_request(tiny_request("search", 9))[0]))
            artifact.write_text('{"schema":"openhyra-feature-program.v1","features":[],"features":[]}')
            rejected = subprocess.run(
                [sys.executable, str(EVALUATOR_PATH), str(artifact)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("duplicate JSON key", rejected.stdout)

    def test_seed_solution_runs_and_task_contract_matches(self):
        task_dir = ROOT / "tasks" / EVALUATOR.TASK_NAME
        task = json.loads((task_dir / "task.json").read_text())
        self.assertEqual(task["protocol"], EVALUATOR.TASK_PROTOCOL)
        self.assertEqual(task["editable_files"], ["feature_program.json"])
        self.assertEqual(task["evaluation"]["audit_stage"]["direction"], "min")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "feature_program.json").write_bytes((task_dir / "seed_solution" / "feature_program.json").read_bytes())
            (destination / "solve.sh").write_bytes((task_dir / "seed_solution" / "solve.sh").read_bytes())
            subprocess.run(["bash", "solve.sh"], cwd=destination, check=True)
            EVALUATOR.validate_feature_program(json.loads((destination / "solution.json").read_text()))


if __name__ == "__main__":
    unittest.main()
