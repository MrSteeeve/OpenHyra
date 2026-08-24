import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tasks" / "sums_diffs" / "research_verifier.py"
SPEC = importlib.util.spec_from_file_location(
    "sums_diffs_research_verifier",
    VERIFIER_PATH,
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _valid_payload():
    return {
        "schema": VERIFIER.SCHEMA,
        "base": 10,
        "digits": [0, 1, 3],
        "levels": 3,
        "check_levels": [1, 2, 3],
        "obligations": [
            {"id": "NC", "type": "sum_no_carry"},
            {"id": "SB", "type": "signed_digit_bound"},
            {
                "id": "SP",
                "type": "product_formula",
                "operation": "sum",
            },
            {
                "id": "DP",
                "type": "product_formula",
                "operation": "difference",
            },
            {
                "id": "L2",
                "type": "level_counts",
                "level": 2,
                "expected_n": 9,
                "expected_sum_count": 36,
                "expected_diff_count": 49,
            },
        ],
    }


class DigitProductPositiveTests(unittest.TestCase):
    def test_parameterized_construction_and_exact_counts(self):
        evidence = VERIFIER.verify_digit_product(_valid_payload())

        self.assertEqual(evidence["schema"], VERIFIER.EVIDENCE_SCHEMA)
        self.assertEqual(evidence["status"], "bounded_checked")
        self.assertEqual(
            evidence["scope"]["exactly_checked_levels"],
            [1, 2, 3],
        )
        self.assertFalse(evidence["scope"]["asymptotic_claims_checked"])
        self.assertEqual(
            evidence["digit_counts"],
            {"digits": 3, "digit_sums": 6, "digit_differences": 7},
        )

        level_two = evidence["levels"][1]
        self.assertEqual(
            level_two["A"],
            [0, 1, 3, 10, 11, 13, 30, 31, 33],
        )
        self.assertEqual(
            (level_two["n"], level_two["sums"], level_two["diffs"]),
            (9, 36, 49),
        )
        self.assertTrue(level_two["sum_product_formula_holds"])
        self.assertTrue(level_two["diff_product_formula_holds"])

        level_three = evidence["levels"][2]
        self.assertEqual(
            (level_three["n"], level_three["sums"], level_three["diffs"]),
            (27, 216, 343),
        )
        self.assertEqual(
            evidence["summary"],
            {
                "obligation_count": 5,
                "bounded_checked_count": 5,
                "refuted_count": 0,
            },
        )
        self.assertIsNone(evidence["minimal_counterexample"])
        self.assertEqual(len(evidence["construction_sha256"]), 64)
        self.assertEqual(len(evidence["evidence_sha256"]), 64)

    def test_public_generation_and_count_primitives(self):
        values = VERIFIER.generate_level(10, [3, 0, 1], 2)
        self.assertEqual(
            values,
            (0, 1, 3, 10, 11, 13, 30, 31, 33),
        )
        counts = VERIFIER.exact_sum_diff_counts(values)
        self.assertEqual(counts["n"], 9)
        self.assertEqual(counts["sums"], 36)
        self.assertEqual(counts["diffs"], 49)

        conditions = VERIFIER.check_no_carry_conditions(10, [0, 1, 3])
        self.assertTrue(
            conditions["ordinary_sum_no_carry"]["holds"]
        )
        self.assertTrue(
            conditions["signed_digit_injective_sufficient"]["holds"]
        )


class DigitProductCounterexampleTests(unittest.TestCase):
    def test_smallest_checked_failure_has_explicit_collision(self):
        payload = {
            "schema": VERIFIER.SCHEMA,
            "base": 4,
            "digits": [0, 1, 3],
            "levels": 2,
            "check_levels": [2, 1],
            "obligations": [{
                "id": "SUM",
                "type": "product_formula",
                "operation": "sum",
            }],
        }
        evidence = VERIFIER.verify_digit_product(payload)

        self.assertEqual(evidence["status"], "contains_refutation")
        counterexample = evidence["minimal_counterexample"]
        self.assertEqual(counterexample["obligation_id"], "SUM")
        self.assertEqual(counterexample["level"], 2)
        self.assertEqual(counterexample["expected"], 36)
        self.assertEqual(counterexample["actual"], 28)

        collision = counterexample["digit_expansion_collision"]
        self.assertEqual(collision["status"], "found")
        self.assertEqual(collision["operation"], "sum")
        first = collision["first"]
        second = collision["second"]
        self.assertNotEqual(
            first["operation_digits_little_endian"],
            second["operation_digits_little_endian"],
        )
        self.assertEqual(first["left"] + first["right"], collision["value"])
        self.assertEqual(second["left"] + second["right"], collision["value"])

        checks = evidence["obligations"][0]["checks"]
        self.assertEqual([item["level"] for item in checks], [1, 2])
        self.assertTrue(checks[0]["matches"])
        self.assertFalse(checks[1]["matches"])

    def test_no_carry_returns_lexicographically_first_digit_witness(self):
        report = VERIFIER.check_no_carry_conditions(4, [0, 1, 3])
        self.assertFalse(report["ordinary_sum_no_carry"]["holds"])
        self.assertEqual(
            report["ordinary_sum_no_carry"]["counterexample"],
            {
                "left_digit": 1,
                "right_digit": 3,
                "digit_sum": 4,
                "base": 4,
            },
        )

    def test_structural_failure_is_not_mislabelled_as_a_level_minimum(self):
        payload = {
            "schema": VERIFIER.SCHEMA,
            "base": 4,
            "digits": [0, 1, 3],
            "levels": 1,
            "check_levels": [1],
            "obligations": [{
                "id": "NC",
                "type": "sum_no_carry",
            }],
        }
        evidence = VERIFIER.verify_digit_product(payload)

        self.assertIsNone(evidence["minimal_counterexample"])
        self.assertEqual(
            evidence["structural_refutations"][0]["obligation_id"],
            "NC",
        )


class DigitProductStrictnessTests(unittest.TestCase):
    def test_schema_names_are_unversioned_and_versioned_aliases_fail(self):
        self.assertEqual(VERIFIER.SCHEMA, "openhyra-digit-product")
        self.assertEqual(
            VERIFIER.EVIDENCE_SCHEMA,
            "openhyra-digit-product-evidence",
        )
        for suffix in ("-v1", "-v2"):
            payload = _valid_payload()
            payload["schema"] = VERIFIER.SCHEMA + suffix
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "schema"):
                    VERIFIER.verify_digit_product(payload)

    def test_integer_fields_reject_bools_strings_and_floats(self):
        mutations = [
            ("base", True),
            ("base", "10"),
            ("base", 10.0),
            ("levels", False),
            ("levels", "3"),
            ("levels", 3.0),
        ]
        for field, value in mutations:
            payload = _valid_payload()
            payload[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    VERIFIER.verify_digit_product(payload)

        for value in (True, "1", 1.0):
            payload = _valid_payload()
            payload["digits"][1] = value
            with self.subTest(digit=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    VERIFIER.verify_digit_product(payload)

        for value in (True, "2", 2.0):
            payload = _valid_payload()
            payload["check_levels"][1] = value
            with self.subTest(check_level=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    VERIFIER.verify_digit_product(payload)

        for value in (True, "36", 36.0):
            payload = _valid_payload()
            level_obligation = next(
                item
                for item in payload["obligations"]
                if item["type"] == "level_counts"
            )
            level_obligation["expected_sum_count"] = value
            with self.subTest(expected_sum_count=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    VERIFIER.verify_digit_product(payload)

    def test_unknown_fields_and_duplicate_digits_are_rejected(self):
        payload = _valid_payload()
        payload["trusted"] = True
        with self.assertRaisesRegex(ValueError, "unknown field"):
            VERIFIER.verify_digit_product(payload)

        payload = _valid_payload()
        payload["digits"] = [0, 1, 1]
        with self.assertRaisesRegex(ValueError, "distinct digits"):
            VERIFIER.verify_digit_product(payload)

    def test_resource_limits_fail_before_exact_enumeration(self):
        payload = {
            "schema": VERIFIER.SCHEMA,
            "base": 20,
            "digits": list(range(11)),
            "levels": 3,
            "check_levels": [1, 2, 3],
        }
        with self.assertRaisesRegex(
            ValueError,
            "exact verification limit",
        ):
            VERIFIER.verify_digit_product(payload)

        too_many_levels = _valid_payload()
        too_many_levels["levels"] = 9
        too_many_levels["check_levels"] = list(range(1, 10))
        with self.assertRaisesRegex(ValueError, "check_levels exceeds"):
            VERIFIER.verify_digit_product(too_many_levels)

        oversized_base = _valid_payload()
        oversized_base["base"] = VERIFIER.MAX_BASE + 1
        with self.assertRaisesRegex(ValueError, "must be <="):
            VERIFIER.verify_digit_product(oversized_base)

    def test_sparse_check_levels_are_rejected(self):
        payload = _valid_payload()
        payload["check_levels"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            VERIFIER.verify_digit_product(payload)

    def test_inputs_are_normalized_without_mutating_the_submission(self):
        payload = _valid_payload()
        payload["digits"] = [3, 0, 1]
        payload["check_levels"] = [3, 1, 2]
        original = copy.deepcopy(payload)

        evidence = VERIFIER.verify_digit_product(payload)

        self.assertEqual(payload, original)
        self.assertEqual(evidence["construction"]["digits"], [0, 1, 3])
        self.assertEqual(
            evidence["construction"]["check_levels"],
            [1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
