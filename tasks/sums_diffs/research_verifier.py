#!/usr/bin/env python3
"""Trusted, bounded verification primitives for digit-product constructions.

This module deliberately verifies only finite, machine-readable facts.  A
digit-product construction is described by a base ``b``, a set of base-``b``
digits ``D``, a declared maximum number of levels, and a bounded list of levels
to check.  For a checked level ``l`` it constructs

    A_l = {sum(d_i * b**i for i in range(l)) : d_i in D}

and recomputes ``|A_l + A_l|`` and ``|A_l - A_l|`` exactly.

Public API:

``generate_level(base, digits, level)``
    Return the sorted tuple representing ``A_level``.

``exact_sum_diff_counts(values)``
    Return exact cardinalities and deterministic hashes for a finite set.

``check_no_carry_conditions(base, digits)``
    Check ordinary addition without carries and a signed-digit injectivity
    sufficient condition.

``verify_digit_product(payload)``
    Strictly validate an untrusted construction plus optional obligations and
    return structured, bounded evidence with smallest-checked-level
    counterexamples.

The verifier does not infer an asymptotic result from passing small levels.
There are no third-party dependencies, model calls, or executable callbacks.
"""

import hashlib
import itertools
import json
import re


SCHEMA = "openhyra-digit-product"
EVIDENCE_SCHEMA = "openhyra-digit-product-evidence"

MIN_BASE = 2
MAX_BASE = 1_000_000
MAX_DIGITS = 64
MAX_LEVEL = 16
MAX_CHECK_LEVELS = 8
MAX_EXACT_SET_SIZE = 1_024
MAX_PAIR_EVALUATIONS = MAX_EXACT_SET_SIZE * MAX_EXACT_SET_SIZE
MAX_EXACT_ABS_VALUE = MAX_BASE ** MAX_LEVEL
MAX_OBLIGATIONS = 32
MAX_COLLISION_EXPANSIONS = 200_000

OBLIGATION_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
OBLIGATION_TYPES = {
    "level_counts",
    "product_formula",
    "sum_no_carry",
    "signed_digit_bound",
}
OPERATIONS = {"sum", "difference"}


def _strict_int(value, *, path, minimum=None, maximum=None):
    """Return an integer while rejecting bools and coercible lookalikes."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _strict_keys(payload, *, required, allowed, path):
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{path} keys must be strings")
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(missing)}")


def _canonical_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(payload):
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_digits(base, raw_digits, *, path="digits"):
    if not isinstance(raw_digits, (list, tuple)):
        raise ValueError(f"{path} must be a list of integers")
    if not 2 <= len(raw_digits) <= MAX_DIGITS:
        raise ValueError(
            f"{path} must contain between 2 and {MAX_DIGITS} entries"
        )
    digits = [
        _strict_int(
            digit,
            path=f"{path}[{index}]",
            minimum=0,
            maximum=base - 1,
        )
        for index, digit in enumerate(raw_digits)
    ]
    if len(digits) != len(set(digits)):
        raise ValueError(f"{path} must contain distinct digits")
    return tuple(sorted(digits))


def _normalize_check_levels(raw_levels, *, declared_levels):
    if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
        raise ValueError("check_levels must be a non-empty list")
    if len(raw_levels) > MAX_CHECK_LEVELS:
        raise ValueError(
            f"check_levels exceeds {MAX_CHECK_LEVELS} entries"
        )
    levels = [
        _strict_int(
            level,
            path=f"check_levels[{index}]",
            minimum=1,
            maximum=declared_levels,
        )
        for index, level in enumerate(raw_levels)
    ]
    if len(levels) != len(set(levels)):
        raise ValueError("check_levels must contain distinct levels")
    normalized = tuple(sorted(levels))
    expected_prefix = tuple(range(1, normalized[-1] + 1))
    if normalized != expected_prefix:
        raise ValueError(
            "check_levels must be the contiguous prefix 1..k so the "
            "first counterexample is meaningful"
        )
    return normalized


def _ensure_exact_level_fits(digit_count, level):
    cardinality = digit_count ** level
    if cardinality > MAX_EXACT_SET_SIZE:
        raise ValueError(
            f"level {level} would contain {cardinality} values; "
            f"the exact verification limit is {MAX_EXACT_SET_SIZE}"
        )
    if cardinality * cardinality > MAX_PAIR_EVALUATIONS:
        raise ValueError(
            f"level {level} exceeds the exact pair-evaluation limit"
        )
    return cardinality


def generate_level(base, digits, level):
    """Generate a small digit-product level exactly and deterministically."""
    base = _strict_int(
        base, path="base", minimum=MIN_BASE, maximum=MAX_BASE
    )
    normalized_digits = _normalize_digits(base, digits)
    level = _strict_int(
        level, path="level", minimum=1, maximum=MAX_LEVEL
    )
    expected_size = _ensure_exact_level_fits(len(normalized_digits), level)

    values = (0,)
    place = 1
    for _ in range(level):
        values = tuple(
            prefix + digit * place
            for prefix in values
            for digit in normalized_digits
        )
        place *= base

    # Valid base digits make representations unique.  Retain the independent
    # check so future edits cannot silently weaken that invariant.
    unique_values = tuple(sorted(set(values)))
    if len(unique_values) != expected_size:
        raise RuntimeError("digit-product generation produced a collision")
    return unique_values


def exact_sum_diff_counts(values):
    """Recompute exact finite-set, sumset, and difference-set evidence."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("values must be a non-empty list of distinct integers")
    if len(values) > MAX_EXACT_SET_SIZE:
        raise ValueError(
            f"values exceeds the exact verification limit "
            f"{MAX_EXACT_SET_SIZE}"
        )
    normalized = []
    for index, value in enumerate(values):
        normalized.append(_strict_int(
            value,
            path=f"values[{index}]",
            minimum=-MAX_EXACT_ABS_VALUE,
            maximum=MAX_EXACT_ABS_VALUE,
        ))
    if len(normalized) != len(set(normalized)):
        raise ValueError("values must contain distinct integers")
    normalized.sort()
    if len(normalized) * len(normalized) > MAX_PAIR_EVALUATIONS:
        raise ValueError("values exceeds the exact pair-evaluation limit")

    sumset = set()
    diffset = set()
    for left in normalized:
        for right in normalized:
            sumset.add(left + right)
            diffset.add(left - right)
    sorted_sums = sorted(sumset)
    sorted_diffs = sorted(diffset)
    return {
        "n": len(normalized),
        "sums": len(sorted_sums),
        "diffs": len(sorted_diffs),
        "set_sha256": _sha256_json(normalized),
        "sumset_sha256": _sha256_json(sorted_sums),
        "diffset_sha256": _sha256_json(sorted_diffs),
    }


def _first_sum_carry(base, digits):
    for left in digits:
        for right in digits:
            if left + right >= base:
                return {
                    "left_digit": left,
                    "right_digit": right,
                    "digit_sum": left + right,
                    "base": base,
                }
    return None


def check_no_carry_conditions(base, digits):
    """Return exact finite checks and conservative injectivity guarantees.

    ``ordinary_sum_no_carry`` is the familiar condition
    ``d + e < base`` for every pair of digits.  For signed difference digits,
    ``base > 2 * (max(D) - min(D))`` is a simple sufficient condition ensuring
    that two distinct signed-digit vectors cannot represent the same integer.
    The latter bound also suffices for uniqueness of digitwise sum expansions.
    Failing either sufficient condition does not itself prove that a product
    formula is false; exact checked levels decide that separately.
    """
    base = _strict_int(
        base, path="base", minimum=MIN_BASE, maximum=MAX_BASE
    )
    normalized_digits = _normalize_digits(base, digits)
    span = normalized_digits[-1] - normalized_digits[0]
    carry = _first_sum_carry(base, normalized_digits)
    signed_bound_holds = base > 2 * span
    return {
        "ordinary_sum_no_carry": {
            "holds": carry is None,
            "condition": "max_pair_sum < base",
            "max_pair_sum": 2 * normalized_digits[-1],
            "base": base,
            "counterexample": carry,
        },
        "signed_digit_injective_sufficient": {
            "holds": signed_bound_holds,
            "condition": "base > 2 * digit_span",
            "base": base,
            "digit_span": span,
            "required_strict_lower_bound": 2 * span,
            "guarantees_when_true": [
                "sum_product_formula_all_levels",
                "difference_product_formula_all_levels",
            ],
            "counterexample": (
                None
                if signed_bound_holds
                else {
                    "base": base,
                    "twice_digit_span": 2 * span,
                    "relation": "base <= 2 * digit_span",
                }
            ),
        },
        "both_product_formulas_certified_by_signed_digit_bound": (
            signed_bound_holds
        ),
    }


def _normalize_obligations(raw_obligations, *, check_levels):
    if raw_obligations is None:
        return []
    if not isinstance(raw_obligations, list):
        raise ValueError("obligations must be a list")
    if len(raw_obligations) > MAX_OBLIGATIONS:
        raise ValueError(f"obligations exceeds {MAX_OBLIGATIONS} entries")

    normalized = []
    seen_ids = set()
    for index, raw in enumerate(raw_obligations):
        path = f"obligations[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be an object")
        obligation_type = raw.get("type")
        common_required = {"id", "type"}
        if obligation_type == "level_counts":
            required = common_required | {
                "level",
                "expected_n",
                "expected_sum_count",
                "expected_diff_count",
            }
            allowed = required
        elif obligation_type == "product_formula":
            required = common_required | {"operation"}
            allowed = required
        elif obligation_type in {"sum_no_carry", "signed_digit_bound"}:
            required = common_required
            allowed = required
        else:
            _strict_keys(
                raw,
                required=common_required,
                allowed=set(raw) | common_required,
                path=path,
            )
            raise ValueError(
                f"{path}.type must be one of "
                + ", ".join(sorted(OBLIGATION_TYPES))
            )
        _strict_keys(raw, required=required, allowed=allowed, path=path)

        obligation_id = raw["id"]
        if (
            not isinstance(obligation_id, str)
            or not OBLIGATION_ID_RE.fullmatch(obligation_id)
        ):
            raise ValueError(f"{path}.id has invalid syntax")
        if obligation_id in seen_ids:
            raise ValueError(f"duplicate obligation id: {obligation_id}")
        seen_ids.add(obligation_id)

        item = {"id": obligation_id, "type": obligation_type}
        if obligation_type == "level_counts":
            level = _strict_int(
                raw["level"],
                path=f"{path}.level",
                minimum=1,
                maximum=MAX_LEVEL,
            )
            if level not in check_levels:
                raise ValueError(
                    f"{path}.level must be present in check_levels"
                )
            item.update({
                "level": level,
                "expected_n": _strict_int(
                    raw["expected_n"],
                    path=f"{path}.expected_n",
                    minimum=1,
                    maximum=MAX_EXACT_SET_SIZE,
                ),
                "expected_sum_count": _strict_int(
                    raw["expected_sum_count"],
                    path=f"{path}.expected_sum_count",
                    minimum=1,
                    maximum=MAX_PAIR_EVALUATIONS,
                ),
                "expected_diff_count": _strict_int(
                    raw["expected_diff_count"],
                    path=f"{path}.expected_diff_count",
                    minimum=1,
                    maximum=MAX_PAIR_EVALUATIONS,
                ),
            })
        elif obligation_type == "product_formula":
            operation = raw["operation"]
            if operation not in OPERATIONS:
                raise ValueError(
                    f"{path}.operation must be 'sum' or 'difference'"
                )
            item["operation"] = operation
        normalized.append(item)
    normalized.sort(key=lambda item: item["id"])
    return normalized


def _digit_operation_pairs(digits, operation):
    pairs = {}
    for left in digits:
        for right in digits:
            value = (
                left + right if operation == "sum" else left - right
            )
            pairs.setdefault(value, (left, right))
    return pairs


def _render_digit_representation(sequence, pair_by_digit, base):
    pairs = [pair_by_digit[digit] for digit in sequence]
    left_digits = [pair[0] for pair in pairs]
    right_digits = [pair[1] for pair in pairs]
    places = [base ** index for index in range(len(sequence))]
    left = sum(digit * place for digit, place in zip(left_digits, places))
    right = sum(digit * place for digit, place in zip(right_digits, places))
    return {
        "operation_digits_little_endian": list(sequence),
        "left_digits_little_endian": left_digits,
        "right_digits_little_endian": right_digits,
        "left": left,
        "right": right,
    }


def _find_digit_expansion_collision(base, digits, level, operation):
    pair_by_digit = _digit_operation_pairs(digits, operation)
    operation_digits = tuple(sorted(pair_by_digit))
    expansion_count = len(operation_digits) ** level
    if expansion_count > MAX_COLLISION_EXPANSIONS:
        return {
            "status": "omitted_limit",
            "expansion_count": expansion_count,
            "limit": MAX_COLLISION_EXPANSIONS,
        }

    places = tuple(base ** index for index in range(level))
    seen = {}
    for sequence in itertools.product(operation_digits, repeat=level):
        value = sum(
            digit * place for digit, place in zip(sequence, places)
        )
        previous = seen.get(value)
        if previous is not None and previous != sequence:
            return {
                "status": "found",
                "operation": operation,
                "value": value,
                "first": _render_digit_representation(
                    previous, pair_by_digit, base
                ),
                "second": _render_digit_representation(
                    sequence, pair_by_digit, base
                ),
            }
        seen[value] = sequence
    return {"status": "not_found"}


def _level_counterexample(obligation, actual):
    mismatches = {}
    for expected_key, actual_key in (
        ("expected_n", "n"),
        ("expected_sum_count", "sums"),
        ("expected_diff_count", "diffs"),
    ):
        expected = obligation[expected_key]
        observed = actual[actual_key]
        if expected != observed:
            mismatches[actual_key] = {
                "expected": expected,
                "actual": observed,
            }
    return {
        "kind": "exact_count_mismatch",
        "level": obligation["level"],
        "mismatches": mismatches,
    }


def _verify_obligations(
    obligations,
    *,
    base,
    digits,
    check_levels,
    level_results,
    digit_sum_count,
    digit_diff_count,
    no_carry,
):
    by_level = {item["level"]: item for item in level_results}
    outcomes = []
    for obligation in obligations:
        obligation_type = obligation["type"]
        outcome = {
            "id": obligation["id"],
            "type": obligation_type,
            "obligation_sha256": _sha256_json(obligation),
        }
        if obligation_type == "level_counts":
            actual = by_level[obligation["level"]]
            matches = (
                obligation["expected_n"] == actual["n"]
                and obligation["expected_sum_count"] == actual["sums"]
                and obligation["expected_diff_count"] == actual["diffs"]
            )
            outcome.update({
                "status": "bounded_checked" if matches else "refuted",
                "checked_levels": [obligation["level"]],
                "expected": {
                    "n": obligation["expected_n"],
                    "sums": obligation["expected_sum_count"],
                    "diffs": obligation["expected_diff_count"],
                },
                "actual": {
                    "n": actual["n"],
                    "sums": actual["sums"],
                    "diffs": actual["diffs"],
                },
                "counterexample": (
                    None
                    if matches
                    else _level_counterexample(obligation, actual)
                ),
            })
        elif obligation_type == "product_formula":
            operation = obligation["operation"]
            count_key = "sums" if operation == "sum" else "diffs"
            digit_count = (
                digit_sum_count if operation == "sum" else digit_diff_count
            )
            first_failure = None
            checked = []
            for level in check_levels:
                actual_count = by_level[level][count_key]
                expected_count = digit_count ** level
                checked.append({
                    "level": level,
                    "expected": expected_count,
                    "actual": actual_count,
                    "matches": expected_count == actual_count,
                })
                if first_failure is None and expected_count != actual_count:
                    first_failure = {
                        "kind": "product_formula_count_mismatch",
                        "level": level,
                        "operation": operation,
                        "expected": expected_count,
                        "actual": actual_count,
                        "digit_expansion_collision": (
                            _find_digit_expansion_collision(
                                base, digits, level, operation
                            )
                        ),
                    }
            outcome.update({
                "operation": operation,
                "status": (
                    "bounded_checked"
                    if first_failure is None
                    else "refuted"
                ),
                "checked_levels": list(check_levels),
                "checks": checked,
                "counterexample": first_failure,
            })
        elif obligation_type == "sum_no_carry":
            report = no_carry["ordinary_sum_no_carry"]
            outcome.update({
                "status": (
                    "bounded_checked" if report["holds"] else "refuted"
                ),
                "checked_levels": [],
                "condition": report["condition"],
                "counterexample": report["counterexample"],
            })
        elif obligation_type == "signed_digit_bound":
            report = no_carry["signed_digit_injective_sufficient"]
            outcome.update({
                "status": (
                    "bounded_checked" if report["holds"] else "refuted"
                ),
                "checked_levels": [],
                "condition": report["condition"],
                "counterexample": report["counterexample"],
            })
        outcomes.append(outcome)
    return outcomes


def verify_digit_product(payload):
    """Validate and exactly check a bounded digit-product research artifact."""
    _strict_keys(
        payload,
        required={"schema", "base", "digits", "levels", "check_levels"},
        allowed={
            "schema",
            "base",
            "digits",
            "levels",
            "check_levels",
            "obligations",
        },
        path="digit_product",
    )
    if payload["schema"] != SCHEMA:
        raise ValueError(f"digit_product.schema must be {SCHEMA!r}")
    base = _strict_int(
        payload["base"],
        path="digit_product.base",
        minimum=MIN_BASE,
        maximum=MAX_BASE,
    )
    digits = _normalize_digits(
        base, payload["digits"], path="digit_product.digits"
    )
    declared_levels = _strict_int(
        payload["levels"],
        path="digit_product.levels",
        minimum=1,
        maximum=MAX_LEVEL,
    )
    check_levels = _normalize_check_levels(
        payload["check_levels"], declared_levels=declared_levels
    )
    for level in check_levels:
        _ensure_exact_level_fits(len(digits), level)
    obligations = _normalize_obligations(
        payload.get("obligations"), check_levels=check_levels
    )

    normalized_construction = {
        "schema": SCHEMA,
        "base": base,
        "digits": list(digits),
        "levels": declared_levels,
        "check_levels": list(check_levels),
        "obligations": obligations,
    }
    digit_sums = {
        left + right for left in digits for right in digits
    }
    digit_diffs = {
        left - right for left in digits for right in digits
    }
    no_carry = check_no_carry_conditions(base, digits)

    level_results = []
    for level in check_levels:
        values = generate_level(base, digits, level)
        counts = exact_sum_diff_counts(values)
        expected_sum_product = len(digit_sums) ** level
        expected_diff_product = len(digit_diffs) ** level
        level_results.append({
            "level": level,
            "A": list(values),
            **counts,
            "digitwise_expected_sums": expected_sum_product,
            "digitwise_expected_diffs": expected_diff_product,
            "sum_product_formula_holds": (
                counts["sums"] == expected_sum_product
            ),
            "diff_product_formula_holds": (
                counts["diffs"] == expected_diff_product
            ),
        })

    outcomes = _verify_obligations(
        obligations,
        base=base,
        digits=digits,
        check_levels=check_levels,
        level_results=level_results,
        digit_sum_count=len(digit_sums),
        digit_diff_count=len(digit_diffs),
        no_carry=no_carry,
    )
    refutations = [
        outcome for outcome in outcomes if outcome["status"] == "refuted"
    ]
    level_refutations = [
        outcome
        for outcome in refutations
        if isinstance((outcome.get("counterexample") or {}).get("level"), int)
    ]
    level_refutations.sort(
        key=lambda outcome: (
            outcome["counterexample"]["level"],
            outcome["id"],
        )
    )
    structural_refutations = [
        {
            "obligation_id": outcome["id"],
            **outcome["counterexample"],
        }
        for outcome in sorted(refutations, key=lambda item: item["id"])
        if not isinstance(
            (outcome.get("counterexample") or {}).get("level"),
            int,
        )
    ]
    result = {
        "schema": EVIDENCE_SCHEMA,
        "status": (
            "contains_refutation" if refutations else "bounded_checked"
        ),
        "scope": {
            "declared_levels": declared_levels,
            "exactly_checked_levels": list(check_levels),
            "asymptotic_claims_checked": False,
        },
        "construction": normalized_construction,
        "construction_sha256": _sha256_json(normalized_construction),
        "digit_counts": {
            "digits": len(digits),
            "digit_sums": len(digit_sums),
            "digit_differences": len(digit_diffs),
        },
        "no_carry_conditions": no_carry,
        "levels": level_results,
        "obligations": outcomes,
        "structural_refutations": structural_refutations,
        "summary": {
            "obligation_count": len(outcomes),
            "bounded_checked_count": sum(
                outcome["status"] == "bounded_checked"
                for outcome in outcomes
            ),
            "refuted_count": len(refutations),
        },
        "minimal_counterexample": (
            None
            if not level_refutations
            else {
                "obligation_id": level_refutations[0]["id"],
                **level_refutations[0]["counterexample"],
            }
        ),
    }
    result["evidence_sha256"] = _sha256_json(result)
    return result


__all__ = [
    "EVIDENCE_SCHEMA",
    "MAX_BASE",
    "MAX_CHECK_LEVELS",
    "MAX_DIGITS",
    "MAX_EXACT_SET_SIZE",
    "MAX_LEVEL",
    "SCHEMA",
    "check_no_carry_conditions",
    "exact_sum_diff_counts",
    "generate_level",
    "verify_digit_product",
]
