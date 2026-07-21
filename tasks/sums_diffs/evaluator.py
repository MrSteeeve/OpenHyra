#!/usr/bin/env python3
"""Trusted SimpleTES-compatible evaluator for the sum-difference task."""

import hashlib
import json
import math
import sys
from functools import reduce
from math import gcd
from pathlib import Path

MIN_N = 2
MAX_N = 512
MIN_INT = -1_000_000
MAX_INT = 1_000_000


def fail(msg):
    print(json.dumps({"error": msg}))
    raise SystemExit(0)


def canonical_values(values):
    """Normalize translation, integer scale, and reflection symmetries."""
    vals = sorted(values)
    shifted = [x - vals[0] for x in vals]
    scale = reduce(gcd, shifted[1:], 0) or 1
    normalized = [x // scale for x in shifted]
    reflected = [normalized[-1] - x for x in reversed(normalized)]
    return min(normalized, reflected)


def canonical_hash(values):
    payload = json.dumps(canonical_values(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate_values(values):
    if not isinstance(values, list) or not values:
        raise ValueError('solution.json must contain a non-empty list "A"')
    normalized = []
    for index, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"element {index} is not numeric") from None
        if not math.isfinite(numeric) or abs(numeric - round(numeric)) > 1e-9:
            raise ValueError(f"element {index} is not an integer")
        integer = int(round(numeric))
        if integer < MIN_INT or integer > MAX_INT:
            raise ValueError(f"elements must be in [{MIN_INT}, {MAX_INT}]")
        normalized.append(integer)
    values = sorted(set(normalized))
    n = len(values)
    if not (MIN_N <= n <= MAX_N):
        raise ValueError(f"|A| must be in [{MIN_N}, {MAX_N}], got {n}")
    sumset = {a + b for a in values for b in values}
    diffset = {a - b for a in values for b in values}
    sums, diffs = len(sumset), len(diffset)
    if sums <= n or diffs <= n:
        raise ValueError("both |A+A|/|A| and |A-A|/|A| must be > 1")
    score = math.log(sums / n) / math.log(diffs / n)
    vals = sorted(values)
    return score, {
        "n": n,
        "sums": sums,
        "diffs": diffs,
        "span": vals[-1] - vals[0],
        "set_hash": canonical_hash(vals),
    }, values


def main():
    target = Path(sys.argv[1])
    solution_path = target / "solution.json" if target.is_dir() else target
    if not solution_path.exists():
        fail("solution.json not found")
    try:
        data = json.loads(solution_path.read_text())
        score, metrics, normalized = evaluate_values(data.get("A"))
    except (OSError, ValueError, TypeError) as exc:
        fail(str(exc))
    # Preserve the full IEEE-754 double in the Experience Bank. Formatting is
    # a presentation concern and must not alter parent selection.
    print(json.dumps({"score": score, "metrics": metrics, "normalized_A": normalized}))


if __name__ == "__main__":
    main()
