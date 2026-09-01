"""Exhaustive search over unions of two affine copies of the incumbent."""

import json
import math
import os
import time

INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEED = 4014
SEARCH_SECONDS = 165.0
OFFSET = 4096


def cardinalities(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    return len(values), len(sums), len(diffs)


def score(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def mask(values):
    result = 0
    for value in values:
        result |= 1 << (value + OFFSET)
    return result


def shifted(bits, amount):
    return bits << amount if amount >= 0 else bits >> -amount


def canonicalize(values):
    values = sorted(set(values))
    origin = values[0]
    values = [value - origin for value in values]
    divisor = 0
    for value in values[1:]:
        divisor = math.gcd(divisor, value)
    return tuple(value // divisor for value in values)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    # Fail loudly if the hard-coded fallback is not the evaluated sol_0001 parent.
    initial_metrics = cardinalities(INITIAL_SET)
    assert initial_metrics == (17, 67, 61), initial_metrics

    best = INITIAL_SET
    best_score = score(*initial_metrics)
    write_solution(best)

    deadline = time.monotonic() + SEARCH_SECONDS
    signed_scales = tuple(range(-20, 0)) + tuple(range(1, 21))

    for r in range(1, 21):
        first = tuple(r * a for a in INITIAL_SET)
        first_mask = mask(first)
        first_sums = mask({a + b for a in first for b in first})
        first_diffs = mask({a - b for a in first for b in first})

        for s in signed_scales:
            second = tuple(s * a for a in INITIAL_SET)
            second_mask = mask(second)
            second_sums = mask({a + b for a in second for b in second})
            second_diffs = mask({a - b for a in second for b in second})
            cross_sums = mask({a + b for a in first for b in second})
            forward_diffs = mask({a - b for a in first for b in second})
            reverse_diffs = mask({b - a for a in first for b in second})
            fixed_diffs = first_diffs | second_diffs

            for t in range(-512, 513):
                elements = first_mask | shifted(second_mask, t)
                n = elements.bit_count()
                sums = (
                    first_sums
                    | shifted(cross_sums, t)
                    | shifted(second_sums, 2 * t)
                ).bit_count()
                diffs = (
                    fixed_diffs
                    | shifted(forward_diffs, -t)
                    | shifted(reverse_diffs, t)
                ).bit_count()
                candidate_score = score(n, sums, diffs)

                if candidate_score > best_score:
                    candidate = canonicalize(first + tuple(t + a for a in second))
                    # The score is affine-invariant; verify the emitted form exactly.
                    candidate_metrics = cardinalities(candidate)
                    candidate_score = score(*candidate_metrics)
                    if candidate_score > best_score:
                        best = candidate
                        best_score = candidate_score
                        write_solution(best)

            if time.monotonic() >= deadline:
                write_solution(best)
                print(f"wrote affine-union best: n={len(best)} score={best_score:.9f}")
                return

    write_solution(best)
    print(f"wrote affine-union best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
