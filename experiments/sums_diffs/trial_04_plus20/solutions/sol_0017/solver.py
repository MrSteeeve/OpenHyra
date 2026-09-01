"""Exact sweep over unions of two affine copies of the incumbent set."""

import json
import math
import os
import time


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEARCH_SECONDS = 165.0
ORIGIN = 4096


def exact_metrics(values):
    values = tuple(set(values))
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    return len(values), len(sums), len(diffs)


def score_counts(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def mask(values):
    result = 0
    for value in values:
        result |= 1 << (ORIGIN + value)
    return result


def shifted(bits, amount):
    return bits << amount if amount >= 0 else bits >> -amount


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(set(values))}, stream)
    os.replace(temporary, path)


def main():
    best = INITIAL_SET
    best_n, best_sums, best_diffs = exact_metrics(best)
    best_score = score_counts(best_n, best_sums, best_diffs)
    write_solution(best)

    deadline = time.monotonic() + SEARCH_SECONDS
    base = INITIAL_SET
    base_sums = tuple(a + b for a in base for b in base)
    base_diffs = tuple(a - b for a in base for b in base)
    scales_s = tuple(range(-20, 0)) + tuple(range(1, 21))
    stopped = False

    for r in range(1, 21):
        x_values = tuple(r * a for a in base)
        x_mask = mask(x_values)
        xx_sum = mask(r * value for value in base_sums)
        xx_diff = mask(r * value for value in base_diffs)

        for s in scales_s:
            y_mask = mask(s * a for a in base)
            yy_sum = mask(s * value for value in base_sums)
            yy_diff = mask(s * value for value in base_diffs)
            xy_sum = mask(r * a + s * b for a in base for b in base)
            xy_diff = mask(r * a - s * b for a in base for b in base)
            yx_diff = mask(s * a - r * b for a in base for b in base)

            for t in range(-512, 513):
                union_mask = x_mask | shifted(y_mask, t)
                n = union_mask.bit_count()
                sums = (
                    xx_sum
                    | shifted(xy_sum, t)
                    | shifted(yy_sum, 2 * t)
                ).bit_count()
                diffs = (
                    xx_diff
                    | yy_diff
                    | shifted(xy_diff, -t)
                    | shifted(yx_diff, t)
                ).bit_count()
                candidate_score = score_counts(n, sums, diffs)

                if candidate_score > best_score:
                    candidate = tuple(sorted(set(x_values) | {t + s * a for a in base}))
                    check_n, check_sums, check_diffs = exact_metrics(candidate)
                    if (check_n, check_sums, check_diffs) == (n, sums, diffs):
                        best = candidate
                        best_n, best_sums, best_diffs = n, sums, diffs
                        best_score = candidate_score
                        write_solution(best)

                if (t & 63) == 0 and time.monotonic() >= deadline:
                    stopped = True
                    break

            if stopped:
                break
        if stopped:
            break

    write_solution(best)
    print(
        f"wrote affine-copy best: n={best_n} sums={best_sums} "
        f"diffs={best_diffs} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
