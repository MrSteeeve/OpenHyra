"""Exhaustive search over normalized unions of three integer intervals."""

import itertools
import json
import math
import os
import time


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEARCH_SECONDS = 165.0
DOMAIN_MAX = 48
MASK_WIDTH = 2 * DOMAIN_MAX + 1


def exact_score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def interval_masks():
    """Return masks whose bits encode every closed interval in [0, 96]."""
    rows = []
    for low in range(MASK_WIDTH):
        row = [0] * MASK_WIDTH
        for high in range(low, MASK_WIDTH):
            row[high] = ((1 << (high - low + 1)) - 1) << low
        rows.append(tuple(row))
    return tuple(rows)


def main():
    best = INITIAL_SET
    best_score = exact_score(best)
    write_solution(best)

    masks = interval_masks()
    zero_masks = masks[0]
    score_cache = {}
    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    examined = 0
    completed = True

    # The five transitions e1 < a2 < e2 < a3 < e3 uniquely describe
    # [0,e1-1] U [a2,e2-1] U [a3,e3-1].  Strict inequalities enforce
    # nonempty intervals separated by at least one absent integer.
    transitions = itertools.combinations(range(1, DOMAIN_MAX + 2), 5)
    for e1, a2, e2, a3, e3 in transitions:
        b1 = e1 - 1
        b2 = e2 - 1
        b3 = e3 - 1
        n = e1 + (e2 - a2) + (e3 - a3)

        sum_mask = (
            zero_masks[2 * b1]
            | masks[a2][b1 + b2]
            | masks[a3][b1 + b3]
            | masks[2 * a2][2 * b2]
            | masks[a2 + a3][b2 + b3]
            | masks[2 * a3][2 * b3]
        )
        sum_count = sum_mask.bit_count()

        longest_interval = max(e1, e2 - a2, e3 - a3)
        positive_diff_mask = (
            zero_masks[longest_interval - 1]
            | masks[a2 - b1][b2]
            | masks[a3 - b1][b3]
            | masks[a3 - b2][b3 - a2]
        )
        diff_count = 2 * positive_diff_mask.bit_count() - 1

        key = (n, sum_count, diff_count)
        candidate_score = score_cache.get(key)
        if candidate_score is None:
            candidate_score = math.log(sum_count / n) / math.log(diff_count / n)
            score_cache[key] = candidate_score

        if candidate_score > best_score:
            best = tuple(
                itertools.chain(range(e1), range(a2, e2), range(a3, e3))
            )
            best_score = candidate_score
            write_solution(best)

        examined += 1
        if examined & 8191 == 0 and time.monotonic() >= deadline:
            completed = False
            break

    write_solution(best)
    status = "complete" if completed else "timed out"
    print(
        f"wrote three-interval best: n={len(best)} score={best_score:.9f} "
        f"examined={examined} status={status}"
    )


if __name__ == "__main__":
    main()
