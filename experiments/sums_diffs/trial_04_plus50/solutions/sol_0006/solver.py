"""Exhaustive search of an interval-plus-fringes sum-dominant family."""

import json
import math
import os
import time
from array import array

INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
SEED = 4006  # The exhaustive experiment itself uses no randomness.
SEARCH_SECONDS = 165.0
FRINGE_WIDTH = 10
FRINGE_COUNT = 1 << (FRINGE_WIDTH - 1)
MIN_M = 20
MAX_M = 160


def score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def interval_mask(low, high):
    if high < low:
        return 0
    return ((1 << (high - low + 1)) - 1) << low


def elements(mask):
    return tuple(i for i in range(FRINGE_WIDTH) if mask & (1 << i))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    best = tuple(INITIAL_SET)
    best_score = score(best)
    write_solution(best)

    start = time.monotonic()
    deadline = start + SEARCH_SECONDS

    # Each mask contains 0; its other nine membership bits are exhaustive.
    fringes = tuple(1 | (tail << 1) for tail in range(FRINGE_COUNT))
    fringe_elements = tuple(elements(mask) for mask in fringes)
    fringe_sizes = tuple(mask.bit_count() for mask in fringes)

    self_sums = []
    self_positive_diffs = []
    reflected_self_sums = []
    reflected_fringe_masks = []
    for mask, members in zip(fringes, fringe_elements):
        sum_mask = 0
        diff_mask = 0
        for left in members:
            sum_mask |= mask << left
            for right in members:
                if left > right:
                    diff_mask |= 1 << (left - right)
        self_sums.append(sum_mask)
        self_positive_diffs.append(diff_mask)
        reflected_self_sums.append(
            sum(1 << (18 - bit) for bit in range(19) if sum_mask & (1 << bit))
        )
        reflected_fringe_masks.append(
            sum(1 << (9 - bit) for bit in members)
        )

    # Flat tables indexed by 512 * left_index + right_index.  Their bit
    # positions are respectively 9+l-r and 18-l-r.
    cross_deltas = array("I")
    reflected_cross_sums = array("I")
    for left_members in fringe_elements:
        for reflected_right in reflected_fringe_masks:
            delta_mask = 0
            reflected_sum_mask = 0
            for left in left_members:
                delta_mask |= reflected_right << left
                reflected_sum_mask |= reflected_right << (9 - left)
            cross_deltas.append(delta_mask)
            reflected_cross_sums.append(reflected_sum_mask)
        if time.monotonic() >= deadline:
            print(f"wrote fringe best: n={len(best)} score={best_score:.9f}")
            return

    pair_counter = 0
    stopped = False
    for m in range(MIN_M, MAX_M + 1):
        # Sums involving one middle element and one fringe element.  The
        # minus masks also describe both positive M-L and U-M differences.
        plus_intervals = tuple(interval_mask(10 + x, m - 10 + x) for x in range(10))
        minus_intervals = tuple(interval_mask(10 - x, m - 10 - x) for x in range(10))
        plus_masks = []
        minus_masks = []
        for members in fringe_elements:
            plus_mask = 0
            minus_mask = 0
            for member in members:
                plus_mask |= plus_intervals[member]
                minus_mask |= minus_intervals[member]
            plus_masks.append(plus_mask)
            minus_masks.append(minus_mask)

        middle_size = m - 19
        middle_sum_mask = interval_mask(20, 2 * m - 20)
        middle_positive_diff_mask = interval_mask(1, m - 20)

        # Avoid millions of repeated logarithm calls.  For a fixed m there
        # are only nineteen possible candidate sizes.
        log_tables = {}
        for n in range(middle_size + 2, middle_size + 21):
            table = [0.0] * (2 * m + 2)
            for cardinality in range(n + 1, 2 * m + 2):
                table[cardinality] = math.log(cardinality / n)
            log_tables[n] = table

        for left_index in range(FRINGE_COUNT):
            left_sum_base = self_sums[left_index] | plus_masks[left_index] | middle_sum_mask
            left_diff_base = (
                self_positive_diffs[left_index]
                | minus_masks[left_index]
                | middle_positive_diff_mask
            )
            left_size = fringe_sizes[left_index]
            pair_base = left_index * FRINGE_COUNT

            for right_index in range(FRINGE_COUNT):
                pair_index = pair_base + right_index
                sum_mask = (
                    left_sum_base
                    | (minus_masks[right_index] << m)
                    | (reflected_self_sums[right_index] << (2 * m - 18))
                    | (cross_deltas[pair_index] << (m - 9))
                )
                positive_diff_mask = (
                    left_diff_base
                    | self_positive_diffs[right_index]
                    | minus_masks[right_index]
                    | (reflected_cross_sums[pair_index] << (m - 18))
                )
                n = middle_size + left_size + fringe_sizes[right_index]
                sums_count = sum_mask.bit_count()
                diffs_count = 2 * positive_diff_mask.bit_count() + 1
                table = log_tables[n]
                candidate_score = table[sums_count] / table[diffs_count]

                if candidate_score > best_score:
                    left = fringe_elements[left_index]
                    right = fringe_elements[right_index]
                    candidate = tuple(left) + tuple(range(10, m - 9)) + tuple(
                        m - value for value in reversed(right)
                    )
                    best = candidate
                    best_score = candidate_score
                    write_solution(best)

                pair_counter += 1
                if pair_counter & 4095 == 0 and time.monotonic() >= deadline:
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break

    write_solution(best)
    print(f"wrote fringe best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
