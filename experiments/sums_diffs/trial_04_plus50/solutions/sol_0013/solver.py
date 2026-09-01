"""Exhaustive carry-interacting digit-product search."""

import json
import math
import os
import time


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEARCH_SECONDS = 165.0
DIGIT_LIMIT = 16


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def verify_fallback():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    with open(path) as stream:
        saved = json.load(stream)
    if saved != {"A": list(INITIAL_SET)}:
        raise RuntimeError("failed to preserve the incumbent")


def shifted_union(pattern, blocks):
    result = 0
    while pattern:
        low = pattern & -pattern
        result |= blocks[low.bit_length() - 1]
        pattern ^= low
    return result


def main():
    best = INITIAL_SET
    write_solution(best)
    verify_fallback()

    base_mask = sum(1 << value for value in INITIAL_SET)
    base_sums = 0
    base_diffs = 0
    for left in INITIAL_SET:
        for right in INITIAL_SET:
            base_sums |= 1 << (left + right)
            base_diffs |= 1 << (left - right + INITIAL_SET[-1])

    mask_count = 1 << DIGIT_LIMIT
    digit_sums = [0] * mask_count
    digit_diffs = [0] * mask_count
    eligible = []
    for mask in range(1, mask_count):
        low = mask & -mask
        digit = low.bit_length() - 1
        rest = mask ^ low
        digit_sums[mask] = digit_sums[rest] | (mask << digit)
        differences = 1 << 15
        remaining = rest
        while remaining:
            other_low = remaining & -remaining
            other = other_low.bit_length() - 1
            differences |= 1 << (digit - other + 15)
            differences |= 1 << (other - digit + 15)
            remaining ^= other_low
        digit_diffs[mask] = digit_diffs[rest] | differences
        if 2 <= mask.bit_count() <= 12:
            eligible.append(mask)

    max_count = 2 * (INITIAL_SET[-1] + 40 * (DIGIT_LIMIT - 1)) + 1
    log_ratio = [[0.0] * (max_count + 1) for _ in range(205)]
    for size in range(2, 205):
        row = log_ratio[size]
        for count in range(size + 1, max_count + 1):
            row[count] = math.log(count / size)

    initial_sum_count = base_sums.bit_count()
    initial_diff_count = base_diffs.bit_count()
    best_score = (log_ratio[len(best)][initial_sum_count] /
                  log_ratio[len(best)][initial_diff_count])
    deadline = time.monotonic() + SEARCH_SECONDS
    checked = 0

    for q in range(2, 41):
        value_blocks = [base_mask << (q * digit) for digit in range(DIGIT_LIMIT)]
        sum_blocks = [base_sums << (q * digit) for digit in range(31)]
        diff_blocks = [base_diffs << (q * digit) for digit in range(31)]

        for mask in eligible:
            values_mask = shifted_union(mask, value_blocks)
            size = values_mask.bit_count()
            sum_count = shifted_union(digit_sums[mask], sum_blocks).bit_count()
            diff_count = shifted_union(digit_diffs[mask], diff_blocks).bit_count()
            candidate_score = log_ratio[size][sum_count] / log_ratio[size][diff_count]
            checked += 1

            if candidate_score > best_score:
                values = tuple(index for index in range(values_mask.bit_length())
                               if values_mask & (1 << index))
                best = values
                best_score = candidate_score
                write_solution(best)

            if checked & 8191 == 0 and time.monotonic() >= deadline:
                write_solution(best)
                print(f"time-limited digit sweep: checked={checked} n={len(best)} "
                      f"score={best_score:.9f}")
                return

    write_solution(best)
    print(f"completed digit sweep: checked={checked} n={len(best)} "
          f"score={best_score:.9f}")


if __name__ == "__main__":
    main()
