"""Exhaustively enumerate small normalized sets by Gray code."""

import json
import math
import os
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
P = tuple(sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if i not in MISSING))
FALLBACK = tuple(sorted(set(P).union(x + 1568 for x in P)))
SEARCH_SECONDS = 145.0


def exact_score(values):
    n = len(values)
    sums = {x + y for x in values for y in values}
    diffs = {x - y for x in values for y in values}
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def search_span(m, deadline, best_score, best_values):
    # Ordered-pair multiplicities make a toggle cost O(m), while Gray order
    # changes exactly one of the m-1 interior membership bits per state.
    present = [False] * (m + 1)
    present[0] = present[m] = True
    mask = (1 << 0) | (1 << m)
    sum_counts = [0] * (2 * m + 1)
    diff_counts = [0] * (m + 1)
    sum_counts[0] = sum_counts[2 * m] = 1
    sum_counts[m] = 2
    diff_counts[0] = 2
    diff_counts[m] = 2
    sum_size, positive_diff_size, n = 3, 1, 2

    limit = 1 << (m - 1)
    for state in range(limit):
        if state:
            bit = (state & -state).bit_length() - 1
            x = bit + 1
            adding = not present[x]
            if adding:
                bits = mask
                while bits:
                    low = bits & -bits
                    y = low.bit_length() - 1
                    s = x + y
                    if sum_counts[s] == 0:
                        sum_size += 1
                    sum_counts[s] += 2
                    d = abs(x - y)
                    if diff_counts[d] == 0:
                        positive_diff_size += 1
                    diff_counts[d] += 2
                    bits ^= low
                if sum_counts[2 * x] == 0:
                    sum_size += 1
                sum_counts[2 * x] += 1
                diff_counts[0] += 1
                present[x] = True
                mask |= 1 << x
                n += 1
            else:
                sum_counts[2 * x] -= 1
                if sum_counts[2 * x] == 0:
                    sum_size -= 1
                diff_counts[0] -= 1
                bits = mask ^ (1 << x)
                while bits:
                    low = bits & -bits
                    y = low.bit_length() - 1
                    s = x + y
                    sum_counts[s] -= 2
                    if sum_counts[s] == 0:
                        sum_size -= 1
                    d = abs(x - y)
                    diff_counts[d] -= 2
                    if diff_counts[d] == 0:
                        positive_diff_size -= 1
                    bits ^= low
                present[x] = False
                mask ^= 1 << x
                n -= 1

        diff_size = 2 * positive_diff_size + 1
        candidate_score = math.log(sum_size / n) / math.log(diff_size / n)
        if candidate_score > best_score:
            values = tuple(i for i in range(m + 1) if present[i])
            # Recheck before allowing an incremental bookkeeping result to win.
            checked = exact_score(values)
            if checked > best_score:
                best_score, best_values = checked, values
                write_solution(values)

        if state & 4095 == 0 and time.monotonic() >= deadline:
            return best_score, best_values, False
    return best_score, best_values, True


def main():
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = FALLBACK
    best_score = exact_score(FALLBACK)
    write_solution(FALLBACK)

    for m in range(12, 25):
        best_score, best_values, completed = search_span(
            m, deadline, best_score, best_values
        )
        if not completed:
            break

    write_solution(best_values)
    print(f"wrote exhaustive best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
