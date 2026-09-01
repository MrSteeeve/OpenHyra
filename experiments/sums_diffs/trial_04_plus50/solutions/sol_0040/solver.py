"""Exhaustive alternating one-sided fringe optimization around sol_0036."""

import json
import math
import os
import time
import heapq


FALLBACK = (0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 132, 133, 134, 137,
            139, 143, 147, 151, 155, 156, 157, 158)
SEED = 5039
SEARCH_SECONDS = 150.0
KEEP = 32


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def shifted_parent(delta):
    """Move both core boundaries by delta while retaining fringe patterns."""
    width, core = 53 + delta, 53
    span = 2 * width + core - 1
    left = {x for x in FALLBACK if x < 53}
    right = {158 - x for x in FALLBACK if x > 105}
    values = set(range(width, width + core))
    values.update(left)
    values.update(span - x for x in right)
    return tuple(sorted(values)), span


def exhaustive_side(parent, positions, deadline):
    """Return the leading exact scores among all masks on positions."""
    variable = set(positions)
    fixed = set(parent) - variable
    heap = []
    total = 1 << len(positions)
    for bits in range(total):
        if not (bits & 1023) and time.monotonic() >= deadline:
            break
        values = tuple(sorted(fixed.union(
            positions[j] for j in range(20) if bits & (1 << j))))
        value = score(values)
        item = (value, values)
        if len(heap) < KEEP:
            heapq.heappush(heap, item)
        elif value > heap[0][0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)


def main():
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = FALLBACK
    best_score = score(FALLBACK)
    write_solution(FALLBACK)

    # Alternate orientation; the ordering also ensures d=0 is completed first.
    jobs = [(d, flip) for d in (0, -1, 1, -2, 2, -3, 3)
            for flip in (False, True)]
    for delta, flip in jobs:
        if time.monotonic() >= deadline:
            break
        parent, span = shifted_parent(delta)
        left = tuple(range(20))
        right = tuple(range(span - 19, span + 1))
        first, second = (right, left) if flip else (left, right)
        leaders = exhaustive_side(parent, first, deadline)
        for first_score, state in leaders:
            if first_score > best_score:
                best_score, best_values = first_score, state
                write_solution(state)
            if time.monotonic() >= deadline:
                break
            replies = exhaustive_side(state, second, deadline)
            if replies and replies[0][0] > best_score:
                best_score, best_values = replies[0]
                write_solution(best_values)

    write_solution(best_values)
    print(f"wrote exhaustive-fringe best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
