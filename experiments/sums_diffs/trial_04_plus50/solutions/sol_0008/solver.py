"""Exhaustive two-for-two exchange search for a sum-dominant set."""

import heapq
import json
import math
import os
import random
import time

INITIAL_SET = [2, 3, 5, 6, 7, 10, 14, 15, 18, 22, 23, 26, 30, 31, 33, 34, 35]
SEED = 4007
SEARCH_SECONDS = 155.0
MAX_VALUE = 96
BEAM_WIDTH = 32
TIME_CHECK_INTERVAL = 2048


def canonicalize(values):
    """Remove translation and dilation symmetries from a candidate."""
    ordered = sorted(values)
    origin = ordered[0]
    shifted = tuple(value - origin for value in ordered)
    divisor = 0
    for value in shifted[1:]:
        divisor = math.gcd(divisor, value)
    if divisor > 1:
        shifted = tuple(value // divisor for value in shifted)
    return shifted


def score(values):
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


def exchange_neighborhood(
    values, deadline, seen, leaders, leader_keys, score_cache, best_state
):
    """Add this set's best unseen two-for-two exchanges to a shared top beam."""
    n = len(values)
    occupied = set(values)
    attempts = 0

    for first in range(n - 1):
        for second in range(first + 1, n):
            retained = tuple(
                value
                for index, value in enumerate(values)
                if index != first and index != second
            )
            retained_set = set(retained)
            retained_bits = sum(1 << value for value in retained)

            sum_mask = 0
            diff_mask = 0
            for a in retained:
                for b in retained:
                    sum_mask |= 1 << (a + b)
                    diff_mask |= 1 << (a - b + MAX_VALUE)

            available = [
                value for value in range(MAX_VALUE + 1) if value not in retained_set
            ]
            sum_extra = {}
            diff_extra = {}
            for value in available:
                sum_extra[value] = (retained_bits << value) | (1 << (2 * value))
                mask = 1 << MAX_VALUE
                for retained_value in retained:
                    mask |= 1 << (value - retained_value + MAX_VALUE)
                    mask |= 1 << (retained_value - value + MAX_VALUE)
                diff_extra[value] = mask

            for left_index, left in enumerate(available[:-1]):
                left_sums = sum_mask | sum_extra[left]
                left_diffs = diff_mask | diff_extra[left]
                for right in available[left_index + 1 :]:
                    attempts += 1
                    if attempts % TIME_CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
                        return False

                    sums = (left_sums | sum_extra[right] | (1 << (left + right))).bit_count()
                    distance = right - left
                    diffs = (
                        left_diffs
                        | diff_extra[right]
                        | (1 << (MAX_VALUE + distance))
                        | (1 << (MAX_VALUE - distance))
                    ).bit_count()
                    counts = (sums, diffs)
                    candidate_score = score_cache.get(counts)
                    if candidate_score is None:
                        candidate_score = math.log(sums / n) / math.log(diffs / n)
                        score_cache[counts] = candidate_score

                    threshold = leaders[0][0] if len(leaders) == BEAM_WIDTH else -math.inf
                    if candidate_score < threshold and candidate_score <= best_state[1]:
                        continue

                    candidate = canonicalize(retained + (left, right))
                    if candidate_score > best_state[1]:
                        best_state[0] = candidate
                        best_state[1] = candidate_score
                        write_solution(candidate)

                    if candidate in seen or candidate in leader_keys:
                        continue
                    entry = (candidate_score, candidate)
                    if len(leaders) < BEAM_WIDTH:
                        heapq.heappush(leaders, entry)
                        leader_keys.add(candidate)
                    elif entry > leaders[0]:
                        removed = heapq.heapreplace(leaders, entry)
                        leader_keys.remove(removed[1])
                        leader_keys.add(candidate)

    return True


def main():
    rng = random.Random(SEED)
    parent = canonicalize(INITIAL_SET)
    best_state = [parent, score(parent)]
    beam = [parent]
    seen = {parent}
    score_cache = {}
    deadline = time.monotonic() + SEARCH_SECONDS
    write_solution(parent)

    while beam and time.monotonic() < deadline:
        rng.shuffle(beam)
        leaders = []
        leader_keys = set()
        complete = True
        for values in beam:
            if not exchange_neighborhood(
                values,
                deadline,
                seen,
                leaders,
                leader_keys,
                score_cache,
                best_state,
            ):
                complete = False
                break
        if not complete:
            break
        ranked = sorted(leaders, reverse=True)
        beam = [values for _, values in ranked]
        seen.update(beam)

    write_solution(best_state[0])
    print(
        f"wrote two-exchange best: n={len(best_state[0])} "
        f"score={best_state[1]:.9f}"
    )


if __name__ == "__main__":
    main()
