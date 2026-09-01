"""Iterated tabu search for a fixed-size sum-dominant integer set."""

import json
import math
import os
import random
import time

INITIAL_SET = [2, 3, 5, 6, 7, 10, 14, 15, 18, 22, 23, 26, 30, 31, 33, 34, 35]
SEED = 4001
SEARCH_SECONDS = 155.0
MAX_VALUE = 64
TABU_TENURE = 50
KICK_AFTER = 20
CACHE_LIMIT = 100_000


def canonicalize(values):
    """Remove translation and dilation symmetries from a candidate."""
    ordered = sorted(values)
    origin = ordered[0]
    shifted = [value - origin for value in ordered]
    divisor = 0
    for value in shifted[1:]:
        divisor = math.gcd(divisor, value)
    return tuple(value // divisor for value in shifted)


def score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def random_two_replacement(values, rng):
    retained = set(values)
    retained.difference_update(rng.sample(tuple(values), 2))
    available = tuple(value for value in range(MAX_VALUE + 1) if value not in values)
    retained.update(rng.sample(available, 2))
    return canonicalize(retained)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    current = canonicalize(INITIAL_SET)
    current_score = score(current)
    best = current
    best_score = current_score
    score_cache = {current: current_score}
    tabu_until = {current: TABU_TENURE}
    iteration = 0
    nonimproving = 0
    deadline = time.monotonic() + SEARCH_SECONDS
    write_solution(best)

    while time.monotonic() < deadline:
        iteration += 1
        chosen = None
        chosen_score = -math.inf
        occupied = set(current)

        for removed in current:
            retained = occupied - {removed}
            for replacement in range(MAX_VALUE + 1):
                if replacement in occupied:
                    continue
                candidate = canonicalize(retained | {replacement})
                if tabu_until.get(candidate, 0) >= iteration:
                    continue
                candidate_score = score_cache.get(candidate)
                if candidate_score is None:
                    candidate_score = score(candidate)
                    score_cache[candidate] = candidate_score
                if (candidate_score, candidate) > (chosen_score, chosen or ()):
                    chosen = candidate
                    chosen_score = candidate_score

        if chosen is None:
            chosen = random_two_replacement(current, rng)
            chosen_score = score_cache.setdefault(chosen, score(chosen))

        current = chosen
        current_score = chosen_score
        tabu_until[current] = iteration + TABU_TENURE

        if current_score > best_score:
            best = current
            best_score = current_score
            nonimproving = 0
            write_solution(best)
        else:
            nonimproving += 1

        if nonimproving >= KICK_AFTER:
            current = random_two_replacement(current, rng)
            current_score = score_cache.get(current)
            if current_score is None:
                current_score = score(current)
                score_cache[current] = current_score
            tabu_until[current] = iteration + TABU_TENURE
            nonimproving = 0
            if current_score > best_score:
                best = current
                best_score = current_score
                write_solution(best)

        if len(score_cache) > CACHE_LIMIT:
            score_cache = {best: best_score, current: current_score}

    write_solution(best)
    print(f"wrote tabu best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
