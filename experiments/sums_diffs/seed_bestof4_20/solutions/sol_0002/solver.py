"""Deterministic simulated annealing for a sum-dominant integer set."""

import json
import math
import os
import random
import time


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26,
               28, 29, 33, 36, 37, 38, 40, 41)
SEED = 20260783782170
DOMAIN = tuple(range(121))
RESTARTS = 4
SEARCH_SECONDS = 170.0
MIN_SIZE = 12
MAX_SIZE = 64
COMPOUND_SIZES = (2, 3, 4, 6, 8)


def quality(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def write_solution(path, values):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    best = frozenset(INITIAL_SET)
    best_score = quality(best)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    slice_seconds = SEARCH_SECONDS / RESTARTS

    for restart in range(RESTARTS):
        current = frozenset(INITIAL_SET)
        if restart:
            # Give each long trajectory a distinct, reproducible basin.
            for _ in range(restart):
                flips = set(rng.sample(DOMAIN, rng.choice(COMPOUND_SIZES)))
                perturbed = current ^ flips
                if MIN_SIZE <= len(perturbed) <= MAX_SIZE:
                    current = frozenset(perturbed)
        current_score = quality(current)
        slice_end = min(deadline, started + (restart + 1) * slice_seconds)

        while time.monotonic() < slice_end:
            elapsed = time.monotonic() - started
            temperature = 0.015 * max(0.0, 1.0 - elapsed / SEARCH_SECONDS) + 0.0002
            k = rng.choice(COMPOUND_SIZES) if rng.random() < 0.45 else 1
            candidate = current ^ set(rng.sample(DOMAIN, k))
            if not MIN_SIZE <= len(candidate) <= MAX_SIZE:
                continue
            candidate = frozenset(candidate)
            candidate_score = quality(candidate)
            delta = candidate_score - current_score
            if delta >= 0.0 or (temperature > 0.0 and
                                rng.random() < math.exp(delta / temperature)):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = current
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
