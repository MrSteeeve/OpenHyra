"""Simulated-annealing search for a sum-dominant integer set."""

import json
import math
import os
import random
import time

INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
SEED = 4
SEARCH_SECONDS = 160.0
RESTARTS = 32
MIN_SIZE = 12
MAX_SIZE = 40
MAX_VALUE = 127


def score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def mutate(values, rng):
    candidate = set(values)
    move = rng.random()
    if move < 0.70:
        candidate.remove(rng.choice(tuple(candidate)))
        available = tuple(set(range(MAX_VALUE + 1)) - candidate)
        candidate.add(rng.choice(available))
    elif move < 0.85 and len(candidate) < MAX_SIZE:
        available = tuple(set(range(MAX_VALUE + 1)) - candidate)
        candidate.add(rng.choice(available))
    elif len(candidate) > MIN_SIZE:
        candidate.remove(rng.choice(tuple(candidate)))
    else:
        candidate.remove(rng.choice(tuple(candidate)))
        available = tuple(set(range(MAX_VALUE + 1)) - candidate)
        candidate.add(rng.choice(available))
    return tuple(sorted(candidate))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    best = tuple(INITIAL_SET)
    best_score = score(best)
    write_solution(best)

    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    slice_seconds = SEARCH_SECONDS / RESTARTS

    for restart in range(RESTARTS):
        current = tuple(INITIAL_SET)
        current_score = score(current)
        slice_deadline = min(deadline, start + (restart + 1) * slice_seconds)

        while time.monotonic() < slice_deadline:
            candidate = mutate(current, rng)
            candidate_score = score(candidate)
            elapsed = time.monotonic() - start
            temperature = max(1e-12, 0.02 * (1.0 - elapsed / SEARCH_SECONDS))
            delta = candidate_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = current
                    best_score = current_score
                    write_solution(best)

    write_solution(best)
    print(f"wrote annealing best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
