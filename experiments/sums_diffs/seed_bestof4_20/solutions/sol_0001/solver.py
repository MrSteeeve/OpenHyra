"""Deterministic simulated annealing for a sum-dominant integer set."""

import json
import math
import os
import random
import time


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEED = 20260782782167
DOMAIN = tuple(range(81))
RESTARTS = 32
SEARCH_SECONDS = 170.0


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
        current_score = quality(current)
        slice_end = min(deadline, started + (restart + 1) * slice_seconds)

        while time.monotonic() < slice_end:
            elapsed = time.monotonic() - started
            temperature = 0.02 * max(0.0, 1.0 - elapsed / SEARCH_SECONDS)
            n = len(current)
            move = rng.randrange(3)

            if move == 0 and n < 40:
                available = [x for x in DOMAIN if x not in current]
                candidate = current | {rng.choice(available)}
            elif move == 1 and n > 8:
                candidate = current - {rng.choice(tuple(current))}
            else:
                available = [x for x in DOMAIN if x not in current]
                if not available:
                    continue
                candidate = (current - {rng.choice(tuple(current))}) | {rng.choice(available)}

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
