"""Simulated-annealing search from the official SimpleTES seed."""

import json
import math
import os
import random
import time
from pathlib import Path

INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
MIN_N = 12
MAX_N = 32
MIN_VALUE = -128
MAX_VALUE = 128
RESTARTS = 48
SEARCH_SECONDS = 165.0
T0 = 0.01
T1 = 1e-6
OUTPUT = Path(__file__).with_name("solution.json")


def score(values):
    values = sorted(values)
    n = len(values)
    sums = {x + y for i, x in enumerate(values) for y in values[i:]}
    diffs = {x - y for x in values for y in values}
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def write_solution(values):
    temporary = OUTPUT.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump({"A": sorted(values)}, stream)
    os.replace(temporary, OUTPUT)


def mutate(values, rng):
    candidate = set(values)
    moves = ["replace"]
    if len(candidate) < MAX_N:
        moves.append("add")
    if len(candidate) > MIN_N:
        moves.append("delete")
    move = rng.choice(moves)

    removed = None
    if move != "add":
        removed = rng.choice(sorted(candidate))
        candidate.remove(removed)
    if move != "delete":
        new_value = rng.randint(MIN_VALUE, MAX_VALUE)
        while new_value in candidate or new_value == removed:
            new_value = rng.randint(MIN_VALUE, MAX_VALUE)
        candidate.add(new_value)
    return candidate


def main():
    rng = random.Random(5)
    started = time.monotonic()
    deadline = started + SEARCH_SECONDS

    best = set(INITIAL_SET)
    best_score = score(best)
    write_solution(best)

    for restart in range(RESTARTS):
        restart_start = time.monotonic()
        restart_end = started + SEARCH_SECONDS * (restart + 1) / RESTARTS
        current = set(INITIAL_SET)
        current_score = score(current)

        while time.monotonic() < min(restart_end, deadline):
            now = time.monotonic()
            progress = min(1.0, (now - restart_start) / max(1e-9, restart_end - restart_start))
            temperature = T0 * (T1 / T0) ** progress
            candidate = mutate(current, rng)
            candidate_score = score(candidate)
            delta = candidate_score - current_score

            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
                    write_solution(best)

    write_solution(best)
    print(f"wrote annealing best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
