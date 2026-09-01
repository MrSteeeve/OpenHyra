"""Large-neighborhood beam search for a sum-dominant integer set."""

import json
import math
import os
import random
import time

INITIAL_SET = [2, 3, 5, 6, 7, 10, 14, 15, 18, 22, 23, 26, 30, 31, 33, 34, 35]
SEED = 4003
SEARCH_SECONDS = 155.0
CHAINS = 24
BEAM_WIDTH = 64
MAX_VALUE = 96
DESTROY_SIZES = (3, 4, 5)


def score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def canonicalize(values):
    values = tuple(sorted(values))
    origin = values[0]
    translated = tuple(value - origin for value in values)
    divisor = 0
    for value in translated:
        divisor = math.gcd(divisor, value)
    if divisor > 1:
        translated = tuple(value // divisor for value in translated)
    return translated


def repair(base, target_size, deadline):
    beam = [canonicalize(base)]
    attempts = 0

    while len(beam[0]) < target_size:
        candidates = {}
        for values in beam:
            occupied = set(values)
            for value in range(MAX_VALUE + 1):
                if value in occupied:
                    continue
                attempts += 1
                if attempts % 128 == 0 and time.monotonic() >= deadline:
                    return None
                candidate = canonicalize(values + (value,))
                if candidate not in candidates:
                    candidates[candidate] = score(candidate)

        if not candidates:
            return None
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        beam = [values for values, _ in ranked[:BEAM_WIDTH]]

    return beam[0]


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    parent = canonicalize(INITIAL_SET)
    best = parent
    best_score = score(best)
    write_solution(best)

    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    slice_seconds = SEARCH_SECONDS / CHAINS

    for chain in range(CHAINS):
        current = parent
        slice_deadline = min(deadline, start + (chain + 1) * slice_seconds)

        while time.monotonic() < slice_deadline:
            destroy_size = rng.choice(DESTROY_SIZES)
            removed = set(rng.sample(current, destroy_size))
            base = tuple(value for value in current if value not in removed)
            candidate = repair(base, len(parent), slice_deadline)
            if candidate is None:
                break
            current = candidate
            candidate_score = score(candidate)
            if candidate_score > best_score:
                best = candidate
                best_score = candidate_score
                write_solution(best)

    write_solution(best)
    print(f"wrote beam-search best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
