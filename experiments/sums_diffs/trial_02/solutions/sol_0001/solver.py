"""Simulated annealing around the official SimpleTES seed."""

import json
import math
import os
import random
import time

INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
MAX_SPAN = 64
RESTARTS = 32
TIME_LIMIT = 165.0
T0 = 0.02
TMIN = 1e-5
SEED = 2
FULL_MASK = (1 << (MAX_SPAN + 1)) - 1
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")


def score(mask):
    """Return C(A), computing both cardinalities exactly with bit operations."""
    n = mask.bit_count()

    sums = 0
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        sums |= mask << (bit.bit_length() - 1)
        remaining ^= bit

    positive_differences = 0
    for distance in range(1, MAX_SPAN + 1):
        positive_differences += bool(mask & (mask >> distance))
    differences = 1 + 2 * positive_differences

    return math.log(sums.bit_count() / n) / math.log(differences / n)


def random_bit(mask, rng):
    """Choose a uniformly random set bit from mask."""
    index = rng.randrange(mask.bit_count())
    while index:
        mask &= mask - 1
        index -= 1
    return mask & -mask


def mutate(mask, rng):
    """Apply an equally likely feasible add, delete, or swap move."""
    n = mask.bit_count()
    absent = FULL_MASK ^ mask
    while True:
        operation = rng.randrange(3)
        if operation == 0 and absent:
            return mask | random_bit(absent, rng)
        if operation == 1 and n > 2:
            return mask ^ random_bit(mask, rng)
        if operation == 2 and absent:
            return (mask ^ random_bit(mask, rng)) | random_bit(absent, rng)


def as_list(mask):
    return [value for value in range(MAX_SPAN + 1) if mask & (1 << value)]


def emit(mask):
    """Atomically preserve the best result found so far."""
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": as_list(mask)}, stream)
    os.replace(temporary, OUTPUT)


def main():
    rng = random.Random(SEED)
    initial = sum(1 << value for value in INITIAL_SET)
    best = initial
    best_score = score(best)
    emit(best)

    started = time.monotonic()
    for restart in range(RESTARTS):
        current = initial
        current_score = score(current)
        restart_start = started + TIME_LIMIT * restart / RESTARTS
        restart_end = started + TIME_LIMIT * (restart + 1) / RESTARTS

        iterations = 0
        while True:
            now = time.monotonic()
            if now >= restart_end:
                break
            progress = (now - restart_start) / (restart_end - restart_start)
            temperature = T0 * (TMIN / T0) ** min(1.0, max(0.0, progress))

            candidate = mutate(current, rng)
            candidate_score = score(candidate)
            delta = candidate_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = current
                    best_score = current_score
                    emit(best)
            iterations += 1

    emit(best)
    print(f"wrote annealed set: n={best.bit_count()} score={best_score:.9f}")


if __name__ == "__main__":
    main()
