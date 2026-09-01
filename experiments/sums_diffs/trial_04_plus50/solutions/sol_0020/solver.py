"""Anneal nonlinear row-warp embeddings of the best tensor subset."""

import json
import math
import os
import random
import time


BASE_SET = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
# These are exactly the 12 cells absent from the evaluated sol_0012 parent.
EXCLUDED = frozenset(
    {
        (3, 3),
        (13, 3),
        (21, 3),
        (31, 3),
        (3, 13),
        (31, 13),
        (3, 21),
        (31, 21),
        (3, 31),
        (13, 31),
        (21, 31),
        (31, 31),
    }
)
ROWS = tuple(
    (b, tuple(a for a in BASE_SET if (a, b) not in EXCLUDED)) for b in BASE_SET
)
SEED = 4018
SEARCH_SECONDS = 155.0
RESTARTS = 32
MIN_BASE = 40
MAX_BASE = 94
MIN_OFFSET = -24
MAX_OFFSET = 24
START_TEMPERATURE = 0.025
END_TEMPERATURE = 0.00001


def exact_score(base, offsets):
    """Return the exact objective and deduplicated embedded set."""
    value_bits = 0
    for row_index, (b, columns) in enumerate(ROWS):
        origin = base * b + offsets[row_index]
        for a in columns:
            value_bits |= 1 << (origin + a)

    values = []
    remaining = value_bits
    while remaining:
        low_bit = remaining & -remaining
        values.append(low_bit.bit_length() - 1)
        remaining ^= low_bit

    sum_bits = 0
    nonnegative_diff_bits = 0
    for value in values:
        sum_bits |= value_bits << value
        nonnegative_diff_bits |= value_bits >> value

    n = len(values)
    sums = sum_bits.bit_count()
    diffs = 2 * nonnegative_diff_bits.bit_count() - 1
    score = math.log(sums / n) / math.log(diffs / n)
    return score, tuple(values)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def initial_states(rng):
    """Build 32 deterministic restarts, including the exact parent."""
    zero_offsets = (0,) * len(ROWS)
    parent_score, parent_values = exact_score(67, zero_offsets)
    states = [[67, zero_offsets, parent_score, parent_values]]

    for restart in range(1, RESTARTS):
        base = rng.randint(MIN_BASE, MAX_BASE)
        # Mix local starts with broad starts so several chains remain near the
        # known good carry-free embedding while others test nonlinear geometry.
        radius = (2, 6, 12, 24)[restart % 4]
        offsets = [0]
        offsets.extend(rng.randint(-radius, radius) for _ in ROWS[1:])
        score, values = exact_score(base, offsets)
        states.append([base, tuple(offsets), score, values])
    return states


def propose(rng, base, offsets):
    """Mutate one to three row offsets, with occasional B +/- 1 moves."""
    if rng.random() < 0.10:
        step = -1 if rng.random() < 0.5 else 1
        candidate_base = base + step
        if not MIN_BASE <= candidate_base <= MAX_BASE:
            candidate_base = base - step
        return candidate_base, offsets

    candidate = list(offsets)
    count = rng.randint(1, 3)
    for row_index in rng.sample(range(1, len(ROWS)), count):
        if rng.random() < 0.85:
            change = rng.choice((-3, -2, -1, 1, 2, 3))
            candidate[row_index] = max(
                MIN_OFFSET, min(MAX_OFFSET, candidate[row_index] + change)
            )
        else:
            candidate[row_index] = rng.randint(MIN_OFFSET, MAX_OFFSET)
    return base, tuple(candidate)


def main():
    rng = random.Random(SEED)
    states = initial_states(rng)

    best_score = states[0][2]
    best_values = states[0][3]
    write_solution(best_values)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    proposals = 0

    while True:
        for state in states:
            now = time.monotonic()
            if now >= deadline:
                write_solution(best_values)
                print(
                    f"wrote row-warp best: n={len(best_values)} "
                    f"score={best_score:.9f} proposals={proposals}"
                )
                return

            progress = (now - started) / SEARCH_SECONDS
            temperature = START_TEMPERATURE * (
                END_TEMPERATURE / START_TEMPERATURE
            ) ** progress

            candidate_base, candidate_offsets = propose(rng, state[0], state[1])
            candidate_score, candidate_values = exact_score(
                candidate_base, candidate_offsets
            )
            proposals += 1
            delta = candidate_score - state[2]

            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state[:] = [
                    candidate_base,
                    candidate_offsets,
                    candidate_score,
                    candidate_values,
                ]
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_values = candidate_values
                    write_solution(best_values)


if __name__ == "__main__":
    main()
