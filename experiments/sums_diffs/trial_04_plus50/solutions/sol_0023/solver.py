"""Nonlinear projections of the best carry-free tensor subset."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
SEED = 5022
SEARCH_SECONDS = 155.0


def image(mask, q, c):
    values = {
        x + q * y + c * x * y
        for i, (x, y) in enumerate(CELLS)
        if (mask >> i) & 1
    }
    return tuple(sorted(values))


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    lo = values[0]
    shifted = [x - lo for x in values]
    sum_bits = 0
    distance_bits = 1
    for i, x in enumerate(shifted):
        for y in shifted[i:]:
            sum_bits |= 1 << (x + y)
        for y in shifted[:i]:
            distance_bits |= 1 << (x - y)
    sums = sum_bits.bit_count()
    diffs = 2 * distance_bits.bit_count() - 1
    return math.log(sums / n) / math.log(diffs / n)


def evaluate(mask, q, c):
    values = image(mask, q, c)
    return score_values(values), values


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    best_score, best_values = evaluate(PARENT_MASK, 67, 0)
    best_mask, best_q, best_c = PARENT_MASK, 67, 0
    write_solution(best_values)

    # Exhaustive nonlinear projection sweep of the exact occupied cell set.
    for q in range(20, 81):
        for c in range(-8, 9):
            candidate_score, candidate_values = evaluate(PARENT_MASK, q, c)
            if candidate_score > best_score:
                best_score, best_values = candidate_score, candidate_values
                best_mask, best_q, best_c = PARENT_MASK, q, c
                write_solution(best_values)

    # Anneal projection parameters and small edits of the 17 by 17 cell mask.
    mask, q, c = best_mask, best_q, best_c
    current_score, current_values = evaluate(mask, q, c)
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        new_mask, new_q, new_c = mask, q, c
        move = rng.random()
        if move < 0.25:
            new_q = min(80, max(20, q + rng.choice((-3, -2, -1, 1, 2, 3))))
        elif move < 0.50:
            new_c = min(8, max(-8, c + rng.choice((-2, -1, 1, 2))))
        else:
            for index in rng.sample(range(len(CELLS)), rng.randint(1, 3)):
                new_mask ^= 1 << index

        new_score, new_values = evaluate(new_mask, new_q, new_c)
        elapsed = SEARCH_SECONDS - max(0.0, deadline - time.monotonic())
        fraction = min(1.0, elapsed / SEARCH_SECONDS)
        temperature = 0.006 * (1.0 - fraction) + 0.00003
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            mask, q, c = new_mask, new_q, new_c
            current_score, current_values = new_score, new_values
            if current_score > best_score:
                best_score, best_values = current_score, current_values
                best_mask, best_q, best_c = mask, q, c
                write_solution(best_values)

        # Periodically return to the best basin instead of cooling in a weak one.
        if iteration % 4000 == 0:
            mask, q, c = best_mask, best_q, best_c
            current_score, current_values = best_score, best_values

    write_solution(best_values)
    print(
        f"wrote nonlinear tensor best: n={len(best_values)} "
        f"q={best_q} c={best_c} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
