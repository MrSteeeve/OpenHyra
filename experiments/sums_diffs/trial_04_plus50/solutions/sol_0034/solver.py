"""Quadratic projections of the best carry-free tensor subset."""

import heapq
import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
SEED = 5033
SEARCH_SECONDS = 150.0


def image(mask, q, c, d, e):
    return tuple(sorted({
        q * x + y + c * x * y + d * x * x + e * y * y
        for i, (x, y) in enumerate(CELLS) if (mask >> i) & 1
    }))


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    lo = values[0]
    shifted = tuple(v - lo for v in values)
    span = shifted[-1]
    bits = sum(1 << v for v in shifted)
    reverse = sum(1 << (span - v) for v in shifted)
    sum_bits = 0
    difference_bits = 0
    for v in shifted:
        sum_bits |= bits << v
        difference_bits |= reverse << v
    sums = sum_bits.bit_count()
    diffs = difference_bits.bit_count()
    return math.log(sums / n) / math.log(diffs / n)


def evaluate(mask, q, c, d, e):
    values = image(mask, q, c, d, e)
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

    # This is the exact sol_0023 projection and is written before any search.
    best_score, best_values = evaluate(PARENT_MASK, 56, 0, 0, 0)
    best_state = (PARENT_MASK, 56, 0, 0, 0)
    write_solution(best_values)

    # Keep the 32 best distinct images from the complete coefficient box.
    leaders = []
    serial = 0
    for q in range(32, 81):
        for c in range(-6, 7):
            for d in range(-6, 7):
                for e in range(-6, 7):
                    values = image(PARENT_MASK, q, c, d, e)
                    candidate_score = score_values(values)
                    state = (PARENT_MASK, q, c, d, e)
                    serial += 1
                    item = (candidate_score, serial, state, values)
                    if len(leaders) < 32:
                        heapq.heappush(leaders, item)
                    elif candidate_score > leaders[0][0]:
                        heapq.heapreplace(leaders, item)
                    if candidate_score > best_score:
                        best_score, best_values, best_state = candidate_score, values, state
                        write_solution(best_values)

    # Anneal all retained basins, changing coefficients or 1--3 cells.
    replicas = []
    for candidate_score, _, state, values in sorted(leaders, reverse=True):
        replicas.append([state, candidate_score, values])
    iteration = 0
    while replicas and time.monotonic() < deadline:
        iteration += 1
        replica = replicas[iteration % len(replicas)]
        (mask, q, c, d, e), current_score, _ = replica
        new_mask, new_q, new_c, new_d, new_e = mask, q, c, d, e
        move = rng.random()
        if move < 0.18:
            new_q = min(96, max(20, q + rng.choice((-3, -2, -1, 1, 2, 3))))
        elif move < 0.36:
            new_c = min(10, max(-10, c + rng.choice((-2, -1, 1, 2))))
        elif move < 0.54:
            new_d = min(10, max(-10, d + rng.choice((-2, -1, 1, 2))))
        elif move < 0.72:
            new_e = min(10, max(-10, e + rng.choice((-2, -1, 1, 2))))
        else:
            for index in rng.sample(range(len(CELLS)), rng.randint(1, 3)):
                new_mask ^= 1 << index
        new_state = (new_mask, new_q, new_c, new_d, new_e)
        new_score, new_values = evaluate(*new_state)
        remaining = max(0.0, deadline - time.monotonic())
        temperature = 0.004 * (remaining / SEARCH_SECONDS) + 0.00002
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            replica[:] = [new_state, new_score, new_values]
            if new_score > best_score:
                best_score, best_values, best_state = new_score, new_values, new_state
                write_solution(best_values)

    write_solution(best_values)
    _, q, c, d, e = best_state
    print(
        f"wrote quadratic tensor best: n={len(best_values)} q={q} "
        f"c={c} d={d} e={e} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
