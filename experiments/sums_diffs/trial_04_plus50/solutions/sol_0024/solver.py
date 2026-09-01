"""Parallel tempering around the best linear tensor projection."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
SEED = 5023
SEARCH_SECONDS = 155.0
REPLICAS = 24
Q_MIN = 48
Q_MAX = 64


def image(mask, q):
    values = {
        x + q * y
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


def evaluate(mask, q):
    values = image(mask, q)
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

    best_score, best_values = evaluate(PARENT_MASK, 56)
    best_mask, best_q = PARENT_MASK, 56
    write_solution(best_values)

    temperatures = [
        0.00003 * (0.012 / 0.00003) ** (i / (REPLICAS - 1))
        for i in range(REPLICAS)
    ]
    states = []
    for i in range(REPLICAS):
        q = Q_MIN + (i * (Q_MAX - Q_MIN)) // (REPLICAS - 1)
        candidate_score, candidate_values = evaluate(PARENT_MASK, q)
        states.append([PARENT_MASK, q, candidate_score, candidate_values])
        if candidate_score > best_score:
            best_score, best_values = candidate_score, candidate_values
            best_mask, best_q = PARENT_MASK, q
            write_solution(best_values)

    all_indices = tuple(range(len(CELLS)))
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        replica = iteration % REPLICAS
        mask, q, current_score, current_values = states[replica]
        new_mask, new_q = mask, q
        move = rng.random()
        if move < 0.15:
            new_q = min(Q_MAX, max(Q_MIN, q + rng.choice((-2, -1, 1, 2))))
        elif move < 0.60:
            for index in rng.sample(all_indices, rng.randint(1, 6)):
                new_mask ^= 1 << index
        elif move < 0.82:
            occupied = [i for i in all_indices if (mask >> i) & 1]
            empty = [i for i in all_indices if not (mask >> i) & 1]
            if occupied and empty:
                new_mask ^= (1 << rng.choice(occupied)) | (1 << rng.choice(empty))
        else:
            occupied = [i for i in all_indices if (mask >> i) & 1]
            empty = [i for i in all_indices if not (mask >> i) & 1]
            count = min(rng.randint(2, 6), len(occupied), len(empty))
            for index in rng.sample(occupied, count) + rng.sample(empty, count):
                new_mask ^= 1 << index

        population = new_mask.bit_count()
        if 200 <= population <= 350:
            new_score, new_values = evaluate(new_mask, new_q)
            delta = new_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperatures[replica]):
                states[replica] = [new_mask, new_q, new_score, new_values]
                if new_score > best_score:
                    best_score, best_values = new_score, new_values
                    best_mask, best_q = new_mask, new_q
                    write_solution(best_values)

        # Adjacent-temperature exchanges let hot replicas feed new basins down.
        if iteration % REPLICAS == 0:
            parity = (iteration // REPLICAS) & 1
            for low in range(parity, REPLICAS - 1, 2):
                high = low + 1
                delta = (states[high][2] - states[low][2]) * (
                    1.0 / temperatures[low] - 1.0 / temperatures[high]
                )
                if delta >= 0.0 or rng.random() < math.exp(delta):
                    states[low], states[high] = states[high], states[low]

        if iteration % 12000 == 0:
            states[0] = [best_mask, best_q, best_score, best_values]

    write_solution(best_values)
    print(
        f"wrote tempered tensor best: n={len(best_values)} "
        f"q={best_q} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
