"""Anneal asymmetric fringes around a long interval core."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
PARENT = tuple(
    sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if (PARENT_MASK >> i) & 1)
)
INCUMBENT = tuple(sorted(set(PARENT).union(x + 1568 for x in PARENT)))
WIDTHS = (16, 24, 32, 40, 48, 64)
SEED = 5034
SEARCH_SECONDS = 150.0


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = 0
    distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    sum_count = sums.bit_count()
    diff_count = 2 * distances.bit_count() - 1
    return math.log(sum_count / n) / math.log(diff_count / n)


def make_values(w, m, left, right):
    values = [i for i in range(w) if (left >> i) & 1]
    values.extend(range(w, w + m))
    values.extend(m + w + i for i in range(w) if (right >> i) & 1)
    return tuple(values)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    best_values = INCUMBENT
    best_score = score_values(best_values)
    write_solution(best_values)

    # Four independently randomized chains at each allowed fringe width.
    chains = []
    for w in WIDTHS:
        for density in (0.25, 0.45, 0.65, 0.85):
            left = sum(1 << i for i in range(w) if rng.random() < density)
            right = sum(1 << i for i in range(w) if rng.random() < density)
            max_m = min(400, 512 - left.bit_count() - right.bit_count())
            m = rng.randint(64, max_m)
            values = make_values(w, m, left, right)
            chains.append([score_values(values), w, m, left, right, values])

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        old_score, w, m, left, right, old_values = chain
        new_m, new_left, new_right = m, left, right
        move = rng.random()

        if move < 0.58:
            bit = 1 << rng.randrange(w)
            if rng.random() < 0.5:
                new_left ^= bit
            else:
                new_right ^= bit
        elif move < 0.86:
            # Coupled flips can move occupancy between the two boundaries.
            li = 1 << rng.randrange(w)
            ri = 1 << rng.randrange(w)
            new_left ^= li
            new_right ^= ri
        else:
            step = rng.choice((-16, -8, -4, -2, -1, 1, 2, 4, 8, 16))
            capacity = 512 - new_left.bit_count() - new_right.bit_count()
            new_m = min(400, capacity, max(64, m + step))

        new_values = make_values(w, new_m, new_left, new_right)
        new_score = score_values(new_values)
        elapsed = SEARCH_SECONDS - max(0.0, deadline - time.monotonic())
        fraction = min(1.0, elapsed / SEARCH_SECONDS)
        temperature = 0.0025 * (1.0 - fraction) + 0.000015
        if new_score >= old_score or rng.random() < math.exp((new_score - old_score) / temperature):
            chain[:] = [new_score, w, new_m, new_left, new_right, new_values]
            if new_score > best_score:
                best_score, best_values = new_score, new_values
                write_solution(best_values)

        # Reheat the weakest chain without discarding strong discovered basins.
        if iteration % 12000 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            _, rw, _, _, _, _ = chains[weakest]
            density = rng.uniform(0.15, 0.9)
            rl = sum(1 << i for i in range(rw) if rng.random() < density)
            rr = sum(1 << i for i in range(rw) if rng.random() < density)
            max_m = min(400, 512 - rl.bit_count() - rr.bit_count())
            rm = rng.randint(64, max_m)
            rv = make_values(rw, rm, rl, rr)
            chains[weakest] = [score_values(rv), rw, rm, rl, rr, rv]

    write_solution(best_values)
    print(f"wrote core-fringe best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
