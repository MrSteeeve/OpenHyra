"""Anneal dense intervals with independently evolving asymmetric fringes."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
P = tuple(sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if i not in MISSING))
FALLBACK = tuple(sorted(set(P).union(x + 1568 for x in P)))
SEED = 5035
SEARCH_SECONDS = 150.0


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log((2 * distances.bit_count() - 1) / n)


def materialize(width, core, left, right):
    span = 2 * width + core - 1
    values = set(range(width, width + core))
    values.update(left)
    values.update(span - x for x in right)
    return tuple(sorted(values))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def fresh(rng):
    width = rng.randint(16, 64)
    core = rng.randint(48, 260)
    p = rng.uniform(0.25, 0.75)
    left = {x for x in range(width) if rng.random() < p}
    right = {x for x in range(width) if rng.random() < p}
    left.add(0)
    right.add(0)
    values = materialize(width, core, left, right)
    return [score(values), width, core, left, right, values]


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = FALLBACK
    best_score = score(FALLBACK)
    write_solution(FALLBACK)

    # Include the classical small MSTD fringe as a useful structured basin.
    seed = set(BASE)
    chains = []
    for core in (48, 72, 96, 128):
        values = materialize(34, core, seed, {33 - x for x in seed})
        chains.append([score(values), 34, core, set(seed), {33 - x for x in seed}, values])
    chains.extend(fresh(rng) for _ in range(28))

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        old_score, width, core, left, right, _ = chain
        new_width, new_core = width, core
        new_left, new_right = set(left), set(right)
        move = rng.random()
        if move < 0.66:
            side = new_left if rng.random() < 0.5 else new_right
            x = rng.randrange(width)
            if x == 0:
                continue
            side.symmetric_difference_update((x,))
        elif move < 0.88:
            # Correlated cross-fringe flips often preserve differences while
            # breaking just enough symmetry to gain sums.
            x = rng.randrange(1, width)
            y = width - 1 - x
            new_left.symmetric_difference_update((x,))
            if y:
                new_right.symmetric_difference_update((y,))
        elif move < 0.96:
            new_core = max(24, min(400, core + rng.choice((-4, -2, -1, 1, 2, 4))))
        else:
            delta = rng.choice((-1, 1))
            new_width = max(12, min(72, width + delta))
            if delta > 0:
                if rng.random() < 0.5:
                    new_left.add(width)
                if rng.random() < 0.5:
                    new_right.add(width)
            else:
                new_left.discard(new_width)
                new_right.discard(new_width)

        values = materialize(new_width, new_core, new_left, new_right)
        new_score = score(values)
        remaining = max(0.0, deadline - time.monotonic())
        temperature = 0.003 * (remaining / SEARCH_SECONDS) + 0.000015
        if new_score >= old_score or rng.random() < math.exp((new_score - old_score) / temperature):
            chain[:] = [new_score, new_width, new_core, new_left, new_right, values]
            if new_score > best_score:
                best_score, best_values = new_score, values
                write_solution(values)

        if iteration % 5000 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            if rng.random() < 0.6:
                leader = max(chains, key=lambda z: z[0])
                clone = [leader[0], leader[1], leader[2], set(leader[3]), set(leader[4]), leader[5]]
                for _ in range(8):
                    side = clone[3] if rng.random() < 0.5 else clone[4]
                    side.symmetric_difference_update((rng.randrange(1, clone[1]),))
                clone[5] = materialize(clone[1], clone[2], clone[3], clone[4])
                clone[0] = score(clone[5])
                chains[weakest] = clone
            else:
                chains[weakest] = fresh(rng)

    write_solution(best_values)
    print(f"wrote dense-fringe best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
