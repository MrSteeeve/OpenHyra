"""Focused multi-point refinement of the exact sol_0036 dense-fringe set."""

import json
import math
import os
import random
import time


FALLBACK = (0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 132, 133, 134, 137,
            139, 143, 147, 151, 155, 156, 157, 158)
SEED = 5048
SEARCH_SECONDS = 145.0
DOMAIN = tuple(range(0, 181))


def score(values):
    n = len(values)
    if not 48 <= n <= 90:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    best = FALLBACK
    best_score = score(best)
    write_solution(best)

    # All replicas begin close to the proven basin, but at different radii.
    chains = []
    base = set(FALLBACK)
    for i in range(32):
        state = set(base)
        for _ in range(i // 4):
            x = rng.choice(DOMAIN)
            if x != 0:
                state.symmetric_difference_update((x,))
        values = tuple(sorted(state))
        chains.append([values, score(values)])

    steps = 0
    while time.monotonic() < deadline:
        steps += 1
        k = steps & 31
        old, old_score = chains[k]
        state = set(old)
        move = rng.random()

        if move < 0.30:
            # Ordinary toggle, concentrated around the two core boundaries.
            if rng.random() < 0.72:
                x = rng.choice(tuple(range(42, 64)) + tuple(range(96, 119)))
            else:
                x = rng.choice(DOMAIN)
            if x:
                state.symmetric_difference_update((x,))
        elif move < 0.55:
            # A balanced replacement preserves n while crossing swap barriers.
            count = rng.choice((2, 2, 3, 4))
            present = list(state - {0})
            absent = list(set(DOMAIN) - state)
            if len(present) >= count and len(absent) >= count:
                state.difference_update(rng.sample(present, count))
                state.update(rng.sample(absent, count))
        elif move < 0.78:
            # Shift a short run at either edge of the dense core.
            left = rng.random() < 0.5
            anchor = 53 if left else 105
            length = rng.randint(2, 7)
            delta = rng.choice((-3, -2, -1, 1, 2, 3))
            run = range(anchor, anchor + length) if left else range(anchor - length + 1, anchor + 1)
            for x in run:
                state.discard(x)
                y = x + delta
                if 0 < y <= 180:
                    state.add(y)
        else:
            # Coordinated mirror/complement flips in the two empty gaps.
            count = rng.randint(2, 6)
            for _ in range(count):
                x = rng.randint(1, 52)
                y = 158 - x
                state.symmetric_difference_update((x,))
                if 0 < y <= 180:
                    state.symmetric_difference_update((y,))

        state.add(0)
        values = tuple(sorted(state))
        new_score = score(values)
        progress = (time.monotonic() - start) / SEARCH_SECONDS
        # Replica-dependent temperatures plus slow cooling retain barrier crossing.
        temperature = (0.00002 + 0.0018 * (k / 31) ** 2) * (1.0 - 0.65 * progress)
        if new_score >= old_score or rng.random() < math.exp((new_score - old_score) / temperature):
            chains[k] = [values, new_score]
            if new_score > best_score + 1e-15:
                best, best_score = values, new_score
                write_solution(best)

        # Periodically inject the incumbent into hot replicas with a 3--6 point kick.
        if steps % 12000 == 0:
            target = rng.randrange(12, 32)
            state = set(best)
            for _ in range(rng.randint(3, 6)):
                x = rng.randrange(1, 181)
                state.symmetric_difference_update((x,))
            values = tuple(sorted(state))
            chains[target] = [values, score(values)]

    write_solution(best)
    print(f"wrote focused dense-fringe best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
