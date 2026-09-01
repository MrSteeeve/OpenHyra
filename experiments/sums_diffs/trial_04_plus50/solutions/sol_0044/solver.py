"""Anneal subsets of three-digit mixed-radix product sets."""

import json
import math
import os
import random
import time


FALLBACK = (
    0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 132, 133, 134, 137, 139, 143, 147,
    151, 155, 156, 157, 158,
)
CLASSIC = (0, 2, 3, 4, 7, 11, 12, 14)
SEED = 5043
SEARCH_SECONDS = 150.0


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    bits = sum(1 << x for x in values)
    sums = 0
    nonnegative_differences = 0
    for x in values:
        sums |= bits << x
        nonnegative_differences |= bits >> x
    sn = sums.bit_count()
    dn = 2 * nonnegative_differences.bit_count() - 1
    return math.log(sn / n) / math.log(dn / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def digit_score(digits):
    return score(tuple(sorted(digits)))


def make_digit_sets(rng):
    """Quickly generate locally strong 6--8 point seeds in [0, 24]."""
    pool = {CLASSIC}
    finish = time.monotonic() + 1.5
    while time.monotonic() < finish:
        size = rng.randint(6, 8)
        current = set(rng.sample(range(25), size))
        current.add(0)
        while len(current) > size:
            current.remove(rng.choice(tuple(current - {0})))
        old = digit_score(current)
        for _ in range(80):
            remove = rng.choice(tuple(current - {0}))
            add = rng.choice(tuple(set(range(25)) - current))
            trial = current - {remove} | {add}
            new = digit_score(trial)
            if new >= old or rng.random() < 0.025:
                current, old = trial, new
        pool.add(tuple(sorted(current)))
    return sorted(pool, key=digit_score, reverse=True)[:12]


def coordinates(digits, q, occupancy):
    cells = [x + q * y + q * q * z for z in digits for y in digits for x in digits]
    return tuple(cells[i] for i in range(len(cells)) if occupancy >> i & 1)


def initial_mask(count, rng):
    if count <= 512:
        return (1 << count) - 1
    chosen = rng.sample(range(count), 500)
    return sum(1 << i for i in chosen)


def main():
    rng = random.Random(SEED)
    best_values = FALLBACK
    best_score = score(FALLBACK)
    write_solution(FALLBACK)

    digit_sets = make_digit_sets(rng)
    deadline = time.monotonic() + SEARCH_SECONDS
    chains = []
    for i in range(16):
        digits = digit_sets[i % len(digit_sets)]
        q = rng.randint(32, 64)
        mask = initial_mask(len(digits) ** 3, rng)
        values = coordinates(digits, q, mask)
        state_score = score(values)
        chains.append([state_score, digits, q, mask, values])
        if state_score > best_score:
            best_score, best_values = state_score, values
            write_solution(values)

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        old_score, digits, q, mask, _ = chain
        side = len(digits)
        count = side ** 3
        new_q = q
        new_mask = mask
        move = rng.random()
        if move < 0.78:
            for _ in range(rng.randint(1, 8)):
                new_mask ^= 1 << rng.randrange(count)
        elif move < 0.94:
            # Replace all occupancy decisions in one x/y/z slice.
            axis = rng.randrange(3)
            layer = rng.randrange(side)
            density = min(0.98, max(0.20, mask.bit_count() / count + rng.uniform(-0.18, 0.18)))
            for z in range(side):
                for y in range(side):
                    for x in range(side):
                        if (x, y, z)[axis] == layer:
                            bit = 1 << (x + side * (y + side * z))
                            if rng.random() < density:
                                new_mask |= bit
                            else:
                                new_mask &= ~bit
        else:
            new_q = max(32, min(64, q + rng.choice((-3, -2, -1, 1, 2, 3))))

        n = new_mask.bit_count()
        if n < 48 or n > 512:
            continue
        values = coordinates(digits, new_q, new_mask)
        new_score = score(values)
        fraction = max(0.0, (deadline - time.monotonic()) / SEARCH_SECONDS)
        temperature = 0.006 * fraction + 0.00002
        if new_score >= old_score or rng.random() < math.exp((new_score - old_score) / temperature):
            chain[:] = [new_score, digits, new_q, new_mask, values]
            if new_score > best_score:
                best_score, best_values = new_score, values
                write_solution(values)

        if iteration % 4000 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            leader = max(chains, key=lambda state: state[0])
            clone = [leader[0], leader[1], leader[2], leader[3], leader[4]]
            for _ in range(16):
                clone[3] ^= 1 << rng.randrange(len(clone[1]) ** 3)
            clone[4] = coordinates(clone[1], clone[2], clone[3])
            clone[0] = score(clone[4])
            chains[weakest] = clone

    write_solution(best_values)
    print(f"wrote mixed-radix best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
