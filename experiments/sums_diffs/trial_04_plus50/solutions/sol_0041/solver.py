"""Search unions of translated and reflected copies of sol_0036."""

import json
import math
import os
import random
import time


PARENT = (0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
          67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
          82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
          97, 98, 99, 100, 101, 102, 103, 104, 105, 132, 133, 134, 137,
          139, 143, 147, 151, 155, 156, 157, 158)
SEED = 5040
SEARCH_SECONDS = 150.0
REPLICAS = 32


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    lo = values[0]
    shifted = tuple(x - lo for x in values)
    mask = sum(1 << x for x in shifted)
    sums = positive_differences = 0
    for x in shifted:
        sums |= mask << x
        positive_differences |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * positive_differences.bit_count() - 1) / n)


def materialize(copies):
    values = set()
    for orientation, translation in copies:
        if orientation == 1:
            values.update(x + translation for x in PARENT)
        else:
            values.update(translation - x for x in PARENT)
    return tuple(sorted(values))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def canonical(copies):
    # The anchored first copy removes irrelevant global translation. Duplicate
    # copy descriptions cannot change the union, so discard them.
    rest = sorted(set(copies[1:]))
    return tuple([(1, 0)] + rest)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = PARENT
    best_score = score(PARENT)
    write_solution(PARENT)

    # Exhaust every placement of a second copy before starting the stochastic
    # phase. Keep diverse high-scoring placements as tempering seeds.
    seeds = []
    for orientation in (-1, 1):
        for translation in range(-600, 601):
            copies = canonical(((1, 0), (orientation, translation)))
            if len(copies) != 2:
                continue
            values = materialize(copies)
            value = score(values)
            seeds.append((value, copies, values))
            if value > best_score:
                best_score, best_values = value, values
                write_solution(values)
    seeds.sort(key=lambda item: item[0], reverse=True)

    # Spread initial replicas across leading placements rather than cloning a
    # single basin. Each chain stores score, copy descriptions, and its union.
    chains = []
    stride = max(1, min(20, len(seeds) // REPLICAS))
    for i in range(REPLICAS):
        value, copies, values = seeds[min(i * stride, len(seeds) - 1)]
        chains.append([value, copies, values])
    temperatures = [0.00002 * (1000.0 ** (i / (REPLICAS - 1)))
                    for i in range(REPLICAS)]

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        r = iteration % REPLICAS
        old_score, old_copies, _ = chains[r]
        copies = list(old_copies)
        move = rng.random()

        if move < 0.15 and len(copies) < 7:
            copies.append((rng.choice((-1, 1)), rng.randint(-600, 600)))
        elif move < 0.25 and len(copies) > 2:
            del copies[rng.randrange(1, len(copies))]
        elif move < 0.40:
            j = rng.randrange(1, len(copies))
            copies[j] = (-copies[j][0], copies[j][1])
        else:
            j = rng.randrange(1, len(copies))
            orientation, translation = copies[j]
            if move < 0.82:
                translation += rng.choice((-32, -16, -8, -4, -2, -1,
                                            1, 2, 4, 8, 16, 32))
            else:
                translation = rng.randint(-600, 600)
            translation = max(-600, min(600, translation))
            copies[j] = (orientation, translation)

        new_copies = canonical(copies)
        if not 2 <= len(new_copies) <= 7:
            continue
        values = materialize(new_copies)
        if len(values) > 512:
            continue
        new_score = score(values)
        temperature = temperatures[r]
        if (new_score >= old_score or
                rng.random() < math.exp((new_score - old_score) / temperature)):
            chains[r] = [new_score, new_copies, values]
            if new_score > best_score:
                best_score, best_values = new_score, values
                write_solution(values)

        # Standard adjacent replica exchange lets cold replicas inherit large
        # structural moves discovered by the hotter chains.
        if iteration % REPLICAS == 0:
            parity = (iteration // REPLICAS) & 1
            for j in range(parity, REPLICAS - 1, 2):
                s0, s1 = chains[j][0], chains[j + 1][0]
                exponent = (1.0 / temperatures[j] - 1.0 / temperatures[j + 1]) * (s1 - s0)
                if exponent >= 0.0 or rng.random() < math.exp(max(-700.0, exponent)):
                    chains[j], chains[j + 1] = chains[j + 1], chains[j]

    write_solution(best_values)
    print(f"wrote copy-union best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
