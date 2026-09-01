"""Island genetic search over dense cores with wide asymmetric fringes."""

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
SEED = 5041
SEARCH_SECONDS = 150.0
ISLANDS = 16
ISLAND_SIZE = 32


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n)


def materialize(width, core_end, left, right):
    values = set(range(width, width + core_end + 1))
    values.update(x for x in range(width) if left >> x & 1)
    values.update(2 * width + core_end - x for x in range(width)
                  if right >> x & 1)
    return tuple(sorted(values))


def make(width, core_end, left, right):
    fringe_mask = (1 << width) - 1
    left &= fringe_mask
    right &= fringe_mask
    values = materialize(width, core_end, left, right)
    return (score(values), width, core_end, left, right, values)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def fresh(rng, island):
    width = rng.randint(40, 180)
    core_end = rng.randint(20, min(300, 510 - width // 2))
    density = rng.uniform(0.06, min(0.72, (490 - core_end) / (2 * width)))
    left = right = 0
    for x in range(width):
        if rng.random() < density:
            left |= 1 << x
        if rng.random() < density:
            right |= 1 << x
    # Give different islands distinct starting topology biases.
    if island & 1:
        left |= 1
    if island & 2:
        right |= 1
    return make(width, core_end, left, right)


def tournament(population, rng):
    choices = rng.sample(population, 4)
    return max(choices, key=lambda individual: individual[0])


def child_of(first, second, rng):
    geometry = first if rng.random() < 0.5 else second
    width, core_end = geometry[1], geometry[2]
    a, b = sorted(rng.sample(range(width + 1), 2))
    middle = ((1 << b) - 1) ^ ((1 << a) - 1)
    full = (1 << width) - 1
    left = (first[3] & ~middle) | (second[3] & middle)
    right = (second[4] & ~middle) | (first[4] & middle)
    left &= full
    right &= full

    # Mutate a correlated group of fringe coordinates.  Reflected and nearby
    # partners let useful wide-fringe features move jointly.
    for _ in range(rng.randint(2, 12)):
        x = rng.randrange(width)
        side = rng.randrange(2)
        if side:
            right ^= 1 << x
        else:
            left ^= 1 << x
        if rng.random() < 0.55:
            partner = width - 1 - x
            if side:
                left ^= 1 << partner
            else:
                right ^= 1 << partner
        if rng.random() < 0.18:
            nearby = max(0, min(width - 1, x + rng.choice((-2, -1, 1, 2))))
            if side:
                right ^= 1 << nearby
            else:
                left ^= 1 << nearby

    if rng.random() < 0.12:
        new_width = max(40, min(180, width + rng.choice((-8, -4, -2, 2, 4, 8))))
        if new_width > width:
            for x in range(width, new_width):
                if rng.random() < 0.25:
                    left |= 1 << x
                if rng.random() < 0.25:
                    right |= 1 << x
        width = new_width
    if rng.random() < 0.16:
        core_end = max(20, min(300, core_end + rng.choice(
            (-16, -8, -4, -2, 2, 4, 8, 16))))
    return make(width, core_end, left, right)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best = FALLBACK
    best_score = score(best)
    write_solution(best)

    islands = [[fresh(rng, i) for _ in range(ISLAND_SIZE)]
               for i in range(ISLANDS)]
    for population in islands:
        for individual in population:
            if individual[0] > best_score + 1e-15:
                best_score, best = individual[0], individual[5]
                write_solution(best)
    generation = 0
    while time.monotonic() < deadline:
        generation += 1
        for island_number, population in enumerate(islands):
            if time.monotonic() >= deadline:
                break
            population.sort(key=lambda individual: individual[0], reverse=True)
            next_population = population[:4]
            while len(next_population) < ISLAND_SIZE and time.monotonic() < deadline:
                first = tournament(population, rng)
                second = tournament(population, rng)
                child = child_of(first, second, rng)
                next_population.append(child)
                if child[0] > best_score + 1e-15:
                    best_score, best = child[0], child[5]
                    write_solution(best)
            while len(next_population) < ISLAND_SIZE:
                next_population.append(population[len(next_population)])
            islands[island_number] = next_population

        # Ring migration prevents early convergence while retaining separate
        # fringe-topology niches.
        if generation % 8 == 0:
            migrants = [max(population, key=lambda individual: individual[0])
                        for population in islands]
            for i in range(ISLANDS):
                weakest = min(range(ISLAND_SIZE),
                              key=lambda j: islands[(i + 1) % ISLANDS][j][0])
                islands[(i + 1) % ISLANDS][weakest] = migrants[i]
        if generation % 24 == 0:
            for i in range(ISLANDS):
                weakest = min(range(ISLAND_SIZE), key=lambda j: islands[i][j][0])
                islands[i][weakest] = fresh(rng, i)

    write_solution(best)
    print(f"wrote island-genetic best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
