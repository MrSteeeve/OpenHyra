"""Deterministic island genetic search for small sum-dominant templates."""

import json
import math
import os
import random
import time


SEED = 20260795782205
SEARCH_SECONDS = 170.0
ISLANDS = 16
POPULATION = 128
MIN_N = 8
MAX_N = 40
LIMIT = 120

def parent_set():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_solution.json")
    with open(path) as stream:
        return frozenset(json.load(stream)["A"])


def canonical(values):
    values = sorted(set(values))
    if not values:
        return ()
    offset = values[0]
    values = [x - offset for x in values]
    divisor = 0
    for x in values[1:]:
        divisor = math.gcd(divisor, x)
    if divisor > 1:
        values = [x // divisor for x in values]
    return tuple(values)


def metrics(candidate):
    sums = {a + b for a in candidate for b in candidate}
    diffs = {a - b for a in candidate for b in candidate}
    n = len(candidate)
    score = math.log(len(sums) / n) / math.log(len(diffs) / n)
    return score, len(sums), len(diffs)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def random_candidate(rng):
    n = rng.randint(MIN_N, MAX_N)
    span = rng.randint(max(n - 1, 20), LIMIT)
    return canonical((0, span, *rng.sample(range(1, span), n - 2)))


def repair(values, rng):
    values = {x for x in values if 0 <= x <= LIMIT}
    while len(values) > MAX_N:
        values.remove(rng.choice(tuple(values)))
    while len(values) < MIN_N:
        values.add(rng.randrange(LIMIT + 1))
    return canonical(values)


def crossover(left, right, rng):
    move = rng.randrange(3)
    if move == 0:
        child = set(left) | set(right)
    elif move == 1:
        child = set(left) & set(right)
        if len(child) < MIN_N:
            child |= set(rng.sample(left, min(len(left), MIN_N - len(child))))
            child |= set(rng.sample(right, min(len(right), MIN_N - len(child))))
    else:
        # Align spans before cutting, so the crossover is geometric rather
        # than dependent on the parents having identical end points.
        cut = rng.randint(1, LIMIT - 1)
        lspan, rspan = max(left), max(right)
        lscaled = {round(x * LIMIT / lspan) for x in left} if lspan else {0}
        rscaled = {round(x * LIMIT / rspan) for x in right} if rspan else {0}
        child = {x for x in lscaled if x <= cut} | {x for x in rscaled if x > cut}
    return repair(child, rng)


def mutate(candidate, rng):
    values = set(candidate)
    for _ in range(rng.randint(1, 4)):
        choices = ["swap"]
        if len(values) < MAX_N:
            choices.append("add")
        if len(values) > MIN_N:
            choices.append("delete")
        move = rng.choice(choices)
        if move in ("delete", "swap"):
            values.remove(rng.choice(tuple(values)))
        if move in ("add", "swap"):
            values.add(rng.randrange(LIMIT + 1))
    return repair(values, rng)


def tournament(population, scored, rng):
    contestants = rng.sample(population, 4)
    return max(contestants, key=lambda x: scored[x][0])


def select_diverse(pool, scored):
    ordered = sorted(set(pool), key=lambda x: scored[x][0], reverse=True)
    selected = []
    signatures = set()
    for candidate in ordered:
        _, sums, diffs = scored[candidate]
        signature = (len(candidate), sums, diffs)
        if signature not in signatures:
            selected.append(candidate)
            signatures.add(signature)
            if len(selected) == POPULATION:
                return selected
    for candidate in ordered:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == POPULATION:
                break
    for candidate in pool:
        if len(selected) == POPULATION:
            break
        selected.append(candidate)
    return selected


def main():
    rng = random.Random(SEED)
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    best = parent_set()
    best_score = metrics(best)[0]
    write_solution(output, best)

    scored = {}
    islands = []
    for _ in range(ISLANDS):
        population = []
        while len(population) < POPULATION:
            candidate = random_candidate(rng)
            if candidate not in scored:
                scored[candidate] = metrics(candidate)
            population.append(candidate)
        islands.append(population)

    deadline = time.monotonic() + SEARCH_SECONDS
    generation = 0
    while time.monotonic() < deadline:
        for index, population in enumerate(islands):
            offspring = []
            for _ in range(POPULATION):
                left = tournament(population, scored, rng)
                right = tournament(population, scored, rng)
                child = mutate(crossover(left, right, rng), rng)
                if child not in scored:
                    scored[child] = metrics(child)
                offspring.append(child)
            islands[index] = select_diverse(population + offspring, scored)

            champion = max(islands[index], key=lambda x: scored[x][0])
            if scored[champion][0] > best_score:
                best, best_score = frozenset(champion), scored[champion][0]
                write_solution(output, best)
            if time.monotonic() >= deadline:
                break

        generation += 1
        if generation % 200 == 0 and len(islands) == ISLANDS:
            elites = [max(pop, key=lambda x: scored[x][0]) for pop in islands]
            for index in range(ISLANDS):
                islands[index][-1] = elites[index - 1]

    write_solution(output, best)
    print(f"island GA complete: generations={generation} n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
