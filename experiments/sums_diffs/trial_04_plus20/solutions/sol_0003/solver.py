"""Evolutionary search over interval sets with independently chosen fringes."""

import json
import math
import os
import random
import time

INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEED = 4002
SEARCH_SECONDS = 165.0
POPULATION_SIZE = 256
ELITE_SIZE = 32
TOURNAMENT_SIZE = 4
MUTATION_RATE = 0.08
MIN_M = 30
MAX_M = 160
MIN_K = 8
MAX_K = 24
GENOME_MASK = (1 << MAX_K) - 1


def values_from_individual(individual):
    m, k, left, right = individual
    values = set()
    if k <= m - k:
        values.update(range(k, m - k + 1))
    for r in range(k):
        if left & (1 << r):
            values.add(r)
        if right & (1 << r):
            values.add(m - r)
    return tuple(sorted(values))


def exact_score(individual):
    values = values_from_individual(individual)
    if len(values) < 2:
        return float("-inf")

    mask = 0
    for value in values:
        mask |= 1 << value

    sums = 0
    diffs = 0
    m = individual[0]
    for value in values:
        sums |= mask << value
        diffs |= mask << (m - value)

    n = len(values)
    return math.log(sums.bit_count() / n) / math.log(diffs.bit_count() / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def random_individual(rng):
    return (
        rng.randint(MIN_M, MAX_M),
        rng.randint(MIN_K, MAX_K),
        rng.getrandbits(MAX_K),
        rng.getrandbits(MAX_K),
    )


def tournament(population, rng):
    return max((rng.choice(population) for _ in range(TOURNAMENT_SIZE)), key=lambda item: item[0])[1]


def breed(first, second, rng):
    # m and k remain inside the requested ranges because they come from parents.
    m = first[0] if rng.getrandbits(1) else second[0]
    k = first[1] if rng.getrandbits(1) else second[1]
    crossover = rng.getrandbits(MAX_K)
    left = (first[2] & crossover) | (second[2] & (GENOME_MASK ^ crossover))
    crossover = rng.getrandbits(MAX_K)
    right = (first[3] & crossover) | (second[3] & (GENOME_MASK ^ crossover))

    left_mutations = 0
    right_mutations = 0
    for bit in range(MAX_K):
        if rng.random() < MUTATION_RATE:
            left_mutations |= 1 << bit
        if rng.random() < MUTATION_RATE:
            right_mutations |= 1 << bit
    return m, k, left ^ left_mutations, right ^ right_mutations


def main():
    rng = random.Random(SEED)
    best_values = INITIAL_SET
    initial_mask = sum(1 << value for value in INITIAL_SET)
    initial_sums = 0
    initial_diffs = 0
    for value in INITIAL_SET:
        initial_sums |= initial_mask << value
        initial_diffs |= initial_mask << (INITIAL_SET[-1] - value)
    best_score = math.log(initial_sums.bit_count() / len(INITIAL_SET)) / math.log(
        initial_diffs.bit_count() / len(INITIAL_SET)
    )
    write_solution(best_values)

    deadline = time.monotonic() + SEARCH_SECONDS
    population = []
    score_cache = {}

    # Seed the family with the known fringes, shifted to the smallest allowed m.
    left_seed = sum(1 << r for r in (0, 1, 2, 4, 5, 9))
    right_seed = sum(1 << r for r in (0, 1, 3, 4, 5, 8))
    seeds = [(30, 12, left_seed, right_seed)]
    seeds.extend(random_individual(rng) for _ in range(POPULATION_SIZE - len(seeds)))

    for individual in seeds:
        active = (1 << individual[1]) - 1
        key = (individual[0], individual[1], individual[2] & active, individual[3] & active)
        candidate_score = score_cache.get(key)
        if candidate_score is None:
            candidate_score = exact_score(key)
            score_cache[key] = candidate_score
        population.append((candidate_score, individual))
        if candidate_score > best_score:
            best_score = candidate_score
            best_values = values_from_individual(key)
            write_solution(best_values)

    while time.monotonic() < deadline:
        population.sort(key=lambda item: item[0], reverse=True)
        next_population = population[:ELITE_SIZE]

        while len(next_population) < POPULATION_SIZE:
            first = tournament(population, rng)
            second = tournament(population, rng)
            child = breed(first, second, rng)
            active = (1 << child[1]) - 1
            key = (child[0], child[1], child[2] & active, child[3] & active)
            candidate_score = score_cache.get(key)
            if candidate_score is None:
                candidate_score = exact_score(key)
                score_cache[key] = candidate_score
            next_population.append((candidate_score, child))

            if candidate_score > best_score:
                best_score = candidate_score
                best_values = values_from_individual(key)
                write_solution(best_values)

        population = next_population
        if len(score_cache) > 300000:
            score_cache.clear()

    write_solution(best_values)
    print(f"wrote evolutionary best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
