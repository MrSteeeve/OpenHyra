"""Pareto annealing of sum/difference tradeoffs on a small integer domain."""

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
SEED = 5046
SEARCH_SECONDS = 145.0
DOMAIN = 221
MIN_N = 50
MAX_N = 90
LAMBDAS = tuple(1.06 + i * (1.18 - 1.06) / 23 for i in range(24))


def metrics(values):
    """Return exact (n, number of sums, number of differences)."""
    n = len(values)
    bits = sum(1 << x for x in values)
    sums = 0
    nonnegative_differences = 0
    for x in values:
        sums |= bits << x
        nonnegative_differences |= bits >> x
    return n, sums.bit_count(), 2 * nonnegative_differences.bit_count() - 1


def true_score(triple):
    n, sums, differences = triple
    return math.log(sums / n) / math.log(differences / n)


def coordinates(triple):
    n, sums, differences = triple
    return math.log(sums / n), math.log(differences / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def add_archive(archive, values, triple):
    """Maintain the nondominated frontier: larger sum growth, smaller diff growth."""
    x, y = coordinates(triple)
    key = triple
    old = archive.get(key)
    if old is not None:
        return
    for _, ax, ay in archive.values():
        if ax >= x and ay <= y and (ax > x or ay < y):
            return
    dead = [k for k, (_, ax, ay) in archive.items()
            if x >= ax and y <= ay and (x > ax or y < ay)]
    for k in dead:
        del archive[k]
    archive[key] = (tuple(values), x, y)


def mutate(rng, values):
    result = set(values)
    move = rng.random()
    if move < 0.45:  # single toggle
        x = rng.randrange(DOMAIN)
        if x in result:
            if len(result) > MIN_N:
                result.remove(x)
        elif len(result) < MAX_N:
            result.add(x)
    elif move < 0.75:  # size-preserving swap
        result.remove(rng.choice(tuple(result)))
        missing = rng.randrange(DOMAIN)
        while missing in result:
            missing = rng.randrange(DOMAIN)
        result.add(missing)
    else:  # toggle a contiguous block of 2--6 coordinates
        width = rng.randint(2, 6)
        start = rng.randrange(DOMAIN - width + 1)
        block = set(range(start, start + width))
        candidate = result.symmetric_difference(block)
        if MIN_N <= len(candidate) <= MAX_N:
            result = candidate
    return tuple(sorted(result))


def main():
    rng = random.Random(SEED)
    fallback_metrics = metrics(FALLBACK)
    if fallback_metrics != (66, 265, 235):
        raise RuntimeError("incorrect sol_0036 fallback")
    best_values = FALLBACK
    best_score = true_score(fallback_metrics)
    write_solution(best_values)

    deadline = time.monotonic() + SEARCH_SECONDS
    archive = {}
    add_archive(archive, FALLBACK, fallback_metrics)
    chains = []
    for index, lam in enumerate(LAMBDAS):
        values = set(FALLBACK)
        # Give replicas distinct nearby starts without risking the checkpoint.
        for _ in range(2 + index % 7):
            values = set(mutate(rng, tuple(values)))
        values = tuple(sorted(values))
        triple = metrics(values)
        x, y = coordinates(triple)
        chains.append([values, triple, x - lam * y])
        add_archive(archive, values, triple)

    proposals = 0
    while time.monotonic() < deadline:
        index = proposals % len(chains)
        lam = LAMBDAS[index]
        values, old_triple, old_objective = chains[index]
        candidate = mutate(rng, values)
        if candidate != values:
            triple = metrics(candidate)
            x, y = coordinates(triple)
            objective = x - lam * y
            remaining = max(0.0, deadline - time.monotonic()) / SEARCH_SECONDS
            temperature = 0.012 * remaining + 0.00008
            if (objective >= old_objective or
                    rng.random() < math.exp((objective - old_objective) / temperature)):
                chains[index] = [candidate, triple, objective]
                add_archive(archive, candidate, triple)
                candidate_score = true_score(triple)
                if candidate_score > best_score + 1e-15:
                    # Recompute from the actual emitted tuple before replacing fallback.
                    verified = metrics(candidate)
                    verified_score = true_score(verified)
                    if verified == triple and verified_score > best_score + 1e-15:
                        best_values, best_score = candidate, verified_score
                        write_solution(best_values)

        proposals += 1
        if proposals % 2000 == 0 and archive:
            frontier = list(archive.values())
            # Exchange each replica with the archive point best for its scalarization.
            for i, lam_i in enumerate(LAMBDAS):
                if rng.random() < 0.5:
                    state, ax, ay = max(frontier, key=lambda item: item[1] - lam_i * item[2])
                else:
                    state, ax, ay = rng.choice(frontier)
                chains[i] = [state, metrics(state), ax - lam_i * ay]

    # A final write makes interruption-free completion unambiguous.
    if metrics(best_values) != fallback_metrics and true_score(metrics(best_values)) <= true_score(fallback_metrics):
        best_values, best_score = FALLBACK, true_score(fallback_metrics)
    write_solution(best_values)
    print(f"wrote Pareto best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
