"""Search for a sum-dominant set by deterministic simulated annealing."""

import json
import math
import os
import random
import time
from pathlib import Path


INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
N_MIN = 12
N_MAX = 40
COORD_MAX = 96
RESTARTS = 32
T0 = 0.01
COOLING = 0.99995
TIME_LIMIT = 165.0
RANDOM_SEED = 3


def normalize(values):
    values = sorted(values)
    offset = values[0]
    return tuple(value - offset for value in values)


def score(values):
    sums = {a + b for i, a in enumerate(values) for b in values[i:]}
    positive_differences = {
        a - b for i, a in enumerate(values) for b in values[:i]
    }
    n = len(values)
    sum_count = len(sums)
    difference_count = 2 * len(positive_differences) + 1
    exponent = math.log(sum_count / n) / math.log(difference_count / n)
    return exponent, sum_count, difference_count


def mutate(values, rng):
    candidate = list(values)
    occupied = set(candidate)
    choice = rng.random()

    if choice < 0.70:
        index = rng.randrange(len(candidate))
        occupied.remove(candidate[index])
        replacement = rng.randrange(COORD_MAX + 1)
        while replacement in occupied:
            replacement = rng.randrange(COORD_MAX + 1)
        candidate[index] = replacement
    elif choice < 0.85 and len(candidate) < N_MAX:
        addition = rng.randrange(COORD_MAX + 1)
        while addition in occupied:
            addition = rng.randrange(COORD_MAX + 1)
        candidate.append(addition)
    elif choice >= 0.85 and len(candidate) > N_MIN:
        del candidate[rng.randrange(len(candidate))]
    else:
        index = rng.randrange(len(candidate))
        occupied.remove(candidate[index])
        replacement = rng.randrange(COORD_MAX + 1)
        while replacement in occupied:
            replacement = rng.randrange(COORD_MAX + 1)
        candidate[index] = replacement

    return normalize(candidate)


def save(values, destination):
    temporary = destination.with_name(".solution.json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, destination)


def random_restart(rng, incumbent, restart):
    if restart == 0:
        return incumbent
    # Alternate broad random starts with randomized kicks from the best set.
    if restart % 2 == 0:
        n = rng.randint(N_MIN, N_MAX)
        return normalize(rng.sample(range(COORD_MAX + 1), n))
    values = incumbent
    for _ in range(4 + restart):
        values = mutate(values, rng)
    return values


def main():
    rng = random.Random(RANDOM_SEED)
    destination = Path(__file__).resolve().parent / "solution.json"
    deadline = time.monotonic() + TIME_LIMIT

    best = normalize(INITIAL_SET)
    best_score, best_sums, best_differences = score(best)
    save(best, destination)
    evaluations = 1

    for restart in range(RESTARTS):
        now = time.monotonic()
        if now >= deadline:
            break
        restart_deadline = now + (deadline - now) / (RESTARTS - restart)

        current = random_restart(rng, best, restart)
        current_score, current_sums, current_differences = score(current)
        evaluations += 1
        if current_score > best_score:
            best = current
            best_score = current_score
            best_sums = current_sums
            best_differences = current_differences
            save(best, destination)
        temperature = T0

        while time.monotonic() < restart_deadline:
            candidate = mutate(current, rng)
            candidate_score, candidate_sums, candidate_differences = score(candidate)
            evaluations += 1
            delta = candidate_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score

            if candidate_score > best_score:
                best = candidate
                best_score = candidate_score
                best_sums = candidate_sums
                best_differences = candidate_differences
                save(best, destination)

            temperature *= COOLING

    print(
        f"best C={best_score:.9f}: n={len(best)} sums={best_sums} "
        f"diffs={best_differences} evaluations={evaluations}"
    )


if __name__ == "__main__":
    main()
