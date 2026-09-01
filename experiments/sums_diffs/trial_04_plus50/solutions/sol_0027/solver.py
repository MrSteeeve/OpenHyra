"""Parallel tempering directly on the best set's integer coordinates."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
SEED = 5026
SEARCH_SECONDS = 155.0
REPLICAS = 16


def parent_values():
    return tuple(sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if i not in MISSING))


def canonical(values):
    values = tuple(sorted(values))
    if len(values) != len(set(values)):
        return None
    origin = values[0]
    shifted = tuple(x - origin for x in values)
    divisor = 0
    for x in shifted[1:]:
        divisor = math.gcd(divisor, x)
    if divisor > 1:
        shifted = tuple(x // divisor for x in shifted)
    if shifted[-1] > 4000:
        return None
    return shifted


def score_values(values):
    n = len(values)
    sum_bits = 0
    distance_bits = 1
    for i, x in enumerate(values):
        for y in values[i:]:
            sum_bits |= 1 << (x + y)
        for y in values[:i]:
            distance_bits |= 1 << (x - y)
    sums = sum_bits.bit_count()
    diffs = 2 * distance_bits.bit_count() - 1
    return math.log(sums / n) / math.log(diffs / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def displaced(values, rng):
    result = list(values)
    move = rng.random()
    if move < 0.50:
        indices = (rng.randrange(len(result)),)
        delta = rng.randint(1, 32) * rng.choice((-1, 1))
        for i in indices:
            result[i] += delta
    elif move < 0.80:
        length = rng.randint(2, 16)
        start = rng.randrange(len(result) - length + 1)
        delta = rng.randint(1, 32) * rng.choice((-1, 1))
        for i in range(start, start + length):
            result[i] += delta
    else:
        first_length = rng.randint(2, 16)
        second_length = rng.randint(2, 16)
        first = rng.randrange(len(result) - first_length + 1)
        # Choose two disjoint rank blocks, retrying briefly when they overlap.
        second = rng.randrange(len(result) - second_length + 1)
        for _ in range(12):
            if first + first_length <= second or second + second_length <= first:
                break
            second = rng.randrange(len(result) - second_length + 1)
        if not (first + first_length <= second or second + second_length <= first):
            return None
        delta = rng.randint(1, 32) * rng.choice((-1, 1))
        for i in range(first, first + first_length):
            result[i] += delta
        for i in range(second, second + second_length):
            result[i] -= delta
    if min(result) < 0 or max(result) > 4000:
        return None
    return canonical(result)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    parent = canonical(parent_values())
    best_values = parent
    best_score = score_values(parent)
    write_solution(best_values)

    temperatures = tuple(
        0.00002 * (0.02 / 0.00002) ** (i / (REPLICAS - 1))
        for i in range(REPLICAS)
    )
    states = [parent] * REPLICAS
    scores = [best_score] * REPLICAS
    iteration = 0

    while time.monotonic() < deadline:
        iteration += 1
        for replica, temperature in enumerate(temperatures):
            candidate = displaced(states[replica], rng)
            if candidate is None:
                continue
            candidate_score = score_values(candidate)
            delta = candidate_score - scores[replica]
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                states[replica] = candidate
                scores[replica] = candidate_score
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_values = candidate
                    write_solution(best_values)

        # Alternate the swap parity so every neighboring temperature mixes.
        for low in range(iteration & 1, REPLICAS - 1, 2):
            high = low + 1
            exponent = (1.0 / temperatures[low] - 1.0 / temperatures[high]) * (
                scores[high] - scores[low]
            )
            if exponent >= 0.0 or rng.random() < math.exp(max(-700.0, exponent)):
                states[low], states[high] = states[high], states[low]
                scores[low], scores[high] = scores[high], scores[low]

    write_solution(best_values)
    print(f"wrote coordinate PT best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
