"""Anneal nearly reflection-symmetric sets with a few exceptions."""

import json
import math
import os
import random
import time

INITIAL_SET = [0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33]
SEED = 4012
SEARCH_SECONDS = 160.0
REPLICAS = 32
MIN_M = 24
MAX_M = 160
MAX_EXCEPTIONS = 8


def materialize(state):
    m, symmetric, exceptions = state
    values = 0
    remaining = symmetric
    while remaining:
        low = remaining & -remaining
        i = low.bit_length() - 1
        values |= (1 << i) | (1 << (m - i))
        remaining ^= low
    return values ^ exceptions


def score_mask(values):
    n = values.bit_count()
    if n < 2:
        return -1.0

    sums = 0
    nonnegative_differences = 0
    remaining = values
    while remaining:
        low = remaining & -remaining
        a = low.bit_length() - 1
        sums |= values << a
        nonnegative_differences |= values >> a
        remaining ^= low

    sum_count = sums.bit_count()
    difference_count = 2 * nonnegative_differences.bit_count() - 1
    return math.log(sum_count / n) / math.log(difference_count / n)


def state_from_values(values, m):
    present = set(values)
    symmetric = 0
    exceptions = 0
    for i in range(m // 2 + 1):
        j = m - i
        left = i in present
        right = j in present
        if left and right:
            symmetric |= 1 << i
        elif left:
            exceptions |= 1 << i
        elif right:
            exceptions |= 1 << j
    return m, symmetric, exceptions


def valid_exception_points(m, exceptions):
    center = m // 2 if m % 2 == 0 else -1
    return [
        x for x in range(m + 1)
        if x != center
        and not (exceptions & (1 << x))
        and not (exceptions & (1 << (m - x)))
    ]


def mutate(state, rng):
    m, symmetric, exceptions = state
    move = rng.random()

    if move < 0.60:
        symmetric ^= 1 << rng.randrange(m // 2 + 1)
    elif move < 0.90:
        count = exceptions.bit_count()
        choices = []
        available = valid_exception_points(m, exceptions)
        if count < MAX_EXCEPTIONS and available:
            choices.append("add")
        if count:
            choices.append("remove")
            if available:
                choices.append("move")
        if not choices:
            return state
        kind = rng.choice(choices)
        if kind in ("remove", "move"):
            occupied = [x for x in range(m + 1) if exceptions & (1 << x)]
            exceptions ^= 1 << rng.choice(occupied)
        if kind in ("add", "move"):
            available = valid_exception_points(m, exceptions)
            exceptions |= 1 << rng.choice(available)
    else:
        step = rng.randint(1, 4)
        new_m = m + (step if rng.random() < 0.5 else -step)
        if new_m < MIN_M or new_m > MAX_M:
            new_m = m - (new_m - m)
        new_m = min(MAX_M, max(MIN_M, new_m))

        symmetric &= (1 << (new_m // 2 + 1)) - 1
        exceptions &= (1 << (new_m + 1)) - 1
        if new_m % 2 == 0:
            exceptions &= ~(1 << (new_m // 2))
        for x in range(new_m // 2 + 1):
            y = new_m - x
            if x != y and exceptions & (1 << x) and exceptions & (1 << y):
                exceptions &= ~(1 << (x if rng.random() < 0.5 else y))
        m = new_m

    return m, symmetric, exceptions


def write_solution(values_mask):
    values = [i for i in range(values_mask.bit_length()) if values_mask & (1 << i)]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": values}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    initial = state_from_values(INITIAL_SET, 33)
    initial_mask = materialize(initial)
    initial_score = score_mask(initial_mask)

    states = [initial for _ in range(REPLICAS)]
    scores = [initial_score for _ in range(REPLICAS)]
    best_mask = initial_mask
    best_score = initial_score
    write_solution(best_mask)

    low_temperature = 0.00015
    high_temperature = 0.05
    temperatures = [
        low_temperature * (high_temperature / low_temperature) ** (i / (REPLICAS - 1))
        for i in range(REPLICAS)
    ]

    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    sweep = 0
    while time.monotonic() < deadline:
        elapsed_fraction = min(1.0, (time.monotonic() - start) / SEARCH_SECONDS)
        cooling = 1.0 - 0.85 * elapsed_fraction

        for i in range(REPLICAS):
            candidate = mutate(states[i], rng)
            candidate_mask = materialize(candidate)
            candidate_score = score_mask(candidate_mask)
            delta = candidate_score - scores[i]
            temperature = temperatures[i] * cooling
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                states[i] = candidate
                scores[i] = candidate_score
                if candidate_score > best_score:
                    best_mask = candidate_mask
                    best_score = candidate_score
                    write_solution(best_mask)

            if time.monotonic() >= deadline:
                break

        parity = sweep & 1
        for i in range(parity, REPLICAS - 1, 2):
            left_temperature = temperatures[i] * cooling
            right_temperature = temperatures[i + 1] * cooling
            exponent = (scores[i + 1] - scores[i]) * (
                1.0 / left_temperature - 1.0 / right_temperature
            )
            if exponent >= 0.0 or rng.random() < math.exp(exponent):
                states[i], states[i + 1] = states[i + 1], states[i]
                scores[i], scores[i + 1] = scores[i + 1], scores[i]
        sweep += 1

    write_solution(best_mask)
    print(f"wrote symmetric-exception best: n={best_mask.bit_count()} score={best_score:.9f}")


if __name__ == "__main__":
    main()
