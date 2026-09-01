"""Parallel-tempering search for a sum-dominant integer set."""

import json
import math
import os
import random
import time


INITIAL_SET = [0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33]
SEED = 4004
SEARCH_SECONDS = 155.0
REPLICAS = 24
MIN_TEMPERATURE = 0.0001
MAX_TEMPERATURE = 0.05
SWAP_INTERVAL = 200
MIN_SIZE = 12
MAX_SIZE = 32
MAX_VALUE = 160


def score(values):
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def canonicalize(values):
    values = sorted(set(values))
    origin = values[0]
    values = [value - origin for value in values]
    divisor = 0
    for value in values[1:]:
        divisor = math.gcd(divisor, value)
    if divisor > 1:
        values = [value // divisor for value in values]
    return tuple(values)


def replacement(values, rng, count=1):
    candidate = set(values)
    count = min(count, len(candidate))
    for value in rng.sample(sorted(candidate), count):
        candidate.remove(value)
    available = [value for value in range(MAX_VALUE + 1) if value not in candidate]
    candidate.update(rng.sample(available, count))
    return canonicalize(candidate)


def subset_move(values, rng):
    count = rng.randint(1, len(values) - 1)
    selected = set(rng.sample(values, count))
    fixed = set(values) - selected

    for _ in range(20):
        if rng.random() < 0.5:
            low = -min(selected)
            high = MAX_VALUE - max(selected)
            offset = rng.randint(low, high)
            if offset == 0:
                continue
            moved = {value + offset for value in selected}
        else:
            low = max(selected)
            high = min(selected) + MAX_VALUE
            axis_twice = rng.randint(low, high)
            moved = {axis_twice - value for value in selected}
        if len(moved) == len(selected) and moved.isdisjoint(fixed):
            return canonicalize(fixed | moved)
    return replacement(values, rng)


def mutate(values, rng):
    move = rng.random()
    if move < 0.35:
        return replacement(values, rng)
    if move < 0.75:
        return replacement(values, rng, rng.randint(2, min(4, len(values))))
    if move < 0.90:
        return subset_move(values, rng)

    candidate = set(values)
    if len(candidate) <= MIN_SIZE:
        add = True
    elif len(candidate) >= MAX_SIZE:
        add = False
    else:
        add = rng.random() < 0.5
    if add:
        available = [value for value in range(MAX_VALUE + 1) if value not in candidate]
        candidate.add(rng.choice(available))
    else:
        candidate.remove(rng.choice(sorted(candidate)))
    return canonicalize(candidate)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    temperatures = [
        MIN_TEMPERATURE
        * (MAX_TEMPERATURE / MIN_TEMPERATURE) ** (index / (REPLICAS - 1))
        for index in range(REPLICAS)
    ]

    best = canonicalize(INITIAL_SET)
    best_score = score(best)
    write_solution(best)

    states = []
    scores = []
    for index in range(REPLICAS):
        state = best
        for _ in range(index):
            state = mutate(state, rng)
        state_score = score(state)
        states.append(state)
        scores.append(state_score)
        if state_score > best_score:
            best, best_score = state, state_score
            write_solution(best)

    deadline = time.monotonic() + SEARCH_SECONDS
    proposals_since_swap = 0
    swap_parity = 0

    while time.monotonic() < deadline:
        for index, temperature in enumerate(temperatures):
            candidate = mutate(states[index], rng)
            candidate_score = score(candidate)
            delta = candidate_score - scores[index]
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                states[index] = candidate
                scores[index] = candidate_score
                if candidate_score > best_score:
                    best, best_score = candidate, candidate_score
                    write_solution(best)
            proposals_since_swap += 1

        if proposals_since_swap >= SWAP_INTERVAL:
            proposals_since_swap %= SWAP_INTERVAL
            for left in range(swap_parity, REPLICAS - 1, 2):
                right = left + 1
                log_acceptance = (
                    (1.0 / temperatures[left] - 1.0 / temperatures[right])
                    * (scores[right] - scores[left])
                )
                if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
                    states[left], states[right] = states[right], states[left]
                    scores[left], scores[right] = scores[right], scores[left]
            swap_parity ^= 1

    write_solution(best)
    print(f"wrote tempering best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
