"""Parallel-tempering search over gap vectors for a sum-dominant set."""

import json
import math
import os
import random
import time

INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEED = 4009
SEARCH_SECONDS = 155.0
REPLICAS = 32
MIN_GAP = 1
MAX_GAP = 8
MIN_SPAN = 25
MAX_SPAN = 80
MIN_TEMPERATURE = 0.0001
MAX_TEMPERATURE = 0.05


def values_from_gaps(gaps):
    values = [0]
    for gap in gaps:
        values.append(values[-1] + gap)
    return tuple(values)


def score_gaps(gaps):
    values = values_from_gaps(gaps)
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def random_gaps(rng, count):
    while True:
        gaps = tuple(rng.randint(MIN_GAP, MAX_GAP) for _ in range(count))
        if MIN_SPAN <= sum(gaps) <= MAX_SPAN:
            return gaps


def mutate(gaps, rng):
    candidate = list(gaps)
    if rng.random() < 0.5:
        while True:
            index = rng.randrange(len(candidate))
            change = rng.choice((-3, -2, -1, 1, 2, 3))
            new_gap = candidate[index] + change
            new_span = sum(candidate) + change
            if MIN_GAP <= new_gap <= MAX_GAP and MIN_SPAN <= new_span <= MAX_SPAN:
                candidate[index] = new_gap
                return tuple(candidate)
    else:
        while True:
            donor = rng.randrange(len(candidate))
            receiver = rng.randrange(len(candidate) - 1)
            if receiver >= donor:
                receiver += 1
            amount = rng.randint(1, 3)
            if candidate[donor] - amount >= MIN_GAP and candidate[receiver] + amount <= MAX_GAP:
                candidate[donor] -= amount
                candidate[receiver] += amount
                return tuple(candidate)


def write_solution(gaps):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values_from_gaps(gaps))}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    initial_gaps = tuple(b - a for a, b in zip(INITIAL_SET, INITIAL_SET[1:]))
    temperatures = tuple(
        MIN_TEMPERATURE
        * (MAX_TEMPERATURE / MIN_TEMPERATURE) ** (index / (REPLICAS - 1))
        for index in range(REPLICAS)
    )

    states = [initial_gaps]
    states.extend(random_gaps(rng, len(initial_gaps)) for _ in range(REPLICAS - 1))
    scores = [score_gaps(gaps) for gaps in states]
    best_index = max(range(REPLICAS), key=scores.__getitem__)
    best = states[best_index]
    best_score = scores[best_index]
    write_solution(best)

    deadline = time.monotonic() + SEARCH_SECONDS
    sweep = 0
    while time.monotonic() < deadline:
        for index, temperature in enumerate(temperatures):
            candidate = mutate(states[index], rng)
            candidate_score = score_gaps(candidate)
            delta = candidate_score - scores[index]
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                states[index] = candidate
                scores[index] = candidate_score
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    write_solution(best)

        start = sweep & 1
        for index in range(start, REPLICAS - 1, 2):
            exponent = (scores[index + 1] - scores[index]) * (
                1.0 / temperatures[index] - 1.0 / temperatures[index + 1]
            )
            if exponent >= 0.0 or rng.random() < math.exp(exponent):
                states[index], states[index + 1] = states[index + 1], states[index]
                scores[index], scores[index + 1] = scores[index + 1], scores[index]
        sweep += 1

    write_solution(best)
    print(f"wrote gap-tempering best: n={len(best) + 1} score={best_score:.9f}")


if __name__ == "__main__":
    main()
