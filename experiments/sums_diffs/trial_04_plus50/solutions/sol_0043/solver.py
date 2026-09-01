"""Parallel tempering over unrestricted bounded gap vectors."""

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
SEED = 5042
SEARCH_SECONDS = 150.0
REPLICAS = 32
LOW_N, HIGH_N = 40, 120
LOW_SPAN, HIGH_SPAN = 120, 400
MAX_GAP = 24


def values_from_gaps(gaps):
    values = [0]
    for gap in gaps:
        values.append(values[-1] + gap)
    return tuple(values)


def score(values):
    n = len(values)
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def valid(gaps):
    return (LOW_N <= len(gaps) + 1 <= HIGH_N and
            LOW_SPAN <= sum(gaps) <= HIGH_SPAN and
            all(1 <= gap <= MAX_GAP for gap in gaps))


def fresh(rng):
    n = rng.randint(40, 100)
    target = rng.randint(max(120, n - 1), min(400, MAX_GAP * (n - 1)))
    gaps = [1] * (n - 1)
    extra = target - n + 1
    while extra:
        i = rng.randrange(len(gaps))
        amount = min(extra, MAX_GAP - gaps[i], rng.randint(1, 8))
        gaps[i] += amount
        extra -= amount
    rng.shuffle(gaps)
    values = values_from_gaps(gaps)
    return [score(values), gaps, values]


def mutate(gaps, rng):
    result = list(gaps)
    move = rng.random()
    if move < 0.40:
        # Replace a contiguous block, allowing coordinated cluster motion.
        length = rng.randint(1, min(12, len(result)))
        start = rng.randrange(len(result) - length + 1)
        result[start:start + length] = [rng.randint(1, MAX_GAP)
                                        for _ in range(length)]
    elif move < 0.70:
        length = rng.randint(2, min(16, len(result)))
        start = rng.randrange(len(result) - length + 1)
        block = result[start:start + length]
        if rng.random() < 0.5:
            block.reverse()
        else:
            rng.shuffle(block)
        result[start:start + length] = block
    elif move < 0.90:
        # Splitting a gap inserts a point; merging adjacent gaps removes one.
        splittable = [i for i, gap in enumerate(result) if gap >= 2]
        if ((rng.random() < 0.5 and len(result) + 1 < HIGH_N) or
                len(result) + 1 <= LOW_N) and splittable:
            i = rng.choice(splittable)
            cut = rng.randint(1, result[i] - 1)
            result[i:i + 1] = [cut, result[i] - cut]
        elif len(result) + 1 > LOW_N:
            mergeable = [i for i in range(len(result) - 1)
                         if result[i] + result[i + 1] <= MAX_GAP]
            if mergeable:
                i = rng.choice(mergeable)
                result[i:i + 2] = [result[i] + result[i + 1]]
    else:
        # Size changes at either boundary or by point insertion/deletion.
        if rng.random() < 0.5 and len(result) + 1 < HIGH_N:
            gap = rng.randint(1, MAX_GAP)
            result.insert(0 if rng.random() < 0.5 else len(result), gap)
        elif len(result) + 1 > LOW_N:
            result.pop(0 if rng.random() < 0.5 else -1)
    return result


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = FALLBACK
    best_score = score(FALLBACK)
    write_solution(FALLBACK)

    fallback_gaps = [b - a for a, b in zip(FALLBACK, FALLBACK[1:])]
    chains = [[best_score, fallback_gaps, FALLBACK]]
    chains.extend(fresh(rng) for _ in range(REPLICAS - 1))
    temperatures = [0.00002 * (1000.0 ** (i / (REPLICAS - 1)))
                    for i in range(REPLICAS)]
    iteration = 0

    while time.monotonic() < deadline:
        iteration += 1
        for i, temperature in enumerate(temperatures):
            old_score, gaps, _ = chains[i]
            candidate_gaps = mutate(gaps, rng)
            if not valid(candidate_gaps):
                continue
            candidate_values = values_from_gaps(candidate_gaps)
            candidate_score = score(candidate_values)
            delta = candidate_score - old_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                chains[i] = [candidate_score, candidate_gaps, candidate_values]
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_values = candidate_values
                    write_solution(best_values)

        # Standard adjacent temperature exchanges, alternating parity.
        parity = iteration & 1
        for i in range(parity, REPLICAS - 1, 2):
            left, right = chains[i], chains[i + 1]
            exponent = ((1.0 / temperatures[i] - 1.0 / temperatures[i + 1]) *
                        (right[0] - left[0]))
            if exponent >= 0.0 or rng.random() < math.exp(exponent):
                chains[i], chains[i + 1] = right, left

    write_solution(best_values)
    print(f"wrote gap-vector best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
