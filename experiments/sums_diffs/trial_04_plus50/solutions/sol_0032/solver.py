"""Anneal the relative dilation of two affine copies of the tensor parent."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
PARENT = tuple(
    sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if (PARENT_MASK >> i) & 1)
)
SEED = 5031
SEARCH_SECONDS = 150.0
TRANSLATION_LIMIT = 12000


def affine_union(r, s, orientation, translation):
    values = {r * x for x in PARENT}
    values.update(orientation * s * x + translation for x in PARENT)
    return tuple(sorted(values))


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    shifted = tuple(x - values[0] for x in values)
    mask = sum(1 << x for x in shifted)
    sums = 0
    distances = 0
    for x in shifted:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n
    )


def evaluate(r, s, orientation, translation):
    values = affine_union(r, s, orientation, translation)
    return score_values(values), values


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    # Exact sol_0029 incumbent: P union (P - 1568).
    best_values = affine_union(1, 1, 1, -1568)
    best_score = score_values(best_values)
    write_solution(best_values)

    # Seed all dilation basins at the natural endpoint alignments.  This is
    # deterministic and gives scale moves valid overlapping states to inhabit.
    leaders = [(best_score, 1, 1, 1, -1568, best_values)]
    endpoint = PARENT[-1]
    for r in range(1, 17):
        for s in range(1, 17):
            if math.gcd(r, s) != 1:
                continue
            for orientation in (-1, 1):
                offsets = (0, r * endpoint, -orientation * s * endpoint,
                           r * endpoint - orientation * s * endpoint)
                for translation in offsets:
                    candidate_score, values = evaluate(r, s, orientation, translation)
                    if candidate_score < 0.0:
                        continue
                    item = (candidate_score, r, s, orientation, translation, values)
                    leaders.append(item)
                    if candidate_score > best_score:
                        best_score, best_values = candidate_score, values
                        write_solution(best_values)
    leaders.sort(key=lambda item: item[0], reverse=True)
    leaders = leaders[:32]

    # Parallel tempering retains exploratory hot replicas while the cold ones
    # refine translations around the strongest relative-scale geometries.
    temperatures = [0.00002 * (500.0 ** (i / 31.0)) for i in range(32)]
    chains = [list(leaders[i % len(leaders)]) for i in range(32)]
    steps = (-256, -128, -64, -32, -16, -8, -4, -2, -1,
             1, 2, 4, 8, 16, 32, 64, 128, 256)
    iteration = 0
    while time.monotonic() < deadline:
        index = iteration & 31
        current_score, r, s, orientation, translation, current_values = chains[index]
        new_r, new_s = r, s
        new_orientation, new_translation = orientation, translation
        move = rng.random()
        if move < 0.28:
            new_r = rng.randint(1, 16)
            if math.gcd(new_r, new_s) != 1:
                iteration += 1
                continue
        elif move < 0.56:
            new_s = rng.randint(1, 16)
            if math.gcd(new_r, new_s) != 1:
                iteration += 1
                continue
        elif move < 0.68:
            new_orientation = -orientation
        else:
            new_translation = max(
                -TRANSLATION_LIMIT,
                min(TRANSLATION_LIMIT, translation + rng.choice(steps)),
            )

        new_score, new_values = evaluate(
            new_r, new_s, new_orientation, new_translation
        )
        if new_score >= 0.0:
            delta = new_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperatures[index]):
                chains[index] = [new_score, new_r, new_s, new_orientation,
                                 new_translation, new_values]
                if new_score > best_score:
                    best_score, best_values = new_score, new_values
                    write_solution(best_values)

        iteration += 1
        if iteration % 256 == 0:
            parity = (iteration // 256) & 1
            for low in range(parity, 31, 2):
                high = low + 1
                delta = chains[high][0] - chains[low][0]
                exponent = delta * (1.0 / temperatures[low] - 1.0 / temperatures[high])
                if exponent >= 0.0 or rng.random() < math.exp(exponent):
                    chains[low], chains[high] = chains[high], chains[low]

    write_solution(best_values)
    print(f"wrote relative-dilation best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
