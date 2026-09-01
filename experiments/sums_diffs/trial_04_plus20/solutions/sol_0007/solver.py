"""Cross-entropy search over inclusion masks for a sum-dominant set."""

import json
import math
import os
import time

import numpy as np


INCUMBENT = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
SEED = 4005
SEARCH_SECONDS = 155.0
DOMAIN_SIZE = 64
SAMPLES = 2048
ELITE_COUNT = 64
MUTATIONS_PER_ELITE = 128
ALPHA = 0.20
MIN_SIZE = 12
MAX_SIZE = 40
STAGNATION_LIMIT = 25
CACHE_LIMIT = 300_000


def values_to_mask(values):
    mask = 0
    for value in values:
        mask |= 1 << value
    return mask


def mask_to_values(mask):
    values = []
    while mask:
        bit = mask & -mask
        values.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(values)


def score_mask(mask):
    values = mask_to_values(mask)
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    n = len(values)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def rows_to_masks(rows):
    packed = np.packbits(rows, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def sample_masks(probabilities, rng, count):
    masks = []
    seen = set()
    while len(masks) < count:
        batch_size = max(256, count - len(masks))
        rows = rng.random((batch_size, DOMAIN_SIZE)) < probabilities
        sizes = rows.sum(axis=1)
        valid = rows[(sizes >= MIN_SIZE) & (sizes <= MAX_SIZE)]
        for mask in rows_to_masks(valid):
            if mask not in seen:
                seen.add(mask)
                masks.append(mask)
                if len(masks) == count:
                    break
    return set(masks)


def mutated_masks(elites, rng):
    mutations = set()
    bit_numbers = np.arange(DOMAIN_SIZE, dtype=np.uint64)
    for elite in elites:
        base = ((np.uint64(elite) >> bit_numbers) & np.uint64(1)).astype(bool)
        local = set()
        while len(local) < MUTATIONS_PER_ELITE:
            batch_size = max(32, MUTATIONS_PER_ELITE - len(local))
            rates = rng.uniform(0.025, 0.10, size=(batch_size, 1))
            flips = rng.random((batch_size, DOMAIN_SIZE)) < rates
            empty = np.flatnonzero(~flips.any(axis=1))
            if len(empty):
                flips[empty, rng.integers(0, DOMAIN_SIZE, size=len(empty))] = True
            rows = np.logical_xor(base, flips)
            sizes = rows.sum(axis=1)
            valid = rows[(sizes >= MIN_SIZE) & (sizes <= MAX_SIZE)]
            for mask in rows_to_masks(valid):
                if mask != elite:
                    local.add(mask)
                    if len(local) == MUTATIONS_PER_ELITE:
                        break
        mutations.update(local)
    return mutations


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = np.random.default_rng(SEED)
    incumbent_mask = values_to_mask(INCUMBENT)
    best_mask = incumbent_mask
    best_score = score_mask(best_mask)
    score_cache = {best_mask: best_score}
    write_solution(INCUMBENT)

    probabilities = np.full(DOMAIN_SIZE, 0.30, dtype=float)
    deadline = time.monotonic() + SEARCH_SECONDS
    stagnant_generations = 0
    generations = 0

    def rank_masks(masks):
        nonlocal best_mask, best_score, score_cache
        ranked = []
        for index, mask in enumerate(sorted(masks)):
            if index % 128 == 0 and time.monotonic() >= deadline - 0.35:
                break
            candidate_score = score_cache.get(mask)
            if candidate_score is None:
                candidate_score = score_mask(mask)
                if len(score_cache) >= CACHE_LIMIT:
                    score_cache = {best_mask: best_score}
                score_cache[mask] = candidate_score
            ranked.append((candidate_score, mask))
            if candidate_score > best_score:
                best_mask = mask
                best_score = candidate_score
                write_solution(mask_to_values(best_mask))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    while time.monotonic() < deadline - 0.75:
        score_before_generation = best_score
        candidates = sample_masks(probabilities, rng, SAMPLES)
        candidates.add(incumbent_mask)

        provisional = rank_masks(candidates)
        if not provisional or time.monotonic() >= deadline - 0.75:
            break
        provisional_elites = [mask for _, mask in provisional[:ELITE_COUNT]]

        candidates.update(mutated_masks(provisional_elites, rng))
        ranked = rank_masks(candidates)
        if not ranked:
            break
        elites = [mask for _, mask in ranked[:ELITE_COUNT]]

        elite_rows = np.zeros((len(elites), DOMAIN_SIZE), dtype=float)
        for row, mask in enumerate(elites):
            for bit in mask_to_values(mask):
                elite_rows[row, bit] = 1.0
        target = elite_rows.mean(axis=0)
        probabilities = (1.0 - ALPHA) * probabilities + ALPHA * target
        probabilities = np.clip(probabilities, 0.03, 0.97)

        generations += 1
        if best_score > score_before_generation:
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        if stagnant_generations >= STAGNATION_LIMIT:
            probabilities.fill(0.30)
            stagnant_generations = 0

    write_solution(mask_to_values(best_mask))
    print(
        f"wrote cross-entropy best: n={best_mask.bit_count()} "
        f"score={best_score:.9f} generations={generations}"
    )


if __name__ == "__main__":
    main()
