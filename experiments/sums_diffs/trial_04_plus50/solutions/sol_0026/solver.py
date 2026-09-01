"""Coupled row/column large-neighborhood search of the best tensor subset."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
SEED = 5025
SEARCH_SECONDS = 155.0
Q = 56
WIDTH = 32
SAMPLES = 128


def image(mask, q=Q):
    values = {
        x + q * y
        for i, (x, y) in enumerate(CELLS)
        if (mask >> i) & 1
    }
    return tuple(sorted(values))


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    lo = values[0]
    shifted = [x - lo for x in values]
    sum_bits = 0
    distance_bits = 1
    for i, x in enumerate(shifted):
        for y in shifted[i:]:
            sum_bits |= 1 << (x + y)
        for y in shifted[:i]:
            distance_bits |= 1 << (x - y)
    sums = sum_bits.bit_count()
    diffs = 2 * distance_bits.bit_count() - 1
    return math.log(sums / n) / math.log(diffs / n)


def evaluate(mask):
    values = image(mask)
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

    best_score, best_values = evaluate(PARENT_MASK)
    best_mask = PARENT_MASK
    write_solution(best_values)

    mask = best_mask
    current_score = best_score
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        row_count = rng.randint(2, 5)
        column_count = rng.randint(2, 5)
        rows = rng.sample(range(17), row_count)
        columns = rng.sample(range(17), column_count)
        cross = tuple(
            i for i in range(len(CELLS))
            if i // 17 in rows or i % 17 in columns
        )
        cross_bits = sum(1 << i for i in cross)
        outside = mask & ~cross_bits

        candidates = []
        for sample in range(SAMPLES):
            # Row/column latent variables create coherent holes across both axes.
            row_bias = [rng.gauss(0.0, 0.75) for _ in range(17)]
            column_bias = [rng.gauss(0.0, 0.75) for _ in range(17)]
            density = rng.uniform(1.2, 3.8)
            candidate = outside
            for i in cross:
                y, x = divmod(i, 17)
                old = 1 if (mask >> i) & 1 else -1
                latent = density + row_bias[y] + column_bias[x]
                latent += rng.uniform(-0.8, 0.8) + 0.35 * old
                if latent > 0.0:
                    candidate |= 1 << i
            n = candidate.bit_count()
            if 240 <= n <= 320:
                candidate_score, candidate_values = evaluate(candidate)
                candidates.append((candidate_score, candidate, candidate_values))
            if time.monotonic() >= deadline:
                break

        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = candidates[:WIDTH]

        # Greedily refine each beam member along a shuffled part of the cross.
        refined = []
        for candidate_score, candidate, candidate_values in beam:
            trial_indices = rng.sample(cross, min(16, len(cross)))
            for index in trial_indices:
                trial = candidate ^ (1 << index)
                if not 240 <= trial.bit_count() <= 320:
                    continue
                trial_score, trial_values = evaluate(trial)
                if trial_score > candidate_score:
                    candidate_score = trial_score
                    candidate = trial
                    candidate_values = trial_values
                if time.monotonic() >= deadline:
                    break
            refined.append((candidate_score, candidate, candidate_values))
            if time.monotonic() >= deadline:
                break

        new_score, new_mask, new_values = max(refined, key=lambda item: item[0])
        elapsed = SEARCH_SECONDS - max(0.0, deadline - time.monotonic())
        fraction = min(1.0, elapsed / SEARCH_SECONDS)
        temperature = 0.003 * (1.0 - fraction) + 0.00005
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score, best_values = current_score, new_values
                best_mask = mask
                write_solution(best_values)

        if iteration % 20 == 0:
            mask = best_mask
            current_score = best_score

    write_solution(best_values)
    print(
        f"wrote coupled-cross tensor best: n={len(best_values)} "
        f"q={Q} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
