"""Alternating row/column repair of the best tensor subset."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
SEED = 5024
SEARCH_SECONDS = 155.0
Q = 56
LINE = len(BASE)


def image(mask, q, c):
    values = {
        x + q * y + c * x * y
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


def evaluate(mask, q, c):
    values = image(mask, q, c)
    return score_values(values), values


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def replace_line(mask, axis, line, pattern):
    """Replace one row or column of the 17 by 17 tensor mask."""
    for offset in range(LINE):
        index = line * LINE + offset if axis == 0 else offset * LINE + line
        if (pattern >> offset) & 1:
            mask |= 1 << index
        else:
            mask &= ~(1 << index)
    return mask


def line_pattern(mask, axis, line):
    pattern = 0
    for offset in range(LINE):
        index = line * LINE + offset if axis == 0 else offset * LINE + line
        pattern |= ((mask >> index) & 1) << offset
    return pattern


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    best_score, best_values = evaluate(PARENT_MASK, Q, 0)
    best_mask = PARENT_MASK
    write_solution(best_values)

    mask = best_mask
    current_score, current_values = best_score, best_values
    iteration = 0
    all_bits = (1 << LINE) - 1
    while time.monotonic() < deadline - 0.5:
        iteration += 1
        axis = iteration & 1
        line = rng.randrange(LINE)
        old_pattern = line_pattern(mask, axis, line)
        patterns = {old_pattern, 0, all_bits}
        while len(patterns) < 256:
            if rng.random() < 0.70:
                pattern = old_pattern
                for bit in rng.sample(range(LINE), rng.randint(1, 7)):
                    pattern ^= 1 << bit
            else:
                density = rng.uniform(0.65, 1.0)
                pattern = sum((rng.random() < density) << bit for bit in range(LINE))
            patterns.add(pattern)

        beam = []
        for pattern in patterns:
            candidate_mask = replace_line(mask, axis, line, pattern)
            if not 240 <= candidate_mask.bit_count() <= 320:
                continue
            candidate_score, candidate_values = evaluate(candidate_mask, Q, 0)
            beam.append((candidate_score, candidate_mask, candidate_values))
        beam.sort(key=lambda item: item[0], reverse=True)
        beam = beam[:64]

        # Greedily polish the strongest beam members with every one-cell flip.
        polished = beam[:]
        for _, candidate_mask, _ in beam[:8]:
            if time.monotonic() >= deadline - 0.5:
                break
            local_best = None
            for bit in range(LINE):
                pattern = line_pattern(candidate_mask, axis, line) ^ (1 << bit)
                refined_mask = replace_line(candidate_mask, axis, line, pattern)
                if not 240 <= refined_mask.bit_count() <= 320:
                    continue
                refined_score, refined_values = evaluate(refined_mask, Q, 0)
                item = (refined_score, refined_mask, refined_values)
                if local_best is None or item[0] > local_best[0]:
                    local_best = item
            if local_best is not None:
                polished.append(local_best)
        new_score, new_mask, new_values = max(polished, key=lambda item: item[0])

        elapsed = SEARCH_SECONDS - max(0.0, deadline - time.monotonic())
        fraction = min(1.0, elapsed / SEARCH_SECONDS)
        temperature = 0.003 * (1.0 - fraction) + 0.00005 * fraction
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            mask = new_mask
            current_score, current_values = new_score, new_values
            if current_score > best_score:
                best_score, best_values = current_score, current_values
                best_mask = mask
                write_solution(best_values)

        if iteration % 20 == 0:
            mask = best_mask
            current_score, current_values = best_score, best_values

    write_solution(best_values)
    print(
        f"wrote row/column repair best: n={len(best_values)} "
        f"q={Q} score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
