"""Anneal carry-free subsets of a 32 by 16 integer grid."""

import functools
import json
import math
import os
import random
import time


SEED = 20260790782190
SEARCH_SECONDS = 168.0
WIDTH = 32
HEIGHT = 16
RESTARTS = 6
FULL = (1 << WIDTH) - 1


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def quality_counts(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def integer_quality(candidate):
    sums = {a + b for a in candidate for b in candidate}
    diffs = {a - b for a in candidate for b in candidate}
    return quality_counts(len(candidate), len(sums), len(diffs))


@functools.lru_cache(maxsize=200000)
def xsum(mask_a, mask_b):
    result = 0
    mask = mask_a
    while mask:
        bit = mask & -mask
        result |= mask_b << (bit.bit_length() - 1)
        mask -= bit
    return result


@functools.lru_cache(maxsize=200000)
def xdiff(mask_a, mask_b):
    # Difference dx is stored at bit dx + 31.
    result = 0
    mask = mask_a
    while mask:
        bit = mask & -mask
        x = bit.bit_length() - 1
        other = mask_b
        while other:
            b = other & -other
            result |= 1 << (x - (b.bit_length() - 1) + WIDTH - 1)
            other -= b
        mask -= bit
    return result


@functools.lru_cache(maxsize=120000)
def grid_score(rows):
    n = sum(mask.bit_count() for mask in rows)
    if n < 2 or n > 512:
        return -10.0

    sum_lines = [0] * (2 * HEIGHT - 1)
    diff_lines = [0] * (2 * HEIGHT - 1)
    for y1, first in enumerate(rows):
        if not first:
            continue
        for y2, second in enumerate(rows):
            if not second:
                continue
            sum_lines[y1 + y2] |= xsum(first, second)
            diff_lines[y1 - y2 + HEIGHT - 1] |= xdiff(first, second)
    sums = sum(line.bit_count() for line in sum_lines)
    diffs = sum(line.bit_count() for line in diff_lines)
    return quality_counts(n, sums, diffs)


def emit(rows):
    return {
        x + 1000 * y
        for y, mask in enumerate(rows)
        for x in range(WIDTH)
        if mask >> x & 1
    }


def initial_state(restart, rng):
    # Structured starts keep some additive regularity; later starts are freer.
    seed_x = (0, 1, 3, 4, 5, 8, 9, 12, 13, 16, 20, 24, 25, 28)
    seed_y = (0, 1, 3, 4, 5, 8, 9, 12, 13)
    xmask = sum(1 << x for x in seed_x)
    if restart == 0:
        rows = [xmask if y in seed_y else 0 for y in range(HEIGHT)]
    elif restart < 3:
        rows = [xmask if y in seed_y else 0 for y in range(HEIGHT)]
        for _ in range(35 + 20 * restart):
            y, x = rng.randrange(HEIGHT), rng.randrange(WIDTH)
            rows[y] ^= 1 << x
    else:
        density = (0.20, 0.30, 0.42)[restart - 3]
        rows = [sum(1 << x for x in range(WIDTH) if rng.random() < density)
                for _ in range(HEIGHT)]
    return tuple(rows)


def mutate(state, rng):
    rows = list(state)
    move = rng.random()
    if move < 0.54:  # one or occasionally two independent cell flips
        for _ in range(1 if rng.random() < 0.88 else 2):
            y, x = rng.randrange(HEIGHT), rng.randrange(WIDTH)
            rows[y] ^= 1 << x
    elif move < 0.72:  # occupancy-preserving swap
        occupied = [(y, x) for y, mask in enumerate(rows)
                    for x in range(WIDTH) if mask >> x & 1]
        empty = [(y, x) for y, mask in enumerate(rows)
                 for x in range(WIDTH) if not (mask >> x & 1)]
        if occupied and empty:
            y1, x1 = rng.choice(occupied)
            y2, x2 = rng.choice(empty)
            rows[y1] ^= 1 << x1
            rows[y2] ^= 1 << x2
    elif move < 0.84:  # small rectangle toggle
        y0 = rng.randrange(HEIGHT)
        x0 = rng.randrange(WIDTH)
        h, w = rng.choice(((1, 2), (2, 1), (2, 2), (1, 3), (3, 1)))
        rectangle = ((1 << min(w, WIDTH - x0)) - 1) << x0
        for y in range(y0, min(HEIGHT, y0 + h)):
            rows[y] ^= rectangle
    elif move < 0.93:  # row copy, with a small mutation to avoid clones
        source, target = rng.sample(range(HEIGHT), 2)
        rows[target] = rows[source] ^ (1 << rng.randrange(WIDTH))
    else:  # column copy
        source, target = rng.sample(range(WIDTH), 2)
        for y in range(HEIGHT):
            if rows[y] >> source & 1:
                rows[y] |= 1 << target
            else:
                rows[y] &= ~(1 << target)
    return tuple(mask & FULL for mask in rows)


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        parent = set(json.load(stream)["A"])
    best = parent
    best_score = integer_quality(parent)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    last_write = started
    for restart in range(RESTARTS):
        state = initial_state(restart, rng)
        score = grid_score(state)
        slice_start = started + restart * SEARCH_SECONDS / RESTARTS
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)
        while time.monotonic() < slice_end:
            trial = mutate(state, rng)
            trial_score = grid_score(trial)
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.010 * (1.0 - progress) + 0.00015
            delta = trial_score - score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state, score = trial, trial_score
                if score > best_score:
                    best, best_score = emit(state), score
            now = time.monotonic()
            if now - last_write > 12.0:
                write_solution(output, best)
                last_write = now
        write_solution(output, best)

    write_solution(output, best)
    print(f"carry-free grid annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
