"""Anneal controlled-carry two-digit sum-dominant constructions."""

import json
import math
import os
import random
import time


SEED = 20260791782194
SEARCH_SECONDS = 170.0
RESTARTS = 10
MIN_DIGITS = 8
MAX_DIGITS = 24
MIN_BASE = 8
MAX_BASE = 80
MAX_X = 96
MAX_Y = 36


def values(state):
    xs, ys, base = state
    return frozenset(x + base * y for x in xs for y in ys)


def quality(candidate):
    shifted = frozenset(a - min(candidate) for a in candidate)
    bits = sum(1 << a for a in shifted)
    sums = 0
    positive_diffs = 0
    for a in shifted:
        sums |= bits << a
        positive_diffs |= bits >> a
    n = len(shifted)
    sum_count = sums.bit_count()
    diff_count = 2 * positive_diffs.bit_count() - 1
    return math.log(sum_count / n) / math.log(diff_count / n)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def valid(state):
    xs, ys, base = state
    if not (MIN_DIGITS <= len(xs) <= MAX_DIGITS and MIN_DIGITS <= len(ys) <= MAX_DIGITS):
        return False
    if not (MIN_BASE <= base <= MAX_BASE):
        return False
    if xs[0] != 0 or ys[0] != 0 or xs[-1] > MAX_X or ys[-1] > MAX_Y:
        return False
    candidate = values(state)
    return 2 <= len(candidate) <= 512 and max(candidate) <= 1000000


def initial_state(restart, rng):
    # Dense, slightly wider-than-base low digits deliberately create carries.
    base = rng.randint(10, 48)
    nx = rng.randint(9, 18)
    ny = rng.randint(9, 18)
    x_limit = min(MAX_X, max(nx - 1, base + rng.randint(2, 20)))
    y_limit = rng.randint(ny, min(MAX_Y, ny + 13))
    xs = tuple(sorted({0} | set(rng.sample(range(1, x_limit + 1), nx - 1))))
    ys = tuple(sorted({0} | set(rng.sample(range(1, y_limit + 1), ny - 1))))
    state = (xs, ys, base)
    while not valid(state):
        if len(xs) >= len(ys) and len(xs) > MIN_DIGITS:
            xs = xs[:-1]
        elif len(ys) > MIN_DIGITS:
            ys = ys[:-1]
        else:
            base = min(MAX_BASE, base + 1)
        state = (xs, ys, base)
    return state


def change_digit(digits, limit, rng):
    result = set(digits)
    move = rng.random()
    if move < 0.20 and len(result) < MAX_DIGITS:
        result.add(rng.randint(1, limit))
    elif move < 0.38 and len(result) > MIN_DIGITS:
        result.remove(rng.choice(tuple(result - {0})))
    else:
        old = rng.choice(tuple(result - {0}))
        result.remove(old)
        if move < 0.82:
            result.add(max(1, min(limit, old + rng.choice((-3, -2, -1, 1, 2, 3)))))
        else:
            result.add(rng.randint(1, limit))
    return tuple(sorted(result))


def mutate(state, rng):
    xs, ys, base = state
    move = rng.random()
    if move < 0.43:
        xs = change_digit(xs, MAX_X, rng)
    elif move < 0.86:
        ys = change_digit(ys, MAX_Y, rng)
    elif move < 0.96:
        base = max(MIN_BASE, min(MAX_BASE, base + rng.choice((-3, -2, -1, 1, 2, 3))))
    else:
        base = rng.randint(MIN_BASE, MAX_BASE)
    trial = (xs, ys, base)
    return trial if valid(trial) else state


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        best = frozenset(json.load(stream)["A"])
    best_score = quality(best)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    cache = {}
    for restart in range(RESTARTS):
        state = initial_state(restart, rng)
        current_score = quality(values(state))
        cache[state] = current_score
        slice_start = started + restart * SEARCH_SECONDS / RESTARTS
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)

        while time.monotonic() < slice_end:
            trial = mutate(state, rng)
            if trial == state:
                continue
            trial_score = cache.get(trial)
            if trial_score is None:
                trial_score = quality(values(trial))
                cache[trial] = trial_score
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.012 * max(0.015, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state = trial
                current_score = trial_score
                if current_score > best_score:
                    best = values(state)
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"controlled-carry annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
