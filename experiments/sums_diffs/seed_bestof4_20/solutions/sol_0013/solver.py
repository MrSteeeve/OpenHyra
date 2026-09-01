"""Anneal independent fringes around a guaranteed full middle interval."""

import json
import math
import os
import random
import time


SEED = 20260794782202
SEARCH_SECONDS = 170.0
RESTARTS = 32
MS = (80, 120, 160, 240)
KS = (12, 16, 24, 32)


def values(state):
    m, k, left, right = state
    result = set(range(k, m - k + 1))
    result.update(i for i in range(k) if left >> i & 1)
    result.update(m - i for i in range(k) if right >> i & 1)
    return result


def quality(candidate):
    """Exact score using integer bitsets for the two pair-set enumerations."""
    bits = 0
    for value in candidate:
        bits |= 1 << value
    sums = 0
    positive_diffs = 0
    for value in candidate:
        sums |= bits << value
        positive_diffs |= bits >> value
    n = len(candidate)
    sum_count = sums.bit_count()
    diff_count = 2 * positive_diffs.bit_count() - 1
    return math.log(sum_count / n) / math.log(diff_count / n)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def initial_state(restart, rng):
    # Two independent starts for every prescribed (m, k) geometry.
    m, k = [(m, k) for m in MS for k in KS][restart % 16]
    density = 0.35 + 0.30 * rng.random()
    left = sum((rng.random() < density) << i for i in range(k))
    right = sum((rng.random() < density) << i for i in range(k))
    # Keep the extreme endpoints present, avoiding irrelevant translations.
    return m, k, left | 1, right | 1


def mutate(state, rng):
    m, k, left, right = state
    move = rng.random()
    if move < 0.62:
        side = rng.randrange(2)
        bit = 1 << rng.randrange(1, k)
        if side == 0:
            left ^= bit
        else:
            right ^= bit
    elif move < 0.88:
        for _ in range(2):
            side = rng.randrange(2)
            bit = 1 << rng.randrange(1, k)
            if side == 0:
                left ^= bit
            else:
                right ^= bit
    else:
        # Swap equal contiguous fringe blocks, either across or within sides.
        width = rng.randint(1, min(6, k - 1))
        a = rng.randint(1, k - width)
        b = rng.randint(1, k - width)
        mask = (1 << width) - 1
        if rng.random() < 0.65:
            x, y = (left >> a) & mask, (right >> b) & mask
            left = (left & ~(mask << a)) | (y << a)
            right = (right & ~(mask << b)) | (x << b)
        else:
            choose_left = rng.random() < 0.5
            side = left if choose_left else right
            x, y = (side >> a) & mask, (side >> b) & mask
            side = (side & ~(mask << a) & ~(mask << b)) | (x << b) | (y << a)
            if choose_left:
                left = side
            else:
                right = side
    return m, k, left | 1, right | 1


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        best = set(json.load(stream)["A"])
    best_score = quality(best)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    for restart in range(RESTARTS):
        state = initial_state(restart, rng)
        candidate = values(state)
        current_score = quality(candidate)
        slice_start = started + restart * SEARCH_SECONDS / RESTARTS
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)
        cache = {state: current_score}

        while time.monotonic() < slice_end:
            trial = mutate(state, rng)
            if trial == state:
                continue
            trial_score = cache.get(trial)
            if trial_score is None:
                trial_score = quality(values(trial))
                cache[trial] = trial_score
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.006 * max(0.015, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state = trial
                current_score = trial_score
                if current_score > best_score:
                    best = values(state)
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"dense-middle fringe annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
