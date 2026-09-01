"""Anneal several independent boundary rows around a periodic core."""

import json
import math
import os
import random
import time


SEED = 20260786782180
SEARCH_SECONDS = 168.0
RESTARTS = 8
MIN_WIDTH = 8
MAX_WIDTH = 20
MIN_ROWS = 40
MAX_ROWS = 100
MIN_DEPTH = 2
MAX_DEPTH = 8

LOWER = frozenset((0, 1, 3, 4, 5, 8))
MIDDLE = frozenset((0, 1, 4, 5, 8))
UPPER = frozenset((0, 1, 3, 4, 5))
PARENT = (12, 60, 2, (LOWER, MIDDLE), MIDDLE, (MIDDLE, UPPER))


def values(state):
    width, rows, depth, lower, middle, upper = state
    result = set()
    for row, mask in enumerate(lower):
        result.update(row * width + bit for bit in mask)
    for row in range(depth, rows - depth):
        result.update(row * width + bit for bit in middle)
    for offset, mask in enumerate(upper):
        row = rows - depth + offset
        result.update(row * width + bit for bit in mask)
    return frozenset(result)


def quality(candidate):
    sums = {a + b for a in candidate for b in candidate}
    diffs = {a - b for a in candidate for b in candidate}
    n = len(candidate)
    return math.log(len(sums) / n) / math.log(len(diffs) / n)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def valid(state):
    width, rows, depth, lower, middle, upper = state
    masks = lower + (middle,) + upper
    size = sum(map(len, lower)) + (rows - 2 * depth) * len(middle) + sum(map(len, upper))
    return (
        MIN_WIDTH <= width <= MAX_WIDTH
        and MIN_ROWS <= rows <= MAX_ROWS
        and MIN_DEPTH <= depth <= MAX_DEPTH
        and 2 * depth < rows
        and len(lower) == len(upper) == depth
        and 2 <= size <= 512
        and all(masks)
        and all(0 <= bit < width for mask in masks for bit in mask)
    )


def scaled(mask, width):
    return frozenset(min(width - 1, (bit * width + 6) // 12) for bit in mask)


def initial_state(restart, rng):
    if restart == 0:
        return PARENT
    width = rng.randint(MIN_WIDTH, MAX_WIDTH)
    depth = rng.randint(MIN_DEPTH, MAX_DEPTH)
    middle = scaled(MIDDLE, width)
    lower = [middle for _ in range(depth)]
    upper = [middle for _ in range(depth)]
    lower[0] = scaled(LOWER, width)
    upper[-1] = scaled(UPPER, width)
    # Long cores are the useful regime, but retain variety across restarts.
    rows = rng.randint(64, MAX_ROWS)
    state = (width, rows, depth, tuple(lower), middle, tuple(upper))
    while not valid(state) and rows > MIN_ROWS:
        rows -= 1
        state = (width, rows, depth, tuple(lower), middle, tuple(upper))
    while not valid(state):
        target = rng.choice(lower + [middle] + upper)
        if len(target) > 1:
            replacement = frozenset(list(target)[:-1])
            if target is middle:
                middle = replacement
            else:
                for masks in (lower, upper):
                    for index, mask in enumerate(masks):
                        if mask is target:
                            masks[index] = replacement
                            break
        state = (width, rows, depth, tuple(lower), middle, tuple(upper))
    return state


def mutate(state, rng):
    width, rows, depth, lower0, middle0, upper0 = state
    lower, middle, upper = list(lower0), middle0, list(upper0)
    move = rng.random()

    if move < 0.08:
        new_width = max(MIN_WIDTH, min(MAX_WIDTH, width + rng.choice((-1, 1))))
        if new_width < width:
            lower = [frozenset(bit for bit in mask if bit < new_width) for mask in lower]
            middle = frozenset(bit for bit in middle if bit < new_width)
            upper = [frozenset(bit for bit in mask if bit < new_width) for mask in upper]
        width = new_width
    elif move < 0.20:
        rows += rng.choice((-3, -2, -1, 1, 2, 3, 5))
        rows = max(MIN_ROWS, min(MAX_ROWS, rows))
    elif move < 0.28:
        new_depth = max(MIN_DEPTH, min(MAX_DEPTH, depth + rng.choice((-1, 1))))
        if new_depth > depth:
            lower.append(middle)
            upper.insert(0, middle)
        elif new_depth < depth:
            lower.pop()
            upper.pop(0)
        depth = new_depth
    elif move < 0.91:
        location = rng.randrange(2 * depth + 1)
        mask = set((lower + [middle] + upper)[location])
        for _ in range(1 if rng.random() < 0.84 else 2):
            bit = rng.randrange(width)
            if bit in mask and len(mask) > 1:
                mask.remove(bit)
            else:
                mask.add(bit)
        replacement = frozenset(mask)
        if location < depth:
            lower[location] = replacement
        elif location == depth:
            middle = replacement
        else:
            upper[location - depth - 1] = replacement
    else:
        masks = lower + [middle] + upper
        source, target = rng.sample(range(len(masks)), 2)
        replacement = masks[source]
        if target < depth:
            lower[target] = replacement
        elif target == depth:
            middle = replacement
        else:
            upper[target - depth - 1] = replacement

    trial = (width, rows, depth, tuple(lower), middle, tuple(upper))
    return trial if valid(trial) else state


def main():
    rng = random.Random(SEED)
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    best = values(PARENT)
    best_score = quality(best)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    cache = {PARENT: best_score}
    for restart in range(RESTARTS):
        state = initial_state(restart, rng)
        current = values(state)
        current_score = cache.setdefault(state, quality(current))
        slice_start = started + restart * SEARCH_SECONDS / RESTARTS
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)

        while time.monotonic() < slice_end:
            trial_state = mutate(state, rng)
            if trial_state == state:
                continue
            trial_score = cache.get(trial_state)
            if trial_score is None:
                trial_score = quality(values(trial_state))
                cache[trial_state] = trial_score
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.008 * max(0.02, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state = trial_state
                current_score = trial_score
                if current_score > best_score:
                    best = values(state)
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"boundary-layer annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
