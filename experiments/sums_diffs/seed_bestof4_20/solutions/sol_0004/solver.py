"""Anneal endcap masks around a repeated periodic core."""

import json
import math
import os
import random
import time


SEED = 20260785782178
SEARCH_SECONDS = 170.0
RESTARTS = 10
MIN_WIDTH = 8
MAX_WIDTH = 24
MIN_ROWS = 3
MAX_ROWS = 60
PARENT = (
    12,
    5,
    frozenset((0, 1, 3, 4, 5, 8)),
    frozenset((0, 1, 4, 5, 8)),
    frozenset((0, 1, 3, 4, 5)),
)


def values(state):
    width, rows, lower, middle, upper = state
    result = set(lower)
    for row in range(1, rows - 1):
        result.update(row * width + bit for bit in middle)
    result.update((rows - 1) * width + bit for bit in upper)
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
    width, rows, lower, middle, upper = state
    size = len(lower) + (rows - 2) * len(middle) + len(upper)
    return (
        MIN_WIDTH <= width <= MAX_WIDTH
        and MIN_ROWS <= rows <= MAX_ROWS
        and 2 <= size <= 512
        and lower and middle and upper
        and all(0 <= bit < width for mask in (lower, middle, upper) for bit in mask)
    )


def initial_state(restart, rng):
    if restart == 0:
        return PARENT
    width = rng.randint(MIN_WIDTH, MAX_WIDTH)
    rows = rng.randint(MIN_ROWS, MAX_ROWS)
    # Scale the proven width-12 masks into the new period, then perturb them.
    masks = []
    for source in PARENT[2:]:
        mask = {min(width - 1, (bit * width + 6) // 12) for bit in source}
        masks.append(mask)
    for _ in range(1 + restart % 4):
        mask = masks[rng.randrange(3)]
        bit = rng.randrange(width)
        if bit in mask and len(mask) > 1:
            mask.remove(bit)
        else:
            mask.add(bit)
    state = (width, rows, *(frozenset(mask) for mask in masks))
    while not valid(state):
        rows -= 1
        state = (width, rows, *state[2:])
    return state


def mutate(state, rng):
    width, rows, lower, middle, upper = state
    masks = [set(lower), set(middle), set(upper)]
    move = rng.random()

    if move < 0.13:  # Change the period, retaining all still-valid residues.
        step = rng.choice((-2, -1, 1, 2))
        new_width = max(MIN_WIDTH, min(MAX_WIDTH, width + step))
        if new_width < width:
            masks = [{bit for bit in mask if bit < new_width} for mask in masks]
        width = new_width
    elif move < 0.27:  # Mostly walk in R, with occasional long jumps.
        if rng.random() < 0.8:
            rows += rng.choice((-2, -1, 1, 2))
        else:
            rows = rng.randint(MIN_ROWS, MAX_ROWS)
        rows = max(MIN_ROWS, min(MAX_ROWS, rows))
    elif move < 0.88:  # Toggle one to three bits in one template mask.
        mask = masks[rng.randrange(3)]
        for _ in range(1 if rng.random() < 0.72 else rng.choice((2, 3))):
            bit = rng.randrange(width)
            if bit in mask and len(mask) > 1:
                mask.remove(bit)
            else:
                mask.add(bit)
    else:  # Copy selected occupancy information between masks.
        source, target = rng.sample(range(3), 2)
        if rng.random() < 0.3:
            masks[target] = set(masks[source])
        else:
            for bit in rng.sample(range(width), rng.randint(1, min(3, width))):
                if bit in masks[source]:
                    masks[target].add(bit)
                elif bit in masks[target] and len(masks[target]) > 1:
                    masks[target].remove(bit)

    trial = (width, rows, *(frozenset(mask) for mask in masks))
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
                trial = values(trial_state)
                trial_score = quality(trial)
                cache[trial_state] = trial_score
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.010 * max(0.025, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state = trial_state
                current_score = trial_score
                if current_score > best_score:
                    best = values(state)
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"template annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
