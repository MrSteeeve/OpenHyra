"""Anneal periodic multi-mask cores with independently mutable end rows."""

import json
import math
import os
import random
import time


SEED = 20260788782184
SEARCH_SECONDS = 167.0
RESTARTS = 10
MIN_WIDTH, MAX_WIDTH = 4, 10
MIN_PERIOD, MAX_PERIOD = 2, 8
MIN_ROWS, MAX_ROWS = 30, 120
MAX_END = 8

# The successful width-12 construction supplies a useful density/profile seed.
BASE = frozenset((0, 1, 4, 5, 8))
LOWER = frozenset((0, 1, 3, 4, 5, 8))
UPPER = frozenset((0, 1, 3, 4, 5))


def values(state):
    width, rows, period, depth, phases, lower, upper = state
    result = set()
    for row in range(rows):
        if row < depth:
            mask = lower[row]
        elif row >= rows - depth:
            mask = upper[row - (rows - depth)]
        else:
            mask = phases[(row - depth) % period]
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
    width, rows, period, depth, phases, lower, upper = state
    masks = phases + lower + upper
    if not (
        MIN_WIDTH <= width <= MAX_WIDTH
        and MIN_ROWS <= rows <= MAX_ROWS
        and MIN_PERIOD <= period <= MAX_PERIOD
        and period == len(phases)
        and 0 <= depth <= MAX_END
        and len(lower) == len(upper) == depth
        and 2 * depth < rows
        and all(masks)
        and all(0 <= bit < width for mask in masks for bit in mask)
    ):
        return False
    # Rows occupy disjoint width-sized blocks, so this is the exact cardinality.
    size = sum(map(len, lower)) + sum(map(len, upper))
    size += sum(len(phases[(row - depth) % period]) for row in range(depth, rows - depth))
    return 2 <= size <= 512


def scaled(mask, width):
    return frozenset(min(width - 1, (bit * width + 6) // 12) for bit in mask)


def initial_state(restart, rng):
    width = 8 if restart == 0 else rng.randint(MIN_WIDTH, MAX_WIDTH)
    period = 2 if restart == 0 else rng.randint(MIN_PERIOD, MAX_PERIOD)
    depth = 2 if restart == 0 else rng.randint(0, MAX_END)
    core = scaled(BASE, width)
    phases = [core for _ in range(period)]
    # Seed complementary phase types, while leaving homogeneous starts in half
    # the restarts so the experiment can decide whether periodicity is useful.
    if restart == 0 or restart % 2:
        for phase in range(1, period, 2):
            mask = set(core)
            bit = (phase * 3 + width // 2) % width
            if bit in mask and len(mask) > 1:
                mask.remove(bit)
            else:
                mask.add(bit)
            phases[phase] = frozenset(mask)
    lower = [core for _ in range(depth)]
    upper = [core for _ in range(depth)]
    if depth:
        lower[0] = scaled(LOWER, width)
        upper[-1] = scaled(UPPER, width)
    rows = 96 if restart == 0 else rng.randint(48, MAX_ROWS)
    state = (width, rows, period, depth, tuple(phases), tuple(lower), tuple(upper))
    while not valid(state) and rows > MIN_ROWS:
        rows -= 1
        state = (width, rows, period, depth, tuple(phases), tuple(lower), tuple(upper))
    return state


def mutate(state, rng):
    width, rows, period, depth, phases0, lower0, upper0 = state
    phases, lower, upper = list(phases0), list(lower0), list(upper0)
    move = rng.random()

    if move < 0.07:
        new_width = max(MIN_WIDTH, min(MAX_WIDTH, width + rng.choice((-1, 1))))
        if new_width < width:
            phases = [frozenset(bit for bit in mask if bit < new_width) for mask in phases]
            lower = [frozenset(bit for bit in mask if bit < new_width) for mask in lower]
            upper = [frozenset(bit for bit in mask if bit < new_width) for mask in upper]
        width = new_width
    elif move < 0.17:
        rows = max(MIN_ROWS, min(MAX_ROWS, rows + rng.choice((-5, -3, -1, 1, 3, 5))))
    elif move < 0.25:
        if rng.random() < 0.5 and period < MAX_PERIOD:
            phases.insert(rng.randrange(period + 1), phases[rng.randrange(period)])
            period += 1
        elif period > MIN_PERIOD:
            phases.pop(rng.randrange(period))
            period -= 1
    elif move < 0.33:
        new_depth = max(0, min(MAX_END, depth + rng.choice((-1, 1))))
        if new_depth > depth:
            lower.append(phases[0])
            upper.insert(0, phases[-1])
        elif new_depth < depth:
            lower.pop()
            upper.pop(0)
        depth = new_depth
    elif move < 0.87:
        masks = phases + lower + upper
        location = rng.randrange(len(masks))
        mask = set(masks[location])
        for _ in range(1 if rng.random() < 0.85 else 2):
            bit = rng.randrange(width)
            if bit in mask and len(mask) > 1:
                mask.remove(bit)
            else:
                mask.add(bit)
        replacement = frozenset(mask)
        if location < period:
            phases[location] = replacement
        elif location < period + depth:
            lower[location - period] = replacement
        else:
            upper[location - period - depth] = replacement
    elif move < 0.94:
        source, target = rng.sample(range(period), 2)
        phases[target] = phases[source]
    else:
        first, second = rng.sample(range(period), 2)
        phases[first], phases[second] = phases[second], phases[first]

    trial = (width, rows, period, depth, tuple(phases), tuple(lower), tuple(upper))
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
        current = values(state)
        current_score = quality(current)
        cache[state] = current_score
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
            temperature = 0.009 * max(0.015, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state, current_score = trial_state, trial_score
                if current_score > best_score:
                    best, best_score = values(state), current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"periodic-core annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
