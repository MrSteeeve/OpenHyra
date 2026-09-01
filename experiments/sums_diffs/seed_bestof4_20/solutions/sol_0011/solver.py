"""Anneal wide modular intervals with independently mutable fringe rows."""

import json
import math
import os
import random
import time


SEED = 20260792782196
SEARCH_SECONDS = 168.0
MODULI = (9, 12, 16, 20, 24, 28, 34, 40)
MIN_DEPTH = 2
MAX_DEPTH = 10


def row_count(core):
    return 480 // len(core)


def values(state):
    modulus, depth, lower, core, upper = state
    rows = row_count(core)
    result = set()
    for row, mask in enumerate(lower):
        result.update(row * modulus + residue for residue in mask)
    for row in range(depth, rows - depth):
        result.update(row * modulus + residue for residue in core)
    for offset, mask in enumerate(upper):
        row = rows - depth + offset
        result.update(row * modulus + residue for residue in mask)
    return frozenset(result)


def quality(candidate):
    mask = 0
    for value in candidate:
        mask |= 1 << value
    sum_mask = 0
    nonnegative_diff_mask = 0
    for value in candidate:
        sum_mask |= mask << value
        nonnegative_diff_mask |= mask >> value
    n = len(candidate)
    sums = sum_mask.bit_count()
    diffs = 2 * nonnegative_diff_mask.bit_count() - 1
    return math.log(sums / n) / math.log(diffs / n)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def valid(state):
    modulus, depth, lower, core, upper = state
    if not (9 <= modulus <= 40 and MIN_DEPTH <= depth <= MAX_DEPTH):
        return False
    if not core or len(lower) != depth or len(upper) != depth:
        return False
    rows = row_count(core)
    if rows <= 2 * depth:
        return False
    masks = lower + (core,) + upper
    if any(not mask for mask in masks):
        return False
    if any(residue < 0 or residue >= modulus for mask in masks for residue in mask):
        return False
    size = sum(map(len, lower)) + (rows - 2 * depth) * len(core) + sum(map(len, upper))
    return 2 <= size <= 512


def initial_state(modulus, rng):
    # Seed each width with a perturbed, repeated 4-column motif, then let the
    # annealer discover an ordinary (carry-allowing) residue pattern.
    core = {r for r in range(modulus) if r % 4 in (0, 1)}
    for _ in range(1 + modulus // 10):
        residue = rng.randrange(modulus)
        if residue in core and len(core) > 2:
            core.remove(residue)
        else:
            core.add(residue)
    core = frozenset(core)
    depth = rng.randint(3, min(MAX_DEPTH, max(3, row_count(core) // 4)))
    lower = [core for _ in range(depth)]
    upper = [core for _ in range(depth)]
    for masks in (lower, upper):
        for index in range(depth):
            mask = set(core)
            for _ in range(1 + (index == 0)):
                residue = rng.randrange(modulus)
                if residue in mask and len(mask) > 1:
                    mask.remove(residue)
                else:
                    mask.add(residue)
            masks[index] = frozenset(mask)
    state = (modulus, depth, tuple(lower), core, tuple(upper))
    return state if valid(state) else (modulus, MIN_DEPTH, (core,) * MIN_DEPTH, core, (core,) * MIN_DEPTH)


def mutate(state, rng):
    modulus, depth, lower0, core, upper0 = state
    lower, upper = list(lower0), list(upper0)
    move = rng.random()

    if move < 0.13:
        new_depth = max(MIN_DEPTH, min(MAX_DEPTH, depth + rng.choice((-1, 1))))
        if new_depth > depth:
            lower.append(core)
            upper.insert(0, core)
        elif new_depth < depth:
            lower.pop()
            upper.pop(0)
        depth = new_depth
    elif move < 0.43:
        mask = set(core)
        for _ in range(1 if rng.random() < 0.88 else 2):
            residue = rng.randrange(modulus)
            if residue in mask and len(mask) > 1:
                mask.remove(residue)
            else:
                mask.add(residue)
        core = frozenset(mask)
    elif move < 0.94:
        location = rng.randrange(2 * depth)
        target = lower if location < depth else upper
        index = location if location < depth else location - depth
        mask = set(target[index])
        for _ in range(1 if rng.random() < 0.90 else 2):
            residue = rng.randrange(modulus)
            if residue in mask and len(mask) > 1:
                mask.remove(residue)
            else:
                mask.add(residue)
        target[index] = frozenset(mask)
    else:
        masks = lower + [core] + upper
        source = rng.choice(masks)
        location = rng.randrange(2 * depth)
        if location < depth:
            lower[location] = source
        else:
            upper[location - depth] = source

    trial = (modulus, depth, tuple(lower), core, tuple(upper))
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
    for restart, modulus in enumerate(MODULI):
        state = initial_state(modulus, rng)
        current = values(state)
        current_score = quality(current)
        slice_start = started + restart * SEARCH_SECONDS / len(MODULI)
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / len(MODULI))
        cache = {state: current_score}

        while time.monotonic() < slice_end:
            trial_state = mutate(state, rng)
            if trial_state == state:
                continue
            trial_score = cache.get(trial_state)
            if trial_score is None:
                trial_score = quality(values(trial_state))
                cache[trial_state] = trial_score
            progress = (time.monotonic() - slice_start) / max(0.001, slice_end - slice_start)
            temperature = 0.006 * max(0.015, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state = trial_state
                current_score = trial_score
                if current_score > best_score:
                    best = values(state)
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"wide modular-fringe annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
