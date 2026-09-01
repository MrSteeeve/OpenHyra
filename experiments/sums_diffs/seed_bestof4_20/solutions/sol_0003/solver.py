"""Anneal unions of residue masks in consecutive rows of width 12."""

import json
import math
import os
import random
import time


SEED = 20260784782174
SEARCH_SECONDS = 170.0
RESTARTS = 12
WIDTH = 12
MIN_ROWS = 3
MAX_ROWS = 40
PARENT_ROWS = (
    frozenset((0, 1, 2, 4, 5, 9)),
    frozenset((0, 1, 2, 4, 5, 9)),
    frozenset((0, 1, 2, 4, 5, 9)),
    frozenset((0, 1, 2, 4, 5)),
)


def values(rows):
    return frozenset(WIDTH * r + b for r, mask in enumerate(rows) for b in mask)


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


def initial_rows(restart, rng):
    if restart == 0:
        return list(PARENT_ROWS)
    row_count = rng.randint(MIN_ROWS, MAX_ROWS)
    rows = [PARENT_ROWS[r % len(PARENT_ROWS)] for r in range(row_count)]
    # Give each restart a few structured imperfections without destroying the seed.
    for _ in range(1 + restart // 3):
        r = rng.randrange(row_count)
        mask = set(rows[r])
        if rng.random() < 0.5 and len(mask) < 10:
            mask.add(rng.randrange(WIDTH))
        elif len(mask) > 2:
            mask.remove(rng.choice(tuple(mask)))
        rows[r] = frozenset(mask)
    return rows


def mutate(rows, rng):
    candidate = list(rows)
    move = rng.randrange(4)

    if move == 0:  # Copy a whole row, sometimes inserting it to explore R.
        source = candidate[rng.randrange(len(candidate))]
        if len(candidate) < MAX_ROWS and rng.random() < 0.35:
            candidate.insert(rng.randrange(len(candidate) + 1), source)
        else:
            candidate[rng.randrange(len(candidate))] = source
    elif move == 1:  # Delete a whole row.
        if len(candidate) > MIN_ROWS:
            del candidate[rng.randrange(len(candidate))]
        else:
            candidate[rng.randrange(len(candidate))] = candidate[rng.randrange(len(candidate))]
    elif move == 2:  # Shift every residue in one row together.
        r = rng.randrange(len(candidate))
        direction = rng.choice((-1, 1))
        shifted = {b + direction for b in candidate[r] if 0 <= b + direction < WIDTH}
        if 2 <= len(shifted) <= 10:
            candidate[r] = frozenset(shifted)
    else:  # Replace a row mask, biased toward mutations of an existing mask.
        r = rng.randrange(len(candidate))
        if rng.random() < 0.8:
            mask = set(candidate[rng.randrange(len(candidate))])
            changes = 1 if rng.random() < 0.75 else 2
            for _ in range(changes):
                if rng.random() < 0.5 and len(mask) > 2:
                    mask.remove(rng.choice(tuple(mask)))
                elif len(mask) < 10:
                    mask.add(rng.randrange(WIDTH))
        else:
            mask = set(rng.sample(range(WIDTH), rng.randint(2, 10)))
        if 2 <= len(mask) <= 10:
            candidate[r] = frozenset(mask)

    return candidate


def main():
    rng = random.Random(SEED)
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    best = values(PARENT_ROWS)
    best_score = quality(best)
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    for restart in range(RESTARTS):
        rows = initial_rows(restart, rng)
        current = values(rows)
        current_score = quality(current)
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)

        while time.monotonic() < slice_end:
            trial_rows = mutate(rows, rng)
            trial = values(trial_rows)
            trial_score = quality(trial)
            progress = (time.monotonic() - (slice_end - SEARCH_SECONDS / RESTARTS)) / (SEARCH_SECONDS / RESTARTS)
            temperature = 0.012 * max(0.03, 1.0 - progress)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                rows = trial_rows
                current = trial
                current_score = trial_score
                if current_score > best_score:
                    best = current
                    best_score = current_score

        write_solution(output, best)

    write_solution(output, best)
    print(f"row annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
