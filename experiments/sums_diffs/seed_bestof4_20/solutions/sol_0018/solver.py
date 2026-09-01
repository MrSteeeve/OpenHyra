"""Exhaust piecewise-periodic width-four cores, then polish their endcaps."""

import json
import math
import os
import time


SEED = 20260799782217  # The experiment is deterministic; retained for replay identity.
SEARCH_SECONDS = 165.0
WIDTH = 4
MIN_ROWS = 80
MAX_ROWS = 128
BOUNDARY_ROWS = 8


def read_parent(path):
    with open(path) as stream:
        return frozenset(json.load(stream)["A"])


def make_values(rows, change, first_mask, second_mask, left=None, right=None):
    """Decode row masks; change is the first row using second_mask."""
    left = left or (first_mask,) * BOUNDARY_ROWS
    right = right or (second_mask,) * BOUNDARY_ROWS
    answer = []
    for row in range(rows):
        if row < BOUNDARY_ROWS:
            mask = left[row]
        elif row >= rows - BOUNDARY_ROWS:
            mask = right[row - (rows - BOUNDARY_ROWS)]
        else:
            mask = first_mask if row < change else second_mask
        base = WIDTH * row
        for bit in range(WIDTH):
            if mask & (1 << bit):
                answer.append(base + bit)
    return frozenset(answer)


def counts_and_quality(candidate):
    """Exact support counts using Python integers as dense bitsets."""
    bits = 0
    for value in candidate:
        bits |= 1 << value
    sum_bits = 0
    positive_diff_bits = 0
    for value in candidate:
        sum_bits |= bits << value
        positive_diff_bits |= bits >> value
    sums = sum_bits.bit_count()
    diffs = 2 * positive_diff_bits.bit_count() - 1
    n = len(candidate)
    score = math.log(sums / n) / math.log(diffs / n)
    return sums, diffs, score


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def polish(rows, change, first_mask, second_mask, incumbent, incumbent_score, deadline):
    """Deterministic coordinate ascent over all sixteen boundary row masks."""
    left = [first_mask] * BOUNDARY_ROWS
    right = [second_mask] * BOUNDARY_ROWS
    best = incumbent
    best_score = incumbent_score
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for side, index in ((side, index) for side in (0, 1)
                            for index in range(BOUNDARY_ROWS)):
            masks = left if side == 0 else right
            old_mask = masks[index]
            coordinate_score = best_score
            coordinate_mask = old_mask
            coordinate_values = best
            for mask in range(1, 16):
                if time.monotonic() >= deadline:
                    return best, best_score
                masks[index] = mask
                trial = make_values(rows, change, first_mask, second_mask,
                                    tuple(left), tuple(right))
                if len(trial) > 512:
                    continue
                score = counts_and_quality(trial)[2]
                if score > coordinate_score:
                    coordinate_score = score
                    coordinate_mask = mask
                    coordinate_values = trial
            masks[index] = coordinate_mask
            if coordinate_score > best_score:
                best, best_score = coordinate_values, coordinate_score
                improved = True
    return best, best_score


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    parent = read_parent(os.path.join(directory, "parent_solution.json"))
    best = parent
    best_score = counts_and_quality(parent)[2]
    write_solution(output, best)

    started = time.monotonic()
    enumeration_deadline = started + 150.0
    final_deadline = started + SEARCH_SECONDS
    # Keep several structurally different leaders for endcap polishing.  A core
    # is retained only when it sets a new record, avoiding a large candidate pool.
    leaders = []
    core_record = -1.0
    exhausted = True
    for rows in range(MIN_ROWS, MAX_ROWS + 1):
        for change in range(BOUNDARY_ROWS + 1, rows - BOUNDARY_ROWS):
            for first_mask in range(1, 16):
                for second_mask in range(1, 16):
                    if time.monotonic() >= enumeration_deadline:
                        exhausted = False
                        break
                    trial = make_values(rows, change, first_mask, second_mask)
                    if len(trial) > 512:
                        continue
                    score = counts_and_quality(trial)[2]
                    if score > core_record:
                        core_record = score
                        leaders.append((score, rows, change, first_mask, second_mask))
                        leaders = leaders[-24:]
                    if score > best_score:
                        best, best_score = trial, score
                if not exhausted:
                    break
            if not exhausted:
                break
        if not exhausted:
            break

    # Highest records first. Each sweep inherits its own natural periodic ends;
    # global best remains the pinned parent unless a genuinely better set appears.
    for _, rows, change, first_mask, second_mask in reversed(leaders):
        if time.monotonic() >= final_deadline:
            break
        seed = make_values(rows, change, first_mask, second_mask)
        seed_score = counts_and_quality(seed)[2]
        trial, score = polish(rows, change, first_mask, second_mask,
                              seed, seed_score, final_deadline)
        if score > best_score:
            best, best_score = trial, score
            write_solution(output, best)

    write_solution(output, best)
    print(f"piecewise width-4 search complete: n={len(best)} score={best_score:.9f} "
          f"exhausted={exhausted}")


if __name__ == "__main__":
    main()
