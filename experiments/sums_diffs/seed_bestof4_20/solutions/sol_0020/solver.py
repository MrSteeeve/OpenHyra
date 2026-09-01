"""Exact-score beam search for three-segment piecewise-periodic sets."""

import json
import math
import os
import random
import time


SEED = 20260801782223
SEARCH_SECONDS = 165.0
BEAM_WIDTH = 2000
MODULI = range(5, 9)
MIN_SEGMENT = 12
MAX_SEGMENT = 48
SEGMENTS = 3
BOUNDARY_ROWS = 8


def quality(candidate):
    """Exact score, using Python integers as dense sum/difference bitsets."""
    vals = sorted(candidate)
    bits = sum(1 << value for value in vals)
    sums = 0
    positive_diffs = 0
    for value in vals:
        sums |= bits << value
        positive_diffs |= bits >> value
    n = len(vals)
    sum_count = sums.bit_count()
    diff_count = 2 * positive_diffs.bit_count() - 1
    return math.log(sum_count / n) / math.log(diff_count / n)


def make_values(modulus, parts):
    result = set()
    first_row = 0
    for rows, mask in parts:
        residues = tuple(bit for bit in range(modulus) if mask >> bit & 1)
        for row in range(first_row, first_row + rows):
            base = row * modulus
            result.update(base + bit for bit in residues)
        first_row += rows
    return frozenset(result)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def retain(scored):
    """Deduplicate constructions and retain the exact-score beam."""
    unique = {}
    for score, modulus, parts, values in scored:
        key = (modulus, parts)
        old = unique.get(key)
        if old is None or score > old[0]:
            unique[key] = (score, modulus, parts, values)
    return sorted(unique.values(), key=lambda item: item[0], reverse=True)[:BEAM_WIDTH]


def continuation_choices(modulus, parent_rank, stage, rng):
    """A broad deterministic sample of independent mask/length continuations."""
    all_mask = (1 << modulus) - 1
    masks = {all_mask, 1, all_mask ^ 1}
    for shift in range(modulus):
        masks.add(1 << shift)
        masks.add(((1 << shift) | (1 << ((shift + 1) % modulus))))
    # More choices for the best states, while keeping a full 2000-wide beam.
    count = 24 if parent_rank < 160 else (12 if parent_rank < 700 else 6)
    while len(masks) < count:
        masks.add(rng.randint(1, all_mask))
    lengths = {12, 18, 24, 30, 36, 42, 48}
    choices = []
    ordered_masks = sorted(masks)
    for index in range(count):
        mask = ordered_masks[index % len(ordered_masks)]
        length = tuple(sorted(lengths))[(index * 3 + parent_rank + stage) % len(lengths)]
        if index >= len(lengths):
            length = rng.randint(MIN_SEGMENT, MAX_SEGMENT)
        choices.append((length, mask))
    return choices


def beam_search(deadline, rng, incumbent, incumbent_score):
    best, best_score = incumbent, incumbent_score
    global_beam = []
    # Exhaust every first mask and every allowed first-segment length.
    for modulus in MODULI:
        scored = []
        for mask in range(1, 1 << modulus):
            for rows in range(MIN_SEGMENT, MAX_SEGMENT + 1):
                parts = ((rows, mask),)
                values = make_values(modulus, parts)
                scored.append((quality(values), modulus, parts, values))
        global_beam.extend(retain(scored))

    beam = retain(global_beam)
    for stage in range(1, SEGMENTS):
        scored = []
        for rank, (_, modulus, parts, _) in enumerate(beam):
            for rows, mask in continuation_choices(modulus, rank, stage, rng):
                trial_parts = parts + ((rows, mask),)
                values = make_values(modulus, trial_parts)
                if len(values) <= 512:
                    score = quality(values)
                    scored.append((score, modulus, trial_parts, values))
                    if stage == SEGMENTS - 1 and score > best_score:
                        best, best_score = values, score
            if time.monotonic() >= deadline - 25.0:
                break
        beam = retain(scored)
        if not beam:
            break
    return beam, best, best_score


def polish(candidate, deadline, best, best_score):
    """Greedy exact coordinate polishing at both ends and both junctions."""
    modulus, parts = candidate[1], candidate[2]
    rows = [part[0] for part in parts]
    junctions = [rows[0], rows[0] + rows[1]]
    total_rows = sum(rows)
    mutable_rows = set(range(min(BOUNDARY_ROWS, total_rows)))
    mutable_rows.update(range(max(0, total_rows - BOUNDARY_ROWS), total_rows))
    for junction in junctions:
        mutable_rows.update(range(max(0, junction - BOUNDARY_ROWS),
                                  min(total_rows, junction + BOUNDARY_ROWS)))
    current = set(candidate[3])
    current_score = candidate[0]
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for row in sorted(mutable_rows):
            for residue in range(modulus):
                value = row * modulus + residue
                trial = set(current)
                if value in trial:
                    if len(trial) <= 2:
                        continue
                    trial.remove(value)
                elif len(trial) < 512:
                    trial.add(value)
                score = quality(trial)
                if score > current_score:
                    current, current_score = trial, score
                    improved = True
                    if score > best_score:
                        best, best_score = frozenset(trial), score
                if time.monotonic() >= deadline:
                    return best, best_score
    return best, best_score


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        incumbent = frozenset(json.load(stream)["A"])
    incumbent_score = quality(incumbent)
    write_solution(output, incumbent)

    deadline = time.monotonic() + SEARCH_SECONDS
    beam, best, best_score = beam_search(deadline, rng, incumbent, incumbent_score)
    # Polish several distinct leading cores, retaining the pinned parent globally.
    for candidate in beam[:12]:
        if time.monotonic() >= deadline:
            break
        best, best_score = polish(candidate, deadline, best, best_score)
        write_solution(output, best)
    write_solution(output, best)
    print(f"piecewise-periodic beam complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
