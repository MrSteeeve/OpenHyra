"""Exact block coordinate ascent on the width-four incumbent fringes."""

import json
import math
import os
import time


SEARCH_SECONDS = 165.0
WIDTH = 4
FRINGE = 24


def quality_counts(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def exact_quality(candidate):
    sums = {a + b for a in candidate for b in candidate}
    diffs = {a - b for a in candidate for b in candidate}
    return quality_counts(len(candidate), len(sums), len(diffs))


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def optimize_block(candidate, first_row, span, deadline):
    """Exhaust the 2^16 occupancies of four consecutive width-four rows."""
    positions = [WIDTH * first_row + bit for bit in range(16)]
    fixed = candidate.difference(positions)
    fixed_bits = 0
    reverse_bits = 0
    for value in fixed:
        fixed_bits |= 1 << value
        reverse_bits |= 1 << (span - value)

    fixed_sums = 0
    fixed_diffs = 0
    for value in fixed:
        fixed_sums |= fixed_bits << value
        fixed_diffs |= fixed_bits << (span - value)

    cross_sums = [fixed_bits << value for value in positions]
    cross_diffs = [
        (fixed_bits << (span - value)) | (reverse_bits << value)
        for value in positions
    ]

    # Each entry contains all contributions involving at least one selected
    # block cell.  The least-significant-bit recurrence adds its cross terms
    # and its pairs with the cells already present in the smaller mask.
    count = 1 << 16
    sum_parts = [0] * count
    diff_parts = [0] * count
    best_mask = 0
    best_score = -1.0
    fixed_n = len(fixed)

    for mask in range(count):
        if mask and (mask & 2047) == 0 and time.monotonic() >= deadline:
            return candidate, exact_quality(candidate), False
        if mask:
            low = mask & -mask
            index = low.bit_length() - 1
            rest = mask ^ low
            value = positions[index]
            sums = sum_parts[rest] | cross_sums[index] | (1 << (2 * value))
            diffs = diff_parts[rest] | cross_diffs[index] | (1 << span)
            other = rest
            while other:
                other_low = other & -other
                other_value = positions[other_low.bit_length() - 1]
                sums |= 1 << (value + other_value)
                diffs |= (1 << (value - other_value + span)) | (1 << (other_value - value + span))
                other ^= other_low
            sum_parts[mask] = sums
            diff_parts[mask] = diffs

        n = fixed_n + mask.bit_count()
        if n < 2 or n > 512:
            continue
        score = quality_counts(
            n,
            (fixed_sums | sum_parts[mask]).bit_count(),
            (fixed_diffs | diff_parts[mask]).bit_count(),
        )
        if score > best_score:
            best_score = score
            best_mask = mask

    result = set(fixed)
    for index, value in enumerate(positions):
        if best_mask >> index & 1:
            result.add(value)
    return result, best_score, True


def block_starts(rows, offset):
    starts = list(range(offset, FRINGE - 3, 4))
    right = rows - FRINGE
    starts.extend(range(right + offset, rows - 3, 4))
    return starts


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        candidate = set(json.load(stream)["A"])

    best_score = exact_quality(candidate)
    write_solution(output, candidate)
    span = max(candidate)
    rows = span // WIDTH + 1
    deadline = time.monotonic() + SEARCH_SECONDS

    improved = True
    sweeps = 0
    while improved and time.monotonic() < deadline:
        improved = False
        for offset in range(4):
            starts = block_starts(rows, offset)
            # Alternate the end visited first, while retaining deterministic
            # coverage of every legal offset block.
            if (sweeps + offset) & 1:
                starts.reverse()
            for start in starts:
                trial, score, complete = optimize_block(candidate, start, span, deadline)
                if not complete:
                    write_solution(output, candidate)
                    print(f"block ascent timed out: n={len(candidate)} score={best_score:.9f}")
                    return
                if score > best_score + 1e-15:
                    candidate = trial
                    best_score = score
                    improved = True
                    write_solution(output, candidate)
        sweeps += 1

    write_solution(output, candidate)
    print(f"block ascent complete: n={len(candidate)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
