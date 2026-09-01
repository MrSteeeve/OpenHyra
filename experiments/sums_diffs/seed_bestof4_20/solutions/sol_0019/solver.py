"""Exhaust periodic residue masks, then exactly polish their boundary rows."""

import heapq
import json
import math
import os
import time


SEARCH_SECONDS = 165.0
BOUNDARY_ROWS = 12
KEEP = 64


def quality_counts(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def quality(candidate):
    bits = 0
    for value in candidate:
        bits |= 1 << value
    span = max(candidate)
    reverse = 0
    for value in candidate:
        reverse |= 1 << (span - value)
    sums = 0
    diffs = 0
    for value in candidate:
        sums |= bits << value
        diffs |= bits << (span - value)
        diffs |= reverse << value
    return quality_counts(len(candidate), sums.bit_count(), diffs.bit_count())


def periodic_signature(modulus, mask):
    sum_carries = [set() for _ in range(modulus)]
    diff_carries = [set() for _ in range(modulus)]
    for left in mask:
        for right in mask:
            total = left + right
            sum_carries[total % modulus].add(total // modulus)
            delta = left - right
            diff_carries[delta % modulus].add(delta // modulus)
    sum_used = [carries for carries in sum_carries if carries]
    diff_used = [carries for carries in diff_carries if carries]
    sum_extra = sum(max(carries) - min(carries) for carries in sum_used)
    diff_extra = sum(max(carries) - min(carries) for carries in diff_used)
    return len(sum_used), sum_extra, len(diff_used), diff_extra


def periodic_values(modulus, mask, rows):
    return {modulus * row + bit for row in range(rows) for bit in mask}


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def census():
    elite = []
    serial = 0
    for modulus in range(5, 13):
        for encoded in range(1, 1 << modulus):
            mask = tuple(bit for bit in range(modulus) if encoded >> bit & 1)
            population = len(mask)
            sum_residues, sum_extra, diff_residues, diff_extra = periodic_signature(modulus, mask)
            for rows in range(40, 512 // population + 1):
                length = 2 * rows - 1
                sums = sum_residues * length + sum_extra
                diffs = diff_residues * length + diff_extra
                score = quality_counts(population * rows, sums, diffs)
                entry = (score, serial, modulus, mask, rows)
                serial += 1
                if len(elite) < KEEP:
                    heapq.heappush(elite, entry)
                elif score > elite[0][0]:
                    heapq.heapreplace(elite, entry)
    return sorted(elite, reverse=True)


def row_base(other, origin):
    bits = 0
    for value in other:
        bits |= 1 << value
    reverse = 0
    for value in other:
        reverse |= 1 << (origin - value)
    sums = 0
    diffs = 0
    for value in other:
        sums |= bits << value
        diffs |= bits << (origin - value)
        diffs |= reverse << value
    return bits, reverse, origin, sums, diffs


def score_replacement(other_n, base, row_start, encoded, modulus):
    bits, reverse, span, base_sums, base_diffs = base
    additions = [row_start + bit for bit in range(modulus) if encoded >> bit & 1]
    sums = base_sums
    diffs = base_diffs
    for value in additions:
        sums |= bits << value
        diffs |= bits << (span - value)
        # The two orientations share the same arbitrary origin ``span``.
        diffs |= reverse << value
    for left in additions:
        for right in additions:
            sums |= 1 << (left + right)
            diffs |= 1 << (span + left - right)
    return quality_counts(other_n + len(additions), sums.bit_count(), diffs.bit_count()), additions


def polish(candidate, modulus, rows, deadline, incumbent_score, incumbent, output):
    current = set(candidate)
    depth = min(BOUNDARY_ROWS, rows // 2)
    row_order = list(range(depth)) + list(range(rows - depth, rows))
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        for row in row_order:
            if time.monotonic() >= deadline:
                break
            start = row * modulus
            old = {value for value in current if start <= value < start + modulus}
            other = current - old
            base = row_base(other, modulus * rows - 1)
            old_encoded = sum(1 << (value - start) for value in old)
            best_score, old_values = score_replacement(
                len(other), base, start, old_encoded, modulus
            )
            best_values = set(old_values)
            for encoded in range(1, 1 << modulus):
                if len(other) + encoded.bit_count() > 512:
                    continue
                trial_score, additions = score_replacement(len(other), base, start, encoded, modulus)
                if trial_score > best_score:
                    best_score = trial_score
                    best_values = set(additions)
                if encoded & 127 == 0 and time.monotonic() >= deadline:
                    break
            if best_values != old:
                current = other | best_values
                changed = True
            if best_score > incumbent_score:
                incumbent_score = best_score
                incumbent = set(current)
                write_solution(output, incumbent)
    return incumbent_score, incumbent


def main():
    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        incumbent = set(json.load(stream)["A"])
    incumbent_score = quality(incumbent)
    write_solution(output, incumbent)

    elite = census()
    for core_score, _, modulus, mask, rows in elite:
        if time.monotonic() >= deadline:
            break
        candidate = periodic_values(modulus, mask, rows)
        if core_score > incumbent_score:
            incumbent_score, incumbent = core_score, candidate
            write_solution(output, incumbent)
        incumbent_score, incumbent = polish(
            candidate, modulus, rows, deadline, incumbent_score, incumbent, output
        )

    write_solution(output, incumbent)
    print(f"periodic-mask search complete: n={len(incumbent)} score={incumbent_score:.9f}")


if __name__ == "__main__":
    main()
