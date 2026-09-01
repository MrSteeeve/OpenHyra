"""Variable-neighborhood refinement of the exact sol_0036 incumbent."""

import json
import math
import os
import random
import time


FALLBACK = (0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 132, 133, 134, 137,
            139, 143, 147, 151, 155, 156, 157, 158)
SEED = 5037
SEARCH_SECONDS = 150.0
LOW_N, HIGH_N = 60, 74
DOMAIN = range(171)


def score(values):
    n = len(values)
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def consider(candidate, best, best_score):
    candidate = tuple(sorted(candidate))
    if not LOW_N <= len(candidate) <= HIGH_N:
        return best, best_score, False
    value = score(candidate)
    if value > best_score + 1e-15:
        write_solution(candidate)
        return candidate, value, True
    return best, best_score, False


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best = FALLBACK
    best_score = score(best)
    write_solution(best)
    stagnant = 0

    while time.monotonic() < deadline:
        improved = False
        present = set(best)

        # Exhaust the complete one-toggle neighborhood and take its best move.
        single_best, single_score = best, best_score
        for x in DOMAIN:
            if time.monotonic() >= deadline:
                break
            if x in present:
                if len(best) <= LOW_N:
                    continue
                candidate = present - {x}
            else:
                if len(best) >= HIGH_N:
                    continue
                candidate = present | {x}
            value = score(tuple(sorted(candidate)))
            if value > single_score + 1e-15:
                single_best, single_score = tuple(sorted(candidate)), value
        if single_score > best_score + 1e-15:
            best, best_score, improved = single_best, single_score, True
            write_solution(best)
            present = set(best)

        # Best-first exact 1-for-1 swaps. Random tie order changes after restart.
        removed = list(best)
        added = list(set(DOMAIN) - present)
        rng.shuffle(removed)
        rng.shuffle(added)
        swap_best, swap_score = best, best_score
        for old in removed:
            base = present - {old}
            for new in added:
                if time.monotonic() >= deadline:
                    break
                candidate = tuple(sorted(base | {new}))
                value = score(candidate)
                if value > swap_score + 1e-15:
                    swap_best, swap_score = candidate, value
            if time.monotonic() >= deadline:
                break
        if swap_score > best_score + 1e-15:
            best, best_score, improved = swap_best, swap_score, True
            write_solution(best)
            present = set(best)

        # Sample paired toggles, including add/add, delete/delete, and swaps.
        for _ in range(2500):
            if time.monotonic() >= deadline:
                break
            x, y = rng.sample(range(171), 2)
            candidate = set(present)
            candidate.symmetric_difference_update((x, y))
            new_best, new_score, changed = consider(candidate, best, best_score)
            if changed:
                best, best_score, improved = new_best, new_score, True
                present = set(best)

        # Boundary shifts: replace either extreme by a nearby coordinate.
        for old in (best[0], best[-1]):
            for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
                new = old + delta
                if 0 <= new <= 170 and new not in present:
                    candidate = (present - {old}) | {new}
                    best, best_score, changed = consider(candidate, best, best_score)
                    if changed:
                        improved = True
                        present = set(best)

        stagnant = 0 if improved else stagnant + 1
        if stagnant >= 10:
            # A restart reorders best-first ties and draws a fresh deterministic
            # sample; the incumbent itself is never replaced by a worse state.
            stagnant = 0
            rng.shuffle(removed)
            rng.shuffle(added)

    write_solution(best)
    print(f"wrote VNS best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
