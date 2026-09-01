"""Delete and repair a near-limit product of the best dense-fringe set."""

import json
import math
import os
import random
import time


A66 = (0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
       67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
       82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
       97, 98, 99, 100, 101, 102, 103, 104, 105, 132, 133, 134, 137,
       139, 143, 147, 151, 155, 156, 157, 158)
B8 = (0, 2, 3, 4, 7, 11, 12, 14)
PRODUCT = tuple(a + 256 * b for b in B8 for a in A66)
SEED = 5049
SEARCH_SECONDS = 145.0


def score(values):
    n = len(values)
    if n < 2:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = positive_differences = 0
    for x in values:
        sums |= mask << x
        positive_differences |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * positive_differences.bit_count() - 1) / n)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def deletion_choices(values, rng, limit):
    """Mix fringe, layer-balanced, and random deletion choices."""
    candidates = set()
    ordered = sorted(values)
    candidates.update(ordered[:8])
    candidates.update(ordered[-8:])
    for b in B8:
        layer = [x for x in values if 256 * b <= x <= 256 * b + 158]
        if layer:
            candidates.add(layer[0])
            candidates.add(layer[-1])
            candidates.add(rng.choice(layer))
    remaining = list(set(values) - candidates)
    rng.shuffle(remaining)
    candidates.update(remaining[:max(0, limit - len(candidates))])
    choices = list(candidates)
    rng.shuffle(choices)
    return choices[:limit]


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    # The proven 66-point set is always a valid immediate fallback.
    best = A66
    best_score = score(best)
    write_solution(best)

    initial = tuple(sorted(PRODUCT))
    beam = [(score(initial), initial)]
    finals = []

    # Exactly 16 deletions first reach the size limit; continuing to 40 tests
    # whether sacrificing more points produces a better exponent.
    for removed in range(1, 41):
        if time.monotonic() >= deadline - 8.0:
            break
        generated = {}
        choice_limit = 96 if len(beam) == 1 else 48
        for _, values in beam:
            for x in deletion_choices(values, rng, choice_limit):
                candidate = tuple(y for y in values if y != x)
                if candidate not in generated:
                    generated[candidate] = score(candidate)
            if time.monotonic() >= deadline - 8.0:
                break
        ranked = sorted(((s, v) for v, s in generated.items()), reverse=True)
        beam = ranked[:64]
        if removed >= 16:
            finals.extend(beam[:8])
            for candidate_score, candidate in beam:
                if candidate_score > best_score:
                    best_score, best = candidate_score, candidate
                    write_solution(best)

    # Tabu-guided 1-for-1 repairs retain each candidate's cardinality while
    # crossing deletion barriers. All accepted and checkpointed scores are exact.
    pool = sorted(finals, reverse=True)[:32]
    if not pool and len(initial) <= 512:
        pool = [(score(initial), initial)]
    tabu = {}
    iteration = 0
    while pool and time.monotonic() < deadline:
        iteration += 1
        old_score, old = pool[iteration % len(pool)]
        present = set(old)
        removed = list(set(initial) - present)
        if not removed:
            continue
        outgoing = deletion_choices(old, rng, 12)
        rng.shuffle(removed)
        incoming = removed[:12]
        local_best = None
        for x in outgoing:
            if tabu.get(x, 0) > iteration:
                continue
            base = present - {x}
            for y in incoming:
                candidate = tuple(sorted(base | {y}))
                candidate_score = score(candidate)
                if local_best is None or candidate_score > local_best[0]:
                    local_best = (candidate_score, candidate, x, y)
            if time.monotonic() >= deadline:
                break
        if local_best is None:
            continue
        candidate_score, candidate, x, y = local_best
        # Mild tabu-search diversification: accept the best sampled neighbor.
        pool[iteration % len(pool)] = (candidate_score, candidate)
        tabu[y] = iteration + 50
        if candidate_score > best_score:
            best_score, best = candidate_score, candidate
            write_solution(best)
        if iteration % 200 == 0:
            pool.sort(reverse=True)

    write_solution(best)
    print(f"wrote product-deletion best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
