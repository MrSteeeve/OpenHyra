"""Block destroy--repair search around the exact sol_0036 incumbent."""

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
SEED = 5038
SEARCH_SECONDS = 150.0
DOMAIN = tuple(range(166))
LOW_N, HIGH_N = 62, 70


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


def boundary_shift(values, delta):
    """Move both dense-core boundaries together by one or two layers."""
    candidate = set(values)
    amount = abs(delta)
    if delta > 0:
        for j in range(amount):
            candidate.add(52 - j)
            candidate.add(106 + j)
    else:
        for j in range(amount):
            candidate.discard(53 + j)
            candidate.discard(105 - j)
    return tuple(sorted(candidate))


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best = FALLBACK
    best_score = score(best)
    write_solution(best)

    # The pool carries distinct local basins between destroy--repair rounds.
    pool = [(best_score, best)]
    seen = {best}
    stagnant = 0
    round_number = 0

    while time.monotonic() < deadline:
        round_number += 1
        parent = rng.choice(pool)[1] if stagnant < 8 else FALLBACK
        if stagnant >= 8:
            pool = [(best_score, best), (score(FALLBACK), FALLBACK)]
            stagnant = 0

        # Restrict destruction to the outermost twenty coordinates at each end.
        left_edge, right_edge = parent[0] + 19, parent[-1] - 19
        fringe = [x for x in parent if x <= left_edge or x >= right_edge]
        destroy_count = min(len(fringe), rng.randint(3, 8))
        removed = set(rng.sample(fringe, destroy_count))
        base = tuple(x for x in parent if x not in removed)

        # Rebuild to roughly the incumbent scale.  Width 64 permits several
        # mutually incompatible block replacements to survive each layer.
        target = rng.choices(range(LOW_N, HIGH_N + 1),
                             weights=(1, 2, 4, 7, 10, 7, 4, 2, 1))[0]
        target = max(target, len(base))
        beam = [(score(base), base)]
        while len(beam[0][1]) < target and time.monotonic() < deadline:
            candidates = {}
            for _, state in beam:
                occupied = set(state)
                for x in DOMAIN:
                    if x in occupied:
                        continue
                    candidate = tuple(sorted(state + (x,)))
                    if candidate in candidates:
                        continue
                    value = score(candidate)
                    candidates[candidate] = value
                if time.monotonic() >= deadline:
                    break
            if not candidates:
                break
            beam = sorted(((value, state) for state, value in candidates.items()),
                          reverse=True)[:64]

        additions = list(beam)
        # Occasionally make the requested coordinated core-boundary move.
        if round_number % 3 == 0:
            for _, state in beam[:16]:
                for delta in (-2, -1, 1, 2):
                    shifted = boundary_shift(state, delta)
                    if LOW_N <= len(shifted) <= HIGH_N:
                        additions.append((score(shifted), shifted))

        improved = False
        for value, state in additions:
            if state not in seen:
                seen.add(state)
                pool.append((value, state))
            if value > best_score + 1e-15:
                best_score, best = value, state
                write_solution(best)
                improved = True
        pool = sorted(pool, reverse=True)[:16]
        stagnant = 0 if improved else stagnant + 1

    write_solution(best)
    print(f"wrote block-repair best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
