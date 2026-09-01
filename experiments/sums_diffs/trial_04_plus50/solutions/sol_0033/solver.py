"""Search quotient--remainder refoldings of the best tensor construction."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
TENSOR = tuple(
    sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if (PARENT_MASK >> i) & 1)
)
# sol_0029 is the overlap-maximizing translate TENSOR union (TENSOR + 28*56).
PARENT = tuple(sorted(set(TENSOR).union(x + 1568 for x in TENSOR)))
SEED = 5032
SEARCH_SECONDS = 150.0
SWEEP_SECONDS = 112.0


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    shifted = tuple(x - values[0] for x in values)
    mask = sum(1 << x for x in shifted)
    sums = 0
    distances = 0
    for x in shifted:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n
    )


def refold(m, k, reflected):
    if reflected:
        return tuple(sorted({k * (x // m) - (x % m) for x in PARENT}))
    return tuple(sorted({k * (x // m) + (x % m) for x in PARENT}))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def keep_leader(leaders, item):
    key = (item[1], item[2], item[3])
    for old in leaders:
        if (old[1], old[2], old[3]) == key:
            return
    if len(leaders) < 32:
        leaders.append(item)
        leaders.sort(reverse=True, key=lambda z: z[0])
    elif item[0] > leaders[-1][0]:
        leaders[-1] = item
        leaders.sort(reverse=True, key=lambda z: z[0])


def main():
    rng = random.Random(SEED)
    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    sweep_deadline = started + SWEEP_SECONDS

    best_values = PARENT
    best_score = score_values(best_values)
    write_solution(best_values)

    # Deterministic exhaustive-order sweep.  The separate deadline guarantees
    # time for refinement even on slower evaluator machines.
    leaders = []
    stop_sweep = False
    for m in range(16, 513):
        for k in range(1, 513):
            for reflected in (False, True):
                values = refold(m, k, reflected)
                candidate_score = score_values(values)
                keep_leader(leaders, (candidate_score, m, k, reflected, values))
                if candidate_score > best_score:
                    best_score, best_values = candidate_score, values
                    write_solution(best_values)
            if time.monotonic() >= sweep_deadline:
                stop_sweep = True
                break
        if stop_sweep:
            break

    if not leaders:
        values = refold(56, 56, False)
        leaders.append((score_values(values), 56, 56, False, values))

    # Each chain carries a symmetric-difference toggle set.  Parameter moves
    # retain only toggles that remain near the newly refolded coordinate span.
    chains = [list(item) + [set()] for item in leaders]
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        current_score, m, k, reflected, current_values, toggles = chain
        new_m, new_k, new_reflected = m, k, reflected
        new_toggles = set(toggles)
        move = rng.random()
        if move < 0.25:
            new_m = min(512, max(16, m + rng.choice((-8, -4, -2, -1, 1, 2, 4, 8))))
        elif move < 0.50:
            new_k = min(512, max(1, k + rng.choice((-16, -8, -4, -2, -1, 1, 2, 4, 8, 16))))
        elif move < 0.58:
            new_reflected = not reflected
        else:
            base = refold(new_m, new_k, new_reflected)
            lo, hi = base[0], base[-1]
            for _ in range(rng.randint(1, 3)):
                if rng.random() < 0.62:
                    x = rng.choice(base)
                else:
                    x = rng.randint(lo, hi)
                if x in new_toggles:
                    new_toggles.remove(x)
                else:
                    new_toggles.add(x)

        base = refold(new_m, new_k, new_reflected)
        lo, hi = base[0], base[-1]
        new_toggles = {x for x in new_toggles if lo <= x <= hi}
        if len(new_toggles) > 24:
            for x in rng.sample(tuple(new_toggles), len(new_toggles) - 24):
                new_toggles.remove(x)
        new_values = tuple(sorted(set(base).symmetric_difference(new_toggles)))
        new_score = score_values(new_values)
        elapsed = time.monotonic() - started
        temperature = 0.0025 * max(0.0, 1.0 - elapsed / SEARCH_SECONDS) + 0.00002
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            chain[:] = [new_score, new_m, new_k, new_reflected, new_values, new_toggles]
            if new_score > best_score:
                best_score, best_values = new_score, new_values
                write_solution(best_values)

        if iteration % 2400 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            leader = leaders[rng.randrange(min(8, len(leaders)))]
            chains[weakest] = list(leader) + [set()]

    write_solution(best_values)
    print(f"wrote quotient-remainder best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
