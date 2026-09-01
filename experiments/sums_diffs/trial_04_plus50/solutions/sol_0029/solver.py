"""Search overlapping translated and reflected unions of the tensor parent."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_MASK = ((1 << len(CELLS)) - 1) ^ sum(1 << i for i in MISSING)
PARENT = tuple(
    sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if (PARENT_MASK >> i) & 1)
)
SEED = 5028
SEARCH_SECONDS = 150.0
LIMIT = PARENT[-1]


def score_values(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    lo = values[0]
    shifted = tuple(x - lo for x in values)
    mask = sum(1 << x for x in shifted)
    sums = 0
    distances = 0
    for x in shifted:
        sums |= mask << x
        distances |= mask >> x
    sum_count = sums.bit_count()
    diff_count = 2 * distances.bit_count() - 1
    return math.log(sum_count / n) / math.log(diff_count / n)


def affine_union(t, reflected):
    if reflected:
        other = (t - x for x in PARENT)
    else:
        other = (t + x for x in PARENT)
    return tuple(sorted(set(PARENT).union(other)))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    best_values = PARENT
    best_score = score_values(best_values)
    write_solution(best_values)

    # Keep the strongest basins from the complete translated/reflected sweep.
    leaders = []
    for reflected in (False, True):
        for t in range(-LIMIT, LIMIT + 1):
            values = affine_union(t, reflected)
            if len(values) > 512:
                continue
            candidate_score = score_values(values)
            item = (candidate_score, t, reflected, values)
            if len(leaders) < 32:
                leaders.append(item)
                leaders.sort(key=lambda z: z[0], reverse=True)
            elif candidate_score > leaders[-1][0]:
                leaders[-1] = item
                leaders.sort(key=lambda z: z[0], reverse=True)
            if candidate_score > best_score:
                best_score, best_values = candidate_score, values
                write_solution(best_values)

    # Each chain owns an affine union plus a small, reversible deletion set.
    chains = []
    for candidate_score, t, reflected, values in leaders:
        chains.append([candidate_score, t, reflected, set(), values])

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        current_score, t, reflected, deleted, current_values = chain
        new_t, new_reflected = t, reflected
        new_deleted = set(deleted)
        move = rng.random()
        if move < 0.25:
            new_t = min(LIMIT, max(-LIMIT, t + rng.choice((-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32))))
        elif move < 0.35:
            new_reflected = not reflected
        elif move < 0.82:
            available = [x for x in current_values if x not in new_deleted]
            count = min(len(available), rng.randint(1, 4))
            if count:
                new_deleted.update(rng.sample(available, count))
        elif new_deleted:
            count = min(len(new_deleted), rng.randint(1, 4))
            for x in rng.sample(tuple(new_deleted), count):
                new_deleted.remove(x)
        else:
            continue

        base_values = affine_union(new_t, new_reflected)
        new_deleted.intersection_update(base_values)
        if len(new_deleted) > 36:
            for x in rng.sample(tuple(new_deleted), len(new_deleted) - 36):
                new_deleted.remove(x)
        new_values = tuple(x for x in base_values if x not in new_deleted)
        new_score = score_values(new_values)
        fraction = min(1.0, (SEARCH_SECONDS - max(0.0, deadline - time.monotonic())) / SEARCH_SECONDS)
        temperature = 0.003 * (1.0 - fraction) + 0.00002
        delta = new_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            chain[:] = [new_score, new_t, new_reflected, new_deleted, new_values]
            if new_score > best_score:
                best_score, best_values = new_score, new_values
                write_solution(best_values)

        # Periodic basin injection prevents all chains drifting into sparse states.
        if iteration % 3200 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            leader = leaders[rng.randrange(min(8, len(leaders)))]
            chains[weakest] = [leader[0], leader[1], leader[2], set(), leader[3]]

    write_solution(best_values)
    print(f"wrote affine-union best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
