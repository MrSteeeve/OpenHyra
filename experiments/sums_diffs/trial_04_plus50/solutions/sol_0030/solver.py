"""Add a highly overlapping third tensor copy to the affine-union incumbent."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
P = tuple(sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if i not in MISSING))
# sol_0029, reconstructed exactly as P union (P + 1568).
B = tuple(sorted(set(P).union(x + 1568 for x in P)))
SEED = 5029
SEARCH_SECONDS = 150.0


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
    return math.log(sums.bit_count() / n) / math.log((2 * distances.bit_count() - 1) / n)


def third_union(t, reflected):
    other = (t - x for x in P) if reflected else (t + x for x in P)
    return tuple(sorted(set(B).union(other)))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def retain(leaders, item):
    if len(leaders) < 32:
        leaders.append(item)
        leaders.sort(key=lambda z: z[0], reverse=True)
    elif item[0] > leaders[-1][0]:
        leaders[-1] = item
        leaders.sort(key=lambda z: z[0], reverse=True)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = B
    best_score = score_values(B)
    write_solution(B)

    leaders = []
    for reflected in (False, True):
        for t in range(-4000, 4001):
            values = third_union(t, reflected)
            if len(values) > 512:
                continue
            candidate_score = score_values(values)
            retain(leaders, (candidate_score, t, reflected, values))
            if candidate_score > best_score:
                best_score, best_values = candidate_score, values
                write_solution(best_values)

    # Refine each strong shift basin with local shift moves and reversible deletions.
    chains = [[s, t, r, set(), v] for s, t, r, v in leaders]
    iteration = 0
    shift_steps = tuple(range(1, 33))
    while chains and time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        current_score, t, reflected, deleted, current_values = chain
        new_t = t
        new_deleted = set(deleted)
        move = rng.random()
        if move < 0.42:
            new_t = max(-4000, min(4000, t + rng.choice((-1, 1)) * rng.choice(shift_steps)))
        elif move < 0.80:
            available = [x for x in current_values if x not in new_deleted]
            count = min(len(available), rng.randint(1, 6))
            if not count:
                continue
            new_deleted.update(rng.sample(available, count))
        elif new_deleted:
            count = min(len(new_deleted), rng.randint(1, 6))
            for x in rng.sample(tuple(new_deleted), count):
                new_deleted.remove(x)
        else:
            continue

        base_values = third_union(new_t, reflected)
        new_deleted.intersection_update(base_values)
        if len(new_deleted) > 48:
            for x in rng.sample(tuple(new_deleted), len(new_deleted) - 48):
                new_deleted.remove(x)
        new_values = tuple(x for x in base_values if x not in new_deleted)
        new_score = score_values(new_values)
        elapsed = SEARCH_SECONDS - max(0.0, deadline - time.monotonic())
        temperature = 0.0025 * (1.0 - min(1.0, elapsed / SEARCH_SECONDS)) + 0.00002
        if new_score >= current_score or rng.random() < math.exp((new_score - current_score) / temperature):
            chain[:] = [new_score, new_t, reflected, new_deleted, new_values]
            if new_score > best_score:
                best_score, best_values = new_score, new_values
                write_solution(best_values)

        if iteration % 3000 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            leader = leaders[rng.randrange(min(8, len(leaders)))]
            chains[weakest] = [leader[0], leader[1], leader[2], set(), leader[3]]

    write_solution(best_values)
    print(f"wrote third-copy best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
