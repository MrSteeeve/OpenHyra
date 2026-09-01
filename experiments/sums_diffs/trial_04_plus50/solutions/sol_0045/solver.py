"""Anneal seven permutation matchings projected from the 66-point incumbent."""

import json
import math
import os
import random
import time


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((x, y) for y in BASE for x in BASE)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
P = tuple(sorted(x + 56 * y for i, (x, y) in enumerate(CELLS) if i not in MISSING))
FALLBACK = tuple(sorted(set(P).union(x + 1568 for x in P)))
SEED = 5044
SEARCH_SECONDS = 150.0
MATCHINGS = 7


def score(values):
    n = len(values)
    if n < 2 or n > 512:
        return -1.0
    mask = sum(1 << x for x in values)
    sums = distances = 0
    for x in values:
        sums |= mask << x
        distances |= mask >> x
    return math.log(sums.bit_count() / n) / math.log((2 * distances.bit_count() - 1) / n)


def materialize(q, permutations, toggles):
    cells = set(toggles)
    for permutation in permutations:
        cells.update(enumerate(permutation))
    if len(cells) > 512:
        return ()
    return tuple(sorted(P[i] + q * P[j] for i, j in cells))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def new_chain(rng, q=None):
    n = len(P)
    permutations = []
    # Nearby cyclic diagonals give a regular, collision-free starting mask.
    offset = rng.randrange(n)
    for k in range(MATCHINGS):
        step = 2 * k - MATCHINGS + 1
        permutations.append([(i + offset + step) % n for i in range(n)])
    q = q if q is not None else rng.randint(160, 320)
    values = materialize(q, permutations, set())
    return [score(values), q, permutations, set(), values]


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    best_values = FALLBACK
    best_score = score(FALLBACK)
    write_solution(FALLBACK)

    chains = [new_chain(rng, q) for q in range(160, 321, 10)]
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        chain = chains[iteration % len(chains)]
        old_score, q, permutations, toggles, _ = chain
        new_q = q
        new_permutations = [list(p) for p in permutations]
        new_toggles = set(toggles)
        move = rng.random()

        if move < 0.48:
            # A transposition preserves the permutation-matching invariant.
            k = rng.randrange(MATCHINGS)
            i, j = rng.sample(range(len(P)), 2)
            new_permutations[k][i], new_permutations[k][j] = (
                new_permutations[k][j], new_permutations[k][i]
            )
        elif move < 0.66:
            k = rng.randrange(MATCHINGS)
            replacement = list(range(len(P)))
            rng.shuffle(replacement)
            new_permutations[k] = replacement
        elif move < 0.84:
            new_q = max(160, min(320, q + rng.choice((-8, -4, -2, -1, 1, 2, 4, 8))))
        else:
            # Sparse toggles allow small departures from the seven matchings.
            for _ in range(rng.randint(1, 3)):
                cell = (rng.randrange(len(P)), rng.randrange(len(P)))
                new_toggles.symmetric_difference_update((cell,))
            if len(new_toggles) > 24:
                new_toggles.remove(rng.choice(tuple(new_toggles)))

        values = materialize(new_q, new_permutations, new_toggles)
        new_score = score(values) if values else -1.0
        remaining = max(0.0, deadline - time.monotonic())
        temperature = 0.0035 * (remaining / SEARCH_SECONDS) + 0.00002
        if new_score >= old_score or rng.random() < math.exp((new_score - old_score) / temperature):
            chain[:] = [new_score, new_q, new_permutations, new_toggles, values]
            if new_score > best_score:
                best_score, best_values = new_score, values
                write_solution(values)

        if iteration % 2000 == 0:
            weakest = min(range(len(chains)), key=lambda i: chains[i][0])
            leader = max(chains, key=lambda z: z[0])
            if rng.random() < 0.7:
                clone_perms = [list(p) for p in leader[2]]
                for _ in range(12):
                    p = clone_perms[rng.randrange(MATCHINGS)]
                    i, j = rng.sample(range(len(P)), 2)
                    p[i], p[j] = p[j], p[i]
                clone_values = materialize(leader[1], clone_perms, set(leader[3]))
                chains[weakest] = [score(clone_values), leader[1], clone_perms,
                                   set(leader[3]), clone_values]
            else:
                chains[weakest] = new_chain(rng)

    write_solution(best_values)
    print(f"wrote matching-projection best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
