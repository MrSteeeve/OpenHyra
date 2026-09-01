"""Jointly anneal three translated/reflected copies of the tensor parent."""

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
SEED = 5030
SEARCH_SECONDS = 150.0
SHIFT_LIMIT = 4000
REPLICAS = 32
STEPS = (-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64)


def union_values(t1, e1, t2, e2):
    values = set(PARENT)
    values.update(t1 + e1 * x for x in PARENT)
    values.update(t2 + e2 * x for x in PARENT)
    return tuple(sorted(values))


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
    return math.log(sums.bit_count() / n) / math.log(
        (2 * distances.bit_count() - 1) / n
    )


def evaluate(state):
    values = union_values(*state)
    return score_values(values), values


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS

    # sol_0029: the best two-copy union, represented here with a duplicate third copy.
    fallback_state = (-1568, 1, 0, 1)
    best_score, best_values = evaluate(fallback_state)
    write_solution(best_values)

    temperatures = tuple(
        0.000015 * (0.012 / 0.000015) ** (i / (REPLICAS - 1))
        for i in range(REPLICAS)
    )
    chains = []
    anchors = ((-1568, 1), (0, 1), (313, -1), (1881, -1))
    for i in range(REPLICAS):
        # Every initial state is the exact incumbent, with permuted duplicate copies.
        if i & 1:
            state = (0, 1, -1568, 1)
        else:
            state = fallback_state
        score, values = evaluate(state)
        chains.append([score, state, values])

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        index = iteration % REPLICAS
        current_score, state, current_values = chains[index]
        proposal = list(state)
        move = rng.random()
        if move < 0.76:
            which = 0 if rng.random() < 0.5 else 2
            proposal[which] = max(
                -SHIFT_LIMIT, min(SHIFT_LIMIT, proposal[which] + rng.choice(STEPS))
            )
        elif move < 0.92:
            which = 1 if rng.random() < 0.5 else 3
            # Keep the copy's interval fixed while reversing its orientation.
            proposal[which - 1] += proposal[which] * PARENT[-1]
            proposal[which] = -proposal[which]
        else:
            # Occasionally relocate a copy onto/near either established overlap basin.
            which = 0 if rng.random() < 0.5 else 2
            center, orientation = rng.choice(anchors)
            proposal[which] = center + rng.choice(STEPS)
            proposal[which + 1] = orientation
        proposal = tuple(proposal)
        new_score, new_values = evaluate(proposal)
        delta = new_score - current_score
        if new_score >= 0.0 and (
            delta >= 0.0 or rng.random() < math.exp(delta / temperatures[index])
        ):
            chains[index] = [new_score, proposal, new_values]
            if new_score > best_score:
                best_score, best_values = new_score, new_values
                write_solution(best_values)

        # Adjacent replica exchanges preserve each temperature's broad/narrow role.
        if iteration % 64 == 0:
            parity = (iteration // 64) & 1
            for left in range(parity, REPLICAS - 1, 2):
                right = left + 1
                s_left = chains[left][0]
                s_right = chains[right][0]
                exponent = (s_right - s_left) * (
                    1.0 / temperatures[left] - 1.0 / temperatures[right]
                )
                if exponent >= 0.0 or rng.random() < math.exp(exponent):
                    chains[left], chains[right] = chains[right], chains[left]

    write_solution(best_values)
    print(f"wrote joint three-copy best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
