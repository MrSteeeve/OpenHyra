"""Anneal dense-core sets with independently optimized left and right fringes."""

import json
import math
import os
import random
import time

INITIAL_SET = (0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29)
SEED = 4008
SEARCH_SECONDS = 160.0
REPLICAS = 32
MIN_W = 12
MAX_W = 40
MAX_M = 240


def values_from_state(state):
    """Return A = L union [w,m-w] union (m-R)."""
    w, m, left, right = state
    values = [i for i in range(w) if (left >> i) & 1]
    values.extend(range(w, m - w + 1))
    values.extend(m - i for i in range(w) if (right >> i) & 1)
    return tuple(sorted(values))


def score_values(values):
    bits = sum(1 << value for value in values)
    sums = 0
    positive_differences = 0
    pending = bits
    while pending:
        low_bit = pending & -pending
        value = low_bit.bit_length() - 1
        sums |= bits << value
        positive_differences |= bits >> value
        pending ^= low_bit
    n = len(values)
    sum_count = sums.bit_count()
    difference_count = 2 * positive_differences.bit_count() - 1
    return math.log(sum_count / n) / math.log(difference_count / n)


def score_state(state, cache):
    cached = cache.get(state)
    if cached is not None:
        return cached
    result = score_values(values_from_state(state))
    if len(cache) >= 100_000:
        cache.clear()
    cache[state] = result
    return result


def encode_parent(w, m):
    """Project the incumbent onto a legal geometry, retaining its fringe bits."""
    parent = set(INITIAL_SET)
    left = sum(1 << i for i in range(w) if i in parent)
    right = sum(1 << i for i in range(w) if m - i in parent)
    return (w, m, left, right)


def random_state(rng):
    w = rng.randint(MIN_W, MAX_W)
    m = rng.randint(2 * w, MAX_M)
    left = rng.getrandbits(w)
    right = rng.getrandbits(w)
    return (w, m, left, right)


def mutate(state, rng):
    w, m, left, right = state
    if rng.random() < 0.80:
        flips = rng.randint(1, 4)
        positions = rng.sample(range(2 * w), flips)
        for position in positions:
            if position < w:
                left ^= 1 << position
            else:
                right ^= 1 << (position - w)
    else:
        if rng.random() < 0.45:
            if rng.random() < 0.10:
                new_w = rng.randint(MIN_W, MAX_W)
            else:
                new_w = max(MIN_W, min(MAX_W, w + rng.choice((-2, -1, 1, 2))))
            if 2 * new_w > MAX_M:
                new_w = MAX_M // 2
            if new_w > w:
                added = ((1 << (new_w - w)) - 1) << w
                left |= rng.getrandbits(new_w) & added
                right |= rng.getrandbits(new_w) & added
            else:
                left &= (1 << new_w) - 1
                right &= (1 << new_w) - 1
            w = new_w
            m = max(2 * w, m)
        else:
            if rng.random() < 0.10:
                m = rng.randint(2 * w, MAX_M)
            else:
                m = max(2 * w, min(MAX_M, m + rng.choice((-8, -4, -2, -1, 1, 2, 4, 8))))

    candidate = (w, m, left, right)
    if len(values_from_state(candidate)) < 2:
        return state
    return candidate


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    cache = {}
    best = INITIAL_SET
    best_score = score_values(best)
    write_solution(best)

    # Use the best incumbent projections as half the population; the remaining
    # replicas provide broad coverage of the legal geometry and fringe masks.
    projections = []
    for w in range(MIN_W, MAX_W + 1):
        for m in range(2 * w, MAX_M + 1):
            state = encode_parent(w, m)
            projections.append((score_state(state, cache), state))
    projections.sort(reverse=True)

    states = [state for _, state in projections[: REPLICAS // 2]]
    states.extend(random_state(rng) for _ in range(REPLICAS - len(states)))
    scores = [score_state(state, cache) for state in states]
    for state, state_score in zip(states, scores):
        if state_score > best_score:
            best = values_from_state(state)
            best_score = state_score
            write_solution(best)

    start = time.monotonic()
    deadline = start + SEARCH_SECONDS
    steps = 0
    while time.monotonic() < deadline:
        replica = steps % REPLICAS
        candidate = mutate(states[replica], rng)
        candidate_score = score_state(candidate, cache)
        progress = min(1.0, (time.monotonic() - start) / SEARCH_SECONDS)
        base_temperature = 0.012 * (1.0 - progress) + 0.00005
        temperature = base_temperature * (0.35 + 1.65 * replica / (REPLICAS - 1))
        delta = candidate_score - scores[replica]
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            states[replica] = candidate
            scores[replica] = candidate_score
            if candidate_score > best_score:
                best = values_from_state(candidate)
                best_score = candidate_score
                write_solution(best)
        steps += 1

    write_solution(best)
    print(f"wrote dense-core best: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
