"""Anneal independent fringes around a dense interval core."""

import json
import math
import os
import random
import time


SEED = 20260789782188
SEARCH_SECONDS = 168.0
RESTARTS = 12
MIN_N, MAX_N = 300, 510
MIN_K, MAX_K = 20, 100


def candidate_bits(state):
    nmax, k, left, right = state
    core = ((1 << (nmax - 2 * k + 1)) - 1) << k
    reflected = 0
    while right:
        low = right & -right
        reflected |= 1 << (nmax - (low.bit_length() - 1))
        right ^= low
    return left | core | reflected


def quality_bits(bits, nmax):
    sums = 0
    diffs = 0
    remaining = bits
    while remaining:
        low = remaining & -remaining
        value = low.bit_length() - 1
        sums |= bits << value
        diffs |= bits << (nmax - value)
        remaining ^= low
    size = bits.bit_count()
    return math.log(sums.bit_count() / size) / math.log(diffs.bit_count() / size)


def values(bits):
    result = []
    while bits:
        low = bits & -bits
        result.append(low.bit_length() - 1)
        bits ^= low
    return result


def write_solution(path, answer):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": answer}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def initial_state(restart, rng):
    # Include several densities: useful fringes tend to be neither intervals nor tiny sets.
    nmax = rng.randint(MIN_N, MAX_N)
    k = rng.randint(MIN_K, MAX_K)
    density = (0.25, 0.40, 0.55, 0.70)[restart % 4]
    left = sum((rng.random() < density) << bit for bit in range(k))
    right = sum((rng.random() < density) << bit for bit in range(k))
    # Translation/reflection normalization: retain both extreme endpoints.
    return nmax, k, left | 1, right | 1


def mutate(state, rng):
    nmax, k, left, right = state
    move = rng.random()
    if move < 0.055:
        nmax = max(MIN_N, min(MAX_N, nmax + rng.choice((-5, -3, -2, -1, 1, 2, 3, 5))))
    elif move < 0.11:
        new_k = max(MIN_K, min(MAX_K, k + rng.choice((-2, -1, 1, 2))))
        if new_k > k:
            added = ((1 << new_k) - 1) ^ ((1 << k) - 1)
            # The points were in the old core, so keeping them is the smooth mutation.
            left |= added
            right |= added
        else:
            left &= (1 << new_k) - 1
            right &= (1 << new_k) - 1
        k = new_k
    elif move < 0.72:
        target_left = rng.random() < 0.5
        mask = left if target_left else right
        flips = 1 if rng.random() < 0.88 else rng.choice((2, 3))
        for _ in range(flips):
            mask ^= 1 << rng.randrange(1, k)
        if target_left:
            left = mask
        else:
            right = mask
    elif move < 0.91:
        target_left = rng.random() < 0.5
        mask = left if target_left else right
        occupied = [bit for bit in range(1, k) if mask >> bit & 1]
        empty = [bit for bit in range(1, k) if not (mask >> bit & 1)]
        if occupied and empty:
            mask ^= (1 << rng.choice(occupied)) | (1 << rng.choice(empty))
        if target_left:
            left = mask
        else:
            right = mask
    else:
        # Coordinated cross-fringe changes can repair sums without changing density.
        bit1, bit2 = rng.randrange(1, k), rng.randrange(1, k)
        left ^= 1 << bit1
        right ^= 1 << bit2
    return nmax, k, left | 1, right | 1


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        best = sorted(set(json.load(stream)["A"]))
    parent_bits = sum(1 << value for value in best)
    best_score = quality_bits(parent_bits, best[-1])
    write_solution(output, best)

    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    for restart in range(RESTARTS):
        state = initial_state(restart, rng)
        bits = candidate_bits(state)
        score = quality_bits(bits, state[0])
        slice_start = started + restart * SEARCH_SECONDS / RESTARTS
        slice_end = min(deadline, started + (restart + 1) * SEARCH_SECONDS / RESTARTS)
        steps = 0
        while time.monotonic() < slice_end:
            trial = mutate(state, rng)
            trial_bits = candidate_bits(trial)
            trial_score = quality_bits(trial_bits, trial[0])
            progress = (time.monotonic() - slice_start) / (slice_end - slice_start)
            temperature = 0.0045 * max(0.015, (1.0 - progress) ** 2)
            delta = trial_score - score
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                state, bits, score = trial, trial_bits, trial_score
                if score > best_score:
                    best_score = score
                    best = values(bits)
            steps += 1
            if steps % 4096 == 0 and time.monotonic() >= deadline:
                break
        write_solution(output, best)

    write_solution(output, best)
    print(f"fringe-pair annealing complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
