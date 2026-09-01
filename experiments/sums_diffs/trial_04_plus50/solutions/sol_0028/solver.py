"""Width-32 exact marginal-deletion beam for the q=56 tensor image."""

import json
import math
import os
import random
import time
from math import gcd


BASE = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
MISSING = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT = tuple(
    sorted(
        x + 56 * y
        for i, (x, y) in enumerate((p for y in BASE for p in ((x, y) for x in BASE)))
        if i not in MISSING
    )
)
SEED = 5027
SPAN = 1881
WIDTH = 32
SEARCH_SECONDS = 155.0


def objective(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def make_state(values):
    sums = [0] * (2 * SPAN + 1)
    diffs = [0] * (2 * SPAN + 1)
    for x in values:
        for y in values:
            sums[x + y] += 1
            diffs[x - y + SPAN] += 1
    ns = sum(bool(x) for x in sums)
    nd = sum(bool(x) for x in diffs)
    return (tuple(values), sums, diffs, ns, nd, objective(len(values), ns, nd))


def deletion_metrics(state, x):
    values, sums, diffs, ns, nd, unused = state
    sum_delta = {}
    diff_delta = {}
    for y in values:
        s = x + y
        sum_delta[s] = sum_delta.get(s, 0) + (1 if y == x else 2)
        d1 = x - y + SPAN
        diff_delta[d1] = diff_delta.get(d1, 0) + 1
        if y != x:
            d2 = y - x + SPAN
            diff_delta[d2] = diff_delta.get(d2, 0) + 1
    new_ns = ns - sum(sums[k] == v for k, v in sum_delta.items())
    new_nd = nd - sum(diffs[k] == v for k, v in diff_delta.items())
    return objective(len(values) - 1, new_ns, new_nd), new_ns, new_nd, sum_delta, diff_delta


def delete_point(state, x, metrics=None):
    values, sums, diffs, ns, nd, unused = state
    if metrics is None:
        metrics = deletion_metrics(state, x)
    score, new_ns, new_nd, sum_delta, diff_delta = metrics
    new_sums = sums.copy()
    new_diffs = diffs.copy()
    for k, v in sum_delta.items():
        new_sums[k] -= v
    for k, v in diff_delta.items():
        new_diffs[k] -= v
    new_values = tuple(v for v in values if v != x)
    return (new_values, new_sums, new_diffs, new_ns, new_nd, score)


def add_point(state, x):
    values, sums, diffs, ns, nd, unused = state
    sum_delta = {}
    diff_delta = {}
    for y in values:
        s = x + y
        sum_delta[s] = sum_delta.get(s, 0) + 2
        diff_delta[x - y + SPAN] = diff_delta.get(x - y + SPAN, 0) + 1
        diff_delta[y - x + SPAN] = diff_delta.get(y - x + SPAN, 0) + 1
    sum_delta[2 * x] = sum_delta.get(2 * x, 0) + 1
    diff_delta[SPAN] = diff_delta.get(SPAN, 0) + 1
    new_ns = ns + sum(sums[k] == 0 for k in sum_delta)
    new_nd = nd + sum(diffs[k] == 0 for k in diff_delta)
    new_sums = sums.copy()
    new_diffs = diffs.copy()
    for k, v in sum_delta.items():
        new_sums[k] += v
    for k, v in diff_delta.items():
        new_diffs[k] += v
    new_values = tuple(sorted(values + (x,)))
    score = objective(len(new_values), new_ns, new_nd)
    return (new_values, new_sums, new_diffs, new_ns, new_nd, score)


def canonical(values):
    lo = values[0]
    shifted = tuple(x - lo for x in values)
    divisor = 0
    for x in shifted[1:]:
        divisor = gcd(divisor, x)
    if divisor > 1:
        shifted = tuple(x // divisor for x in shifted)
    return shifted


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    deadline = time.monotonic() + SEARCH_SECONDS
    parent = make_state(PARENT)
    beam = [parent]
    best = parent
    write_solution(best[0])
    deletions = 0

    while len(beam[0][0]) > 80 and time.monotonic() < deadline:
        candidates = []
        for state in beam:
            for x in state[0]:
                metrics = deletion_metrics(state, x)
                candidates.append((metrics[0], state, x, metrics))
            if time.monotonic() >= deadline:
                break
        candidates.sort(key=lambda item: item[0], reverse=True)
        new_beam = []
        seen = set()
        for unused, state, x, metrics in candidates:
            child = delete_point(state, x, metrics)
            key = canonical(child[0])
            if key not in seen:
                seen.add(key)
                new_beam.append(child)
                if child[5] > best[5]:
                    best = child
                    write_solution(best[0])
                if len(new_beam) == WIDTH:
                    break
        if not new_beam:
            break
        beam = new_beam
        deletions += 1

        # Tabu-guided delete/add repairs at each eight-level boundary.
        if deletions % 8 == 0:
            current = beam[0]
            tabu = {}
            for step in range(2000):
                if time.monotonic() >= deadline:
                    break
                choices = [x for x in current[0] if tabu.get(x, -1) <= step]
                if not choices:
                    choices = list(current[0])
                remove_pool = rng.sample(choices, min(12, len(choices)))
                removed = max(remove_pool, key=lambda x: deletion_metrics(current, x)[0])
                partial = delete_point(current, removed)
                occupied = set(partial[0])
                add_pool = []
                while len(add_pool) < 24:
                    x = rng.randrange(SPAN + 1)
                    if x not in occupied and x not in add_pool and tabu.get(x, -1) <= step:
                        add_pool.append(x)
                trial = max((add_point(partial, x) for x in add_pool), key=lambda s: s[5])
                if trial[5] > current[5]:
                    added = next(x for x in trial[0] if x not in occupied)
                    tabu[removed] = step + 50
                    tabu[added] = step + 50
                    current = trial
                    if current[5] > best[5]:
                        best = current
                        write_solution(best[0])
            if current[5] > beam[-1][5]:
                beam[-1] = current
                beam.sort(key=lambda s: s[5], reverse=True)

    write_solution(best[0])
    print(f"wrote deletion-beam best: n={len(best[0])} score={best[5]:.9f}")


if __name__ == "__main__":
    main()
