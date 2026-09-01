"""Exhaustive small-template census followed by carry-free product tests."""

import json
import math
import os
import time


SEARCH_SECONDS = 165.0
DOMAIN_END = 20
MAX_SIZE = 512


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def counts(candidate):
    values = tuple(candidate)
    sums = {a + b for a in values for b in values}
    diffs = {a - b for a in values for b in values}
    return len(sums), len(diffs)


def quality(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def bit_counts(mask, values):
    sum_bits = 0
    positive_differences = 0
    for value in values:
        sum_bits |= mask << value
        positive_differences |= mask >> value
    return sum_bits.bit_count(), 2 * positive_differences.bit_count() - 1


def census(deadline):
    """Return every per-size nondominated (sum count, difference count) set."""
    buckets = {n: {} for n in range(6, 19)}
    fixed = (1 << 0) | (1 << DOMAIN_END)
    for interior in range(1 << (DOMAIN_END - 1)):
        if (interior & 8191) == 0 and time.monotonic() >= deadline:
            break
        n = interior.bit_count() + 2
        if not 6 <= n <= 18:
            continue
        mask = fixed | (interior << 1)
        values = tuple(i for i in range(DOMAIN_END + 1) if mask >> i & 1)
        sums, diffs = bit_counts(mask, values)
        buckets[n].setdefault((sums, diffs), values)

    frontier = []
    for n, entries in buckets.items():
        # Increasing differences: retain precisely the records in sum count.
        best_sums = -1
        for (sums, diffs), values in sorted(entries.items(), key=lambda item: (item[0][1], -item[0][0])):
            if sums > best_sums:
                frontier.append((values, sums, diffs))
                best_sums = sums
    return frontier


def product(left, right):
    radix = 2 * (max(left) - min(left)) + 1
    return tuple(sorted(a + radix * b for b in right for a in left))


def main():
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parent_solution.json")
    with open(parent_path) as stream:
        best = tuple(sorted(set(json.load(stream)["A"])))
    best_sums, best_diffs = counts(best)
    best_score = quality(len(best), best_sums, best_diffs)
    write_solution(output, best)

    deadline = time.monotonic() + SEARCH_SECONDS
    frontier = census(deadline)

    # Include all useful single-coordinate deletions as possible product factors.
    factors = {}
    for values, sums, diffs in frontier:
        factors.setdefault(values, (sums, diffs))
        for index in range(len(values)):
            reduced = values[:index] + values[index + 1:]
            if len(reduced) >= 2:
                factors.setdefault(reduced, counts(reduced))

    ranked = []
    for values, (sums, diffs) in factors.items():
        score = quality(len(values), sums, diffs)
        ranked.append((score, values, sums, diffs))
        if score > best_score:
            best, best_sums, best_diffs, best_score = values, sums, diffs, score
    ranked.sort(reverse=True, key=lambda item: item[0])
    # Pareto records contain redundancy; this deterministic cap reserves time for
    # exact deletion tests while retaining the strongest and smallest factors.
    chosen = ranked[:160]
    chosen += sorted(ranked[160:], key=lambda item: (len(item[1]), -item[0]))[:96]

    products = {}
    for _, left, left_sums, left_diffs in chosen:
        for _, right, right_sums, right_diffs in chosen:
            n = len(left) * len(right)
            if n > MAX_SIZE:
                continue
            candidate = product(left, right)
            key = tuple(candidate)
            sums = left_sums * right_sums
            diffs = left_diffs * right_diffs
            score = quality(n, sums, diffs)
            previous = products.get(n)
            if previous is None or score > previous[0]:
                products[n] = (score, key, sums, diffs)
            if score > best_score:
                best, best_sums, best_diffs, best_score = key, sums, diffs, score

    # Add further coordinates (for example 6*6*6) until no cardinality's best
    # product improves.  One representative per size prevents equivalent
    # products from blooming and reserves most of the budget for deletion tests.
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        current_products = list(products.values())
        for _, left, left_sums, left_diffs in current_products:
            for _, right, right_sums, right_diffs in chosen:
                n = len(left) * len(right)
                if n > MAX_SIZE:
                    continue
                sums = left_sums * right_sums
                diffs = left_diffs * right_diffs
                score = quality(n, sums, diffs)
                previous = products.get(n)
                if previous is None or score > previous[0] + 1e-15:
                    candidate = product(left, right)
                    products[n] = (score, candidate, sums, diffs)
                    changed = True
                    if score > best_score:
                        best, best_sums, best_diffs, best_score = candidate, sums, diffs, score

    # Exact one-point deletion tests on the best product at every attainable size.
    for _, candidate, _, _ in sorted(products.values(), reverse=True):
        if time.monotonic() >= deadline:
            break
        for index in range(len(candidate)):
            if (index & 7) == 0 and time.monotonic() >= deadline:
                break
            reduced = candidate[:index] + candidate[index + 1:]
            sums, diffs = counts(reduced)
            score = quality(len(reduced), sums, diffs)
            if score > best_score:
                best, best_sums, best_diffs, best_score = reduced, sums, diffs, score
                write_solution(output, best)

    write_solution(output, best)
    print(f"census/product search complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
