"""Evolve three-level conditional-digit sets with controlled carries."""

import json
import math
import os
import random
import time


SEED = 20260793782199
SEARCH_SECONDS = 170.0
RADICES = (12, 16, 20, 24)
POP_PER_ISLAND = 12
ELITES_PER_ISLAND = 2
Z_CAPS = {12: 12, 16: 10, 20: 8, 24: 4}


def decode(individual):
    q, zmask, y_masks, x_masks = individual
    result = []
    for z, ymask in enumerate(y_masks):
        if not (zmask >> z) & 1:
            continue
        for y in range(q):
            if (ymask >> y) & 1:
                base = q * y + q * q * z
                xmask = x_masks[y]
                result.extend(base + x for x in range(q) if (xmask >> x) & 1)
    return frozenset(result)


def score_values(candidate):
    """Exact cardinalities using shifts of one arbitrary-precision bitset."""
    values = tuple(candidate)
    n = len(values)
    bits = sum(1 << value for value in values)
    sums = 0
    diffs = 0
    offset = max(values)
    for value in values:
        sums |= bits << value
        diffs |= bits << (offset - value)
    sum_count = sums.bit_count()
    diff_count = diffs.bit_count()
    return math.log(sum_count / n) / math.log(diff_count / n)


def write_solution(path, candidate):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(candidate)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def viable(individual):
    q, zmask, y_masks, x_masks = individual
    if not zmask or len(x_masks) != q or len(y_masks) != Z_CAPS[q]:
        return False
    if any(mask == 0 or mask >= 1 << q for mask in x_masks):
        return False
    if any(mask == 0 or mask >= 1 << q for mask in y_masks):
        return False
    size = 0
    x_sizes = tuple(mask.bit_count() for mask in x_masks)
    for z, ymask in enumerate(y_masks):
        if (zmask >> z) & 1:
            size += sum(x_sizes[y] for y in range(q) if (ymask >> y) & 1)
    return 128 <= size <= 512


def random_mask(q, density, rng):
    mask = 0
    for bit in range(q):
        if rng.random() < density:
            mask |= 1 << bit
    return mask or 1 << rng.randrange(q)


def random_individual(q, rng):
    zcap = Z_CAPS[q]
    for _ in range(1000):
        # Correlated rows seed useful structure while row-specific noise makes
        # the construction genuinely conditional and nonperiodic.
        x_density = rng.uniform(0.38, 0.72)
        y_density = rng.uniform(0.32, 0.68)
        x_template = random_mask(q, x_density, rng)
        y_template = random_mask(q, y_density, rng)
        x_masks = []
        for _ in range(q):
            mask = x_template
            for _ in range(rng.randrange(1, 4)):
                mask ^= 1 << rng.randrange(q)
            x_masks.append(mask or x_template)
        y_masks = []
        for _ in range(zcap):
            mask = y_template
            for _ in range(rng.randrange(1, 5)):
                mask ^= 1 << rng.randrange(q)
            y_masks.append(mask or y_template)
        zmask = random_mask(zcap, rng.uniform(0.55, 1.0), rng)
        individual = (q, zmask, tuple(y_masks), tuple(x_masks))
        if viable(individual):
            return individual
    # This dense deterministic shape is only an initialization backstop.
    x_masks = tuple((1 << max(2, q // 2)) - 1 for _ in range(q))
    y_masks = tuple((1 << max(2, q // 2)) - 1 for _ in range(zcap))
    z_needed = max(1, min(zcap, 256 // ((q // 2) ** 2)))
    return (q, (1 << z_needed) - 1, y_masks, x_masks)


def crossover(left, right, rng):
    q = left[0]
    zmask = left[1] if rng.random() < 0.5 else right[1]
    y_masks = tuple(
        left[2][i] if rng.random() < 0.5 else right[2][i]
        for i in range(len(left[2]))
    )
    x_masks = tuple(
        left[3][i] if rng.random() < 0.5 else right[3][i]
        for i in range(q)
    )
    return (q, zmask, y_masks, x_masks)


def mutate(individual, rng):
    q, zmask0, y0, x0 = individual
    for _ in range(24):
        zmask, y_masks, x_masks = zmask0, list(y0), list(x0)
        moves = 1 if rng.random() < 0.72 else rng.randint(2, 5)
        for _ in range(moves):
            move = rng.random()
            if move < 0.08:
                zmask ^= 1 << rng.randrange(len(y_masks))
                zmask = zmask or zmask0
            elif move < 0.50:
                row = rng.randrange(len(y_masks))
                y_masks[row] ^= 1 << rng.randrange(q)
                y_masks[row] = y_masks[row] or y0[row]
            elif move < 0.88:
                row = rng.randrange(q)
                x_masks[row] ^= 1 << rng.randrange(q)
                x_masks[row] = x_masks[row] or x0[row]
            elif move < 0.94:
                source, target = rng.sample(range(len(y_masks)), 2)
                y_masks[target] = y_masks[source]
            else:
                source, target = rng.sample(range(q), 2)
                x_masks[target] = x_masks[source]
        trial = (q, zmask, tuple(y_masks), tuple(x_masks))
        if viable(trial):
            return trial
    return individual


def main():
    rng = random.Random(SEED)
    directory = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(directory, "solution.json")
    with open(os.path.join(directory, "parent_solution.json")) as stream:
        best = frozenset(json.load(stream)["A"])
    best_score = score_values(best)
    write_solution(output, best)

    cache = {}

    def evaluate(individual):
        known = cache.get(individual)
        if known is None:
            values = decode(individual)
            known = (score_values(values), values)
            cache[individual] = known
        return known

    islands = {
        q: [random_individual(q, rng) for _ in range(POP_PER_ISLAND)]
        for q in RADICES
    }
    deadline = time.monotonic() + SEARCH_SECONDS
    generation = 0
    while time.monotonic() < deadline:
        for q in RADICES:
            ranked = sorted(islands[q], key=lambda item: evaluate(item)[0], reverse=True)
            for individual in ranked[:ELITES_PER_ISLAND]:
                value, candidate = evaluate(individual)
                if value > best_score:
                    best_score, best = value, candidate
                    write_solution(output, best)
            next_population = ranked[:ELITES_PER_ISLAND]
            pool = ranked[:8]
            while len(next_population) < POP_PER_ISLAND:
                # Tournament selection, uniform mask crossover, then the
                # prescribed bit-flip/row-copy mutation operators.
                left = max(rng.sample(pool, 3), key=lambda item: evaluate(item)[0])
                right = max(rng.sample(pool, 3), key=lambda item: evaluate(item)[0])
                child = crossover(left, right, rng)
                child = mutate(child, rng)
                if viable(child):
                    next_population.append(child)
            islands[q] = next_population
            if time.monotonic() >= deadline:
                break
        generation += 1
        if generation % 10 == 0:
            write_solution(output, best)

    write_solution(output, best)
    print(f"three-level evolution complete: generations={generation} n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
