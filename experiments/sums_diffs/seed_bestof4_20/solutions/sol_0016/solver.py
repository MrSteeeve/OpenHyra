"""Deterministic beam search for coordinated periodic fringe sequences."""

import json
import math
import os
import time


SEED = 20260797782211
SEARCH_SECONDS = 164.0
BEAM = 20000
WIDTHS = (4, 5, 6, 7, 8)
DEPTH = 14


def score_and_signature(a):
    """Return the exact score and translation-invariant bitset signature."""
    a = sorted(a)
    lo = a[0]
    a = [x - lo for x in a]
    bits = 0
    for x in a:
        bits |= 1 << x
    sums = 0
    diffs = 0
    for x in a:
        sums |= bits << x
        diffs |= bits >> x
    ns = sums.bit_count()
    nd = 2 * diffs.bit_count() - 1
    n = len(a)
    return math.log(ns / n) / math.log(nd / n), (sums, diffs, n)


def materialize(width, rows, core, left, right):
    masks = [core] * rows
    for i, mask in enumerate(left):
        masks[i] = mask
    for i, mask in enumerate(right):
        masks[rows - 1 - i] = mask
    return [r * width + b for r, mask in enumerate(masks)
            for b in range(width) if mask >> b & 1]


def mask_pool(width, core):
    """Order useful masks deterministically; all nonempty masks remain reachable."""
    all_masks = list(range(1, 1 << width))
    def rank(mask):
        distance = (mask ^ core).bit_count()
        edge = -int(mask & 1 != 0) - int(mask >> (width - 1) & 1 != 0)
        # SEED is used only as a stable tie breaker, not as hidden randomness.
        tie = (mask * 1103515245 + SEED) & 0xffff
        return distance, edge, tie
    all_masks.sort(key=rank)
    return all_masks


def save(path, a):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": sorted(a)}, stream, separators=(",", ":"))
    os.replace(temporary, path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(here, "solution.json")
    with open(os.path.join(here, "parent_solution.json")) as stream:
        best = sorted(set(json.load(stream)["A"]))
    best_score, _ = score_and_signature(best)
    save(output, best)

    deadline = time.monotonic() + SEARCH_SECONDS
    # Two nearby densities per width give distinct periodic cores.  The row
    # count is the longest in the requested 40..120 range that respects n<=512.
    jobs = []
    for width in WIDTHS:
        for count in sorted(set((max(1, width // 2), min(width, width // 2 + 1)))):
            core = (1 << count) - 1
            rows = min(120, 512 // count)
            if rows >= 40:
                jobs.append((width, rows, core))

    for job_index, (width, rows, core) in enumerate(jobs):
        if time.monotonic() >= deadline:
            break
        pool = mask_pool(width, core)
        # A state is (left fringe, right fringe, exact score).  Alternating
        # sides avoids the quadratic Cartesian join while still coordinating
        # both signatures at every layer.
        base = materialize(width, rows, core, (), ())
        base_score, _ = score_and_signature(base)
        beam = [((), (), base_score)]
        for layer in range(2 * DEPTH):
            now = time.monotonic()
            if now >= deadline - 0.4:
                break
            # Broad early branching fills the beam; once full, use the closest
            # masks so an iteration has a predictable upper cost.
            branch = min(len(pool), 32 if len(beam) < 500 else 8)
            candidates = {}
            extend_left = layer % 2 == 0
            stopped = False
            for state_index, (left, right, unused) in enumerate(beam):
                if state_index % 64 == 0 and time.monotonic() >= deadline - 1.0:
                    stopped = True
                    break
                for mask in pool[:branch]:
                    nl = left + (mask,) if extend_left else left
                    nr = right if extend_left else right + (mask,)
                    a = materialize(width, rows, core, nl, nr)
                    if not (2 <= len(a) <= 512):
                        continue
                    value, signature = score_and_signature(a)
                    old = candidates.get(signature)
                    if old is None or value > old[2]:
                        candidates[signature] = (nl, nr, value)
                    if value > best_score:
                        best, best_score = a, value
            if not candidates:
                break
            beam = sorted(candidates.values(), key=lambda x: x[2], reverse=True)[:BEAM]
            # Preserve output between expensive layers and divide time fairly.
            save(output, best)
            if stopped or time.monotonic() + (time.monotonic() - now) > deadline:
                break

    save(output, best)
    print(f"row-signature beam complete: n={len(best)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
