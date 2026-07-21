"""Seed solver: classic MSTD (more-sums-than-differences) base set, expanded
carry-free to a larger set. Writes solution.json with the resulting set.

C(A) is invariant under plain base expansion (both logs scale by k), so this
seed only reproduces the base set's exponent — improving beyond it requires
genuinely better constructions or search.
"""

import json

# Conway's MSTD set: |A+A| = 26 > |A-A| = 25
BASE_SET = [0, 2, 3, 4, 7, 11, 12, 14]
BASE = 29     # > 2*max(BASE_SET), so digit sums never carry
LEVELS = 3    # n = 8^3 = 512 elements, max element well within limits


def expand(base_set, base, levels):
    out = [0]
    for level in range(levels):
        scale = base ** level
        out = [x + d * scale for x in out for d in base_set]
    return sorted(set(out))


def main():
    A = expand(BASE_SET, BASE, LEVELS)
    with open("solution.json", "w") as f:
        json.dump({"A": A}, f)
    print(f"wrote solution.json: n={len(A)}, max={max(A)}")


if __name__ == "__main__":
    main()
