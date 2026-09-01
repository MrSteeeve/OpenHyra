"""Anneal residue masks for periodic lifts, with the tensor best as fallback."""

import json
import math
import os
import random
import time


BASE_SET = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
TENSOR_BASE = 67
REMOVED = {204, 214, 222, 232, 874, 902, 1410, 1438, 2080, 2090, 2098, 2108}
INCUMBENT = tuple(
    x
    for x in (a + TENSOR_BASE * b for b in BASE_SET for a in BASE_SET)
    if x not in REMOVED
)

MODULI = (31, 47, 63, 79, 95, 127)
HEIGHTS = (2, 3, 4, 6, 8)
SEED = 4019
SEARCH_SECONDS = 155.0
MIN_SIZE = 32
MAX_SIZE = 512
COOLING_STEPS = 12000
HOT_TEMPERATURE = 0.012
COLD_TEMPERATURE = 0.000015


def score_from_counts(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def exact_score(values):
    n = len(values)
    sums = len({x + y for x in values for y in values})
    diffs = len({x - y for x in values for y in values})
    return score_from_counts(n, sums, diffs)


class LiftState:
    """Residue mask with exact incremental sum/difference carry counts."""

    def __init__(self, modulus, height, residues):
        self.modulus = modulus
        self.height = height
        self.chosen = [False] * modulus
        self.selected = []
        self.position = [-1] * modulus
        self.sum_counts = [[0] * modulus for _ in range(2)]
        self.diff_counts = [[0] * modulus for _ in range(2)]
        self.score = 0.0
        for residue in residues:
            self.add(residue)
        self.recompute_score()

    def add(self, residue):
        modulus = self.modulus
        for other in self.selected:
            carry, slot = divmod(residue + other, modulus)
            self.sum_counts[carry][slot] += 2

            carry, slot = divmod(residue - other, modulus)
            self.diff_counts[carry + 1][slot] += 1
            carry, slot = divmod(other - residue, modulus)
            self.diff_counts[carry + 1][slot] += 1

        carry, slot = divmod(2 * residue, modulus)
        self.sum_counts[carry][slot] += 1
        self.diff_counts[1][0] += 1

        self.chosen[residue] = True
        self.position[residue] = len(self.selected)
        self.selected.append(residue)

    def remove(self, residue):
        modulus = self.modulus
        for other in self.selected:
            if other == residue:
                continue
            carry, slot = divmod(residue + other, modulus)
            self.sum_counts[carry][slot] -= 2

            carry, slot = divmod(residue - other, modulus)
            self.diff_counts[carry + 1][slot] -= 1
            carry, slot = divmod(other - residue, modulus)
            self.diff_counts[carry + 1][slot] -= 1

        carry, slot = divmod(2 * residue, modulus)
        self.sum_counts[carry][slot] -= 1
        self.diff_counts[1][0] -= 1

        position = self.position[residue]
        last = self.selected.pop()
        if last != residue:
            self.selected[position] = last
            self.position[last] = position
        self.position[residue] = -1
        self.chosen[residue] = False

    def toggle(self, residue):
        if self.chosen[residue]:
            self.remove(residue)
        else:
            self.add(residue)

    def recompute_score(self):
        width = 2 * self.height - 1
        sums = 0
        diffs = 0
        for slot in range(self.modulus):
            sum_low = self.sum_counts[0][slot] > 0
            sum_high = self.sum_counts[1][slot] > 0
            if sum_low or sum_high:
                sums += width + (sum_low and sum_high)

            diff_low = self.diff_counts[0][slot] > 0
            diff_high = self.diff_counts[1][slot] > 0
            if diff_low or diff_high:
                diffs += width + (diff_low and diff_high)

        n = len(self.selected) * self.height
        self.score = score_from_counts(n, sums, diffs)

    def mask(self):
        return tuple(sorted(self.selected))

    def values(self):
        return tuple(
            residue + self.modulus * level
            for level in range(self.height)
            for residue in sorted(self.selected)
        )


def random_residues(rng, modulus, height, density=None):
    minimum = (MIN_SIZE + height - 1) // height
    maximum = min(modulus, MAX_SIZE // height)
    if density is None:
        count = rng.randint(minimum, maximum)
    else:
        count = round(density * modulus)
        count = max(minimum, min(maximum, count))
    return rng.sample(range(modulus), count)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    best_values = INCUMBENT
    best_score = exact_score(best_values)
    write_solution(best_values)

    chains = []
    densities = (0.24, 0.38, 0.52, 0.66, 0.80)
    for modulus in MODULI:
        for height_index, height in enumerate(HEIGHTS):
            residues = random_residues(
                rng, modulus, height, densities[height_index]
            )
            state = LiftState(modulus, height, residues)
            chains.append(
                {
                    "state": state,
                    "visits": rng.randrange(COOLING_STEPS),
                    "best_score": state.score,
                    "best_mask": state.mask(),
                    "restarts": 0,
                }
            )

    deadline = time.monotonic() + SEARCH_SECONDS
    chain_index = 0
    while time.monotonic() < deadline:
        chain = chains[chain_index]
        chain_index = (chain_index + 1) % len(chains)
        state = chain["state"]

        phase = chain["visits"] % COOLING_STEPS
        temperature = HOT_TEMPERATURE * (
            COLD_TEMPERATURE / HOT_TEMPERATURE
        ) ** (phase / (COOLING_STEPS - 1))

        draw = rng.random()
        if draw < 0.72:
            toggles = [rng.randrange(state.modulus)]
        elif draw < 0.92:
            toggles = rng.sample(range(state.modulus), 2)
        else:
            absent = [r for r in range(state.modulus) if not state.chosen[r]]
            if absent and state.selected:
                toggles = [rng.choice(state.selected), rng.choice(absent)]
            else:
                toggles = [rng.randrange(state.modulus)]

        new_count = len(state.selected) + sum(
            -1 if state.chosen[residue] else 1 for residue in toggles
        )
        new_size = new_count * state.height
        if MIN_SIZE <= new_size <= MAX_SIZE:
            old_score = state.score
            for residue in toggles:
                state.toggle(residue)
            state.recompute_score()
            delta = state.score - old_score

            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                if state.score > chain["best_score"]:
                    chain["best_score"] = state.score
                    chain["best_mask"] = state.mask()
                if state.score > best_score:
                    candidate = state.values()
                    candidate_score = exact_score(candidate)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_values = candidate
                        write_solution(best_values)
            else:
                for residue in reversed(toggles):
                    state.toggle(residue)
                state.score = old_score

        chain["visits"] += 1
        if chain["visits"] % COOLING_STEPS == 0:
            chain["restarts"] += 1
            if chain["restarts"] % 3:
                residues = list(chain["best_mask"])
                perturbations = max(2, state.modulus // 12)
                for _ in range(perturbations):
                    residue = rng.randrange(state.modulus)
                    if residue in residues:
                        residues.remove(residue)
                    else:
                        residues.append(residue)
                minimum = (MIN_SIZE + state.height - 1) // state.height
                maximum = min(state.modulus, MAX_SIZE // state.height)
                if not minimum <= len(residues) <= maximum:
                    residues = random_residues(rng, state.modulus, state.height)
            else:
                residues = random_residues(rng, state.modulus, state.height)
            chain["state"] = LiftState(state.modulus, state.height, residues)

    write_solution(best_values)
    print(f"wrote periodic-lift best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
