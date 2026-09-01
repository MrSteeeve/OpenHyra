"""Simulated annealing search for a high sum-vs-difference exponent."""

import json
import math
import os
import random
import time


INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
N_CHOICES = [16, 17, 18, 19, 20, 24, 32]
RESTARTS = 64
STEPS_PER_RESTART = 20_000
TIME_LIMIT = 168.0


def objective(n, sum_count, diff_count):
    return math.log(sum_count / n) / math.log(diff_count / n)


class PairCounts:
    """Maintain exact distinct sum and difference counts under local edits."""

    def __init__(self, values):
        self.values = set(values)
        self.sums = {}
        self.diffs = {}
        ordered = sorted(self.values)
        for i, x in enumerate(ordered):
            self._bump(self.sums, 2 * x, 1)
            for y in ordered[i + 1 :]:
                self._bump(self.sums, x + y, 1)
                self._bump(self.diffs, y - x, 1)

    @staticmethod
    def _bump(table, key, amount):
        updated = table.get(key, 0) + amount
        if updated:
            table[key] = updated
        else:
            del table[key]

    def add(self, x):
        for y in self.values:
            self._bump(self.sums, x + y, 1)
            self._bump(self.diffs, abs(x - y), 1)
        self._bump(self.sums, 2 * x, 1)
        self.values.add(x)

    def remove(self, x):
        self.values.remove(x)
        self._bump(self.sums, 2 * x, -1)
        for y in self.values:
            self._bump(self.sums, x + y, -1)
            self._bump(self.diffs, abs(x - y), -1)

    def replace(self, old, new):
        self.remove(old)
        self.add(new)

    def score(self):
        # Positive gaps account for both signs; zero occurs exactly once.
        return objective(len(self.values), len(self.sums), 2 * len(self.diffs) + 1)


def write_solution(path, values):
    normalized = sorted(x - min(values) for x in values)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump({"A": normalized}, stream)
    os.replace(temporary, path)


def initial_candidate(target, rng):
    values = set(INITIAL_SET)
    while len(values) > target:
        values.remove(rng.choice(sorted(values)))
    while len(values) < target:
        low, high = min(values), max(values)
        candidate = rng.randint(low - 3, high + 3)
        if candidate not in values:
            values.add(candidate)
    return values


def proposal(state, target, rng):
    """Return (kind, old, new), or None when a sampled edit is invalid."""
    size = len(state.values)
    lower = max(2, target - 2)
    upper = min(512, target + 2)
    draw = rng.random()

    if draw < 0.42:  # one-element replacement
        old = rng.choice(tuple(state.values))
        low, high = min(state.values), max(state.values)
        new = rng.randint(low - 6, high + 6)
        kind = "replace"
    elif draw < 0.70:  # +/-1 or +/-2 shift
        old = rng.choice(tuple(state.values))
        new = old + rng.choice((-2, -1, 1, 2))
        kind = "replace"
    elif draw < 0.85 and size < upper:  # insertion
        low, high = min(state.values), max(state.values)
        old = None
        new = rng.randint(low - 5, high + 5)
        kind = "add"
    elif size > lower:  # deletion
        old = rng.choice(tuple(state.values))
        new = None
        kind = "remove"
    else:
        return None

    if new is not None and new in state.values:
        return None
    prospective = (state.values - ({old} if old is not None else set()))
    if new is not None:
        prospective.add(new)
    if max(prospective) - min(prospective) > max(48, 4 * target):
        return None
    return kind, old, new


def apply_edit(state, edit):
    kind, old, new = edit
    if kind == "replace":
        state.replace(old, new)
    elif kind == "add":
        state.add(new)
    else:
        state.remove(old)


def undo_edit(state, edit):
    kind, old, new = edit
    if kind == "replace":
        state.replace(new, old)
    elif kind == "add":
        state.remove(new)
    else:
        state.add(old)


def main():
    started = time.monotonic()
    deadline = started + TIME_LIMIT
    rng = random.Random(1)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")

    best_values = set(INITIAL_SET)
    best_score = PairCounts(best_values).score()
    write_solution(output_path, best_values)

    for restart in range(RESTARTS):
        if time.monotonic() >= deadline:
            break
        target = N_CHOICES[restart % len(N_CHOICES)]
        state = PairCounts(initial_candidate(target, rng))
        current = state.score()
        if current > best_score:
            best_score, best_values = current, set(state.values)

        for step in range(STEPS_PER_RESTART):
            if (step & 255) == 0 and time.monotonic() >= deadline:
                write_solution(output_path, best_values)
                print(f"annealing stopped safely: n={len(best_values)} score={best_score:.9f}")
                return
            edit = proposal(state, target, rng)
            if edit is None:
                continue
            apply_edit(state, edit)
            candidate = state.score()
            fraction = step / STEPS_PER_RESTART
            temperature = 0.018 * (0.00025 / 0.018) ** fraction
            if candidate >= current or rng.random() < math.exp((candidate - current) / temperature):
                current = candidate
                if candidate > best_score:
                    best_score, best_values = candidate, set(state.values)
            else:
                undo_edit(state, edit)

    write_solution(output_path, best_values)
    print(f"annealing complete: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
