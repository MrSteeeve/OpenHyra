"""Parallel-tempering search over subsets of a carry-free tensor square."""

import json
import math
import os
import random
import time


BASE_SET = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
TENSOR_BASE = 67
POINTS = tuple(a + TENSOR_BASE * b for b in BASE_SET for a in BASE_SET)
SEED = 4011
SEARCH_SECONDS = 155.0
REPLICAS = 24
MIN_SIZE = 64
MIN_TEMPERATURE = 1e-5
MAX_TEMPERATURE = 0.02
SWAP_INTERVAL = 200
DIFF_OFFSET = POINTS[-1]
SUM_SLOTS = 2 * POINTS[-1] + 1
DIFF_SLOTS = 2 * POINTS[-1] + 1


def objective(n, sum_bits, diff_bits):
    sums = sum_bits.bit_count()
    diffs = diff_bits.bit_count()
    return math.log(sums / n) / math.log(diffs / n)


class State:
    """An exact subset score maintained under point toggles."""

    def __init__(self):
        count = len(POINTS)
        self.chosen = [True] * count
        self.selected = list(range(count))
        self.position = list(range(count))
        self.sum_counts = [0] * SUM_SLOTS
        self.diff_counts = [0] * DIFF_SLOTS
        self.sum_bits = 0
        self.diff_bits = 0

        for i, x in enumerate(POINTS):
            for j in range(i, count):
                slot = x + POINTS[j]
                self.sum_counts[slot] += 1
                self.sum_bits |= 1 << slot
            for y in POINTS:
                slot = x - y + DIFF_OFFSET
                self.diff_counts[slot] += 1
                self.diff_bits |= 1 << slot
        self.score = objective(count, self.sum_bits, self.diff_bits)

    def clone(self):
        other = object.__new__(State)
        other.chosen = self.chosen.copy()
        other.selected = self.selected.copy()
        other.position = self.position.copy()
        other.sum_counts = self.sum_counts.copy()
        other.diff_counts = self.diff_counts.copy()
        other.sum_bits = self.sum_bits
        other.diff_bits = self.diff_bits
        other.score = self.score
        return other

    def _increment_sum(self, slot):
        if self.sum_counts[slot] == 0:
            self.sum_bits |= 1 << slot
        self.sum_counts[slot] += 1

    def _decrement_sum(self, slot):
        self.sum_counts[slot] -= 1
        if self.sum_counts[slot] == 0:
            self.sum_bits &= ~(1 << slot)

    def _increment_diff(self, slot):
        if self.diff_counts[slot] == 0:
            self.diff_bits |= 1 << slot
        self.diff_counts[slot] += 1

    def _decrement_diff(self, slot):
        self.diff_counts[slot] -= 1
        if self.diff_counts[slot] == 0:
            self.diff_bits &= ~(1 << slot)

    def add(self, index):
        x = POINTS[index]
        for other in self.selected:
            y = POINTS[other]
            self._increment_sum(x + y)
            self._increment_diff(x - y + DIFF_OFFSET)
            self._increment_diff(y - x + DIFF_OFFSET)
        self._increment_sum(2 * x)
        self._increment_diff(DIFF_OFFSET)

        self.chosen[index] = True
        self.position[index] = len(self.selected)
        self.selected.append(index)

    def remove(self, index):
        x = POINTS[index]
        for other in self.selected:
            if other == index:
                continue
            y = POINTS[other]
            self._decrement_sum(x + y)
            self._decrement_diff(x - y + DIFF_OFFSET)
            self._decrement_diff(y - x + DIFF_OFFSET)
        self._decrement_sum(2 * x)
        self._decrement_diff(DIFF_OFFSET)

        position = self.position[index]
        last = self.selected.pop()
        if last != index:
            self.selected[position] = last
            self.position[last] = position
        self.position[index] = -1
        self.chosen[index] = False

    def toggle(self, index):
        if self.chosen[index]:
            self.remove(index)
        else:
            self.add(index)

    def recompute_score(self):
        self.score = objective(len(self.selected), self.sum_bits, self.diff_bits)

    def values(self):
        return tuple(POINTS[index] for index in sorted(self.selected))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    initial = State()
    states = [initial] + [initial.clone() for _ in range(REPLICAS - 1)]
    temperatures = [
        MIN_TEMPERATURE
        * (MAX_TEMPERATURE / MIN_TEMPERATURE) ** (i / (REPLICAS - 1))
        for i in range(REPLICAS)
    ]

    best_score = initial.score
    best_values = initial.values()
    write_solution(best_values)

    deadline = time.monotonic() + SEARCH_SECONDS
    sweep = 0
    swap_parity = 0
    point_range = range(len(POINTS))

    while time.monotonic() < deadline:
        for replica, temperature in enumerate(temperatures):
            if time.monotonic() >= deadline:
                break
            state = states[replica]
            toggles = rng.sample(point_range, rng.randint(1, 4))
            new_size = len(state.selected) + sum(
                -1 if state.chosen[index] else 1 for index in toggles
            )
            if new_size < MIN_SIZE:
                continue

            old_score = state.score
            for index in toggles:
                state.toggle(index)
            state.recompute_score()
            delta = state.score - old_score

            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                if state.score > best_score:
                    best_score = state.score
                    best_values = state.values()
                    write_solution(best_values)
            else:
                for index in reversed(toggles):
                    state.toggle(index)
                state.score = old_score

        sweep += 1
        if sweep % SWAP_INTERVAL == 0:
            for left in range(swap_parity, REPLICAS - 1, 2):
                right = left + 1
                exponent = (states[right].score - states[left].score) * (
                    1.0 / temperatures[left] - 1.0 / temperatures[right]
                )
                if exponent >= 0.0 or rng.random() < math.exp(exponent):
                    states[left], states[right] = states[right], states[left]
            swap_parity ^= 1

    write_solution(best_values)
    print(f"wrote tensor-subset best: n={len(best_values)} score={best_score:.9f}")


if __name__ == "__main__":
    main()
