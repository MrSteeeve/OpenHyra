"""Compress the sol_0012 tensor subset with an exact deletion beam."""

import bisect
import heapq
import json
import math
import os
import random
import time


BASE_SET = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
TENSOR_BASE = 67
ALL_TENSOR_POINTS = tuple(
    a + TENSOR_BASE * b for b in BASE_SET for a in BASE_SET
)
MISSING_INDICES = frozenset(
    (36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252)
)
PARENT = tuple(
    value for index, value in enumerate(ALL_TENSOR_POINTS)
    if index not in MISSING_INDICES
)

SEED = 4016
SEARCH_SECONDS = 165.0
BEAM_WIDTH = 16
MIN_SIZE = 32
MAX_COORDINATE = 2244
OFFSET = MAX_COORDINATE
SLOTS = 2 * MAX_COORDINATE + 1
PARENT_INDEX = {value: index for index, value in enumerate(PARENT)}


def objective(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


class State:
    """A set with exact unordered-sum and ordered-difference multiplicities."""

    def __init__(self, values):
        self.selected = sorted(values)
        self.sum_counts = [0] * SLOTS
        self.diff_counts = [0] * SLOTS
        self.sum_size = 0
        self.diff_size = 0

        for position, x in enumerate(self.selected):
            for y in self.selected[position:]:
                slot = x + y
                if self.sum_counts[slot] == 0:
                    self.sum_size += 1
                self.sum_counts[slot] += 1
            for y in self.selected:
                slot = x - y + OFFSET
                if self.diff_counts[slot] == 0:
                    self.diff_size += 1
                self.diff_counts[slot] += 1

        self.score = objective(len(self.selected), self.sum_size, self.diff_size)
        mask = 0
        for value in self.selected:
            mask |= 1 << PARENT_INDEX[value]
        self.mask = mask

    def clone(self):
        other = object.__new__(State)
        other.selected = self.selected.copy()
        other.sum_counts = self.sum_counts.copy()
        other.diff_counts = self.diff_counts.copy()
        other.sum_size = self.sum_size
        other.diff_size = self.diff_size
        other.score = self.score
        other.mask = self.mask
        return other

    def deletion_score(self, position):
        """Score after deleting one value, without changing this state."""
        values = self.selected
        x = values[position]

        lost_sums = 0
        for y in values:
            if self.sum_counts[x + y] == 1:
                lost_sums += 1

        # A nonzero difference involving x is removed once, or twice when
        # equally distant selected values flank x.  Counts are symmetric, so
        # it suffices to inspect positive distances and double the result.
        lost_positive_diffs = 0
        left = position - 1
        right = position + 1
        infinity = MAX_COORDINATE + 1
        while left >= 0 or right < len(values):
            left_distance = x - values[left] if left >= 0 else infinity
            right_distance = values[right] - x if right < len(values) else infinity
            if left_distance == right_distance:
                distance = left_distance
                multiplicity = 2
                left -= 1
                right += 1
            elif left_distance < right_distance:
                distance = left_distance
                multiplicity = 1
                left -= 1
            else:
                distance = right_distance
                multiplicity = 1
                right += 1
            if self.diff_counts[OFFSET + distance] == multiplicity:
                lost_positive_diffs += 1

        n = len(values) - 1
        sums = self.sum_size - lost_sums
        diffs = self.diff_size - 2 * lost_positive_diffs
        return objective(n, sums, diffs)

    def remove_at(self, position):
        values = self.selected
        x = values[position]
        for other_position, y in enumerate(values):
            sum_slot = x + y
            self.sum_counts[sum_slot] -= 1
            if self.sum_counts[sum_slot] == 0:
                self.sum_size -= 1

            if other_position != position:
                positive = x - y + OFFSET
                negative = y - x + OFFSET
                self.diff_counts[positive] -= 1
                if self.diff_counts[positive] == 0:
                    self.diff_size -= 1
                self.diff_counts[negative] -= 1
                if self.diff_counts[negative] == 0:
                    self.diff_size -= 1

        self.diff_counts[OFFSET] -= 1
        if self.diff_counts[OFFSET] == 0:
            self.diff_size -= 1
        values.pop(position)
        parent_index = PARENT_INDEX.get(x)
        if parent_index is not None:
            self.mask &= ~(1 << parent_index)
        self.score = objective(len(values), self.sum_size, self.diff_size)
        return x

    def addition_score(self, x):
        """Score after adding absent coordinate x, without changing this state."""
        new_sums = int(self.sum_counts[2 * x] == 0)
        for y in self.selected:
            if self.sum_counts[x + y] == 0:
                new_sums += 1

        distances = {abs(x - y) for y in self.selected}
        new_diffs = 2 * sum(
            self.diff_counts[OFFSET + distance] == 0 for distance in distances
        )
        return objective(
            len(self.selected) + 1,
            self.sum_size + new_sums,
            self.diff_size + new_diffs,
        )

    def add(self, x):
        for y in self.selected:
            sum_slot = x + y
            if self.sum_counts[sum_slot] == 0:
                self.sum_size += 1
            self.sum_counts[sum_slot] += 1

            positive = x - y + OFFSET
            negative = y - x + OFFSET
            if self.diff_counts[positive] == 0:
                self.diff_size += 1
            self.diff_counts[positive] += 1
            if self.diff_counts[negative] == 0:
                self.diff_size += 1
            self.diff_counts[negative] += 1

        if self.sum_counts[2 * x] == 0:
            self.sum_size += 1
        self.sum_counts[2 * x] += 1
        if self.diff_counts[OFFSET] == 0:
            self.diff_size += 1
        self.diff_counts[OFFSET] += 1
        bisect.insort(self.selected, x)
        parent_index = PARENT_INDEX.get(x)
        if parent_index is not None:
            self.mask |= 1 << parent_index
        self.score = objective(
            len(self.selected), self.sum_size, self.diff_size
        )

    def values(self):
        return tuple(self.selected)


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    rng = random.Random(SEED)
    initial = State(PARENT)
    beam = [initial]
    best_score = initial.score
    best_values = initial.values()
    write_solution(best_values)
    deadline = time.monotonic() + SEARCH_SECONDS
    timed_out = False

    while len(beam[0].selected) > MIN_SIZE and not timed_out:
        heap = []
        seen = set()
        serial = 0
        for parent_number, state in enumerate(beam):
            for position, x in enumerate(state.selected):
                if (serial & 31) == 0 and time.monotonic() >= deadline:
                    timed_out = True
                    break
                child_mask = state.mask & ~(1 << PARENT_INDEX[x])
                if child_mask in seen:
                    continue
                seen.add(child_mask)
                score = state.deletion_score(position)
                item = (score, child_mask, parent_number, x)
                if len(heap) < BEAM_WIDTH:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
                serial += 1
            if timed_out:
                break
        if timed_out or not heap:
            break

        next_beam = []
        for score, child_mask, parent_number, x in sorted(heap, reverse=True):
            child = beam[parent_number].clone()
            child.remove_at(bisect.bisect_left(child.selected, x))
            child.mask = child_mask
            child.score = score
            next_beam.append(child)
            if score > best_score:
                best_score = score
                best_values = child.values()
                write_solution(best_values)
        beam = next_beam

    # Use the terminal compressed beam for exact single-swap repair.  Each
    # trial removes one random member and exhaustively finds its best insertion
    # over the requested coordinate interval.
    repair_states = beam
    while not timed_out and time.monotonic() < deadline:
        state = repair_states[rng.randrange(len(repair_states))]
        position = rng.randrange(len(state.selected))
        candidate = state.clone()
        removed = candidate.remove_at(position)
        occupied = set(candidate.selected)
        insertion_score = state.score
        insertion = removed

        coordinates = list(range(MAX_COORDINATE + 1))
        start = rng.randrange(MAX_COORDINATE + 1)
        coordinates = coordinates[start:] + coordinates[:start]
        for iteration, x in enumerate(coordinates):
            if iteration & 127 == 0 and time.monotonic() >= deadline:
                timed_out = True
                break
            if x in occupied:
                continue
            score = candidate.addition_score(x)
            if score > insertion_score:
                insertion_score = score
                insertion = x

        if timed_out:
            break
        if insertion_score > state.score:
            candidate.add(insertion)
            candidate.score = insertion_score
            repair_states[repair_states.index(state)] = candidate
            if insertion_score > best_score:
                best_score = insertion_score
                best_values = candidate.values()
                write_solution(best_values)

    write_solution(best_values)
    print(
        f"wrote deletion-beam best: n={len(best_values)} "
        f"score={best_score:.9f}"
    )


if __name__ == "__main__":
    main()
