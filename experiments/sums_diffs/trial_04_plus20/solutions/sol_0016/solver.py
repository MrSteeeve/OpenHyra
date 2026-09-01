"""Projection sweep and annealing over the parent tensor support."""

import json
import math
import os
import random
import time


BASE_SET = (0, 1, 3, 4, 5, 8, 12, 13, 16, 20, 21, 24, 28, 29, 31, 32, 33)
CELLS = tuple((a, b) for b in BASE_SET for a in BASE_SET)
MISSING_PARENT_CELLS = frozenset((36, 41, 44, 48, 121, 133, 172, 184, 240, 245, 248, 252))
PARENT_SELECTED = tuple(i for i in range(len(CELLS)) if i not in MISSING_PARENT_CELLS)
SEED = 4015
SEARCH_SECONDS = 150.0
HARD_SECONDS = 170.0
MIN_SUPPORT = 64
MAX_SUPPORT = 289
MIN_TEMPERATURE = 1e-6
MAX_TEMPERATURE = 0.004
COOLING_STEPS = 150000
STAGNATION_STEPS = 300000


def objective(n, sums, diffs):
    return math.log(sums / n) / math.log(diffs / n)


def exact_score(values):
    """Return the exact score and cardinalities via nonnegative bitsets."""
    values = tuple(sorted(set(values)))
    mask = 0
    for value in values:
        mask |= 1 << value

    sum_bits = 0
    nonnegative_diff_bits = 0
    for value in values:
        sum_bits |= mask << value
        nonnegative_diff_bits |= mask >> value

    sums = sum_bits.bit_count()
    diffs = 2 * nonnegative_diff_bits.bit_count() - 1
    return objective(len(values), sums, diffs), values


class State:
    """Exact projected-set score maintained under support-cell toggles."""

    def __init__(self, projected_points, selected_cells):
        self.projected_points = projected_points
        unique_values = tuple(sorted(set(projected_points)))
        value_ids = {value: i for i, value in enumerate(unique_values)}
        self.values_by_id = unique_values
        self.cell_value_id = tuple(value_ids[value] for value in projected_points)
        self.cell_chosen = [False] * len(projected_points)
        self.multiplicity = [0] * len(unique_values)
        self.active_ids = []
        self.active_position = [-1] * len(unique_values)
        self.support_size = 0

        maximum = unique_values[-1]
        self.diff_offset = maximum
        slots = 2 * maximum + 1
        self.sum_counts = [0] * slots
        self.diff_counts = [0] * slots
        self.sum_bits = 0
        self.diff_bits = 0

        for index in selected_cells:
            self.add(index)
        self.recompute_score()

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

    def _activate(self, value_id):
        x = self.values_by_id[value_id]
        for other_id in self.active_ids:
            y = self.values_by_id[other_id]
            self._increment_sum(x + y)
            self._increment_diff(x - y + self.diff_offset)
            self._increment_diff(y - x + self.diff_offset)
        self._increment_sum(2 * x)
        self._increment_diff(self.diff_offset)
        self.active_position[value_id] = len(self.active_ids)
        self.active_ids.append(value_id)

    def _deactivate(self, value_id):
        x = self.values_by_id[value_id]
        for other_id in self.active_ids:
            if other_id == value_id:
                continue
            y = self.values_by_id[other_id]
            self._decrement_sum(x + y)
            self._decrement_diff(x - y + self.diff_offset)
            self._decrement_diff(y - x + self.diff_offset)
        self._decrement_sum(2 * x)
        self._decrement_diff(self.diff_offset)

        position = self.active_position[value_id]
        last = self.active_ids.pop()
        if last != value_id:
            self.active_ids[position] = last
            self.active_position[last] = position
        self.active_position[value_id] = -1

    def add(self, cell_index):
        value_id = self.cell_value_id[cell_index]
        if self.multiplicity[value_id] == 0:
            self._activate(value_id)
        self.multiplicity[value_id] += 1
        self.cell_chosen[cell_index] = True
        self.support_size += 1

    def remove(self, cell_index):
        value_id = self.cell_value_id[cell_index]
        self.multiplicity[value_id] -= 1
        if self.multiplicity[value_id] == 0:
            self._deactivate(value_id)
        self.cell_chosen[cell_index] = False
        self.support_size -= 1

    def toggle(self, cell_index):
        if self.cell_chosen[cell_index]:
            self.remove(cell_index)
        else:
            self.add(cell_index)

    def recompute_score(self):
        n = len(self.active_ids)
        self.score = objective(n, self.sum_bits.bit_count(), self.diff_bits.bit_count())

    def selected_cells(self):
        return tuple(i for i, chosen in enumerate(self.cell_chosen) if chosen)

    def output_values(self):
        return tuple(sorted(self.values_by_id[i] for i in self.active_ids))


def write_solution(values):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"A": list(values)}, stream)
    os.replace(temporary, path)


def main():
    started = time.monotonic()
    rng = random.Random(SEED)

    # Always leave a valid copy of the 1.079440527 parent before doing any search.
    parent_values = tuple(CELLS[i][0] + 67 * CELLS[i][1] for i in PARENT_SELECTED)
    best_score, best_values = exact_score(parent_values)
    write_solution(best_values)

    projection_score = float("-inf")
    best_u = 1
    best_v = 1
    best_projection_values = ()
    for u in range(1, 49):
        for v in range(1, 49):
            if math.gcd(u, v) != 1:
                continue
            values = (u * CELLS[i][0] + v * CELLS[i][1] for i in PARENT_SELECTED)
            score, projected_values = exact_score(values)
            if len(projected_values) >= MIN_SUPPORT and score > projection_score:
                projection_score = score
                best_u = u
                best_v = v
                best_projection_values = projected_values
            if score > best_score:
                best_score = score
                best_values = projected_values
                write_solution(best_values)

    projected_points = tuple(best_u * a + best_v * b for a, b in CELLS)
    state = State(projected_points, PARENT_SELECTED)
    local_best_score = state.score
    local_best_cells = state.selected_cells()
    if state.output_values() != best_projection_values:
        raise RuntimeError("projection initialization mismatch")

    anneal_deadline = min(time.monotonic() + SEARCH_SECONDS, started + HARD_SECONDS)
    steps = 0
    last_local_improvement = 0
    heat_start = 0
    cell_range = range(len(CELLS))

    while time.monotonic() < anneal_deadline:
        toggles = rng.sample(cell_range, rng.randint(1, 3))
        new_support_size = state.support_size + sum(
            -1 if state.cell_chosen[index] else 1 for index in toggles
        )
        if not MIN_SUPPORT <= new_support_size <= MAX_SUPPORT:
            steps += 1
            continue

        old_score = state.score
        for index in toggles:
            state.toggle(index)
        state.recompute_score()
        if not MIN_SUPPORT <= len(state.active_ids) <= MAX_SUPPORT:
            for index in reversed(toggles):
                state.toggle(index)
            state.score = old_score
            steps += 1
            continue

        phase = min(steps - heat_start, COOLING_STEPS - 1) / (COOLING_STEPS - 1)
        temperature = MAX_TEMPERATURE * (MIN_TEMPERATURE / MAX_TEMPERATURE) ** phase
        delta = state.score - old_score
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            if state.score > local_best_score:
                local_best_score = state.score
                local_best_cells = state.selected_cells()
                last_local_improvement = steps
            if state.score > best_score:
                best_score = state.score
                best_values = state.output_values()
                write_solution(best_values)
        else:
            for index in reversed(toggles):
                state.toggle(index)
            state.score = old_score

        steps += 1
        if steps - last_local_improvement >= STAGNATION_STEPS:
            state = State(projected_points, local_best_cells)
            heat_start = steps
            last_local_improvement = steps

    write_solution(best_values)
    print(
        f"wrote projected tensor best: n={len(best_values)} score={best_score:.9f} "
        f"projection=({best_u},{best_v})"
    )


if __name__ == "__main__":
    main()
