import numpy as np
import copy

from typing import List, NewType, Union

Number = NewType("Number", Union[int, float])


class UniformOrderedCartesianProductSampler:
    def __init__(self, search_space_sets: List[List[Number]]):
        self._search_space_sets = copy.deepcopy(search_space_sets)
        self._sample_size = len(self._search_space_sets)
        self._search_space_sets.append([max(self._search_space_sets[-1]) + 1])
        self._compute_prefix_counts()

    def __call__(self) -> List[Number]:
        new_random_values = []
        previous_selected_index = 0

        for i in reversed(range(self._sample_size)):
            search_space_size = len(self._search_space_sets[i])
            probabilities = [
                self._prefix_counts[i][j]
                / self._prefix_counts[i + 1][previous_selected_index]
                if self._search_space_sets[i][j]
                < self._search_space_sets[i + 1][previous_selected_index]
                else 0
                for j in range(search_space_size)
            ]
            previous_selected_index = np.random.choice(
                range(search_space_size),
                size=1,
                p=probabilities,
            )[0]
            new_random_values.append(
                self._search_space_sets[i][previous_selected_index]
            )
        return new_random_values[::-1]

    def _compute_prefix_counts(self) -> List[List[int]]:
        self._prefix_counts = [[1] * len(self._search_space_sets[0])]
        for i, search_space_set in enumerate(self._search_space_sets[1:], 1):
            self._prefix_counts.append(
                [
                    sum(
                        self._prefix_counts[i - 1][k]
                        for k, previous_value in enumerate(
                            self._search_space_sets[i - 1]
                        )
                        if previous_value < current_value
                    )
                    for j, current_value in enumerate(search_space_set)
                ]
            )
        return self._prefix_counts
