import json
import copy
import re
import random

from typing import Any, Dict, List, NewType, Union, NamedTuple

from uniform_sampler import UniformOrderedCartesianProductSampler

POWERS_OF_10 = [0] + [10 ** j for j in range(20)]
MIN_DAY_OF_MONTH = 1
MAX_DAY_OF_MONTH = 31
BASE_100 = 100
MAX_ATTEMPTS = 10

Number = NewType("Number", Union[int, float])


def extract_numbers_from_text(text: str) -> List[Dict[str, Any]]:
    return [
        {
            "span": (number_match.start(0), number_match.end(0)),
            "value": int(number_match.group(0)),
        }
        for number_match in re.finditer(r"\d+", text)
    ]


def random_choice_in_large_range(
    left: int,
    right: int,
    subsample_size: int,
) -> List[int]:
    """
    Divides the [left, right] range into subsample_size buckets and selects a random integer from each of the bucket.

    The use case: when the range size is too large and np.random.choice is too slow.
    """
    new_samples = []
    bucket_size = (right - left + 1) // subsample_size
    for i in range(subsample_size):
        random_ind = random.randint(0, bucket_size - 1)
        new_samples.append(left + i * bucket_size + random_ind)
    return new_samples


class AugmentationConfig(NamedTuple):
    max_change_times: int
    max_year_difference: int
    year_min: int
    year_max: int
    max_search_space_size: int
    fixpoint_values: List[Number]

    @staticmethod
    def from_config_path(config_path: str):
        with open(config_path, "r") as fp:
            config_dict = json.load(fp)
        return AugmentationConfig.from_config_dict(config_dict)

    @staticmethod
    def from_config_dict(config_dict: Dict[str, Any]):
        return AugmentationConfig(**config_dict)


class QuestionPassageNumbersOrderedAugmenter:
    def __init__(self, augmentation_config: AugmentationConfig):
        self._augmentation_config = augmentation_config

    def __call__(
        self,
        original_passage: str,
        original_question: str,
        passage_numbers: List[Dict[str, Any]],
        question_numbers: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._generate_new_passage(
            original_passage,
            original_question,
            passage_numbers,
            question_numbers,
        )

    @staticmethod
    def from_config_path(config_path: str):
        with open(config_path, "r") as fp:
            config_dict = json.load(fp)
        return QuestionPassageNumbersOrderedAugmenter(
            AugmentationConfig.from_config_dict(config_dict)
        )

    @staticmethod
    def _modify_text(
        original_text: str, numbers: List[Number], map_original_to_new_value
    ):
        modified_text = copy.deepcopy(original_text)
        answer = 0
        shift = 0
        for span_dict in numbers:
            new_value = map_original_to_new_value[span_dict["value"]]
            if "sign" in span_dict:
                answer += span_dict["sign"] * new_value
            start, end = span_dict["span"]
            start += shift
            end += shift
            modified_text = "".join(
                (modified_text[:start], str(new_value), modified_text[end:])
            )
            shift += len(str(new_value)) - (end - start)
        return modified_text, answer

    def _generate_new_passage(
        self,
        original_passage: str,
        original_question: str,
        passage_numbers: List[Dict[str, Any]],
        question_numbers: List[Dict[str, Any]] = None,
        attempts=0,
    ) -> Union[Dict[str, Any], None]:
        if attempts > MAX_ATTEMPTS:
            return None

        if not question_numbers:
            question_numbers = extract_numbers_from_text(original_question)

        original_values = [span_dict["value"] for span_dict in passage_numbers] + [
            span_dict["value"] for span_dict in question_numbers
        ]

        strictly_sorted_values = sorted(set(original_values))
        search_space_sets = self._generate_search_space_sets(strictly_sorted_values)
        sampler = UniformOrderedCartesianProductSampler(search_space_sets)
        map_original_to_new_value = dict(zip(strictly_sorted_values, sampler()))

        passage_text, answer = QuestionPassageNumbersOrderedAugmenter._modify_text(
            original_passage,
            passage_numbers,
            map_original_to_new_value,
        )

        if answer < 0:
            return self._generate_new_passage(
                original_passage,
                original_question,
                passage_numbers,
                question_numbers,
                attempts + 1,
            )

        question_text, answer = QuestionPassageNumbersOrderedAugmenter._modify_text(
            original_question,
            question_numbers,
            map_original_to_new_value,
        )

        return {
            "new_passage": passage_text,
            "new_question": question_text,
            "new_answer": answer,
        }

    def _generate_search_space_sets(
        self, original_values: List[Number]
    ) -> List[List[Number]]:
        return [self._generate_search_space(value) for value in original_values]

    def _generate_search_space(self, original_value: Number) -> List[Number]:
        if (
            isinstance(original_value, float)
            or original_value in self._augmentation_config.fixpoint_values
        ):
            return [original_value]
        for i, power_of_10 in enumerate(POWERS_OF_10):
            if power_of_10 <= original_value < POWERS_OF_10[i + 1]:
                break
        for j, power_of_10 in enumerate(POWERS_OF_10[1:]):
            if original_value % power_of_10 != 0:
                break

        left = max(
            POWERS_OF_10[i] // POWERS_OF_10[j],
            (original_value // POWERS_OF_10[j])
            // self._augmentation_config.max_change_times,
        )
        right = min(
            POWERS_OF_10[i + 1] // POWERS_OF_10[j],
            (original_value // POWERS_OF_10[j])
            * self._augmentation_config.max_change_times,
        )
        if (
            self._augmentation_config.year_min
            <= original_value
            <= self._augmentation_config.year_max
        ):
            if original_value % BASE_100 == 0:
                return list(
                    range(
                        max(
                            original_value
                            - self._augmentation_config.max_year_difference,
                            self._augmentation_config.year_min,
                        ),
                        original_value
                        + self._augmentation_config.max_year_difference
                        + 1,
                    )
                )
            left = max(
                left,
                (original_value - self._augmentation_config.max_year_difference)
                // POWERS_OF_10[j],
            )
            right = min(
                right,
                (original_value + self._augmentation_config.max_year_difference)
                // POWERS_OF_10[j],
            )

        search_space = []
        if right - left > self._augmentation_config.max_search_space_size:
            range_values = random_choice_in_large_range(
                left,
                right + 1,
                self._augmentation_config.max_search_space_size,
            )
        else:
            range_values = list(range(left, right + 1))

        for num in range_values:
            new_value = num * POWERS_OF_10[j]
            if (num % 10 != 0) and (
                original_value < MIN_DAY_OF_MONTH
                or original_value > MAX_DAY_OF_MONTH
                or (MIN_DAY_OF_MONTH <= new_value <= MAX_DAY_OF_MONTH)
            ):
                search_space.append(new_value)
        return search_space
