from augment_numbers_qa import QuestionPassageNumbersOrderedAugmenter

import string
import random
import json
import copy
from tqdm import tqdm
from argparse import ArgumentParser

from typing import Dict, Any

STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip().lower()
    if not text:
        return []
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False


def generate_augmented_drop_data(
    train_data: Dict[str, Dict[str, Any]],
    predictions_train: Dict[str, Dict[str, Any]],
    qa_augmenter: QuestionPassageNumbersOrderedAugmenter,
) -> Dict[str, Dict[str, Any]]:
    new_train_data = copy.deepcopy(train_data)

    for passage_id, passage_qa in tqdm(train_data.items()):
        for question in passage_qa["qa_pairs"]:
            prediction = predictions_train[question["query_id"]]
            if (
                prediction["numbers"]
                and prediction["numbers"][-1]["sign"] == 0
                and _is_number(prediction["predicted_answer"])
                and _is_number(question["answer"]["number"])
                and round(float(prediction["predicted_answer"]), 2)
                == round(float(question["answer"]["number"]), 2)
            ):
                passage_text = " ".join(whitespace_tokenize(passage_qa["passage"]))
                new_passage_item = qa_augmenter(
                    passage_text,
                    question["question"],
                    prediction["numbers"],
                )
                if new_passage_item is None:
                    continue
                new_question = copy.deepcopy(question)
                new_question["query_id"] = f'augmented_{question["query_id"]}'
                new_question["question"] = new_passage_item["new_question"]
                new_question["answer"]["number"] = str(new_passage_item["new_answer"])
                new_train_item = {
                    f'augmented_#_{passage_id}_#_{question["query_id"]}': {
                        "passage": new_passage_item["new_passage"],
                        "qa_pairs": [new_question],
                    }
                }
                new_train_data.update(new_train_item)

    train_set_shuffled = list(new_train_data.items())
    random.shuffle(train_set_shuffled)

    return dict(train_set_shuffled)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--augmentation_config_path")
    parser.add_argument("--train_data_path")
    parser.add_argument("--prediction_train_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    with open(args.train_data_path, "r") as fp:
        train_data = json.load(fp)
    with open(args.prediction_train_path, "r") as fp:
        predictions_train = json.load(fp)
    qa_augmenter = QuestionPassageNumbersOrderedAugmenter.from_config_path(
        args.augmentation_config_path
    )

    augmented_train_data = generate_augmented_drop_data(
        train_data, predictions_train, qa_augmenter
    )
    with open(args.output_path, "w") as fp:
        json.dump(augmented_train_data, fp)
