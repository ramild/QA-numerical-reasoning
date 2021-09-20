import operator
import torch
from functools import partial, reduce, cmp_to_key
import heapq
from typing import Dict, List, Any
import numpy as np

EPS = 1e-6


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False


def get_top_spans_by_min_prob(
    original_text: str,
    offsets,
    text_start,
    span_start_probs,
    span_end_probs,
    class_prob,
    answer_type,
    prob_min=0.01,
) -> List[Dict[str, Any]]:
    if class_prob < EPS:
        return []
    probs_matrix = torch.multiply(
        span_start_probs.reshape(-1, 1), span_end_probs.reshape(1, -1)
    )
    spans = list(zip(*np.where(probs_matrix * class_prob >= prob_min)))
    answers = []
    for span in spans:
        if span[0] > span[1]:
            continue
        start_offset = offsets[span[0] - text_start][0]
        end_offset = offsets[span[1] - text_start][1]
        predicted_answer = original_text[start_offset:end_offset]
        answers.append(
            {
                "answer": predicted_answer,
                "type": answer_type,
                "probability": probs_matrix[span[0]][span[1]].item() * class_prob,
                "explanation": [start_offset, end_offset],
            }
        )
    return answers


def get_probabilities_product(inds, probs):
    return reduce(operator.mul, (prob[ind] for ind, prob in zip(inds, probs)), 1)


def get_next_set(inds):
    return [
        tuple(t - (1 if j == i else 0) for j, t in enumerate(inds))
        for i in range(len(inds))
        if inds[i] > 0
    ]


def get_top_probs(probabilities, prob_min=0.01):
    original_prod = partial(get_probabilities_product, probs=probabilities)

    indexes = [np.array(d).argsort() for d in probabilities]
    probs = [sorted(p) for p in probabilities]
    prod = partial(get_probabilities_product, probs=probs)
    max_inds = [indexes[i][len(dat) - 1] for i, dat in enumerate(probs)]
    if original_prod(max_inds) < prob_min:
        return [], []

    k_smallest = [max_inds]
    possible_k_smallest = []
    next_possible_k_smallest = tuple(len(prob) - 1 for prob in probs)

    while True:
        new_possible = sorted(
            get_next_set(next_possible_k_smallest),
            key=prod,
            reverse=True,
        )
        possible_k_smallest = heapq.merge(
            possible_k_smallest, new_possible, key=prod, reverse=True
        )
        next_possible_k_smallest = next(possible_k_smallest)
        index_smallest = [
            indexes[i][index] for i, index in enumerate(list(next_possible_k_smallest))
        ]
        if original_prod(index_smallest) >= prob_min:
            k_smallest.append(index_smallest)
        else:
            break

    return k_smallest, [original_prod(inds) for inds in k_smallest]


def get_top_arithmetic_by_min_prob(
    original_numbers,
    number_sign_probs,
    class_prob,
    prob_min=0.01,
) -> List[Dict[str, Any]]:
    sign_remap = {0: 0, 1: 1, 2: -1}
    if class_prob < EPS:
        return []
    indexes, prods = get_top_probs(number_sign_probs, prob_min / class_prob)
    answer_to_prob = {}
    for index_list, prod in zip(indexes, prods):
        num_of_numbers = len(original_numbers)
        predicted_answer = sum(
            sign_remap[sign] * number
            for sign, number in zip(index_list[:num_of_numbers], original_numbers)
        )
        if predicted_answer in answer_to_prob:
            answer_to_prob[predicted_answer] += prod.item() * class_prob
        else:
            answer_to_prob[predicted_answer] = prod.item() * class_prob
    return [
        {
            "answer": predicted_answer,
            "type": "arithmetic",
            "probability": float(prob),
            "explanation": indexes,
        }
        for predicted_answer, prob in answer_to_prob.items()
    ]


def get_top_count_by_min_prob(
    count_probs,
    class_prob,
    prob_min=0.01,
) -> List[Dict[str, Any]]:
    if class_prob < EPS:
        return []
    return [
        {
            "answer": count,
            "type": "count",
            "probability": float(prob.item() * class_prob),
        }
        for count, prob in enumerate(count_probs)
        if prob * class_prob >= prob_min
    ]


def accumulate_probabilities(top_predicted_answers):
    ans_probs = {}
    for ans in top_predicted_answers:
        answer = ans["answer"]
        if _is_number(str(answer)):
            answer = round(float(answer), 5)
        if answer not in ans_probs:
            ans_probs[answer] = {
                "probability": ans["probability"],
                "type": [ans["type"]],
            }
        else:
            ans_probs[answer]["probability"] += ans["probability"]
            ans_probs[answer]["type"].append(ans["type"])
    ans_probs = dict(sorted(ans_probs.items(), key=lambda x: -x[1]["probability"]))
    return [
        {
            "answer": ans,
            "probability": value["probability"],
            "type": value["type"],
        }
        for ans, value in ans_probs.items()
    ]


# Можно убрать из answering_abilities multiple spans?
def get_all_top_answers_by_min_prob(
    metadata,
    question_span_start_log_probs,
    question_span_end_log_probs,
    passage_span_start_log_probs,
    passage_span_end_log_probs,
    number_sign_log_probs,
    count_number_log_probs,
    answer_ability_log_probs,
    prob_min=0.01,
) -> List[List[Dict[str, Any]]]:
    answering_abilities = [
        "passage_span_extraction",
        "question_span_extraction",
        "addition_subtraction",
        "counting",
        "multiple_spans",
    ]
    with torch.no_grad():
        question_span_start_probs = torch.exp(question_span_start_log_probs)
        question_span_end_probs = torch.exp(question_span_end_log_probs)
        passage_span_start_probs = torch.exp(passage_span_start_log_probs)
        passage_span_end_probs = torch.exp(passage_span_end_log_probs)
        number_sign_probs = torch.exp(number_sign_log_probs)
        count_probs = torch.exp(count_number_log_probs)
        answer_ability_probs = (
            torch.exp(answer_ability_log_probs).cpu().numpy().tolist()
        )

        all_batch_answers: List[List[Dict[str, Any]]] = []
        batch_size = len(metadata)
        for i in range(batch_size):
            top_passage_span_answers = get_top_spans_by_min_prob(
                original_text=metadata[i]["original_passage"],
                offsets=metadata[i]["passage_token_offsets"],
                text_start=len(metadata[i]["question_tokens"]) + 2,
                span_start_probs=passage_span_start_probs[i],
                span_end_probs=passage_span_end_probs[i],
                class_prob=answer_ability_probs[i][
                    answering_abilities.index("passage_span_extraction")
                ],
                answer_type="passage-span",
                prob_min=prob_min,
            )

            top_question_span_answers = get_top_spans_by_min_prob(
                original_text=metadata[i]["original_question"],
                offsets=metadata[i]["question_token_offsets"],
                text_start=1,
                span_start_probs=question_span_start_probs[i],
                span_end_probs=question_span_end_probs[i],
                class_prob=answer_ability_probs[i][
                    answering_abilities.index("question_span_extraction")
                ],
                answer_type="question-span",
                prob_min=prob_min,
            )

            top_arithmetic_answers = get_top_arithmetic_by_min_prob(
                metadata[i]["original_numbers"],
                number_sign_probs[i],
                class_prob=answer_ability_probs[i][
                    answering_abilities.index("addition_subtraction")
                ],
                prob_min=prob_min,
            )

            top_counting_answers = get_top_count_by_min_prob(
                count_probs[i],
                class_prob=answer_ability_probs[i][
                    answering_abilities.index("counting")
                ],
                prob_min=prob_min,
            )

            sort_by_probability = cmp_to_key(
                lambda first, second: 1
                if first["probability"] < second["probability"]
                else -1
            )

            all_batch_answers.append(
                accumulate_probabilities(
                    sorted(
                        top_passage_span_answers
                        + top_question_span_answers
                        + top_arithmetic_answers
                        + top_counting_answers,
                        key=sort_by_probability,
                    )
                )
            )
    return all_batch_answers
