import json
import logging

from contraqa.entities import QASample
from contraqa.models import openai_api

logger = logging.getLogger(__name__)

EVALUATION_PROMPTS = {
    "AnswerExist": (
        "Determine if the reference short answer appears anywhere in the longer system answer --- even if it's noted as incorrect, anecdotal or implicitly mentioned. "
        'Return a JSON object with a single key, "answer", whose value is either "true" or "false":\n'
        '  - "true" if the reference short answer is present in any form (explicit or implicit).\n'
        '  - "false" otherwise.\n'
        'Question: "{question}"\n'
        'Reference short answer: "{reference_answer}"\n'
        'System answer: "{model_answer}"\n\n'
    ),
    "AnswerExistInPassages": (
        "Determine if the given short answer appears anywhere in the following passages --- even if it's noted as anecdotal or implicitly mentioned. "
        'Return a JSON object with a single key, "answer", whose value is either "true" or "false":\n'
        '  - "true" if the given short answer is present in any form (explicit or implicit).\n'
        '  - "false" otherwise.\n'
        'Question: "{question}"\n'
        'Short answer: "{model_answer}"\n'
        "Passages:\n{passages}\n\n"
    ),
    "ConflictExist": (
        "Your task is to assess whether a pair of short answers to a question are clearly indicated as conflicting answers in a longer answer written by a question answering system. "
        'You should return a JSON object with an "answer" key "true" if the pair of short answers clearly indicated as conflicting in the longer system\'s answer or "false" otherwise.\n'
        'Question: "{question}"\n'
        'Short answer #1: "{short_answer1}"\n'
        'Short answer #2: "{short_answer2}"\n'
        'System answer: "{model_answer}"\n\n'
    ),
    "ConflictExistInPassages": (
        "Your task is to assess whether the two short answers, which are located in the following passages, are clearly indicated as conflicting to the same question. "
        'You should return a JSON object with an "answer" key "true" if the pair of short answers clearly indicated as conflicting or "false" otherwise.\n'
        'Question: "{question}"\n'
        'Short answer #1: "{short_answer1}"\n'
        'Short answer #2: "{short_answer2}"\n'
        "Passages:\n{passages}\n\n"
    ),
    "AnswerDecomposition": (
        "Given the following question and long-form system answer, extract all distinct short answers that directly address the question. "
        "Ensure to extract all such sub-answers. "
        'The output must be a JSON object with a "sub_answers" key whose value is an array of sub-answer strings.\n'
        'Question: "{question}"\n'
        'System answer: "{model_answer}"\n\n'
    ),
    # Unused
    "ConflictExistInReferencePairs": (
        "Your task is to assess whether a pair of short answers to a question are found as conflicting answers in a set of reference conflicting answers. "
        'You should return a JSON object with an "answer" key "true" if the pair of short answers clearly indicated as conflicting in the longer system\'s answer or "false" otherwise.\n'
        'Question: "{question}"\n'
        'Short answer #1: "{short_answer1}"\n'
        'Short answer #2: "{short_answer2}"\n'
        "Reference conflicting answers pairs: \n{conflicting_answers_pairs}\n\n"
        # "Does the pair of reference short answers presented as conflicting answers in the system answer? answer true or false.\n"
    ),
    "NoAnswer": (
        "Your task is to assess whether the system answer to the following question explicitly and clearly states that there is no answer in the given sources. "
        'You should return a JSON object with an "answer" key "true" if the answer clearly states that no answer or "false" otherwise.\n'
        'Question: "{question}"\n'
        'System answer: "{model_answer}"\n\n'
        "Does the system answer clearly states that there is no answer in the sources? answer true or false.\n"
    ),
}

SYSTEM_JUDGE_PROMPT = "You are a judge of question answering systems."


# Either {"answer": true} or {"answer": false}
BOOLEAN_ONLY_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "title": "AnswerResponse",
            "type": "object",
            "properties": {"answer": {"title": "Answer", "type": "boolean"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "name": "AnswerResponse",
        "strict": True,
    },
}

ARRAY_OF_STRINGS_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "title": "SubAnswersResponse",
            "type": "object",
            "properties": {
                "sub_answers": {
                    "title": "SubAnswersArrayOfString",
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["sub_answers"],
            "additionalProperties": False,
        },
        "name": "AnswerResponse",
        "strict": True,
    },
}


def format_passages(passages: list[str]) -> str:
    return "\n\n".join([f"Passage #{i}:\n{p}" for i, p in enumerate(passages, start=1)])


def get_llm_as_a_judge_response(
    prompt: str, judge_model: str, response_format: dict = None
) -> bool | dict:
    model_id, model_kwargs = openai_api.model2id(judge_model)

    if not response_format:
        response_format = BOOLEAN_ONLY_JSON_SCHEMA

    raw_resp = openai_api.completion(
        prompt=prompt,
        system=SYSTEM_JUDGE_PROMPT,
        model=model_id,
        **model_kwargs,
        response_format=response_format,
    )
    resp = json.loads(raw_resp)

    if response_format == BOOLEAN_ONLY_JSON_SCHEMA:
        decision = resp["answer"]
        assert decision in [True, False]
        return decision

    return resp


def get_answer_recall_judge_response(
    question: str, reference_answer: str, model_answer: str, judge_model: str
):
    prompt = EVALUATION_PROMPTS["AnswerExist"].format(
        question=question,
        reference_answer=reference_answer,
        model_answer=model_answer,
    )
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def get_conflicting_pairs_found_judge_response(
    question: str,
    short_answer1: str,
    short_answer2: str,
    model_answer: str,
    judge_model: str,
) -> bool:
    prompt = EVALUATION_PROMPTS["ConflictExist"].format(
        question=question,
        short_answer1=short_answer1,
        short_answer2=short_answer2,
        model_answer=model_answer,
    )
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def get_no_answer_judge_response(
    question: str,
    model_answer: str,
    judge_model: str,
) -> bool:
    prompt = EVALUATION_PROMPTS["NoAnswer"].format(question=question, model_answer=model_answer)
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def get_answer_exist_in_passages_judge_response(
    question: str,
    model_sub_answer: str,
    passages: list[str],
    judge_model: str,
) -> bool:
    passages_text = format_passages(passages)
    prompt = EVALUATION_PROMPTS["AnswerExistInPassages"].format(
        question=question, model_answer=model_sub_answer, passages=passages_text
    )
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def decompose_answer_into_sub_answers_response(
    question: str,
    model_answer: str,
    judge_model: str,
) -> list[str]:
    prompt = EVALUATION_PROMPTS["AnswerDecomposition"].format(
        question=question, model_answer=model_answer
    )
    resp = get_llm_as_a_judge_response(
        prompt=prompt, judge_model=judge_model, response_format=ARRAY_OF_STRINGS_JSON_SCHEMA
    )
    sub_answers = resp["sub_answers"]
    logger.info(f"found {len(sub_answers)} sub answers, checking precision")
    return sub_answers


def evaluate_answer_recall(
    model_answer: str, question: str, reference_answers: list[str], judge_model: str
) -> tuple[float, list[int]]:
    logger.info(f"evaluating model answer recall on {len(reference_answers)} reference answers")

    if not reference_answers:
        return float("nan"), []

    entailed_answers = []
    for ref_answer_id, reference_answer in enumerate(reference_answers):
        is_entailed = get_answer_recall_judge_response(
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
            judge_model=judge_model,
        )
        if is_entailed:
            entailed_answers.append(ref_answer_id)

    return len(entailed_answers) / len(reference_answers), entailed_answers


def evaluate_conflicting_pairs_recall(
    model_answer: str,
    question: str,
    conflicting_pairs: dict[tuple[int, int], tuple[str, str]],
    entailed_answers: list[int],
    judge_model: str,
) -> tuple[float, float, list[tuple[int, int]]]:
    logger.info(f"evaluating model conflicting pairs recall answer on {len(conflicting_pairs)}")

    if not conflicting_pairs:
        return float("nan"), float("nan"), []

    found_conflicting_pairs = []
    for (answer1_id, answer2_id), (
        reference_answer1,
        reference_answer2,
    ) in conflicting_pairs.items():
        is_conflicting_pair_found = get_conflicting_pairs_found_judge_response(
            question=question,
            short_answer1=reference_answer1,
            short_answer2=reference_answer2,
            model_answer=model_answer,
            judge_model=judge_model,
        )
        if is_conflicting_pair_found:
            found_conflicting_pairs.append((answer1_id, answer2_id))  # the found conflicting pair

    strict_conflict_pairs = [
        (a_id1, a_id2)
        for a_id1, a_id2 in found_conflicting_pairs
        if a_id1 in entailed_answers and a_id2 in entailed_answers
    ]

    return (
        len(found_conflicting_pairs) / len(conflicting_pairs),
        len(strict_conflict_pairs) / len(conflicting_pairs),
        found_conflicting_pairs,
    )


def evaluate_no_answer(model_answer: str, question: str, judge_model: str) -> float:
    logger.info(f"evaluating for no answer")

    is_missing = get_no_answer_judge_response(
        question=question, model_answer=model_answer, judge_model=judge_model
    )
    return 1.0 if is_missing else 0.0


def evaluate_answer_precision(
    sub_answers: list[str], question: str, passages: list[str], judge_model: str
) -> tuple[float, dict[str, bool]]:
    subanswer2found = {}
    for sub_answer in sub_answers:
        is_entailed = get_answer_exist_in_passages_judge_response(
            question=question,
            model_sub_answer=sub_answer,
            passages=passages,
            judge_model=judge_model,
        )
        subanswer2found[sub_answer] = int(is_entailed)

    if subanswer2found:
        answer_precision = sum(subanswer2found.values()) / len(subanswer2found)
    else:
        answer_precision = float("nan")
    return answer_precision, subanswer2found


def get_conflicting_pairs_passages_found_judge_response(
    question: str, short_answer1: str, short_answer2: str, passages: list[str], judge_model: str
) -> bool:
    passages_text = format_passages(passages)
    prompt = EVALUATION_PROMPTS["ConflictExistInPassages"].format(
        question=question,
        short_answer1=short_answer1,
        short_answer2=short_answer2,
        passages=passages_text,
    )
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def get_conflicting_pairs_reference_pairs_found_judge_response(
    question: str,
    short_answer1: str,
    short_answer2: str,
    conflicting_answer_pairs: list[tuple[str, str]],
    judge_model: str,
) -> bool:
    conflicting_answers_pairs_text = "\n".join(
        [
            f"{idx}. {ai!r} conflicts with {aj!r}"
            for idx, (ai, aj) in enumerate(conflicting_answer_pairs, start=1)
        ]
    )
    prompt = EVALUATION_PROMPTS["ConflictExistInReferencePairs"].format(
        question=question,
        short_answer1=short_answer1,
        short_answer2=short_answer2,
        conflicting_answers_pairs=conflicting_answers_pairs_text,
    )
    return get_llm_as_a_judge_response(prompt=prompt, judge_model=judge_model)


def evaluate_conflict_precision(
    model_answer: str,
    sub_answers: list[str],
    question: str,
    passages: list[str],
    # conflicting_answer_pairs: list[tuple[str, str]],
    judge_model: str,
) -> tuple[float, dict[tuple[int, int], bool]]:
    conflict_pair2found = {}
    for i, sub_answer1 in enumerate(sub_answers):
        for j, sub_answer2 in enumerate(sub_answers[i + 1 :], start=i + 1):
            is_conflicting_in_model_answer = get_conflicting_pairs_found_judge_response(
                question=question,
                short_answer1=sub_answer1,
                short_answer2=sub_answer2,
                model_answer=model_answer,
                judge_model=judge_model,
            )
            if is_conflicting_in_model_answer:
                # is_conflicting_in_ref_pairs = get_conflicting_pairs_reference_pairs_found_judge_response(
                #     question=question,
                #     short_answer1=sub_answer1,
                #     short_answer2=sub_answer2,
                #     conflicting_answer_pairs=conflicting_answer_pairs,
                #     judge_model=judge_model,
                # )
                # if is_conflicting_in_ref_pairs:
                #     logger.info("found conflicting pair in reference pairs")
                #     conflict_pair2found[(i, j)] = 1.0
                #     continue
                #
                conflict_pair2found[(i, j)] = get_conflicting_pairs_passages_found_judge_response(
                    question=question,
                    short_answer1=sub_answer1,
                    short_answer2=sub_answer2,
                    passages=passages,
                    judge_model=judge_model,
                )

    if conflict_pair2found:
        conflict_precision = sum(conflict_pair2found.values()) / len(conflict_pair2found)
    else:
        conflict_precision = float("nan")
    return conflict_precision, conflict_pair2found


def compute_f1(p: float, r: float) -> float:
    if p + r:
        return 2 * (p * r) / (p + r)
    return float("nan")


def evaluate_model_answer(
    qa_sample: QASample,
    input_sources: list[str],
    model_answer: str,
    judge_model: str,
    answer_quality_only: bool,
) -> dict:
    ref_answers = [a.as_string() for a in qa_sample.answers]

    # evaluate answer recall
    answer_recall, entailed_answers = evaluate_answer_recall(
        model_answer=model_answer,
        question=qa_sample.question,
        reference_answers=ref_answers,
        judge_model=judge_model,
    )

    if qa_sample.answer_type == "Neutral":
        no_answer_score = evaluate_no_answer(
            model_answer=model_answer,
            question=qa_sample.question,
            judge_model=judge_model,
        )
    else:
        no_answer_score = float("nan")

    # Decompose answer
    model_answers_decomposed = decompose_answer_into_sub_answers_response(
        question=qa_sample.question,
        model_answer=model_answer,
        judge_model=judge_model,
    )

    # Answer Precision
    answer_precision, sub_answer_to_entailed = evaluate_answer_precision(
        sub_answers=model_answers_decomposed,
        question=qa_sample.question,
        passages=input_sources,
        judge_model=judge_model,
    )

    # F1
    answer_f1 = compute_f1(answer_precision, answer_recall)

    if not answer_quality_only:
        # evaluate conflicting pairs recall
        # mapping of (id1, id2) -> (ans1, ans2)
        conflicting_pairs = {
            (a_id1, a_id2): (
                qa_sample.answers[a_id1].as_string(),
                qa_sample.answers[a_id2].as_string(),
            )
            for a_id1, a_id2 in qa_sample.conflicting_answer_pairs
        }

        conflict_pairs_recall, strict_conflict_pairs_recall, found_conflicting_pairs = (
            evaluate_conflicting_pairs_recall(
                model_answer=model_answer,
                entailed_answers=entailed_answers,
                question=qa_sample.question,
                conflicting_pairs=conflicting_pairs,
                judge_model=judge_model,
            )
        )

        # Conflict Precision
        conflict_pairs_precision, conflict_sub_answer_pairs_found = evaluate_conflict_precision(
            model_answer=model_answer,
            sub_answers=model_answers_decomposed,
            question=qa_sample.question,
            passages=input_sources,
            # conflicting_answer_pairs=[
            #     (ai.as_string(), aj.as_string())
            #     for ai, aj in qa_sample.get_answer_pairs(is_conflict=True)
            # ],
            judge_model=judge_model,
        )

        # F1
        conflict_f1 = compute_f1(conflict_pairs_precision, conflict_pairs_recall)
        strict_conflict_f1 = compute_f1(conflict_pairs_precision, strict_conflict_pairs_recall)

    else:
        found_conflicting_pairs = float("nan")
        conflict_sub_answer_pairs_found = float("nan")
        strict_conflict_pairs_recall = float("nan")
        strict_conflict_f1 = float("nan")
        conflict_pairs_precision = float("nan")
        conflict_pairs_recall = float("nan")
        conflict_f1 = float("nan")

    return {
        "answer_recall": answer_recall,
        "answer_precision": answer_precision,
        "answer_f1": answer_f1,
        "conflict_pairs_precision": conflict_pairs_precision,
        "conflict_pairs_recall": conflict_pairs_recall,
        "conflict_pairs_f1": conflict_f1,
        "strict_conflict_pairs_recall": strict_conflict_pairs_recall,
        "strict_conflict_pairs_f1": strict_conflict_f1,
        "answers_entailed": entailed_answers,
        "no_answer_score": no_answer_score,
        "sub_answer_to_entailed": sub_answer_to_entailed,
        "found_conflicting_pairs": found_conflicting_pairs,
        "conflict_sub_answer_pairs_found": conflict_sub_answer_pairs_found,
        "model_answers_decomposed": model_answers_decomposed,
        "judge_model": judge_model,
    }


if __name__ == "__main__":
    for name, prompt in EVALUATION_PROMPTS.items():
        print(name)
        print(prompt)
        print("=========")
