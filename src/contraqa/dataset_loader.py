import json
import logging
import re
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from random import Random

import pandas as pd
import rich
from nltk import word_tokenize

from contraqa.entities import Evidence, QASample, Answer
from contraqa.utils import log
from contraqa.utils.configs import config

logger = logging.getLogger(__name__)

DATA_FIXER_MISSING_PARAGRAPH_ID = {
    "healthver_18_claims_11.14": 4,
}
DATA_FIXER = {
    (
        "climate-fever.81",
        2,
    ): (
        "The release of the paper on 22 April 1998 received exceptional media coverage, with many"
        " asking whether it proved that human activity was responsible for climate change. Michael"
        ' Mann responded that it was "highly suggestive" of such a conclusion. He stated, "Our'
        " conclusion was that the warming of the past few decades appears to be closely tied to"
        ' emission of greenhouse gases by humans and not any of the natural factors." Mann noted'
        ' that most proxy data are inherently imprecise, explaining, "We do have error bars. They'
        " are somewhat sizable as one gets farther back in time, and there is reasonable"
        " uncertainty in any given year. There is quite a bit of work to be done in reducing these"
        ' uncertainties." Climatologist Tom Wigley welcomed the progress of the study but expressed'
        " skepticism about whether proxy data could ever be wholly convincing in isolating the"
        " human contribution to climate change."
    ),
    (
        "climate-fever.69",
        3,
    ): (
        "The climate deniers involved in the [[Climatic Research Unit email controversy]]"
        ' ("Climategate") in 2009 claimed that researchers faked the data in their research'
        " publications and suppressed their critics in order to receive more funding (i.e. taxpayer"
        " money). Eight committees investigated these allegations and published reports, each"
        " finding no evidence of fraud or scientific misconduct. According to the Muir Russell"
        ' report, the scientists\' "rigor and honesty as scientists are not in doubt", the'
        ' investigators "did not find any evidence of behavior that might undermine the conclusions'
        ' of the IPCC assessments", but there had been "a consistent pattern of failing to display'
        ' the proper degree of openness." The scientific consensus that climate change is occurring'
        " as a result of human activity remained unchanged at the end of the investigations."
    ),
    (
        "climate-fever.67",
        1,
    ): (
        "Climate and fire experts agree that climate change is a contributing factor to increased"
        " fire frequency and intensity in southeast Australia. While it should not be considered"
        " the sole cause of the 2019–20 Australian bushfires, climate change is considered very"
        " likely to have contributed to the unprecedented extent and severity of those fires."
    ),
    (
        "climate-fever.57",
        2,
    ): (
        "As reported by Pat Michaels on his World Climate Report website, MacCracken stated during"
        ' the hearing that "the last decade is the warmest since 1400," implying the warming was'
        " caused by the greenhouse effect. In response to a question from Walker about the"
        " availability of thermometers in 1400, MacCracken explained the use of biological"
        " materials as temperature proxies. Michaels presented a version of the temperature graph"
        " (based on Bradley & Jones, 1993) with the instrumental temperature curve removed, and"
        ' argued that the "raw data" showed a significant temperature rise in the 1920s—before the'
        ' greenhouse effect had changed substantially—and claimed that "there’s actually been a'
        ' decline since then." He stated that this information had been shared with the Science'
        " Committee, along with a comment from one IPCC reviewer who suggested that the graph was"
        ' misleading and that the text should instead state: "Composite indicators of summer'
        " temperature show that a rapid rise occurred around 1920. This rise was prior to the major"
        " greenhouse emissions. Since then, composite temperatures have dropped slightly on a"
        ' decadal scale." Michaels criticized the IPCC for not incorporating this feedback in the'
        " final report, arguing that it left most readers with the impression that tree-ring data"
        " from the Northern Hemisphere reflected human impact on the atmosphere. He also questioned"
        " the strength of the scientific consensus, noting that only one reviewer out of 2,500"
        " scientists had flagged this issue."
    ),
    (
        "climate-fever.72",
        4,
    ): (
        "This network was used, in combination with satellite altimeter data, to establish that"
        " global mean sea-level rose 19.5 cm between 1870 and 2004 at an average rate of about 1.44"
        " mm/yr (1.7 mm/yr during the 20th century). Data collected by the Commonwealth Scientific"
        " and Industrial Research Organisation (CSIRO) in Australia show that the global mean sea"
        " level currently rises by 3.2 mm per year, at double the average 20th century rate. This"
        " is an important confirmation of climate change simulations which predicted that sea level"
        " rise would accelerate in response to climate change."
    ),
    (
        "climate-fever.67",
        1,
    ): (
        "This network was used, in combination with satellite altimeter data, to establish that"
        " global mean sea-level rose 19.5 cm between 1870 and 2004 at an average rate of about 1.44"
        " mm/yr (1.7 mm/yr during the 20th century). Data collected by the Commonwealth Scientific"
        " and Industrial Research Organisation (CSIRO) in Australia show that the global mean sea"
        " level currently rises by 3.2 mm per year, at double the average 20th century rate. This"
        " is an important confirmation of climate change simulations which predicted that sea level"
        " rise would accelerate in response to climate change."
    ),
}


def load_raw_file(name: str):
    fname = config.root_dir / "raw_data" / name
    with open(fname) as fp:
        return json.load(fp)


def normalize_question(question: str) -> str:
    return question.strip()


def parse_evidences(
    raw_sample: dict, topic_id: str, fixed_paragraph_id: int | None = None
) -> list[Evidence]:
    evidences = []
    for eid, evidence_text in enumerate(raw_sample["evidences"]):
        paragraph_id = raw_sample["paragraph_id"][eid]
        if paragraph_id is None:
            assert fixed_paragraph_id is not None
            logger.info(f"fixing None paragraph id for sample: {topic_id}")
            paragraph_id = fixed_paragraph_id

        assert evidence_text
        evidences.append(
            Evidence(
                text=evidence_text,
                label=raw_sample["labels"][eid],
                paragraph_id=paragraph_id,
            )
        )
    return evidences


def parse_evidence_ids(evidences: list[str]):
    return [int(eid) for eid in evidences if eid]


def parse_wh_answers(wh_answers: list[dict]) -> list[Answer]:
    return [
        Answer(
            answer=ans["answer"].strip(),
            evidence_ids=parse_evidence_ids(ans["evidenceNumbers"]),
        )
        for ans in wh_answers
    ]


def parse_yes_no_answers(yes_no_qa: dict) -> list[Answer]:
    answers = [
        Answer(
            answer=True,
            evidence_ids=parse_evidence_ids(yes_no_qa["yesAnswers"]),
        ),
        Answer(
            answer=False,
            evidence_ids=parse_evidence_ids(yes_no_qa["noAnswers"]),
        ),
    ]
    answers = [ans for ans in answers if ans.evidence_ids]
    return answers


def parse_button_states(button_states: dict) -> list[tuple[int, int]]:
    conflicting_answer_pairs = []
    for btn_state, is_conflict in button_states.items():
        if not is_conflict:
            continue
        ans1, ans2 = [
            int(button_id) for button_id in re.fullmatch(r"btn(\d+)-(\d+)", btn_state).groups()
        ]
        conflicting_answer_pairs.append((ans1 - 1, ans2 - 1))  # 0 indexed numbers
    return conflicting_answer_pairs


def parse_conflict_annotations(raw_sample: dict, topic_id: str) -> list[QASample]:
    paragraphs = raw_sample["paragraphs"]
    fixed_paragraph_id = DATA_FIXER_MISSING_PARAGRAPH_ID.get(topic_id)
    for i, p in enumerate(paragraphs):
        if p is None:
            assert (
                fixed_paragraph_id is None
            ), "cannot fix more than a single paragraph per instance"
            paragraphs[i] = DATA_FIXER[(topic_id, i)]
            fixed_paragraph_id = i

    evidences = parse_evidences(raw_sample, topic_id, fixed_paragraph_id)

    # parse the WH question
    wh_answers = parse_wh_answers(raw_sample["wh_answers"])

    # parse conflict states
    conflicting_answer_pairs = parse_button_states(raw_sample["button_states"])

    # either True/False or yes/no
    is_presupposition = raw_sample["is_presupposition"]
    if isinstance(is_presupposition, str):
        if raw_sample["is_presupposition"].lower() == "yes":
            is_presupposition = True
        elif raw_sample["is_presupposition"].lower() == "no":
            is_presupposition = False

    if not isinstance(is_presupposition, bool):
        logger.warning(
            f"{topic_id}: has non boolean is_presupposition value -"
            f" {raw_sample['is_presupposition']!r}"
        )
        is_presupposition = False

    qa_samples = [
        QASample(
            question_id=f"{topic_id}.1",
            topic_id=topic_id,
            question=normalize_question(raw_sample["wh_question"]),
            paragraphs=deepcopy(paragraphs),
            evidences=deepcopy(evidences),
            answer_type="Conflict",
            answers=wh_answers,
            conflicting_answer_pairs=conflicting_answer_pairs,
            question_type=f"wh-{raw_sample['wh_question_type']}",
            is_presupposition=is_presupposition,
            raw_sample=deepcopy(raw_sample),
        )
    ]

    # parse Yes/No (boolean) questions, can be multiple such
    for q_id, yes_no_qa in enumerate(raw_sample["yes_no_questions"], start=2):
        question_id = f"{topic_id}.{q_id}"
        answers = parse_yes_no_answers(yes_no_qa)

        assert len(answers) == 2
        assert (
            answers[0].evidence_ids and answers[1].evidence_ids
        ), f"No conflicting evidences: {topic_id}"

        qa_samples.append(
            QASample(
                question_id=question_id,
                topic_id=topic_id,
                question=normalize_question(yes_no_qa["question"]),
                paragraphs=deepcopy(paragraphs),
                evidences=deepcopy(evidences),
                answer_type="Conflict",
                answers=answers,
                question_type="boolean",
                # Exactly two answers (True & False), both necessarily conflicting.
                conflicting_answer_pairs=[(0, 1)],
                is_presupposition=None,  # not relevant for yes/no questions
                raw_sample=deepcopy(raw_sample),
            )
        )
    return qa_samples


def parse_neutral_annotations(raw_sample: dict, topic_id: str) -> list[QASample]:
    paragraphs = raw_sample["paragraphs"]
    evidences = parse_evidences(raw_sample, topic_id)
    neutral_samples = []
    question_id = 1

    if raw_sample["wh_question_type"] == "function":
        question_type = "wh-function"
    elif raw_sample["wh_question_type"] == "list":
        question_type = "wh-list"
    else:
        raise ValueError(f"Unknown question type")

    wh_sample = QASample(
        question_id=f"{topic_id}.{question_id}",
        topic_id=topic_id,
        question=normalize_question(raw_sample["wh_question"]),
        paragraphs=deepcopy(paragraphs),
        evidences=deepcopy(evidences),
        answer_type="Neutral",
        answers=[],  # neutral has no answers
        question_type=question_type,
        conflicting_answer_pairs=[],
        is_presupposition=None,
        raw_sample=deepcopy(raw_sample),
    )
    neutral_samples.append(wh_sample)
    question_id += 1

    for question in raw_sample["yes_no_questions"]:
        assert question["noAnswers"] in ([], [""])
        assert question["yesAnswers"] in ([], [""])
        bool_sample = QASample(
            question_id=f"{topic_id}.{question_id}",
            topic_id=topic_id,
            question=normalize_question(question["question"]),
            paragraphs=deepcopy(paragraphs),
            evidences=deepcopy(evidences),
            answer_type="Neutral",
            answers=[],  # neutral has no answers
            question_type="boolean",
            is_presupposition=None,
            conflicting_answer_pairs=[],
            raw_sample=deepcopy(raw_sample),
        )
        neutral_samples.append(bool_sample)
        question_id += 1

    return neutral_samples


def parse_support_annotation(raw_sample: dict, topic_id: str) -> list[QASample]:
    paragraphs = raw_sample["paragraphs"]
    evidences = parse_evidences(raw_sample, topic_id)

    wh_answers = parse_wh_answers(raw_sample["wh_answers"])
    assert len(wh_answers) >= 2, "must have at least one answer"
    assert not any(raw_sample["button_states"].values()), "found conflicting answers in support"

    # WH question
    support_samples = [
        QASample(
            question_id=f"{topic_id}.1",
            topic_id=topic_id,
            question=normalize_question(raw_sample["wh_question"]),
            paragraphs=deepcopy(paragraphs),
            evidences=deepcopy(evidences),
            answer_type="Support",
            answers=wh_answers,
            question_type=f"wh-{raw_sample['wh_question_type']}",
            conflicting_answer_pairs=[],
            is_presupposition=None,  # not relevant / not tagged for support
            raw_sample=deepcopy(raw_sample),
        )
    ]

    # Yes/no questions
    for qid, yes_no_qa in enumerate(raw_sample["yes_no_questions"], start=2):
        answers = parse_yes_no_answers(yes_no_qa)
        assert len(answers) == 1
        support_samples.append(
            QASample(
                question_id=f"{topic_id}.{qid}",
                topic_id=topic_id,
                question=normalize_question(yes_no_qa["question"]),
                paragraphs=deepcopy(paragraphs),
                evidences=deepcopy(evidences),
                answer_type="Support",
                answers=answers,
                question_type="boolean",
                conflicting_answer_pairs=[],
                is_presupposition=None,  # not relevant for yes/no questions
                raw_sample=deepcopy(raw_sample),
            )
        )
    return support_samples


def validate_sample_quality_checks(sample: QASample):
    if not all(p for p in sample.paragraphs):
        raise ValueError(f"sample {sample.question_id} with empty / None paragraph")

    if not all(e.paragraph_id >= 0 for e in sample.evidences):
        raise ValueError(f"sample {sample.question_id} with evidence(s) not found in sources")

    if not all(a.evidence_ids for a in sample.answers):
        raise ValueError(f"sample {sample.question_id} has answer with zero evidence")


def load_data(strict: bool) -> list[QASample]:
    fnames = [
        "healthver_climate-fever_conflcits_with_questions_and_answers.json",
        "healthver_climate-fever_neutral_with_questions_annotated.json",
        "healthver_climate-fever_support_with_questions_and_answers.json",
    ]

    logger.info(f"Loading conflict annotations")
    conflict_qas = []
    last_row_id = 0
    for row_id, row in enumerate(load_raw_file(fnames[0]), start=1):
        topic_id = f'{row["dataset_name"]}.{row_id}'
        conflict_qas += parse_conflict_annotations(row, topic_id)
        last_row_id = row_id

    neutral_qas = []
    logger.info(f"Loading neutral annotations")
    for row_id, row in enumerate(load_raw_file(fnames[1]), start=last_row_id + 1):
        topic_id = f'{row["dataset_name"]}.{row_id}'
        neutral_qas += parse_neutral_annotations(row, topic_id)
        last_row_id = row_id

    support_qas = []
    logger.info(f"Loading support annotations")
    for row_id, row in enumerate(load_raw_file(fnames[2]), start=last_row_id + 1):
        topic_id = f'{row["dataset_name"]}.{row_id}'
        support_qas += parse_support_annotation(row, topic_id)

    dataset = conflict_qas + neutral_qas + support_qas
    Random(42).shuffle(dataset)
    assert len(dataset) == len({x.question_id for x in dataset})

    total_samples = len(dataset)
    if strict:
        valid_samples = []
        for sample in dataset:
            try:
                validate_sample_quality_checks(sample)
                valid_samples.append(sample)
            except ValueError as e:
                logger.warning(f"skipping invalid response: {e!r}")
        dataset = valid_samples

    logger.info(f"loaded {len(dataset)}/{total_samples} valid QA samples")
    qa_types = dict(Counter(f"{s.question_type},{s.answer_type}" for s in dataset))
    logger.info(f"loaded {qa_types} samples")
    return dataset


def save_contraqa_dataset(qa_samples: list[QASample], version: str):
    dataset_dir = config.root_dir / "data" / version
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hash_id = sha256(b"Barkuni")

    for sample in qa_samples:
        hash_id.update(sample.question_id.encode())

    with open(dataset_dir / "ContraQA.json", "w") as fp:
        dataset_dict = {
            "version": version,
            "contamination_id": hash_id.hexdigest(),
            "ContraQA": [qa.model_dump() for qa in qa_samples],
        }
        json.dump(dataset_dict, fp, indent=2)


def load_dataset() -> list[QASample]:
    """Loads ContraQA dataset"""
    with open(config.data_dir / "v1.0" / "ContraQA.json") as fp:
        return [QASample.model_validate(x) for x in json.load(fp)["ContraQA"]]


if __name__ == "__main__":  # pragma: nocover
    log.init("INFO")

    dataset = load_data(strict=True)
    # save_contraqa_dataset(qa_samples=dataset, version="v1.0")

    # Neutral
    rich.print(dataset[0].model_dump(exclude={"raw_sample", "paragraphs"}))
    # Conflict
    rich.print(dataset[4].model_dump(exclude={"raw_sample", "paragraphs"}))
    # Support
    rich.print(dataset[1].model_dump(exclude={"raw_sample", "paragraphs"}))
