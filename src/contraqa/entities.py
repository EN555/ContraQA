from typing import Literal

from pydantic import BaseModel


class Evidence(BaseModel):
    text: str
    label: Literal["Supports", "Refutes", "Neutral"]
    paragraph_id: int


class Answer(BaseModel):
    answer: str | bool
    evidence_ids: list[int]

    def as_string(self):
        if isinstance(self.answer, str):
            return self.answer
        return "yes" if self.answer else "no"


ANSWER_TYPES = Literal["Conflict", "Support", "Neutral"]
QUESTION_TYPES = Literal["wh-function", "wh-list", "boolean"]


class QASample(BaseModel):
    question_id: str
    topic_id: str  # same id for the same paragraphs
    # Input fields
    question: str
    paragraphs: list[str]
    evidences: list[Evidence]
    # Output fields
    answer_type: ANSWER_TYPES
    answers: list[Answer]
    # Metadata fields
    conflicting_answer_pairs: list[tuple[int, int]]
    question_type: QUESTION_TYPES
    is_presupposition: bool | None  # True False or `None` for unknown / not relevant
    raw_sample: dict

    def get_answer_pairs(self, is_conflict: bool) -> list[tuple[Answer, Answer]]:
        return [
            (ans_i, ans_j)
            for i, ans_i in enumerate(self.answers)
            for j, ans_j in enumerate(self.answers[i + 1 :], start=i + 1)
            if ((i, j) in self.conflicting_answer_pairs) == is_conflict
        ]
