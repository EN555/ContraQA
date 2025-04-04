from typing import List

import numpy as np
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment

ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def rouge2_f1(a: str, b: str) -> float:
    """
    Compute ROUGE-2 F1 for a single pair, using the google-research scorer.
    """
    return ROUGE_SCORER.score(a, b)["rouge1"].fmeasure


def average_hungarian_rouge(A: list[str], B: list[str]) -> float:
    """
    1. Build the full |A|×|B| ROUGE-2 F1 matrix with google-research ROUGE.
    2. Find the best one-to-one mapping via the Hungarian algorithm
       (maximising ROUGE ⇒ minimising 1-score).
    3. Return the average over max(|A|, |B|); unmatched items count as 0.
    """
    if not A and not B:
        return 0.0

    m, n = len(A), len(B)
    score_matrix = np.empty((m, n), dtype=np.float32)

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            score_matrix[i, j] = rouge2_f1(a, b)

    # Hungarian assignment – turn maximisation into minimisation
    rows, cols = linear_sum_assignment(1.0 - score_matrix)
    matched_scores = score_matrix[rows, cols]

    total_pairs = max(m, n)
    return float(matched_scores.sum() / total_pairs)


if __name__ == "__main__":
    A1 = ["the cat sat on the mat", "quick brown fox"]
    A2 = ["the cat sat on the mat", "quick brown fox"]
    B1 = ["a quick brown fox", "cat on mat"]
    B2 = ["a quick brown fox jumped", "cat on"]

    for A, B in [(A1, B1), (A2, B2), (A1, B2), (A2, B1), (B1, B2)]:
        avg = average_hungarian_rouge(A, B)
        print("===")
        print(A)
        print(B)
        print(f"Average ROUGE-2 F1: {avg:.4f}")
