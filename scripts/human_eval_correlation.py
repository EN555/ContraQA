import ast
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from contraqa import dataset_loader, evaluate
from contraqa.evaluate import compute_f1
from contraqa.utils import log
from contraqa.utils.configs import config

from scipy.stats import pearsonr, spearmanr

from contraqa.utils.hungarian_rouge import average_hungarian_rouge


def load_data():
    df = pd.read_json(config.data_dir / "data_for_correlation_120_annotated_samples.json")

    assert len(set(df[["question_id", "prompt_template", "model"]].itertuples(index=False))) == len(df), "must be unique"

    df.set_index(["question_id", "prompt_template", "model"], inplace=True)
    df.rename(columns={
        # Recall metrics
        "Found Reference Answers in Model Answer (Recall Answer)": "human_answers_entailed",
        "Conflicting Reference Answer Pairs Found (Recall Conflict)": "human_conflict_pairs_found",
        # Decomposition
        "Model Answers Decomposed": "human_model_answers_decomposed",
        # Precision
        "Found Model Sub-Answers in Reference Answers/Passages (Precision Answer)": "human_sub_answer_to_entailed",
        "Found Conflicting Model's Sub-Answers (Conflict Precision)": "human_conflict_sub_answer_pairs_found",
        # Input Fields
        "question": "question",
        "model_answer.1": "model_answer",
    }, inplace=True)

    df["human_answers_entailed"] = df["human_answers_entailed"].apply(ast.literal_eval)
    df["human_conflict_pairs_found"] = df["human_conflict_pairs_found"].map(
        {None: [], '(0, 1)': [(0, 1)], '(0,1)': [(0, 1)], '(0, 2),(1, 2)': [(0, 2), (1, 2)]}
    ).apply(lambda d: d if isinstance(d, list) else [])

    def _parse_subanswers(x: str) -> list[str]:
        items = []
        for expected_id, line in enumerate(x.splitlines()):
            item_id, txt = line.split(':', 1)
            assert int(item_id) == expected_id
            items.append(txt.strip())
        return items

    df["human_model_answers_decomposed"] = df["human_model_answers_decomposed"].apply(_parse_subanswers)

    df["human_conflict_sub_answer_pairs_found"] = df["human_conflict_sub_answer_pairs_found"].map({
        '[(0, 1) :1]': {(0, 1) :1},
        '(0, 1) :1': {(0, 1) :1},
        '[0, 1) :1]': {(0, 1) :1},
        '[]': {},
        '[(0, 1) :0]': {(0, 1) :0},
    }).apply(lambda d: d if isinstance(d, dict) else {})

    df["human_sub_answer_to_entailed"] = df["human_sub_answer_to_entailed"].apply(ast.literal_eval)
    return df


def _do_eval(sample, row, judge_model):
    assert sample.question == row["question"]
    return evaluate.evaluate_model_answer(
        qa_sample=sample,
        input_sources=sample.paragraphs,
        model_answer=row["model_answer"],
        judge_model=judge_model,
        answer_quality_only=False,
    )


def add_llm_as_a_judge_to_df(df: pd.DataFrame, judge_model = "o4-mini-high") -> pd.DataFrame:
    dataset = dataset_loader.load_dataset()
    id2sample = {x.question_id: x for x in dataset}
    assert len(id2sample) == len(dataset)

    new_data = []

    # init cache
    with ThreadPoolExecutor(max_workers=len(df)) as executor:
        futs = []
        for (question_id, prompt_template, model), row in df.iterrows():
            futs.append(executor.submit(_do_eval, id2sample[question_id], row, judge_model))
        for f in futs:
            f.result()

    for row_id, ((question_id, prompt_template, model), row) in enumerate(df.iterrows()):
        sample = id2sample[question_id]
        assert sample.question == row["question"]
        auto_eval = evaluate.evaluate_model_answer(
            qa_sample=sample,
            input_sources=sample.paragraphs,
            model_answer=row["model_answer"],
            judge_model=judge_model,
            answer_quality_only=False,
        )

        # Recall
        human_answer_recall = len(row["human_answers_entailed"]) / len(sample.answers)
        try:
            human_conflict_pairs_recall = len(row["human_conflict_pairs_found"]) / len(sample.conflicting_answer_pairs)
        except ZeroDivisionError:
            human_conflict_pairs_recall = float("nan")

        # Precision
        human_answer_precision = len(row["human_sub_answer_to_entailed"]) / len(row["human_model_answers_decomposed"])
        # assert len(row["human_conflict_sub_answer_pairs_found"]) == len(row["human_model_answers_decomposed"]) > 0

        try:
            human_conflict_pairs_precision = sum(row["human_conflict_sub_answer_pairs_found"].values()) / len(row["human_conflict_sub_answer_pairs_found"])
        except ZeroDivisionError:
            human_conflict_pairs_precision = float("nan")

        human_conflict_pairs_f1 = compute_f1(human_conflict_pairs_precision, human_conflict_pairs_recall)
        human_answer_f1 = compute_f1(human_answer_precision, human_answer_recall)

        model_answers_decomposed = list(auto_eval["sub_answer_to_entailed"].keys())
        new_data.append({
            "question_id": question_id,
            "prompt_template": prompt_template,
            "model": model,
            "answer_type": sample.answer_type,
            "human_answer_recall": human_answer_recall,
            "human_answer_precision": human_answer_precision,
            "human_answer_f1": human_answer_f1,
            "human_conflict_pairs_recall": human_conflict_pairs_recall,
            "human_conflict_pairs_precision": human_conflict_pairs_precision,
            "human_conflict_pairs_f1": human_conflict_pairs_f1,
            **row.to_dict(),
            **auto_eval,
        })
    return pd.DataFrame(new_data)


def main():
    raw_df = load_data()
    # raw_df = raw_df[raw_df["human_conflict_sub_answer_pairs_found"].apply(bool)]
    print(raw_df.index)

    df = add_llm_as_a_judge_to_df(raw_df)
    print(df)

    all_corr_results = []
    for key in ["answer_recall", "conflict_pairs_recall", "answer_precision", "conflict_pairs_precision", "conflict_pairs_f1", "answer_f1"]:
        human_key = f"human_{key}"
        samples = df[[key, human_key]].copy().dropna()
        print(f'\n\n=== {key} num samples: {len(samples)} ===')
        print(f"human (avg={samples[human_key].mean():.2f}): ", samples[human_key].round(2).values.tolist())
        print(f"model (avg={samples[key].mean():.2f}): ", samples[key].values.round(2).tolist())
        delta_vals = (samples[human_key] - samples[key]).values
        print(f"delta h-m. sum={delta_vals.sum():+.2f}: ", delta_vals.round(2).tolist())
        spearman = spearmanr(samples[human_key], samples[key])
        pearson = pearsonr(samples[human_key], samples[key])
        print('Spearman:', spearman, 'Pearson:', pearson)
        print()
        all_corr_results.append({
            "name": key,
            "spearman": spearman[0],
            "spearman_p_value": spearman[1],
            "pearson": pearson[0],
            "pearson_p_value": pearson[1],
            "human_score": samples[human_key].mean().round(3),
            "model_score": samples[key].mean().round(3),
        })

    key = "model_answers_decomposed"
    human_key = f"human_{key}"
    samples = df[[key, human_key]].copy().dropna()
    print(f'\n\n=== {key} num samples: {len(samples)} ===')
    print("human: ", samples[human_key].apply(len).values.tolist())
    print("model: ", samples[key].apply(len).values.tolist())
    scores = [round(average_hungarian_rouge(row[human_key], row[key]), 2) for _, row in samples.iterrows()]
    avg = sum(scores) / len(scores)
    print(f"ROUGE scores (avg = {avg:.3f}):", scores)

    all_corr_results.append({
        "name": key,
        "rouge": round(avg, 3),
    })

    all_corr_df = pd.DataFrame(all_corr_results).round(5)
    df.to_csv("human_df.csv", index=False)
    all_corr_df.to_csv("human_all_corr_df.csv", index=False)


if __name__ == '__main__':
    log.init()
    main()
