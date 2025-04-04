import argparse
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from contraqa import prompts, models, evaluate, dataset_loader, confidence_intervals
from contraqa.entities import QASample, ANSWER_TYPES, QUESTION_TYPES
from contraqa.utils import log

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"


def format_sample(
    sample: QASample,
    prompt_name: str,
    prompt_templates: dict | None = None,
) -> str:
    prompt_templates = prompt_templates or prompts.CONTRAQA_PROMPTS
    template = prompt_templates[prompt_name]
    paragraphs = evaluate.format_passages(sample.paragraphs)
    return template.format(question=sample.question, paragraphs=paragraphs)


def predict_qa(
    qa_sample: QASample,
    model: str,
    sid: int,
    prompt_template: str,
    perform_evaluation: str,
    judge_model: str | None = None,
) -> tuple[str] | tuple[str, dict]:
    logger.info(f"predicting sample #{sid} with model {model}")
    prompt = format_sample(qa_sample, prompt_template)
    logger.debug(f"prompt: {prompt}")

    resp = models.completion(prompt=prompt, system=DEFAULT_SYSTEM_PROMPT, model=model)
    logger.info(
        f"#{sid}: {qa_sample.question_id} ({qa_sample.question_type}, {qa_sample.answer_type}) -"
        f" {qa_sample.question!r} (reference answers:"
        f" {', '.join(repr(a.answer) for a in qa_sample.answers)}) model answer: {resp!r}"
    )
    logger.debug(
        "evidences:"
        f" {'\n'.join([f'P{e.paragraph_id} ({e.label}): {e.text!r}' for e in qa_sample.evidences])}"
    )

    if perform_evaluation != "none":
        if not judge_model:
            raise ValueError("For evaluation, please provide a judge_model")

        eval_scores = evaluate.evaluate_model_answer(
            qa_sample=qa_sample,
            input_sources=qa_sample.paragraphs,
            model_answer=resp,
            judge_model=judge_model,
            answer_quality_only=perform_evaluation == "answer-quality",
        )
        return resp, eval_scores
    else:
        return (resp,)


def predict(
    model: str,
    prompt_template: str = "Normal",
    num_samples: int | None = None,
    answer_types: list[ANSWER_TYPES] | None = None,
    question_types: list[QUESTION_TYPES] | None = None,
    partially_conflicting_answers_only: bool = False,
    perform_evaluation: Literal["none", "answer-quality", "all"] = "none",
    judge_model: str | None = None,
    max_workers: int = 8,
):
    dataset = dataset_loader.load_dataset()
    logger.info(
        f"predicting {num_samples} samples with model {model} in {prompt_template} mode on sets"
        f" {answer_types}"
    )

    if answer_types is not None:
        dataset = [x for x in dataset if x.answer_type in answer_types]

    if question_types is not None:
        dataset = [x for x in dataset if x.question_type in question_types]

    if partially_conflicting_answers_only:
        if answer_types != ["Conflict"]:
            raise ValueError("conflicting answer pairs only supported for conflicts.")
        dataset = [x for x in dataset if x.get_answer_pairs(True) and x.get_answer_pairs(False)]

    if num_samples:
        dataset = dataset[:num_samples]

    model_answers = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sid = {
            executor.submit(
                predict_qa,
                qa_sample=qa_sample,
                model=model,
                sid=sid,
                prompt_template=prompt_template,
                perform_evaluation=perform_evaluation,
                judge_model=judge_model,
            ): (sid, qa_sample)
            for sid, qa_sample in enumerate(dataset, start=1)
        }

        # As each task completes, grab its result
        for future in tqdm(
            as_completed(future_to_sid), total=len(future_to_sid), desc=f"{model}-{prompt_template}"
        ):
            sid, qa_sample = future_to_sid[future]
            results = future.result()
            logger.info(f"done with #{sid}: {qa_sample.question_id}")
            model_answers.append((qa_sample, *results))

    return model_answers


def answer_to_csv_row(qa_sample: QASample, model, prompt, answer: str, **ctx) -> dict:
    return {
        "question_id": qa_sample.question_id,
        # configuration
        "model": model,
        "prompt_template": prompt,
        # main metadata
        "question_type": qa_sample.question_type,
        "answer_type": qa_sample.answer_type,
        "is_presupposition": qa_sample.is_presupposition,
        "partially_conflicting": bool(
            qa_sample.get_answer_pairs(True) and qa_sample.get_answer_pairs(False)
        ),
        **ctx,
        # input output
        "question": qa_sample.question,
        "model_answer": answer,
        # other (lengthy) metadata
        "reference_answers": "\n".join(str(a.answer) for a in qa_sample.answers),
        "conflicting_answer_pairs": ",".join(str(x) for x in qa_sample.conflicting_answer_pairs),
        # Context:
        "evidences": "\n".join(
            f"#P-{e.paragraph_id}({e.label[:3]}): {e.text}" for e in qa_sample.evidences
        ),
        "paragraphs": "\n\n".join(str(p) for p in qa_sample.paragraphs),
    }


def make_results_df(answer_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(answer_rows)
    df.sort_values(
        [
            "question_type",
            "answer_type",
            "is_presupposition",
            "question_id",
            "model",
            "prompt_template",
        ],
        inplace=True,
    )
    return df


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(
        description="Run predictions over models, prompts, and answer types."
    )

    # multi-option flags for lists
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        # default=["o3-high", "gemini-2.5-pro"],
        help="One or more model names to evaluate. supported models: GPT-4O, o4-mini-high, DeepSeek-R1, qwen3-235B and gemini-2.5-pro.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="The name of the output directory."
    )
    parser.add_argument(
        "-p",
        "--prompt-templates",
        nargs="+",
        default=["Normal", "ContradictAware"],
        help="One or more prompt templates to use.",
    )
    parser.add_argument(
        "-a",
        "--answer-types",
        nargs="+",
        default=["Conflict", "Support"],
        help="One or more answer types (e.g. Conflict, Support).",
    )

    # single-value flags
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to draw (default: None = all).",
    )
    parser.add_argument(
        "-e",
        "--perform-evaluation",
        choices=["none", "answer-quality", "all"],
        default="none",
        metavar="MODE",
        help=(
            "Select evaluation mode after prediction: "
            "'none' (default) skips evaluation, "
            "'answer-quality' runs the answer-quality evaluations only, "
            "'all' runs both evaluations."
        ),
    )
    parser.add_argument(
        "-w", "--max-workers", type=int, default=1, help="Max number of parallel workers."
    )
    parser.add_argument(
        "-j",
        "--judge-model",
        type=str,
        default="o4-mini-high",
        help="The evaluator model for LLM-as-a-judge.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    all_answers_csv = []
    all_answers_json = []

    for model in args.models:
        for prompt in args.prompt_templates:
            exc = None
            for retry in range(1000):
                try:
                    answers = predict(
                        model=model,
                        prompt_template=prompt,
                        num_samples=args.num_samples,
                        answer_types=args.answer_types,
                        partially_conflicting_answers_only=False,
                        perform_evaluation=args.perform_evaluation.lower(),
                        judge_model=args.judge_model,
                        max_workers=args.max_workers,
                    )
                    break
                except OSError as e:
                    logger.error(f"failed with {e!r}, waiting and retrying in 60 secs")
                    time.sleep(60)
                    exc = e
                    continue
            else:
                raise exc

            for qa_sample, *ans_and_eval_scores in answers:
                if args.perform_evaluation != "none":
                    ans, eval_scores = ans_and_eval_scores
                else:
                    ans = ans_and_eval_scores
                    eval_scores = {}

                all_answers_json.append(
                    {
                        # configuration
                        "model": model,
                        "prompt_template": prompt,
                        "model_answer": ans,
                        "sample": qa_sample.model_dump(mode="json"),
                        **eval_scores,
                    }
                )
                all_answers_csv.append(
                    answer_to_csv_row(
                        qa_sample,
                        model,
                        prompt,
                        ans,
                        **eval_scores,
                    )
                )

    # Save results into 3 files
    df = make_results_df(all_answers_csv)
    summary_df = create_summary_df(df)
    json_df = pd.DataFrame(all_answers_json)

    df = df.round(3)
    summary_df = summary_df.round(3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing results to {output_dir}")
    df.to_csv(output_dir / "full-results.csv", index=False)
    summary_df.to_csv(output_dir / "results-summary.csv", index=False)
    json_df.to_json(output_dir / "full-results.jsonl.gz", compression="gzip", orient="records")


def create_summary_df(results_df: pd.DataFrame) -> pd.DataFrame:
    agg_keys = ["model", "prompt_template", "answer_type"]
    score_keys = [
        "answer_recall",
        "answer_precision",
        "answer_f1",
        "no_answer_score",
        "conflict_pairs_recall",
        "conflict_pairs_precision",
        "conflict_pairs_f1",
        "strict_conflict_pairs_recall",
        "strict_conflict_pairs_f1",
    ]
    score_keys = [key for key in score_keys if key in results_df.columns]
    results_df[score_keys] = results_df[score_keys] * 100
    df = results_df[agg_keys + score_keys].groupby(agg_keys)[score_keys].agg(np.nanmean)

    # Compute Paired and Unpaired significant tests
    alpha = 0.05
    for key in score_keys:
        df[f"{key}_ci_paired"] = "MISSING"

        # Iterate over all unique index keys - i.e. all experiments
        for (
            model,
            _,
            answer_type,
        ) in df.index:  # df[agg_keys].drop_duplicates().itertuples(index=False):
            # select the rows of the relevant experiment
            exp_data_normal = results_df[
                (results_df["model"] == model)
                & (results_df["prompt_template"] == "Normal")
                & (results_df["answer_type"] == answer_type)
            ][key].values.copy()
            exp_data_confaware = results_df[
                (results_df["model"] == model)
                & (results_df["prompt_template"] == "ContradictAware")
                & (results_df["answer_type"] == answer_type)
            ][key].values.copy()

            # compute CIs for the selected experiment
            f1_value_normal, ci_normal, _ = confidence_intervals.bootstrap_mean(
                exp_data_normal, alpha=alpha
            )
            f1_value_confaware, ci_confaware, _ = confidence_intervals.bootstrap_mean(
                exp_data_confaware, alpha=alpha
            )
            expected_metric_score_normal = df.loc[model, "Normal", answer_type][key]
            expected_metric_score_confaware = df.loc[model, "ContradictAware", answer_type][key]

            if pd.isna(expected_metric_score_normal) or pd.isna(f1_value_normal):
                significant = "NAN"
            else:
                assert math.isclose(f1_value_normal, expected_metric_score_normal)
                assert math.isclose(f1_value_confaware, expected_metric_score_confaware)

                significant = confidence_intervals.paired_test_mean(
                    exp_data_confaware,
                    exp_data_normal,
                    "ContradictAware",
                    "Normal",
                    unpaired_alpha=alpha,
                )

            for prompt_template in ["Normal", "ContradictAware"]:
                df.loc[(model, prompt_template, answer_type), f"{key}_ci_paired"] = significant

    # Compute unpaired results
    for key in score_keys:
        df[f"{key}_ci"] = np.nan

        # Iterate over all unique index keys - i.e. all experiments
        for (
            model,
            prompt_template,
            answer_type,
        ) in df.index:  # df[agg_keys].drop_duplicates().itertuples(index=False):
            # select the rows of the relevant experiment
            exp_data = results_df[
                (results_df["model"] == model)
                & (results_df["prompt_template"] == prompt_template)
                & (results_df["answer_type"] == answer_type)
            ][key].values.copy()

            # compute CIs for the selected experiment
            f1_value, ci, ci_res = confidence_intervals.bootstrap_mean(exp_data, alpha=0.05)

            expected_metric_score = df.loc[model, prompt_template, answer_type][key]
            if pd.isna(expected_metric_score) or pd.isna(f1_value):
                continue
            assert math.isclose(
                f1_value, expected_metric_score
            ), f"{f1_value} != {expected_metric_score} for {model}, {prompt_template}, {answer_type}"

            df.loc[(model, prompt_template, answer_type), f"{key}_ci"] = ci

    df = df.reset_index()
    return df


if __name__ == "__main__":  # pragma: nocover
    log.init("INFO")
    main()
    # main(shlex.split('--models o3-high GPT-4o gemini-2.5-pro gemini-2.0-flash deepseek-r1 deepseek-v3 qwen3-235b qwen2.5-72b --answer-types "Conflict" --prompt-templates "Normal" "ContradictAware" --perform-eval --output-dir results/dummy'))
