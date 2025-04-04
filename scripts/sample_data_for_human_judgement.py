from random import Random

import pandas as pd

from contraqa.utils.configs import config

SELECTED_MODELS = [
    "o3-high",
    "GPT-4o",
    "deepseek-r1",
    "deepseek-v3",
]
N_SAMPLES = 20


def main():
    fname = config.results_dir / "all8-models-conflict-eval" / "full-results.csv"
    outfile = config.results_dir / "sample_data_for_human_judgement.csv"
    df = pd.read_csv(fname)

    question_ids = sorted(df["question_id"].unique())
    selected_question_ids = Random(1337).sample(question_ids, k=N_SAMPLES)
    sampled_df = df[df["question_id"].isin(selected_question_ids) & df["model"].isin(SELECTED_MODELS)].copy()
    assert len(sampled_df) == N_SAMPLES * len(SELECTED_MODELS) * 2
    sampled_df.to_csv(outfile, index=False)


if __name__ == '__main__':
    main()
