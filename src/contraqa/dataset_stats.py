import pandas as pd
import tiktoken
from nltk import word_tokenize

from contraqa import predict, dataset_loader
from contraqa.entities import QASample


def print_dataset_stats(dataset: list[QASample]):
    df = pd.DataFrame.from_records([d.model_dump() for d in dataset])
    print("Dataset Stats:")
    print("Total Samples", len(df))
    for key in [
        "answer_type",
        "question_type",
        ["answer_type", "question_type"],
        "is_presupposition",
    ]:
        print(f"Total Samples per {key}:")
        print(df[key].value_counts())
        print("===")
    print(df.groupby(["answer_type"])["topic_id"].nunique())
    print("Average number of passages:", df["paragraphs"].apply(len).mean().round(3))
    print()
    print(
        "Average passage tokens count:",
        df["paragraphs"].explode().apply(lambda p: len(word_tokenize(p))).mean().round(3),
    )
    print()
    print(
        "Average instance passages tokens count:",
        df["paragraphs"].apply(lambda ps: sum([len(word_tokenize(p)) for p in ps])).mean().round(3),
    )
    print()
    print("Average number of answers:", df["paragraphs"].apply(len).mean().round(3))


def count_dataset_tokens(dataset: list[QASample], model: str = "o3"):
    dataset = [x for x in dataset if x.answer_type != "Neutral"]

    encoding = tiktoken.encoding_for_model(model)
    for prompt_template in ["Normal", "ContradictAware"]:
        prompts = [predict.format_sample(s, prompt_template, "passages") for s in dataset]
        tokens = sum([len(encoding.encode(p)) for p in prompts])
        print(f"For model {model} with prompt type {prompt_template} total tokens are: {tokens:,}")


def main():
    dataset = dataset_loader.load_dataset()
    print_dataset_stats(dataset)
    count_dataset_tokens(dataset)


if __name__ == "__main__":
    main()
