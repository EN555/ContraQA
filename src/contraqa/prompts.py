# Prompts adapted from Figure 3, templates 1, 4, 5 (2 & 3 are not relevant)
WIKI_CONTRADICT_PROMPTS = {
    "QuestionOnly": "Provide a short answer for the following question based on your internal knowledge.\nQuestion: {question}",
    "Normal": (
        "Provide a short answer for the following question based on the given context.\n"
        "Question: {question}\n"
        "Context: {paragraphs}\n"
    ),
    "ContradictAware": (
        "Provide a short answer for the following question based on the given context. "
        "Carefully investigate the given context and provide a concise response that reflects "
        "the comprehensive view of the context, even if the answer contains contradictory "
        "information reflecting the heterogeneous nature of the context.\n"
        "Question: {question}\n"
        "Context:\n\n{paragraphs}\n"
    ),
}

CONTRAQA_PROMPTS = {
    "Normal": (
        "Provide a concise, single-sentence answer that includes every distinct answer to the following question, based on the given passages from multiple sources.\n"
        "Question: {question}\n"
        "Passages: {paragraphs}\n"
    ),
    "ContradictAware": (
        "Provide a concise, single-sentence answer that includes every distinct answer to the following question, based on the given passages from multiple sources. "
        # "Carefully investigate these passages and provide a concise answer that includes every distinct answer found. "
        "If any answers conflict, clearly indicate which ones are in conflict while remaining objective and neutral.\n"
        "Question: {question}\n"
        "Passages: {paragraphs}\n"
    ),
}


if __name__ == "__main__":  # pragma: nocover
    for k, v in CONTRAQA_PROMPTS.items():
        print(k, v, sep=":\n\n")
