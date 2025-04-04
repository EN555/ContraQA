import time
from functools import cache

from together import Together

from contraqa.utils.configs import config
from contraqa.utils.diskcache import disk_cache

DEEPSEEK_NAME_TO_MODEL = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
}
MODEL_QWEN3_235B = "Qwen/Qwen3-235B-A22B-fp8"
QWEN_NAME_TO_MODEL = {
    "qwen3-235b": "Qwen/Qwen3-235B-A22B-fp8",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
}


@cache
def get_client() -> Together:
    return Together(api_key=config.together_api_key)


@disk_cache(cache_dir=config.cache_dir / "together-ai")
def chat(
    *,
    msgs: list[dict],
    model: str,
    temperature: float | None = 0.0,
    max_tokens: int = 512,
    response_format: dict = None,
):
    client = get_client()

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    time.sleep(1)
    return resp.model_dump(mode="json")


def is_reasoning(model: str):
    return model in [DEEPSEEK_NAME_TO_MODEL["deepseek-r1"], MODEL_QWEN3_235B]


def completion(
    prompt: str,
    model: str,
    system: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    remove_reasoning: bool = True,
    **kwds,
) -> str | list[str] | None:
    assert not system, "DeepSeek R1 trained without system prompt"
    msgs = [{"role": "user", "content": prompt}]
    if chat_history:
        msgs = chat_history + msgs

    resp = chat(msgs=msgs, model=model, **kwds)

    if resp is None:
        return resp

    msgs = [x["message"]["content"] for x in resp["choices"]]
    if remove_reasoning and is_reasoning(model):
        msgs = [msg[msg.find("</think>") + len("</think>") :].lstrip() for msg in msgs]

    if len(resp["choices"]) == 1:
        return msgs[0]
    return msgs


if __name__ == "__main__":  # pragma: nocover
    res = completion(
        prompt="how are you today?",
        model=DEEPSEEK_NAME_TO_MODEL["deepseek-v3"],
        remove_reasoning=False,
    )
    print(f"{res!r}")

    res2 = completion(prompt="how are you today?", model=MODEL_QWEN3_235B, remove_reasoning=False)
    print(f"{res2!r}")

    res3 = completion(
        prompt="/nothink how many t's in strawberrties?",
        model=MODEL_QWEN3_235B,
        remove_reasoning=False,
    )
    print(f"{res3!r}")

    res4 = completion(prompt="how are you today?", model=QWEN_NAME_TO_MODEL["qwen2.5-72b"])
    print(f"{res4!r}")
