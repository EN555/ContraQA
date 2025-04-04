import json
import logging
import re
import time
from functools import cache
from typing import List, Dict, Literal

import openai
import tiktoken
from openai import NOT_GIVEN, BadRequestError
from openai.types.chat import ChatCompletion

from contraqa.utils.configs import config
from contraqa.utils.diskcache import disk_cache

GPT4_MODEL = "gpt-4o-2024-08-06"
GPT4_MINI_MODEL = "gpt-4o-mini-2024-07-18"
GPT_REASON_MODEL = "o3-mini-2025-01-31"
GPT_O4_REASON_MODEL = "o4-mini-2025-04-16"
GPT_O3_REASON_MODEL = "o3-2025-04-16"

logger = logging.getLogger(__name__)


def model2id(model: str) -> tuple[str, dict]:
    """returns model_id and extra keyword arguments for `completion()` parsed from model codename"""
    model = model.lower()
    if model == "gpt-4o":
        return GPT4_MODEL, {}
    elif model == "gpt4o-mini":
        return GPT4_MINI_MODEL, {}
    elif match := re.match(r"^o3-mini-(low|medium|high)$", model):
        return GPT_REASON_MODEL, {"reasoning_effort": match.group(1)}
    elif match := re.match(r"^o4-mini-(low|medium|high)$", model):
        return GPT_O4_REASON_MODEL, {"reasoning_effort": match.group(1)}
    elif match := re.match(r"^o3-(low|medium|high)$", model):
        return GPT_O3_REASON_MODEL, {"reasoning_effort": match.group(1)}
    raise ValueError(f"model {model} is not supported")


class OpenAIAPIResponseError(Exception):
    pass


@cache
def get_client():
    return openai.OpenAI(api_key=config.openai_api_key)


def prompt2msgs(prompt: str, system: str) -> List[Dict[str, str]]:
    msgs = [
        {"role": "developer", "content": system},
        {"role": "user", "content": prompt},
    ]
    if system is None:
        msgs.pop(0)
    return msgs


def completion(
    prompt: str,
    system: str | None,
    model: str,
    chat_history: list[dict[str, str]] | None = None,
    **kwds,
) -> str | list[str] | None:
    msgs = prompt2msgs(prompt, system)
    if chat_history:
        msgs = chat_history + msgs
    resp = chat(msgs=msgs, model=model, **kwds)

    if resp is None:
        return resp

    resp = ChatCompletion.model_validate(resp)
    logger.info(
        f"generated response for model {model} with {resp.usage.completion_tokens_details.reasoning_tokens} tokens"
    )
    if len(resp.choices) == 1:
        # Happens in refusals sometimes, retry fixes this.
        if resp.choices[0].message.content is None:
            raise Exception(resp)
        return resp.choices[0].message.content
    return [x.message.content for x in resp.choices]


@disk_cache(cache_dir=config.cache_dir / "openai-chat")
def chat(
    *,
    msgs: list[dict],
    model: str,
    reasoning_effort: str = NOT_GIVEN,
    temperature: float = 0,
    n: int = NOT_GIVEN,
    max_tokens: int = NOT_GIVEN,
    tools: list[dict] = NOT_GIVEN,
    tool_choice: Literal["none", "auto", "required"] = NOT_GIVEN,
    parallel_tool_calls: bool = NOT_GIVEN,
    response_format: dict = NOT_GIVEN,
):
    client = get_client()

    if model == GPT_REASON_MODEL or reasoning_effort != NOT_GIVEN:
        temperature = NOT_GIVEN  # unsupported

    max_retries = 1
    for retry in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
            )
            break
        # simple retry
        except BadRequestError as exc:
            if retry >= max_retries:
                raise
            logger.warning(f"got error from OpenAI api - {repr(exc)}")
            if exc.status_code == 400:
                return None
            time.sleep(60)

    return resp.to_dict(mode="json")


def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    if model == "gpt-3.5-turbo-16k-0613":
        model = "gpt-3.5-turbo"
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k-0613":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming"
            " gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


if __name__ == "__main__":  # pragma: nocover
    client = get_client()
    print(client.models.list())
