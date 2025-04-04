import logging
from functools import cache

from google import genai
from google.genai import types

from contraqa.utils.configs import config
from contraqa.utils.diskcache import disk_cache

# MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro-exp-03-25"
MODEL_GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash-001"

logger = logging.getLogger(__name__)


@cache
def get_client() -> genai.Client:
    return genai.Client(api_key=config.gemini_api_key)


@disk_cache(cache_dir=config.cache_dir / "gemini-api")
def chat(
    *,
    prompt: str,
    system: str | None,
    model: str,
    temperature: float | None = 0.0,
    thinking_budget: int = 1024,
):
    client = get_client()

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            # max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system,
            thinking_config=(
                types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget else None
            ),
        ),
    )
    return resp.model_dump(mode="json")


def completion(
    prompt: str,
    model: str,
    system: str | None,
    **kwds,
) -> str | list[str] | None:
    resp = chat(prompt=prompt, system=system, model=model, **kwds)

    if resp is None:
        return resp

    res = types.GenerateContentResponse.model_validate(resp)
    logger.info(f"generated response with {res.usage_metadata.thoughts_token_count} thought tokens")
    return res.text


if __name__ == "__main__":  # pragma: nocover
    res = completion(
        prompt="how are you today?", system=None, model=MODEL_GEMINI_2_5_PRO, thinking_budget=10_000
    )
    print(f"{res!r}")
