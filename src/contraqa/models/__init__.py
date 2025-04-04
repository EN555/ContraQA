from contraqa.models import openai_api, togetherai_api, gemini_api


def completion(prompt: str, system: str | None, model: str) -> str | list[str] | None:
    if model.lower().startswith("deepseek-"):
        # no system prompt for DeepSeek-R1
        model_id = togetherai_api.DEEPSEEK_NAME_TO_MODEL[model]
        return togetherai_api.completion(prompt=prompt, system=None, model=model_id)
    elif model.lower().startswith("qwen3-") or model.lower().startswith("qwen2.5-"):
        model_id = togetherai_api.QWEN_NAME_TO_MODEL[model.lower()]
        return togetherai_api.completion(prompt=prompt, system=None, model=model_id)
    elif model.lower() == "gemini-2.5-pro":
        return gemini_api.completion(
            prompt=prompt, system=system, model=gemini_api.MODEL_GEMINI_2_5_PRO
        )
    elif model.lower() == "gemini-2.0-flash":
        return gemini_api.completion(
            prompt=prompt,
            system=system,
            model=gemini_api.MODEL_GEMINI_2_0_FLASH,
            thinking_budget=0,
        )
    # OpenAI Models
    else:
        model_id, model_kwargs = openai_api.model2id(model)
        return openai_api.completion(prompt=prompt, system=system, model=model_id, **model_kwargs)
