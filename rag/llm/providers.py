"""Unified provider interface for structured LLM generation."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

from rag import config

TModel = TypeVar("TModel", bound=BaseModel)


class LLMProvider(str, Enum):
    AISTUDIO = "aistudio"
    OPENROUTER = "openrouter"
    OPENAI = "openai"


def default_model_for_provider(provider: LLMProvider) -> str:
    if provider == LLMProvider.AISTUDIO:
        return config.DEFAULT_AISTUDIO_MODEL
    if provider == LLMProvider.OPENROUTER:
        return config.DEFAULT_OPENROUTER_MODEL
    return config.DEFAULT_OPENAI_MODEL


def _extract_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_structured(raw_text: str, output_model: type[TModel]) -> TModel:
    cleaned = _extract_json_text(raw_text)
    data = json.loads(cleaned)
    return output_model.model_validate(data)


def _usage_dict(prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, raw_text: str = "") -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or 0,
        "raw_response_preview": (raw_text or "")[:500],
    }


def _generate_with_aistudio(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    import google.genai as genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text or ""
    parsed = _parse_structured(raw_text, output_model)

    usage_meta = getattr(response, "usage_metadata", None)
    usage = _usage_dict(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        raw_text=raw_text,
    )
    return parsed, usage


def _generate_with_openrouter(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    parsed = _parse_structured(raw_text, output_model)

    usage_obj = getattr(response, "usage", None)
    usage = _usage_dict(
        prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        raw_text=raw_text,
    )
    return parsed, usage


def _generate_with_openai(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    parsed = _parse_structured(raw_text, output_model)

    usage_obj = getattr(response, "usage", None)
    usage = _usage_dict(
        prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        raw_text=raw_text,
    )
    return parsed, usage


def generate_structured_output(
    *,
    provider: LLMProvider | str,
    model_name: str | None,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    provider_enum = LLMProvider(provider)
    selected_model = model_name or default_model_for_provider(provider_enum)

    if provider_enum == LLMProvider.AISTUDIO:
        return _generate_with_aistudio(
            model_name=selected_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )
    if provider_enum == LLMProvider.OPENROUTER:
        return _generate_with_openrouter(
            model_name=selected_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )
    return _generate_with_openai(
        model_name=selected_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=output_model,
    )


def generate_structured_output_with_fallback(
    *,
    preferred_provider: LLMProvider | str | None,
    model_name: str | None,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    """Generate structured output with provider/model fallback.

    Fallback order:
    1) OpenRouter free tier model
    2) AI Studio default model
    3) OpenRouter standard tier (same model without ':free')
    """

    if preferred_provider is None:
        preferred = LLMProvider.OPENROUTER
    else:
        preferred = LLMProvider(preferred_provider)

    openrouter_free_model = model_name or default_model_for_provider(LLMProvider.OPENROUTER)
    openrouter_paid_model = openrouter_free_model[:-5] if openrouter_free_model.endswith(":free") else openrouter_free_model

    ordered_attempts: list[tuple[LLMProvider, str]] = [
        (LLMProvider.OPENROUTER, openrouter_free_model),
        (LLMProvider.AISTUDIO, default_model_for_provider(LLMProvider.AISTUDIO)),
        (LLMProvider.OPENROUTER, openrouter_paid_model),
    ]

    if preferred == LLMProvider.AISTUDIO:
        ordered_attempts = [
            (LLMProvider.AISTUDIO, default_model_for_provider(LLMProvider.AISTUDIO)),
            (LLMProvider.OPENROUTER, openrouter_free_model),
            (LLMProvider.OPENROUTER, openrouter_paid_model),
        ]
    elif preferred == LLMProvider.OPENAI:
        ordered_attempts = [
            (LLMProvider.OPENAI, model_name or default_model_for_provider(LLMProvider.OPENAI)),
            (LLMProvider.OPENROUTER, openrouter_free_model),
            (LLMProvider.AISTUDIO, default_model_for_provider(LLMProvider.AISTUDIO)),
            (LLMProvider.OPENROUTER, openrouter_paid_model),
        ]

    last_error: Exception | None = None
    attempt_errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for provider, attempt_model in ordered_attempts:
        key = (provider.value, attempt_model)
        if key in seen:
            continue
        seen.add(key)

        try:
            parsed, usage = generate_structured_output(
                provider=provider,
                model_name=attempt_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=output_model,
            )
            usage = {
                **usage,
                "provider": provider.value,
                "model": attempt_model,
                "fallback_attempts": [
                    {"provider": p, "model": m} for (p, m) in [(x[0].value, x[1]) for x in ordered_attempts]
                ],
                "fallback_errors": attempt_errors,
            }
            return parsed, usage
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempt_errors.append(f"{provider.value}:{attempt_model}:{exc}")

    if last_error is None:
        raise RuntimeError("No provider attempt was executed")

    raise RuntimeError(
        "All provider fallback attempts failed. " + " | ".join(attempt_errors)
    ) from last_error
