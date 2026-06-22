"""Unified provider interface for structured LLM generation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
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
    # Some models return JSON `null` when there is no result. Treat that as
    # an empty object so Pydantic can validate into a model with all-None
    # optional fields (which indicates absence rather than a parse error).
    if data is None:
        data = {}
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
    # Mirror the pattern used in data_create/generate_synthetic_summary_aistudio.py
    try:
        import google.genai as genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - import/runtime guard
        raise EnvironmentError(f"google.genai library unavailable: {exc.__class__.__name__}")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set")

    # Lazily create a client instance on module level to reuse connections
    global _aistudio_client
    try:
        _aistudio_client
    except NameError:
        _aistudio_client = None
    if _aistudio_client is None:
        _aistudio_client = genai.Client(api_key=api_key)

    # Default request timeout can be configured via env var
    request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "90"))

    # Some google.genai client versions accept `request_options` while others do not.
    # Try calling with `request_options` first and fall back if the client raises
    # a TypeError for unexpected kwargs (mirrors data_create/generate_synthetic_summary_aistudio.py).
    try:
        response = _aistudio_client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
            request_options={"timeout": request_timeout},
        )
    except TypeError:
        response = _aistudio_client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

    raw_text = getattr(response, "text", "") or ""

    # Try to parse structured JSON, but if parsing fails raise a JSON error to be
    # handled by the caller (so fallback attempts can occur).
    parsed = _parse_structured(raw_text, output_model)

    usage_meta = getattr(response, "usage_metadata", None)
    usage = _usage_dict(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        raw_text=raw_text,
    )
    usage["response_repr"] = repr(response)
    return parsed, usage


def _get_aistudio_api_keys() -> list[tuple[str, str]]:
    key_names = ("GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3")
    return [(key_name, api_key) for key_name in key_names if (api_key := os.environ.get(key_name))]


def _reset_aistudio_client_cache() -> None:
    global _aistudio_client
    try:
        _aistudio_client = None
    except NameError:
        pass


@contextmanager
def _temporary_google_api_key(api_key: str) -> Iterator[None]:
    previous = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    _reset_aistudio_client_cache()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = previous
        _reset_aistudio_client_cache()


def _generate_with_aistudio_rotating_keys(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TModel],
) -> tuple[TModel, dict[str, Any]]:
    api_keys = _get_aistudio_api_keys()
    if not api_keys:
        raise EnvironmentError(
            "No AI Studio API keys are set. Expected GOOGLE_API_KEY, GOOGLE_API_KEY_2, and/or GOOGLE_API_KEY_3."
        )

    errors: list[str] = []
    attempts = [
        {"provider": LLMProvider.AISTUDIO.value, "model": model_name, "key_name": key_name}
        for key_name, _ in api_keys
    ]
    for key_name, api_key in api_keys:
        try:
            with _temporary_google_api_key(api_key):
                parsed, usage = _generate_with_aistudio(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_model=output_model,
                )
            usage = {
                **usage,
                "provider": LLMProvider.AISTUDIO.value,
                "model": model_name,
                "aistudio_key_name": key_name,
                "aistudio_key_attempts": attempts,
                "aistudio_key_errors": errors,
            }
            return parsed, usage
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key_name}:{exc}")

    raise RuntimeError("All AI Studio API key attempts failed. " + " | ".join(errors))


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
    usage["response_repr"] = repr(response)
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
    usage["response_repr"] = repr(response)
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
        return _generate_with_aistudio_rotating_keys(
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
    3) OpenRouter paid tier model (used only if free and AI Studio fail)
    """

    if preferred_provider is None:
        preferred = LLMProvider.OPENROUTER
    else:
        preferred = LLMProvider(preferred_provider)

    openrouter_free_model = model_name or default_model_for_provider(LLMProvider.OPENROUTER)
    openrouter_paid_model = openrouter_free_model[:-5] + ":paid"

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
            if provider == LLMProvider.AISTUDIO:
                parsed, usage = _generate_with_aistudio_rotating_keys(
                    model_name=attempt_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_model=output_model,
                )
            else:
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
