#!/usr/bin/env python3
"""Check LLM provider availability from a .env file.

This script verifies provider API keys in a given .env file and can also
perform lightweight network checks against each provider API endpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import dotenv_values
from pydantic import BaseModel

from rag.llm.providers import LLMProvider, default_model_for_provider, generate_structured_output


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_name: str


@dataclass(frozen=True)
class CheckResult:
    provider: str
    key_name: str
    key_configured: bool
    api_reachable: str
    generation_test: str
    detail: str


class ProviderPingResponse(BaseModel):
    status: str


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(name="aistudio", key_name="GOOGLE_API_KEY"),
    ProviderSpec(name="openrouter", key_name="OPENROUTER_API_KEY"),
    ProviderSpec(name="openai", key_name="OPENAI_API_KEY"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LLM provider availability from .env")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Only validate key presence in .env, skip API requests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each provider check (default: 10)",
    )
    parser.add_argument(
        "--skip-generation-test",
        action="store_true",
        help="Skip sending a sample generation test case",
    )
    parser.add_argument(
        "--debug-raw-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw response preview (first 500 chars) for troubleshooting (default: enabled)",
    )
    return parser.parse_args()


def _clean_env_value(raw: str | None) -> str:
    if raw is None:
        return ""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1].strip()
    return value


def _load_env_values(env_path: Path) -> dict[str, str]:
    file_values = {k: _clean_env_value(v) for k, v in dotenv_values(env_path).items() if k and v is not None}
    # Allow shell env vars to override file values if already set.
    merged = dict(file_values)
    for key, value in os.environ.items():
        if key in merged and value:
            merged[key] = value
    return merged


def _check_openai(api_key: str, timeout: float) -> tuple[str, str]:
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return _interpret_http_result(response.status_code, response.text)


def _check_openrouter(api_key: str, timeout: float) -> tuple[str, str]:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return _interpret_http_result(response.status_code, response.text)


def _check_aistudio(api_key: str, timeout: float) -> tuple[str, str]:
    response = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=timeout,
    )
    return _interpret_http_result(response.status_code, response.text)


def _interpret_http_result(status_code: int, response_text: str) -> tuple[str, str]:
    if 200 <= status_code < 300:
        return "yes", f"HTTP {status_code}"
    if status_code in {401, 403}:
        return "no", f"HTTP {status_code} (auth failed)"

    snippet = " ".join(response_text.split())[:180]
    detail = f"HTTP {status_code}"
    if snippet:
        detail = f"{detail}: {snippet}"
    return "no", detail


def _check_provider(spec: ProviderSpec, env_values: dict[str, str], skip_network: bool, timeout: float) -> CheckResult:
    api_key = _clean_env_value(env_values.get(spec.key_name, ""))
    if not api_key:
        return CheckResult(
            provider=spec.name,
            key_name=spec.key_name,
            key_configured=False,
            api_reachable="n/a",
            generation_test="n/a",
            detail="missing key",
        )

    if skip_network:
        return CheckResult(
            provider=spec.name,
            key_name=spec.key_name,
            key_configured=True,
            api_reachable="skipped",
            generation_test="pending",
            detail="network check skipped",
        )

    try:
        if spec.name == "openai":
            reachable, detail = _check_openai(api_key, timeout)
        elif spec.name == "openrouter":
            reachable, detail = _check_openrouter(api_key, timeout)
        else:
            reachable, detail = _check_aistudio(api_key, timeout)

        return CheckResult(
            provider=spec.name,
            key_name=spec.key_name,
            key_configured=True,
            api_reachable=reachable,
            generation_test="pending",
            detail=detail,
        )
    except requests.RequestException as exc:
        return CheckResult(
            provider=spec.name,
            key_name=spec.key_name,
            key_configured=True,
            api_reachable="no",
            generation_test="pending",
            detail=f"request error: {exc.__class__.__name__}",
        )


def _extract_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _truncate_preview(raw_text: str, size: int = 500) -> str:
    compact = " ".join((raw_text or "").split())
    return compact[:size]


def _aistudio_raw_preview_legacy(model_name: str, system_prompt: str, user_prompt: str) -> str:
    try:
        import google.genai as genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "legacy preview unavailable: missing GOOGLE_API_KEY"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw_text = getattr(response, "text", "") or ""
        return _truncate_preview(raw_text)
    except Exception as exc:  # noqa: BLE001
        return f"legacy preview error: {exc.__class__.__name__}"


def _aistudio_genai_fallback(model_name: str, system_prompt: str, user_prompt: str) -> tuple[bool, str, str]:
    try:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
    except Exception as exc:  # noqa: BLE001
        return False, "", f"google.genai unavailable: {exc.__class__.__name__}"

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "", "google.genai missing GOOGLE_API_KEY"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return False, "", f"google.genai call failed: {exc.__class__.__name__}"

    raw_text = getattr(response, "text", "") or ""
    try:
        data = json.loads(_extract_json_text(raw_text))
    except Exception as exc:  # noqa: BLE001
        return False, raw_text, f"google.genai parse failed: {exc.__class__.__name__}"

    status = str(data.get("status") or "").strip().lower()
    if status == "ok":
        return True, raw_text, "google.genai fallback succeeded"
    return False, raw_text, "google.genai fallback returned non-ok status"


def _run_generation_test(spec: ProviderSpec, *, debug_raw_response: bool) -> tuple[str, str]:
    provider_enum = LLMProvider(spec.name)
    default_model = default_model_for_provider(provider_enum)
    candidate_models = [default_model]
    if provider_enum == LLMProvider.OPENROUTER and default_model.endswith(":free"):
        candidate_models.append(default_model[:-5])

    system_prompt = (
        "Ban la tro ly trich xuat cau truc. "
        "Chi tra ve JSON hop le theo dung schema, khong them giai thich."
    )
    user_prompt_options = [
        (
            "Test ket noi LLM provider voi mot test case don gian theo kieu generation eval. "
            "Tra ve JSON co truong status='ok'."
        ),
        (
            "Return strict JSON only with this shape: {\"status\":\"ok\"}."
        ),
    ]

    errors: list[str] = []
    debug_fragments: list[str] = []
    for model_name in candidate_models:
        for user_prompt in user_prompt_options:
            try:
                output, usage = generate_structured_output(
                    provider=provider_enum,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_model=ProviderPingResponse,
                )
                if output.status.strip().lower() != "ok":
                    errors.append(f"{model_name}: unexpected status={output.status!r}")
                    continue
                usage_summary = ""
                total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
                if total_tokens is not None:
                    usage_summary = f"; total_tokens={total_tokens}"
                raw_preview = ""
                if debug_raw_response and isinstance(usage, dict):
                    preview = (usage.get("raw_response_preview") or "").strip()
                    if preview:
                        raw_preview = f"; raw_preview={_truncate_preview(preview)}"
                return "yes", f"generation ok via {model_name}{usage_summary}{raw_preview}"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_name}: {exc.__class__.__name__}")

    if provider_enum == LLMProvider.AISTUDIO:
        for model_name in candidate_models:
            for user_prompt in user_prompt_options:
                ok, raw_text, message = _aistudio_genai_fallback(model_name, system_prompt, user_prompt)
                if ok:
                    detail = f"generation ok via google.genai fallback ({model_name})"
                    if debug_raw_response:
                        detail = f"{detail}; raw_preview={_truncate_preview(raw_text)}"
                    return "yes", detail
                errors.append(f"{model_name}: {message}")
                if debug_raw_response:
                    if raw_text:
                        debug_fragments.append(f"google.genai raw_preview={_truncate_preview(raw_text)}")
                    else:
                        debug_fragments.append(message)

        if debug_raw_response:
            legacy_preview = _aistudio_raw_preview_legacy(default_model, system_prompt, user_prompt_options[0])
            debug_fragments.append(f"legacy raw_preview={legacy_preview}")

    reason = ", ".join(errors[:4]) if errors else "unknown generation failure"
    if debug_raw_response and debug_fragments:
        reason = f"{reason}; debug={'; '.join(debug_fragments[:2])}"
    return "no", reason


def _apply_generation_tests(
    results: list[CheckResult],
    *,
    skip_generation_test: bool,
    debug_raw_response: bool,
) -> list[CheckResult]:
    updated: list[CheckResult] = []
    for result in results:
        if not result.key_configured:
            updated.append(result)
            continue

        if skip_generation_test:
            updated.append(
                CheckResult(
                    provider=result.provider,
                    key_name=result.key_name,
                    key_configured=result.key_configured,
                    api_reachable=result.api_reachable,
                    generation_test="skipped",
                    detail=result.detail,
                )
            )
            continue

        try:
            generation_status, generation_detail = _run_generation_test(
                ProviderSpec(name=result.provider, key_name=result.key_name),
                debug_raw_response=debug_raw_response,
            )
            merged_detail = f"{result.detail}; {generation_detail}" if result.detail else generation_detail
            updated.append(
                CheckResult(
                    provider=result.provider,
                    key_name=result.key_name,
                    key_configured=result.key_configured,
                    api_reachable=result.api_reachable,
                    generation_test=generation_status,
                    detail=merged_detail,
                )
            )
        except Exception as exc:  # noqa: BLE001
            merged_detail = f"{result.detail}; generation error: {exc.__class__.__name__}"
            updated.append(
                CheckResult(
                    provider=result.provider,
                    key_name=result.key_name,
                    key_configured=result.key_configured,
                    api_reachable=result.api_reachable,
                    generation_test="no",
                    detail=merged_detail,
                )
            )

    return updated


def _print_results(results: list[CheckResult]) -> None:
    headers = ["provider", "key", "configured", "api_reachable", "generation_test", "detail"]
    rows = [
        [
            r.provider,
            r.key_name,
            "yes" if r.key_configured else "no",
            r.api_reachable,
            r.generation_test,
            r.detail,
        ]
        for r in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print(fmt(headers))
    print(separator)
    for row in rows:
        print(fmt(row))


def main() -> int:
    args = _parse_args()
    env_path = Path(args.env_file)

    if not env_path.exists():
        print(f"error: env file not found: {env_path}")
        return 2

    env_values = _load_env_values(env_path)
    for spec in PROVIDERS:
        value = env_values.get(spec.key_name, "")
        if value:
            os.environ[spec.key_name] = value

    results = [_check_provider(spec, env_values, args.skip_network, args.timeout) for spec in PROVIDERS]
    results = _apply_generation_tests(
        results,
        skip_generation_test=args.skip_generation_test,
        debug_raw_response=args.debug_raw_response,
    )
    _print_results(results)

    any_configured = any(r.key_configured for r in results)
    if not any_configured:
        return 1

    checks_ok = True
    if not args.skip_network:
        checks_ok = checks_ok and all((not r.key_configured) or (r.api_reachable == "yes") for r in results)
    if not args.skip_generation_test:
        checks_ok = checks_ok and all((not r.key_configured) or (r.generation_test == "yes") for r in results)

    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())