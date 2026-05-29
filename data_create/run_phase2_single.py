#!/usr/bin/env python3
"""Run a single-file Phase 2 Xu_Ly_Vat_Chung extraction for debugging.

Usage:
  python data_create/run_phase2_single.py /path/to/extracted.json [--model MODEL]

This script mirrors the Phase 2 logic in data_create/patch_processed_json.py but
prints detailed debug info (provider, model, tier, usage, raw preview, and fallbacks).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pydantic import BaseModel

from rag.llm.providers import (
    generate_structured_output_with_fallback,
    default_model_for_provider,
    LLMProvider,
)


class XuLyVatChungOutput(BaseModel):
    Xu_Ly_Vat_Chung: Optional[str] = None


def _flatten_nhan_dinh_text(nhan_dinh: object) -> str:
    if isinstance(nhan_dinh, str):
        return nhan_dinh.strip()
    if isinstance(nhan_dinh, list):
        parts = [_flatten_nhan_dinh_text(item) for item in nhan_dinh]
        return "\n\n".join(part for part in parts if part)
    if isinstance(nhan_dinh, dict):
        parts: list[str] = []
        for key, value in sorted(nhan_dinh.items(), key=lambda item: str(item[0])):
            flattened = _flatten_nhan_dinh_text(value)
            if flattened:
                parts.append(flattened)
        return "\n\n".join(parts)
    return ""


def _extract_xu_ly_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [r"(?i)(Về\s+vật\s+chứng\s*:.*)", r"(?i)(Về\s+xử\s+lý\s+vật\s+chứng\s*:.*)"]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            snippet = m.group(1).strip()
            split_match = re.split(r"\n\s*\[[0-9]+\]|\n\s*Về\s+án\s+phí\s*:", snippet, maxsplit=1)
            return split_match[0].strip()
    return None


def _write_full_response_dump(response_repr: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "last_aistudio_full_response.txt"
    txt_path.write_text(response_repr, encoding="utf-8")
    print(f"Saved full response dump to: {txt_path}")


def run_single(extracted_path: Path, model_name: Optional[str], print_raw: bool) -> int:
    # Load repo .env (if present) so API keys are available to provider clients.
    load_dotenv()

    data = json.loads(extracted_path.read_text(encoding="utf-8"))
    nhan_dinh_text = _flatten_nhan_dinh_text(data.get("NHAN_DINH_CUA_TOA_AN"))

    chosen_model = model_name or default_model_for_provider(LLMProvider.AISTUDIO)

    print("Environment keys:")
    for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        print(f"  {k}: {'SET' if os.environ.get(k) else 'missing'}")
    print(f"Using requested model: {chosen_model}\n")

    system_prompt = (
        "You are a legal extraction assistant. "
        "Extract only the case-level handling of physical evidence from the court decision. "
        "If the decision does not mention handling of physical evidence, return null. "
        "Respond with ONLY valid JSON for the schema. "
        "The JSON object must use the exact field name Xu_Ly_Vat_Chung. "
        "Do not translate or rename the field into English. "
        "Return exactly one JSON object like: {\"Xu_Ly_Vat_Chung\": \"...\"}."
    )

    user_prompt = f"NHAN_DINH_CUA_TOA_AN:\n{nhan_dinh_text.strip()}"

    print("=== Primary LLM attempt ===")
    try:
        result, usage = generate_structured_output_with_fallback(
            preferred_provider=LLMProvider.AISTUDIO,
            model_name=chosen_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=XuLyVatChungOutput,
        )
    except Exception as exc:
        print(f"LLM call raised: {exc.__class__.__name__}: {exc}")
        result = None
        usage = {}

    print("Result object:", result)
    print("Usage metadata:")
    print(json.dumps(usage or {}, ensure_ascii=False, indent=2))
    if print_raw and isinstance(usage, dict):
        raw = usage.get("raw_response_preview")
        if raw:
            print("\n-- RAW RESPONSE PREVIEW --\n")
            print(raw)
        response_repr = usage.get("response_repr")
        if isinstance(response_repr, str) and response_repr.strip():
            _write_full_response_dump(response_repr, extracted_path.parent)

    value = result.Xu_Ly_Vat_Chung if result else None

    if value is None:
        print("=== Falling back to regex/text extraction ===")
        text_fallback = _extract_xu_ly_from_text(nhan_dinh_text)
        print("Text fallback result:", text_fallback)
        final = text_fallback
        final_source = {"provider": "text_fallback", "tier": "n/a", "model": "n/a"}
    else:
        final = str(value).strip() if value else None
        final_source = {
            "provider": usage.get("provider") if isinstance(usage, dict) else None,
            "model": usage.get("model") if isinstance(usage, dict) else None,
            "tier": None,
        }
        # Best-effort tier detection
        model_used = final_source["model"] or chosen_model
        if model_used and model_used.endswith(":free"):
            final_source["tier"] = "free"
        elif model_used and model_used.endswith(":paid"):
            final_source["tier"] = "paid"
        else:
            final_source["tier"] = "unknown"

    print("\n=== FINAL DECISION ===")
    print("Xu_Ly_Vat_Chung:", final)
    print("Source:", json.dumps(final_source, ensure_ascii=False))

    return 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("extracted", help="Path to extracted JSON (input) file")
    p.add_argument("--model", help="Model name to request (optional)")
    p.add_argument("--raw", action="store_true", help="Print raw response preview if available")
    return p.parse_args()


def main():
    args = _parse_args()
    path = Path(args.extracted)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 2
    return run_single(path, args.model, args.raw)


if __name__ == "__main__":
    raise SystemExit(main())
