#!/usr/bin/env python3
"""Generate PHAN_QUYET_CUA_TOA_SO_THAM-style predictions from retrieval outputs.

Pipeline per test JSON file:
1. Read Synthetic_summary and defendant names.
2. Read predicted article signatures from eval_results.json for the same doc_id.
3. Retrieve legal context from law_doc.json with hierarchical rules:
   - dieu-khoan-diem => specific diem context
   - dieu or dieu-khoan (without diem) => full dieu expansion, including all subfields
4. Ask Gemma 4 (AI Studio or OpenRouter) to produce verdict entries per defendant.
5. Write one output JSON per case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from law_clause_retrieve import LawClauseRetriever
from dotenv import load_dotenv

load_dotenv()


class CanCuDieuLuat(BaseModel):
    Diem: str | None = None
    Khoan: str | None = None
    Dieu: str | None = None
    Bo_Luat_Va_Van_Ban_Khac: str | None = None


class PhanQuyetToaSoTham(BaseModel):
    Bi_Cao: str
    Can_Cu_Dieu_Luat: list[CanCuDieuLuat]
    Pham_Toi: str
    Phat_Tu: str | None = None
    Phat_Tien: str | None = None
    An_Phi: str | None = None
    Hinh_Phat_Bo_Sung: str | None = None
    Trach_Nhiem_Dan_Su: str | None = None
    Xu_Ly_Vat_Chung: str | None = None


class VerdictOnlyOutput(BaseModel):
    PHAN_QUYET_CUA_TOA_SO_THAM: list[PhanQuyetToaSoTham]


PREFIX_PATTERN = re.compile(r"^(dieu|\u0111i\u1ec1u|khoan|kho\u1ea3n|diem|\u0111i\u1ec3m)\s+", re.IGNORECASE)


def _normalize_token(token: str, *, lowercase: bool = False) -> str:
    text = str(token).strip()
    text = PREFIX_PATTERN.sub("", text)
    text = text.strip(" .")
    text = re.sub(r"\s+", "", text)
    return text.lower() if lowercase else text


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_model_json(raw_text: str) -> VerdictOnlyOutput:
    cleaned = _extract_json_text(raw_text)
    data = json.loads(cleaned)
    return VerdictOnlyOutput.model_validate(data)


def _get_first(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _sort_index_key(token: str) -> tuple[int, int, str]:
    text = str(token)
    if text.isdigit():
        return (0, int(text), "")
    match = re.match(r"^(\d+)([a-zA-Z].*)$", text)
    if match:
        return (1, int(match.group(1)), match.group(2).lower())
    return (2, 0, text.lower())


def load_law_doc_index(law_doc_path: Path) -> dict[str, dict[str, Any]]:
    with open(law_doc_path, encoding="utf-8") as fh:
        root = json.load(fh)

    if isinstance(root, dict):
        chapters = root.get("Chuong") or root.get("Ch\u01b0\u01a1ng") or []
    elif isinstance(root, list):
        chapters = root
    else:
        chapters = []

    index: dict[str, dict[str, Any]] = {}

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        dieu_list = _get_first(chapter, ["\u0110i\u1ec1u", "Dieu"], default=[])
        if not isinstance(dieu_list, list):
            continue

        for dieu in dieu_list:
            if not isinstance(dieu, dict):
                continue

            dieu_index = _normalize_token(_get_first(dieu, ["index"], ""), lowercase=False)
            if not dieu_index:
                continue

            entry = {
                "dieu": dieu_index,
                "dieu_text": str(_get_first(dieu, ["text"], "") or "").strip(),
                "khoan": {},
            }

            khoan_list = _get_first(dieu, ["Kho\u1ea3n", "Khoan"], default=[])
            if isinstance(khoan_list, list):
                for khoan in khoan_list:
                    if not isinstance(khoan, dict):
                        continue
                    khoan_index = _normalize_token(_get_first(khoan, ["index"], ""), lowercase=False)
                    if not khoan_index:
                        continue

                    khoan_entry = {
                        "khoan": khoan_index,
                        "khoan_text": str(_get_first(khoan, ["text"], "") or "").strip(),
                        "diem": {},
                    }

                    diem_list = _get_first(khoan, ["\u0110i\u1ec3m", "Diem"], default=[])
                    if isinstance(diem_list, list):
                        for diem in diem_list:
                            if not isinstance(diem, dict):
                                continue
                            diem_index = _normalize_token(_get_first(diem, ["index"], ""), lowercase=True)
                            if not diem_index:
                                continue
                            khoan_entry["diem"][diem_index] = {
                                "diem": diem_index,
                                "diem_text": str(_get_first(diem, ["text"], "") or "").strip(),
                            }

                    entry["khoan"][khoan_index] = khoan_entry

            index[dieu_index] = entry

    return index


def _render_full_dieu(entry: dict[str, Any]) -> str:
    lines: list[str] = []
    dieu_idx = entry["dieu"]
    dieu_text = entry.get("dieu_text", "")
    lines.append(f"Dieu {dieu_idx}: {dieu_text}".strip())

    khoan_map = entry.get("khoan", {})
    for khoan_idx in sorted(khoan_map.keys(), key=_sort_index_key):
        khoan_entry = khoan_map[khoan_idx]
        lines.append(f"  Khoan {khoan_idx}: {khoan_entry.get('khoan_text', '')}".rstrip())
        diem_map = khoan_entry.get("diem", {})
        for diem_idx in sorted(diem_map.keys(), key=_sort_index_key):
            diem_text = diem_map[diem_idx].get("diem_text", "")
            lines.append(f"    Diem {diem_idx}: {diem_text}".rstrip())

    return "\n".join(lines).strip()


def _render_specific_diem(entry: dict[str, Any], khoan_idx: str, diem_idx: str) -> str:
    khoan_entry = entry.get("khoan", {}).get(khoan_idx)
    if not khoan_entry:
        return ""
    diem_entry = khoan_entry.get("diem", {}).get(diem_idx)
    if not diem_entry:
        return ""

    dieu_idx = entry["dieu"]
    dieu_text = entry.get("dieu_text", "")
    khoan_text = khoan_entry.get("khoan_text", "")
    diem_text = diem_entry.get("diem_text", "")

    return (
        f"Dieu {dieu_idx}: {dieu_text}\n"
        f"  Khoan {khoan_idx}: {khoan_text}\n"
        f"    Diem {diem_idx}: {diem_text}"
    ).strip()


def _find_diem_across_khoan(entry: dict[str, Any], diem_idx: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for khoan_idx, khoan_entry in entry.get("khoan", {}).items():
        diem_entry = khoan_entry.get("diem", {}).get(diem_idx)
        if diem_entry:
            matches.append((khoan_idx, _render_specific_diem(entry, khoan_idx, diem_idx)))
    matches.sort(key=lambda item: _sort_index_key(item[0]))
    return matches


def _parse_signature(signature: str) -> dict[str, str] | None:
    raw = str(signature).strip()
    if not raw:
        return None

    parts = [p for p in raw.split("-") if p != ""]
    if not parts:
        return None

    dieu = _normalize_token(parts[0], lowercase=False)
    if not dieu:
        return None

    if len(parts) == 1:
        return {"raw": raw, "kind": "dieu", "dieu": dieu, "khoan": "", "diem": ""}

    if len(parts) == 2:
        second_raw = _normalize_token(parts[1], lowercase=False)
        if second_raw.isdigit():
            return {
                "raw": raw,
                "kind": "dieu_khoan",
                "dieu": dieu,
                "khoan": second_raw,
                "diem": "",
            }
        return {
            "raw": raw,
            "kind": "dieu_diem",
            "dieu": dieu,
            "khoan": "",
            "diem": _normalize_token(parts[1], lowercase=True),
        }

    khoan = _normalize_token(parts[1], lowercase=False)
    diem = _normalize_token(parts[2], lowercase=True)
    return {
        "raw": raw,
        "kind": "dieu_khoan_diem",
        "dieu": dieu,
        "khoan": khoan,
        "diem": diem,
    }


def build_law_context(signatures: list[str], law_retriever: LawClauseRetriever) -> dict[str, Any]:
    retrieved_clause_texts: list[dict[str, Any]] = []
    missing_signatures: list[str] = []
    warnings: list[str] = []

    for result in law_retriever.retrieve_many(signatures):
        signature = str(result.get("query") or "").strip()
        found = bool(result.get("found", False))
        if not found:
            if signature:
                missing_signatures.append(signature)
            reason = str(result.get("reason") or "not_found")
            warnings.append(f"{reason}:{signature}")
            continue

        retrieved_clause_texts.append(
            {
                "signature": result.get("normalized") or signature,
                "query": signature,
                "retrieval_level": result.get("level"),
                "dieu": result.get("dieu"),
                "khoan": result.get("khoan"),
                "diem": result.get("diem"),
                "context_text": str(result.get("text") or "").strip(),
            }
        )

    return {
        "signature_inputs": signatures,
        "retrieved_clause_texts": retrieved_clause_texts,
        "missing_signatures": missing_signatures,
        "warnings": warnings,
    }


def load_eval_predictions(eval_results_path: Path) -> dict[str, dict[str, Any]]:
    with open(eval_results_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        per_doc = data.get("per_doc", [])
    elif isinstance(data, list):
        per_doc = data
    else:
        per_doc = []

    output: dict[str, dict[str, Any]] = {}
    if not isinstance(per_doc, list):
        return output

    for item in per_doc:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        output[doc_id] = item

    return output


def extract_predicted_signatures(eval_item: dict[str, Any]) -> list[str]:
    signatures: list[str] = []
    for key in ("predicted_articles", "predicted_articles_full"):
        values = eval_item.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if text:
                signatures.append(text)

    return _dedupe_keep_order(signatures)


def extract_doc_id_and_defendants(test_data: dict[str, Any], fallback_doc_id: str) -> tuple[str, list[str]]:
    thong_tin = test_data.get("THONG_TIN_CHUNG") or {}
    doc_id = str(thong_tin.get("Ma_Ban_An") or test_data.get("Ma_Ban_An") or fallback_doc_id).strip()

    defendant_names: list[str] = []
    thong_tin_bi_cao = thong_tin.get("Thong_Tin_Bi_Cao")
    if isinstance(thong_tin_bi_cao, list):
        for person in thong_tin_bi_cao:
            if isinstance(person, dict):
                name = str(person.get("Ho_Ten") or "").strip()
                if name:
                    defendant_names.append(name)

    return doc_id, _dedupe_keep_order(defendant_names)


def build_prompts(
    doc_id: str,
    synthetic_summary: str,
    defendants: list[str],
    signatures: list[str],
    law_context: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert Vietnamese criminal judgment assistant. "
        "Given case facts and retrieved legal references, produce ONLY valid JSON "
        "with key PHAN_QUYET_CUA_TOA_SO_THAM as a list of verdict objects, one per defendant when applicable. "
        "Each verdict object must include: Bi_Cao, Can_Cu_Dieu_Luat, Pham_Toi, Phat_Tu, An_Phi, Xu_Ly_Vat_Chung. "
        "Use legal references carefully and avoid hallucinating unsupported article details."
    )

    payload = {
        "doc_id": doc_id,
        "defendants": defendants,
        "synthetic_summary": synthetic_summary,
        "predicted_signatures": signatures,
        "law_context": {
            "retrieved_clause_texts": law_context.get("retrieved_clause_texts", []),
            "missing_signatures": law_context.get("missing_signatures", []),
        },
        "required_output_schema": {
            "PHAN_QUYET_CUA_TOA_SO_THAM": [
                {
                    "Bi_Cao": "string",
                    "Can_Cu_Dieu_Luat": [
                        {
                            "Diem": "string|null",
                            "Khoan": "string|null",
                            "Dieu": "string|null",
                            "Bo_Luat_Va_Van_Ban_Khac": "string|null",
                        }
                    ],
                    "Pham_Toi": "string",
                    "Phat_Tu": "string|null",
                    "An_Phi": "string|null",
                    "Xu_Ly_Vat_Chung": "string|null",
                }
            ]
        },
        "instructions": [
            "Return JSON only. No markdown, no commentary.",
            "Predict verdicts per defendant based on facts and provided law context.",
            "Use retrieved_clause_texts as the legal basis context for mapping Can_Cu_Dieu_Luat.",
            "If information is uncertain, keep fields concise and avoid inventing unsupported details.",
        ],
    }

    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def generate_with_aistudio(model_name: str, system_prompt: str, user_prompt: str) -> tuple[VerdictOnlyOutput, dict[str, Any]]:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    response = model.generate_content(
        f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
        generation_config=GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text or ""
    parsed = _parse_model_json(raw_text)

    usage_meta = getattr(response, "usage_metadata", None)
    usage = {
        "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
    }
    usage["raw_response_preview"] = raw_text[:500]

    return parsed, usage


def generate_with_openrouter(model_name: str, system_prompt: str, user_prompt: str) -> tuple[VerdictOnlyOutput, dict[str, Any]]:
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
    parsed = _parse_model_json(raw_text)

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        "raw_response_preview": raw_text[:500],
    }

    return parsed, usage


def parse_filename_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_filename_list_file(list_file_path: str) -> list[str]:
    with open(list_file_path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def resolve_input_files(test_dir: Path, first_n: int | None, file_list: str | None, file_list_path: str | None) -> list[Path]:
    list_mode_enabled = bool(file_list or file_list_path)
    first_n_mode_enabled = first_n is not None

    if list_mode_enabled and first_n_mode_enabled:
        raise ValueError("Use either --first-n or file list mode, not both")

    if list_mode_enabled:
        names: list[str] = []
        if file_list:
            names.extend(parse_filename_list(file_list))
        if file_list_path:
            names.extend(read_filename_list_file(file_list_path))

        unique_names = _dedupe_keep_order(names)
        if not unique_names:
            raise ValueError("No filenames were provided in list mode")

        selected: list[Path] = []
        for name in unique_names:
            candidate = Path(name)
            if candidate.suffix != ".json":
                candidate = Path(f"{name}.json")
            path = test_dir / candidate.name
            if path.exists() and path.is_file():
                selected.append(path)
        return selected

    files = sorted(test_dir.glob("*.json"))
    if first_n is not None:
        if first_n < 1:
            raise ValueError("--first-n must be >= 1")
        files = files[:first_n]
    return files


def process_case(
    case_path: Path,
    eval_map: dict[str, dict[str, Any]],
    law_retriever: LawClauseRetriever,
    output_dir: Path,
    provider: str,
    model_name: str,
    reprocess: bool,
) -> tuple[str, str]:
    output_path = output_dir / case_path.name
    if output_path.exists() and not reprocess:
        return "skipped", "already_exists"

    with open(case_path, encoding="utf-8") as fh:
        test_data = json.load(fh)

    doc_id, defendants = extract_doc_id_and_defendants(test_data, case_path.stem)
    synthetic_summary = str(test_data.get("Synthetic_summary") or "").strip()
    if not synthetic_summary:
        return "skipped", "missing_summary"

    eval_item = eval_map.get(doc_id)
    if not eval_item:
        return "skipped", "missing_eval_doc"

    signatures = extract_predicted_signatures(eval_item)
    if not signatures:
        return "skipped", "empty_predicted_articles"

    law_context = build_law_context(signatures, law_retriever)
    retrieved_clause_texts = law_context.get("retrieved_clause_texts", [])
    if not retrieved_clause_texts:
        return "skipped", "missing_law_context"

    system_prompt, user_prompt = build_prompts(
        doc_id=doc_id,
        synthetic_summary=synthetic_summary,
        defendants=defendants,
        signatures=signatures,
        law_context=law_context,
    )

    warnings = list(law_context.get("warnings", []))
    try:
        if provider == "aistudio":
            parsed, usage = generate_with_aistudio(model_name, system_prompt, user_prompt)
        else:
            parsed, usage = generate_with_openrouter(model_name, system_prompt, user_prompt)
    except (json.JSONDecodeError, ValidationError) as exc:
        output = {
            "doc_id": doc_id,
            "source_file": case_path.name,
            "provider": provider,
            "model": model_name,
            "synthetic_summary": synthetic_summary,
            "defendants": defendants,
            "predicted_articles": eval_item.get("predicted_articles", []),
            "predicted_articles_full": eval_item.get("predicted_articles_full", []),
            "law_context": law_context,
            "generated_verdict": None,
            "_warnings": warnings + [f"parse_error:{exc}"],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=2)
        return "failed", "parse_error"
    except Exception as exc:  # noqa: BLE001
        output = {
            "doc_id": doc_id,
            "source_file": case_path.name,
            "provider": provider,
            "model": model_name,
            "synthetic_summary": synthetic_summary,
            "defendants": defendants,
            "predicted_articles": eval_item.get("predicted_articles", []),
            "predicted_articles_full": eval_item.get("predicted_articles_full", []),
            "law_context": law_context,
            "generated_verdict": None,
            "_warnings": warnings + [f"generation_error:{exc}"],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=2)
        return "failed", "generation_error"

    output = {
        "doc_id": doc_id,
        "source_file": case_path.name,
        "provider": provider,
        "model": model_name,
        "synthetic_summary": synthetic_summary,
        "defendants": defendants,
        "predicted_articles": eval_item.get("predicted_articles", []),
        "predicted_articles_full": eval_item.get("predicted_articles_full", []),
        "law_context": law_context,
        "generated_verdict": parsed.model_dump(exclude_none=True),
        "_usage": usage,
        "_warnings": warnings,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    return "processed", "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PHAN_QUYET_CUA_TOA_SO_THAM-style verdicts from test Synthetic_summary, "
            "eval_results predicted articles, and law_doc.json context."
        )
    )
    parser.add_argument("--test-dir", default="chunk/test", help="Folder containing test JSON files")
    parser.add_argument("--eval-results", default="eval_results.json", help="Path to eval_results.json")
    parser.add_argument("--law-doc", default="law_doc.json", help="Path to law_doc.json")
    parser.add_argument(
        "--output-dir",
        default="output/generated_verdict_from_eval",
        help="Directory for generated output JSON files",
    )
    parser.add_argument(
        "--provider",
        choices=["aistudio", "openrouter"],
        default="aistudio",
        help="Gemma provider backend",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name override. Defaults: aistudio=gemma-4-31b-it, "
            "openrouter=google/gemma-4-31b-it"
        ),
    )
    parser.add_argument("--first-n", type=int, default=None, help="Process first N files")
    parser.add_argument("--file-list", default=None, help="Comma-separated list of filenames to process")
    parser.add_argument("--file-list-path", default=None, help="Path to txt file with one filename per line")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="Overwrite files that already exist in output-dir",
    )

    args = parser.parse_args()

    model_name = args.model
    if not model_name:
        model_name = "gemma-4-31b-it" if args.provider == "aistudio" else "google/gemma-4-31b-it"

    test_dir = Path(args.test_dir)
    eval_results_path = Path(args.eval_results)
    law_doc_path = Path(args.law_doc)
    output_dir = Path(args.output_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    if not eval_results_path.exists():
        raise FileNotFoundError(f"Eval results file does not exist: {eval_results_path}")
    if not law_doc_path.exists():
        raise FileNotFoundError(f"Law doc file does not exist: {law_doc_path}")

    eval_map = load_eval_predictions(eval_results_path)
    law_retriever = LawClauseRetriever(law_doc_path)
    files = resolve_input_files(
        test_dir=test_dir,
        first_n=args.first_n,
        file_list=args.file_list,
        file_list_path=args.file_list_path,
    )

    print(f"Found {len(files)} file(s) to process")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Eval map size: {len(eval_map)}")
    print(f"Law index size (Dieu): {len(getattr(law_retriever, '_dieu_index', {}))}")
    print("-" * 60)

    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()

    for path in files:
        status, reason = process_case(
            case_path=path,
            eval_map=eval_map,
            law_retriever=law_retriever,
            output_dir=output_dir,
            provider=args.provider,
            model_name=model_name,
            reprocess=args.reprocess,
        )
        status_counter[status] += 1
        reason_counter[reason] += 1

        if status == "processed":
            print(f"Processed: {path.name}")
        elif status == "skipped":
            print(f"Skipped:   {path.name} ({reason})")
        else:
            print(f"Failed:    {path.name} ({reason})")

    print("-" * 60)
    print("DONE")
    print(f"Processed: {status_counter.get('processed', 0)}")
    print(f"Skipped:   {status_counter.get('skipped', 0)}")
    print(f"Failed:    {status_counter.get('failed', 0)}")

    if reason_counter:
        print("Reason summary:")
        for reason, count in reason_counter.most_common():
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
