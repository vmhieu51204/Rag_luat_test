"""Patch processed JSON files to match the latest schema.

Phase 1:
- Scan processed files for duplicate defendants in PHAN_QUYET_CUA_TOA_SO_THAM.
- Re-run data_create/fill_template_openrouter.py on matching extracted_fields JSON.

Phase 2:
- Extract case-level Xu_Ly_Vat_Chung from QUYET_DINH using Gemma 4.
- Remove any per-defendant Xu_Ly_Vat_Chung and set top-level Xu_Ly_Vat_Chung.
- Skip files re-extracted in phase 1.

Final normalization:
- Ensure output structure aligns with data_create/schemas.py.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from data_create.schemas import DeNghiVKS, LLMExtractionOutput, build_json_schema_prompt
from rag.llm.providers import (
    LLMProvider,
    default_model_for_provider,
    generate_structured_output,
    generate_structured_output_with_fallback,
)


class XuLyVatChungOutput(BaseModel):
    Xu_Ly_Vat_Chung: Optional[str] = Field(
        None,
        description="Case-level handling of physical evidence from the court decision.",
    )


class DeNghiCuaVienKiemSatOutput(BaseModel):
    De_Nghi_Cua_Vien_Kiem_Sat: list[DeNghiVKS] = Field(default_factory=list)


class PatchReport(BaseModel):
    phase1_reextracted: list[str] = Field(default_factory=list)
    phase2_skipped: list[str] = Field(default_factory=list)
    phase2_updated: list[str] = Field(default_factory=list)
    missing_extracted: list[str] = Field(default_factory=list)
    missing_processed: list[str] = Field(default_factory=list)
    unreadable_files: list[str] = Field(default_factory=list)
    schema_validation_failed: list[str] = Field(default_factory=list)
    normalized_files: list[str] = Field(default_factory=list)
    dry_run: bool = False

    def as_json(self) -> dict:
        return self.model_dump()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _model_tier(model_name: str | None, provider: LLMProvider) -> str:
    if provider == LLMProvider.AISTUDIO:
        return "free"
    if provider == LLMProvider.OPENAI:
        return "paid"
    value = model_name or default_model_for_provider(provider)
    if value.endswith(":free"):
        return "free"
    if value.endswith(":paid"):
        return "paid"
    return "paid"


def _iter_json_files(folder: Path) -> Iterable[Path]:
    return sorted(path for path in folder.glob("*.json") if path.is_file())


def _has_duplicate_defendants(processed_data: dict) -> bool:
    def has_duplicates(items: object) -> bool:
        if not isinstance(items, list):
            return False

        counts: dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Bi_Cao") or "").strip()
            if not name:
                continue
            key = _normalize_name(name)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        return any(count > 1 for count in counts.values())

    verdicts = processed_data.get("PHAN_QUYET_CUA_TOA_SO_THAM")
    de_nghi_vks = processed_data.get("De_Nghi_Cua_Vien_Kiem_Sat")

    return has_duplicates(verdicts) or has_duplicates(de_nghi_vks)


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


def _extract_xu_ly_vat_chung_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"(?i)(Về\s+vật\s+chứng\s*:.*)",
        r"(?i)(Về\s+xử\s+lý\s+vật\s+chứng\s*:.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            snippet = match.group(1).strip()
            # Stop at next section heading if present.
            split_match = re.split(r"\n\s*\[[0-9]+\]|\n\s*Về\s+án\s+phí\s*:", snippet, maxsplit=1)
            return split_match[0].strip()
    return None


def _extract_xu_ly_vat_chung(decision_text: str, nhan_dinh_text: str, model_name: str) -> tuple[Optional[str], dict[str, object]]:
    if not decision_text.strip() and not nhan_dinh_text.strip():
        return None, {"provider": "text_fallback", "tier": "n/a", "model": "n/a"}

    system_prompt = (
        "You are a legal extraction assistant. "
        "Extract only the case-level handling of physical evidence from the court decision. "
        "If the decision does not mention handling of physical evidence, return null. "
        "Respond with ONLY valid JSON for the schema. "
        "Return exactly one JSON object like: {\"Xu_Ly_Vat_Chung\": \"...\"}."
        "Keep the extracted concise, focused on physical evidence handling, and do not include any other information or explanation."
    )
    user_prompt = f"QUYET_DINH:\n{decision_text.strip()}"

    usage: dict[str, object] = {}
    result = None
    try:
        result, usage = generate_structured_output_with_fallback(
            preferred_provider=LLMProvider.AISTUDIO,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=XuLyVatChungOutput,
        )
    except (RuntimeError, ValidationError) as exc:
        print(f"[WARN] Failed to extract Xu_Ly_Vat_Chung: {exc}")
        result = None

    value = result.Xu_Ly_Vat_Chung if result else None
    if value is None:
        if nhan_dinh_text.strip():
            fallback_prompt = f"NHAN_DINH_CUA_TOA_AN:\n{nhan_dinh_text.strip()}"
            try:
                result, usage = generate_structured_output_with_fallback(
                    preferred_provider=LLMProvider.AISTUDIO,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=fallback_prompt,
                    output_model=XuLyVatChungOutput,
                )
                value = result.Xu_Ly_Vat_Chung
            except (RuntimeError, ValidationError) as exc:
                print(f"[WARN] Failed fallback extraction for Xu_Ly_Vat_Chung: {exc}")
                value = None
        if value is None:
            return _extract_xu_ly_vat_chung_from_text(nhan_dinh_text or decision_text), {
                "provider": "text_fallback",
                "tier": "n/a",
                "model": "n/a",
            }

    provider_name = str(usage.get("provider") or LLMProvider.AISTUDIO.value)
    actual_model = str(usage.get("model") or model_name or default_model_for_provider(LLMProvider.AISTUDIO))
    provider_enum = LLMProvider(provider_name) if provider_name in LLMProvider._value2member_map_ else LLMProvider.AISTUDIO
    return (str(value).strip() or None), {
        "provider": provider_name,
        "tier": _model_tier(actual_model, provider_enum),
        "model": actual_model,
    }


def _extract_de_nghi_cua_vien_kiem_sat(
    *,
    danh_sach_bi_cao: object,
    ket_luan_cac_ben: object,
    model_name: str,
) -> tuple[list[dict], dict[str, object]]:
    skeleton = build_json_schema_prompt(DeNghiCuaVienKiemSatOutput)
    system_prompt = (
        "You are a legal extraction assistant. "
        "Extract prosecutor recommendations (De_Nghi_Cua_Vien_Kiem_Sat) based on the provided text. "
        "Use the KET_LUAN_CAC_BEN section as the primary source. "
        "If no prosecutor recommendation is present, return an empty list. "
        "Match Bi_Cao exactly to the defendant list. "
        "Respond with ONLY valid JSON for the schema.\n\n"
        f"JSON SCHEMA:\n{skeleton}"
    )
    user_prompt = (
        "DANH_SACH_BI_CAO:\n"
        f"{json.dumps(danh_sach_bi_cao or [], ensure_ascii=False)}\n\n"
        "KET_LUAN_CAC_BEN:\n"
        f"{json.dumps(ket_luan_cac_ben or '', ensure_ascii=False)}"
    )

    usage: dict[str, object] = {}
    result = None
    try:
        result, usage = generate_structured_output_with_fallback(
            preferred_provider=LLMProvider.AISTUDIO,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=DeNghiCuaVienKiemSatOutput,
        )
    except (RuntimeError, ValidationError) as exc:
        print(f"[WARN] Failed to extract De_Nghi_Cua_Vien_Kiem_Sat: {exc}")
        result = None

    de_nghi = [item.model_dump(exclude_none=True) for item in (result.De_Nghi_Cua_Vien_Kiem_Sat if result else [])]
    provider_name = str(usage.get("provider") or LLMProvider.AISTUDIO.value)
    actual_model = str(usage.get("model") or model_name or default_model_for_provider(LLMProvider.AISTUDIO))
    provider_enum = LLMProvider(provider_name) if provider_name in LLMProvider._value2member_map_ else LLMProvider.AISTUDIO
    return de_nghi, {
        "provider": provider_name,
        "tier": _model_tier(actual_model, provider_enum),
        "model": actual_model,
    }


def _remove_per_defendant_xu_ly(processed_data: dict) -> None:
    verdicts = processed_data.get("PHAN_QUYET_CUA_TOA_SO_THAM")
    if not isinstance(verdicts, list):
        return
    for item in verdicts:
        if isinstance(item, dict) and "Xu_Ly_Vat_Chung" in item:
            item.pop("Xu_Ly_Vat_Chung", None)


def _normalize_pham_toi(processed_data: dict) -> None:
    verdicts = processed_data.get("PHAN_QUYET_CUA_TOA_SO_THAM")
    if not isinstance(verdicts, list):
        return
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        pham_toi = item.get("Pham_Toi")
        if isinstance(pham_toi, str):
            text = pham_toi.strip()
            item["Pham_Toi"] = [text] if text else []


def _normalize_de_nghi_cua_vien_kiem_sat(processed_data: dict) -> None:
    de_nghi = processed_data.get("De_Nghi_Cua_Vien_Kiem_Sat")
    if not isinstance(de_nghi, list):
        de_nghi = []

    for item in de_nghi:
        if not isinstance(item, dict):
            continue
        pham_toi = item.get("Pham_Toi")
        if isinstance(pham_toi, str):
            text = pham_toi.strip()
            item["Pham_Toi"] = [text] if text else []

    processed_data["De_Nghi_Cua_Vien_Kiem_Sat"] = de_nghi


def _missing_de_nghi_cua_vien_kiem_sat(processed_data: dict) -> bool:
    de_nghi = processed_data.get("De_Nghi_Cua_Vien_Kiem_Sat")
    if de_nghi is None:
        return True
    if isinstance(de_nghi, list):
        return len(de_nghi) == 0
    return True


def _validate_structure(processed_data: dict, file_path: Path, report: PatchReport) -> None:
    thong_tin = processed_data.get("THONG_TIN_CHUNG")
    thong_tin = thong_tin if isinstance(thong_tin, dict) else {}

    payload = {
        "Thong_Tin_Bi_Cao": thong_tin.get("Thong_Tin_Bi_Cao") or [],
        "De_Nghi_Cua_Vien_Kiem_Sat": processed_data.get("De_Nghi_Cua_Vien_Kiem_Sat") or [],
        "PHAN_QUYET_CUA_TOA_SO_THAM": processed_data.get("PHAN_QUYET_CUA_TOA_SO_THAM") or [],
        "Xu_Ly_Vat_Chung": processed_data.get("Xu_Ly_Vat_Chung"),
    }

    try:
        LLMExtractionOutput.model_validate(payload)
    except ValidationError as exc:
        print(f"[WARN] Schema validation failed for {file_path.name}: {exc}")
        report.schema_validation_failed.append(file_path.name)


def phase1_reextract(
    *,
    processed_dir: Path,
    extracted_dir: Path,
    dry_run: bool,
    report: PatchReport,
) -> set[str]:
    re_extracted: set[str] = set()

    for processed_path in _iter_json_files(processed_dir):
        try:
            processed_data = _read_json(processed_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable file {processed_path.name}: {exc}")
            report.unreadable_files.append(processed_path.name)
            continue

        # Only backfill De_Nghi_Cua_Vien_Kiem_Sat when missing or empty.
        if not _missing_de_nghi_cua_vien_kiem_sat(processed_data):
            continue

        extracted_path = extracted_dir / processed_path.name
        if not extracted_path.exists():
            print(f"[WARN] Missing extracted file for {processed_path.name}")
            report.missing_extracted.append(processed_path.name)
            continue

        re_extracted.add(processed_path.name)
        if dry_run:
            print(f"[DRY-RUN] Backfill De_Nghi_Cua_Vien_Kiem_Sat for {processed_path.name}")
            report.phase1_reextracted.append(processed_path.name)
            continue

        try:
            extracted_data = _read_json(extracted_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable extracted file {extracted_path.name}: {exc}")
            report.unreadable_files.append(extracted_path.name)
            continue

        de_nghi, usage_meta = _extract_de_nghi_cua_vien_kiem_sat(
            danh_sach_bi_cao=extracted_data.get("Danh_sach_bi_cao"),
            ket_luan_cac_ben=(extracted_data.get("Noi_dung_vu_an") or {}).get("Ket_luan_cac_ben"),
            model_name=default_model_for_provider(LLMProvider.AISTUDIO),
        )

        processed_data["De_Nghi_Cua_Vien_Kiem_Sat"] = de_nghi
        _normalize_de_nghi_cua_vien_kiem_sat(processed_data)
        _write_json(processed_path, processed_data)
        _validate_structure(processed_data, processed_path, report)
        report.phase1_reextracted.append(processed_path.name)
        provider_name = str(usage_meta.get("provider") or "n/a")
        tier = str(usage_meta.get("tier") or "n/a")
        actual_model = str(usage_meta.get("model") or "n/a")
        print(
            f"[INFO] Backfilled De_Nghi_Cua_Vien_Kiem_Sat for {processed_path.name} "
            f"(provider={provider_name}, tier={tier}, model={actual_model})"
        )

    return re_extracted


def phase2_extract_xu_ly(
    *,
    processed_dir: Path,
    extracted_dir: Path,
    skip_files: set[str],
    dry_run: bool,
    model_name: str,
    report: PatchReport,
) -> None:
    for extracted_path in _iter_json_files(extracted_dir):
        if extracted_path.name in skip_files:
            print(f"[INFO] Skip phase 2 for {extracted_path.name} (re-extracted in phase 1)")
            report.phase2_skipped.append(extracted_path.name)
            continue

        processed_path = processed_dir / extracted_path.name
        if not processed_path.exists():
            print(f"[WARN] Missing processed file for {extracted_path.name}")
            report.missing_processed.append(extracted_path.name)
            continue

        try:
            existing_processed = _read_json(processed_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable processed file {processed_path.name}: {exc}")
            report.unreadable_files.append(processed_path.name)
            continue

        if "Xu_Ly_Vat_Chung" in existing_processed:
            print(
                f"[INFO] Skip phase 2 for {processed_path.name} "
                "(Xu_Ly_Vat_Chung key already present)"
            )
            report.phase2_skipped.append(processed_path.name)
            continue

        try:
            extracted_data = _read_json(extracted_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable extracted file {extracted_path.name}: {exc}")
            report.unreadable_files.append(extracted_path.name)
            continue

        decision_text = str(extracted_data.get("QUYET_DINH") or "")
        nhan_dinh_text = _flatten_nhan_dinh_text(extracted_data.get("NHAN_DINH_CUA_TOA_AN"))
        if dry_run:
            print(f"[DRY-RUN] Extract Xu_Ly_Vat_Chung for {extracted_path.name}")
            report.phase2_updated.append(extracted_path.name)
            continue

        xu_ly_vat_chung, usage_meta = _extract_xu_ly_vat_chung(decision_text, nhan_dinh_text, model_name)
        provider_name = str(usage_meta.get("provider") or "text_fallback")
        tier = str(usage_meta.get("tier") or "n/a")
        actual_model = str(usage_meta.get("model") or "n/a")

        processed_data = existing_processed

        _remove_per_defendant_xu_ly(processed_data)
        _normalize_pham_toi(processed_data)
        _normalize_de_nghi_cua_vien_kiem_sat(processed_data)
        processed_data["Xu_Ly_Vat_Chung"] = xu_ly_vat_chung
        _write_json(processed_path, processed_data)
        _validate_structure(processed_data, processed_path, report)
        report.phase2_updated.append(processed_path.name)
        print(
            f"[INFO] Updated Xu_Ly_Vat_Chung for {processed_path.name} "
            f"(provider={provider_name}, tier={tier}, model={actual_model})"
        )


def normalize_outputs(processed_dir: Path, dry_run: bool, report: PatchReport) -> None:
    for processed_path in _iter_json_files(processed_dir):
        try:
            processed_data = _read_json(processed_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable file {processed_path.name}: {exc}")
            report.unreadable_files.append(processed_path.name)
            continue

        _remove_per_defendant_xu_ly(processed_data)
        _normalize_pham_toi(processed_data)
        _normalize_de_nghi_cua_vien_kiem_sat(processed_data)
        if "Xu_Ly_Vat_Chung" not in processed_data:
            processed_data["Xu_Ly_Vat_Chung"] = None

        if dry_run:
            print(f"[DRY-RUN] Normalize structure for {processed_path.name}")
            report.normalized_files.append(processed_path.name)
            continue

        _write_json(processed_path, processed_data)
        _validate_structure(processed_data, processed_path, report)
        report.normalized_files.append(processed_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch processed JSON files to match the latest schema.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data_create/filled_template_openai",
        help="Directory containing processed JSON files",
    )
    parser.add_argument(
        "--extracted-dir",
        default="data_create/extracted_fields",
        help="Directory containing extracted input JSON files",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Phase to run (1, 2, or all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying files or calling the LLM",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the default Gemma model name for phase 2",
    )
    parser.add_argument(
        "--report-path",
        default="data_create/patch_report.json",
        help="Write a JSON summary report to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    extracted_dir = Path(args.extracted_dir)

    model_name = args.model_name or default_model_for_provider(LLMProvider.AISTUDIO)
    report = PatchReport(dry_run=args.dry_run)

    re_extracted: set[str] = set()
    if args.phase in {"1", "all"}:
        re_extracted = phase1_reextract(
            processed_dir=processed_dir,
            extracted_dir=extracted_dir,
            dry_run=args.dry_run,
            report=report,
        )

    if args.phase in {"2", "all"}:
        phase2_extract_xu_ly(
            processed_dir=processed_dir,
            extracted_dir=extracted_dir,
            skip_files=re_extracted,
            dry_run=args.dry_run,
            model_name=model_name,
            report=report,
        )

    if args.phase == "all":
        normalize_outputs(processed_dir, args.dry_run, report)

    report_path = Path(args.report_path)
    _write_json(report_path, report.as_json())
    print(f"[INFO] Wrote report to {report_path}")

    print("[DONE] Patch process completed.")


if __name__ == "__main__":
    main()
