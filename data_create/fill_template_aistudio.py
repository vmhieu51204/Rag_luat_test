import json
import os
import glob
import argparse
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
import google.genai as genai
from google.genai import types

# ==========================================
# 0. MODEL & PRICING CONFIGURATION
# ==========================================
MODEL_NAME = "gemma-4-31b-it"   # Google AI Studio model ID for Gemma 4 31B
INPUT_PRICE_PER_1M_USD  = 0.0   # Free tier on AI Studio
OUTPUT_PRICE_PER_1M_USD = 0.0   # Free tier on AI Studio
 
# Lazy-initialised client handle
_genai_client = None
 
 
def get_model():
    """Lazily initialize and return a Google GenAI client."""
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

# ==========================================
# 1. PYDANTIC SCHEMAS FOR LLM EXTRACTION
# ==========================================
class CanCuDieuLuat(BaseModel):
    Diem: Optional[str] = Field(None, description="Legal point (điểm)")
    Khoan: Optional[str] = Field(None, description="Legal clause (khoản)")
    Dieu: Optional[str] = Field(None, description="Legal article (Điều)")
    Bo_Luat_Va_Van_Ban_Khac: Optional[str] = Field(None, description="Law or other document (Bộ luật)")
 
 
class ThongTinBiCao(BaseModel):
    Ho_Ten: str
    Ngay_Sinh: Optional[str] = None
    Noi_Cu_Tru: Optional[str] = None
    Nghe_Nghiep: Optional[str] = None
    Trinh_Do_Van_Hoa: Optional[str] = None
    Dan_Toc: Optional[str] = None
    Gioi_Tinh: Optional[str] = None
    Ton_Giao: Optional[str] = None
    Quoc_Tich: Optional[str] = None
    Hoan_Canh_Gia_Dinh: Optional[str] = None
    Tien_An: Optional[str] = None
    Tien_Su: Optional[str] = None
    Chi_Tiet_Nhan_Than: Optional[str] = None
    Ngay_Tam_Giam: Optional[str] = None
    Trang_Thai_Co_Mat: Optional[str] = None
 
 
class DeNghiVienKiemSat(BaseModel):
    Bi_Cao: str
    Pham_Toi: str
    Can_Cu_Dieu_Luat: List[CanCuDieuLuat]
    Phat_Tu: Optional[str] = None
    Phat_Tien: Optional[str] = None
    An_Phi: Optional[str] = None
    Hinh_Phat_Bo_Sung: Optional[str] = None
    Trach_Nhiem_Dan_Su: Optional[str] = None
    Xu_Ly_Vat_Chung: Optional[str] = None
 
 
class PhanQuyetToaSoTham(BaseModel):
    Bi_Cao: str
    Can_Cu_Dieu_Luat: List[CanCuDieuLuat]
    Pham_Toi: str
    Phat_Tu: Optional[str] = None
    Phat_Tien: Optional[str] = None
    An_Phi: Optional[str] = None
    Hinh_Phat_Bo_Sung: Optional[str] = None
    Trach_Nhiem_Dan_Su: Optional[str] = None
    Xu_Ly_Vat_Chung: Optional[str] = None
 
 
class LLMExtractionOutput(BaseModel):
    Thong_Tin_Bi_Cao: List[ThongTinBiCao]
    De_Nghi_Cua_Vien_Kiem_Sat: List[DeNghiVienKiemSat]
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]
 
 
class VerdictOnlyOutput(BaseModel):
    PHAN_QUYET_CUA_TOA_SO_THAM: List[PhanQuyetToaSoTham]
 
 
# ==========================================
# 2. HELPERS
# ==========================================
 
# Fields that Google AI Studio's schema validator rejects
_UNSUPPORTED_SCHEMA_FIELDS = {"default", "title", "$schema", "$id", "examples", "contentEncoding"}
 
 
def sanitize_schema(node: dict) -> dict:
    """
    Recursively clean a Pydantic JSON schema into a form accepted by Google AI
    Studio's response_schema validator. Three passes are applied:
 
    Pass 1 - resolve_refs:
        Replace every {"$ref": "#/$defs/Foo"} with the inlined definition so
        the SDK never sees unresolved references.
 
    Pass 2 - flatten_anyof:
        Pydantic emits Optional[X] as {"anyOf": [{"type": "X"}, {"type": "null"}]}.
        Google rejects anyOf entirely. We collapse it:
          - anyOf/oneOf with exactly one non-null branch: unwrap that branch
            and add "nullable": true.
          - anyOf/oneOf that are purely object unions: keep only the first
            non-null branch (best-effort) and mark nullable if null was present.
 
    Pass 3 - clean:
        Strip keys in _UNSUPPORTED_SCHEMA_FIELDS plus "$defs".
    """
    if not isinstance(node, dict):
        return node
 
    # Pass 1: inline all $ref definitions
    defs = node.get("$defs", {})
 
    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                resolved = defs.get(ref_name, obj)
                return resolve_refs(dict(resolved))
            return {k: resolve_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_refs(i) for i in obj]
        return obj
 
    node = resolve_refs(node)
 
    # Pass 2: flatten anyOf / oneOf produced by Optional[X]
    def flatten_anyof(obj):
        if isinstance(obj, list):
            return [flatten_anyof(i) for i in obj]
        if not isinstance(obj, dict):
            return obj
 
        # Recurse children first so nested anyOf are handled bottom-up
        obj = {k: flatten_anyof(v) for k, v in obj.items()}
 
        union_key = None
        if "anyOf" in obj:
            union_key = "anyOf"
        elif "oneOf" in obj:
            union_key = "oneOf"
 
        if union_key:
            branches = obj[union_key]
            null_branches     = [b for b in branches if b.get("type") == "null"]
            non_null_branches = [b for b in branches if b.get("type") != "null"]
            has_null = bool(null_branches)
 
            # Build replacement without the union key
            base = {k: v for k, v in obj.items() if k != union_key}
 
            if len(non_null_branches) == 1:
                # Simple Optional[X] - merge the single branch in
                branch = dict(non_null_branches[0])
                base.update(branch)
            elif non_null_branches:
                # Multiple non-null branches - keep first as best-effort
                branch = dict(non_null_branches[0])
                base.update(branch)
 
            if has_null:
                base["nullable"] = True
 
            obj = base
 
        return obj
 
    node = flatten_anyof(node)
 
    # Pass 3: strip unsupported top-level / nested keys
    def clean(obj):
        if isinstance(obj, dict):
            return {
                k: clean(v)
                for k, v in obj.items()
                if k not in _UNSUPPORTED_SCHEMA_FIELDS and k != "$defs"
            }
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj
 
    return clean(node)
 
 
def get_response_schema(schema: type) -> dict:
    """Return a Google AI Studio-compatible schema dict from a Pydantic model class."""
    raw = schema.model_json_schema()
    return sanitize_schema(raw)
 
 
def build_json_schema_prompt(schema: type) -> str:
    """Return the JSON schema of a Pydantic model as a formatted string for prompts."""
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
 
 
def extract_json_from_response(text: str) -> str:
    """Strip markdown code fences and return raw JSON text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text
 
 
def parse_llm_json(response_text: str, schema: type) -> BaseModel:
    """Parse raw LLM text into a Pydantic model, stripping fences if needed."""
    cleaned = extract_json_from_response(response_text)
    data = json.loads(cleaned)
    return schema.model_validate(data)
 
 
def call_model(
    client,
    system_prompt: str,
    user_content: str,
    schema: type,
) -> tuple[BaseModel, dict]:
    """
    Call the Gemma model via Google AI Studio with JSON-mode output constrained
    to the given Pydantic schema. Returns (parsed_model, usage_dict).
 
    Google AI Studio supports `response_mime_type="application/json"` together
    with `response_schema` to enforce structured output natively — no manual
    fence-stripping required for the happy path, but we keep the fallback parser
    as a safety net.
    """
    # Merge system prompt into the user turn because Gemma instruction-tuned
    # models on AI Studio may not honour a separate system instruction; prepending
    # it is the most reliable approach.
    full_prompt = f"{system_prompt.strip()}\n\n{user_content.strip()}"
 
    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        # Pass a sanitized plain dict — NOT the raw Pydantic class — to avoid
        # "Unknown field for Schema: default" and unresolved $ref errors.
        response_schema=get_response_schema(schema),
    )
 
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
        config=generation_config,
    )
 
    # Extract token usage (metadata may be absent on free tier / some model versions)
    usage_meta = getattr(response, "usage_metadata", None)
    prompt_tokens      = getattr(usage_meta, "prompt_token_count",      0) or 0
    completion_tokens  = getattr(usage_meta, "candidates_token_count",  0) or 0
    total_tokens       = getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens) or 0
 
    usage = {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
    }
 
    # Try to use the native parsed object first; fall back to text parsing.
    raw_text = response.text or ""
    try:
        parsed = parse_llm_json(raw_text, schema)
    except Exception:
        # Last-resort: if response_schema worked, the SDK may expose a parsed
        # candidate directly on some future SDK versions.
        raise ValueError(
            f"Could not parse model response as {schema.__name__}.\n"
            f"Raw response:\n{raw_text[:500]}"
        )
 
    return parsed, usage
 
 
# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================
def process_caselaw_file(input_filepath: str, output_filepath: str, skip_existing: bool = True) -> float:
    """
    Processes a single stage2 JSON file, extracts data using Gemma 4 31B via
    Google AI Studio, saves the result to stage3, and returns the cost in USD.

    If skip_existing is True (default) and the output file already exists,
    the file is skipped and 0.0 is returned.
    """
    # --- Skip already-processed files ---
    if skip_existing and os.path.isfile(output_filepath):
        print(f"Skipped (already exists): {os.path.basename(input_filepath)}")
        return 0.0

    with open(input_filepath, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)
 
    filename = os.path.basename(input_filepath)
    file_cost = 0.0
    model = get_model()
    total_prompt_tokens     = 0
    total_completion_tokens = 0
    total_tokens            = 0
    usage_calls: list       = []
 
    def add_usage_call(call_name: str, usage: dict):
        nonlocal file_cost, total_prompt_tokens, total_completion_tokens, total_tokens
 
        prompt_tokens     = usage.get("prompt_tokens",     0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tok         = usage.get("total_tokens",      prompt_tokens + completion_tokens)
 
        prompt_cost     = (prompt_tokens     / 1_000_000) * INPUT_PRICE_PER_1M_USD
        completion_cost = (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M_USD
        call_cost       = prompt_cost + completion_cost
        file_cost      += call_cost
 
        total_prompt_tokens     += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens            += total_tok
 
        usage_calls.append({
            "name":               call_name,
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "total_tokens":       total_tok,
            "cost_usd":           round(call_cost, 6),
        })
 
    # ------------------------------------------------------------------
    # Build stage3 skeleton from non-LLM fields
    # ------------------------------------------------------------------
    stage3_data = {
        "THONG_TIN_CHUNG": {
            "Ma_Ban_An": filename.replace(".json", ""),
            "Thong_Tin_Bi_Cao": [],
            "Thong_Tin_Nguoi_Tham_Gia_To_Tung": stage2_data.get("Nguoi_tham_gia", ""),
        },
        "NOI_DUNG_VU_AN":        stage2_data.get("Noi_dung_vu_an", {}).get("Qua_trinh_dieu_tra", ""),
        "De_Nghi_Cua_Vien_Kiem_Sat": [],
        "NHAN_DINH_CUA_TOA_AN":  stage2_data.get("Nhan_dinh_cua_toa_an", {}),
        "PHAN_QUYET_CUA_TOA_SO_THAM": [],
    }
 
    llm_input_context = f"""
    --- DANH SÁCH BỊ CÁO VÀ NGƯỜI LIÊN QUAN ---
    {json.dumps(stage2_data.get('Danh_sach_bi_cao', []), ensure_ascii=False)}
 
    --- KẾT LUẬN CỦA CÁC BÊN (VIỆN KIỂM SÁT) ---
    {stage2_data.get('Noi_dung_vu_an', {}).get('Ket_luan_cac_ben', '')}
 
    --- QUYẾT ĐỊNH CỦA TÒA ÁN ---
    {stage2_data.get('QUYET_DINH', '')}
    """
 
    system_prompt = f"""
You are an expert legal data extraction AI. Extract the provided Vietnamese caselaw text into the exact JSON schema requested.
 
CRITICAL INSTRUCTION FOR LEGAL CITATIONS (Can_Cu_Dieu_Luat):
You must handle hierarchical legal citations like a "valid parenthesis" distribution problem.
Higher-level units (Điều) apply to preceding lower-level units (Khoản, Điểm) in the current phrase.
 
Rules for distribution:
1. "điểm s khoản 1, 2 Điều 51"
   -> (Dieu 51, Khoan 1, Diem s) AND (Dieu 51, Khoan 2, Diem null)
   Do NOT apply 'điểm s' to 'khoản 2'.
2. "điểm c, d khoản 2, khoản 5 Điều 355"
   -> (Dieu 355, Khoan 2, Diem c) AND (Dieu 355, Khoan 2, Diem d) AND (Dieu 355, Khoan 5, Diem null)
 
Respond with ONLY a valid JSON object. Do NOT include any explanation or markdown fences.
 
JSON SCHEMA:
{build_json_schema_prompt(LLMExtractionOutput)}
"""
 
    # ------------------------------------------------------------------
    # Main LLM call
    # ------------------------------------------------------------------
    try:
        extracted_data, usage = call_model(
            model, system_prompt, llm_input_context, LLMExtractionOutput
        )
        add_usage_call("llm_extraction", usage)
 
        stage3_data["THONG_TIN_CHUNG"]["Thong_Tin_Bi_Cao"] = [
            d.model_dump(exclude_none=True) for d in extracted_data.Thong_Tin_Bi_Cao
        ]
        stage3_data["De_Nghi_Cua_Vien_Kiem_Sat"] = [
            p.model_dump(exclude_none=True) for p in extracted_data.De_Nghi_Cua_Vien_Kiem_Sat
        ]
        stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"] = [
            v.model_dump(exclude_none=True) for v in extracted_data.PHAN_QUYET_CUA_TOA_SO_THAM
        ]
 
        # ------------------------------------------------------------------
        # Retry: verdict list empty but decision text present
        # ------------------------------------------------------------------
        if stage2_data.get("QUYET_DINH", "").strip() and not stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"]:
            retry_input = f"""
            --- QUYẾT ĐỊNH CỦA TÒA ÁN ---
            {stage2_data.get('QUYET_DINH', '')}
            """
            retry_system_prompt = f"""
You are an expert legal extractor.
Extract ONLY PHAN_QUYET_CUA_TOA_SO_THAM from the decision text.
Return at least one item when the text contains any declaration of guilt, sentence, or legal basis.
Keep legal citations in Can_Cu_Dieu_Luat with fields Diem, Khoan, Dieu, Bo_Luat_Va_Van_Ban_Khac.
 
Respond with ONLY a valid JSON object. Do NOT include any explanation or markdown fences.
 
JSON SCHEMA:
{build_json_schema_prompt(VerdictOnlyOutput)}
"""
            try:
                retry_data, retry_usage = call_model(
                    model, retry_system_prompt, retry_input, VerdictOnlyOutput
                )
                add_usage_call("llm_extraction_verdict_retry", retry_usage)
 
                retry_verdict = [
                    v.model_dump(exclude_none=True) for v in retry_data.PHAN_QUYET_CUA_TOA_SO_THAM
                ]
                if retry_verdict:
                    stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"] = retry_verdict
                else:
                    stage3_data.setdefault("_warnings", []).append(
                        "PHAN_QUYET_CUA_TOA_SO_THAM is empty after retry"
                    )
            except Exception as retry_error:
                print(f"  Retry extraction failed for {filename}: {retry_error}")
                stage3_data.setdefault("_warnings", []).append(
                    "Retry extraction for PHAN_QUYET_CUA_TOA_SO_THAM failed"
                )
 
        # ------------------------------------------------------------------
        # Usage block
        # ------------------------------------------------------------------
        stage3_data["_usage"] = {
            "model": MODEL_NAME,
            "api":   "Google AI Studio (google-genai)",
            "pricing": {
                "input_per_1m_usd":  INPUT_PRICE_PER_1M_USD,
                "output_per_1m_usd": OUTPUT_PRICE_PER_1M_USD,
            },
            "calls": usage_calls,
            "totals": {
                "prompt_tokens":     total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens":      total_tokens,
                "cost_usd":          round(file_cost, 6),
            },
        }
 
    except Exception as e:
        print(f"  Error extracting {filename}: {e}")
        stage3_data.setdefault("_warnings", []).append(f"Extraction failed: {e}")
 
    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(stage3_data, f, ensure_ascii=False, indent=2)
 
    print(f"Processed: {filename} | Cost: ${file_cost:.6f}")
    return file_cost
 
 
# ==========================================
# 4. BATCH PROCESSING & AGGREGATION
# ==========================================
def process_directory(input_dir: str, output_dir: str, skip_existing: bool = True):
    """Process all JSON files in input_dir and write results to output_dir."""
    total_pipeline_cost = 0.0
    skipped_count       = 0
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
 
    print(f"Found {len(input_files)} files to process.")
    if skip_existing:
        print("Mode: skipping already-processed files (use --reprocess to override).")
    print("-" * 40)
 
    for input_file in input_files:
        filename    = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("stage2", "stage3"))
        cost = process_caselaw_file(input_file, output_file, skip_existing=skip_existing)
        if cost == 0.0 and skip_existing and os.path.isfile(output_file):
            skipped_count += 1
        total_pipeline_cost += cost
 
    print("-" * 40)
    print("BATCH COMPLETE")
    print(f"Total Files Found:      {len(input_files)}")
    print(f"Skipped (exist):        {skipped_count}")
    print(f"Processed (new):        {len(input_files) - skipped_count}")
    print(f"FINAL TOTAL COST: ${total_pipeline_cost:.4f}")
 
 
def process_selected_files(input_dir: str, output_dir: str, selected_filenames: List[str], skip_existing: bool = True):
    """Process only the specified filenames from input_dir."""
    total_pipeline_cost = 0.0
    skipped_count       = 0
    existing_files: list = []
    missing_files:  list = []
 
    for filename in selected_filenames:
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            existing_files.append(input_path)
        else:
            missing_files.append(filename)
 
    print(f"Selected {len(selected_filenames)} files.")
    print(f"Found {len(existing_files)} files to process.")
    if skip_existing:
        print("Mode: skipping already-processed files (use --reprocess to override).")
    if missing_files:
        print(f"Missing {len(missing_files)} files: {', '.join(missing_files)}")
    print("-" * 40)
 
    for input_file in existing_files:
        filename    = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("stage2", "stage3"))
        cost = process_caselaw_file(input_file, output_file, skip_existing=skip_existing)
        if cost == 0.0 and skip_existing and os.path.isfile(output_file):
            skipped_count += 1
        total_pipeline_cost += cost
 
    print("-" * 40)
    print("BATCH COMPLETE")
    print(f"Total Files Found:      {len(existing_files)}")
    print(f"Skipped (exist):        {skipped_count}")
    print(f"Processed (new):        {len(existing_files) - skipped_count}")
    print(f"FINAL TOTAL COST: ${total_pipeline_cost:.4f}")
 
 
def parse_filename_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
 
 
def read_filename_list_file(list_file_path: str) -> List[str]:
    with open(list_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
 


# ==========================================
# 5. CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill stage3 template from stage2 fields using Gemma 4 31B via Google AI Studio."
    )
    parser.add_argument(
        "--input-dir",
        default="data_create/Chuong_XXII",
        help="Directory containing input stage2 JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="data_create/Chuong_XXII_aistudio",
        help="Directory to write output stage3 JSON files",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Process the first N files (sorted by filename) from input-dir",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Comma-separated list of filenames to process",
    )
    parser.add_argument(
        "--file-list-path",
        default=None,
        help="Path to a .txt file with one filename per line",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="Re-process and overwrite files that already exist in the output directory",
    )

    args = parser.parse_args()
    skip_existing = not args.reprocess

    list_mode_enabled   = bool(args.file_list or args.file_list_path)
    first_n_mode_enabled = args.first_n is not None

    if list_mode_enabled and first_n_mode_enabled:
        raise ValueError("Use either --first-n or file list mode, not both.")

    if list_mode_enabled:
        selected_filenames: List[str] = []
        if args.file_list:
            selected_filenames.extend(parse_filename_list(args.file_list))
        if args.file_list_path:
            selected_filenames.extend(read_filename_list_file(args.file_list_path))

        # Deduplicate while preserving order
        deduped: List[str] = []
        seen: set = set()
        for name in selected_filenames:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        if not deduped:
            raise ValueError("No filenames were provided in list mode.")

        process_selected_files(args.input_dir, args.output_dir, deduped, skip_existing=skip_existing)

    elif first_n_mode_enabled:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")

        all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        if not all_files:
            raise ValueError(f"No JSON files found in {args.input_dir}")

        selected = [os.path.basename(p) for p in all_files[: args.first_n]]
        process_selected_files(args.input_dir, args.output_dir, selected, skip_existing=skip_existing)

    else:
        process_directory(args.input_dir, args.output_dir, skip_existing=skip_existing)