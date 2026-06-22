import json
import os
import glob
import argparse
import time
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from dotenv import load_dotenv

from data_create.schemas import (
    ThongTinBiCao,
    PhanQuyetToaSoTham,
    LLMExtractionOutput,
    VerdictOnlyOutput,
    build_json_schema_prompt,
)

load_dotenv()
import google.genai as genai
from google.genai import types

# ==========================================
# 0. MODEL CONFIGURATION
# ==========================================
# Lazy-initialised client handles
_client_cache: dict[tuple[str, int | None, int | None], Any] = {}
_openrouter_client = None

DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_RETRY_ATTEMPTS = 1


def get_aistudio_api_keys() -> list[tuple[str, str]]:
    """Return configured AI Studio API keys in the order they should be tried."""
    key_names = ("GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3")
    return [
        (key_name, api_key)
        for key_name in key_names
        if (api_key := os.environ.get(key_name))
    ]


def build_http_options(
    request_timeout_seconds: float | None = None,
    retry_attempts: int | None = None,
) -> tuple[types.HttpOptions | None, int | None, int | None]:
    timeout_ms = None
    if request_timeout_seconds is not None and request_timeout_seconds > 0:
        timeout_ms = int(request_timeout_seconds * 1000)

    resolved_retry_attempts = None
    if retry_attempts is not None:
        resolved_retry_attempts = max(1, int(retry_attempts))

    http_options = None
    if timeout_ms is not None or resolved_retry_attempts is not None:
        http_options = types.HttpOptions(timeout=timeout_ms)
        if resolved_retry_attempts is not None:
            http_options.retry_options = types.HttpRetryOptions(
                attempts=resolved_retry_attempts
            )

    return http_options, timeout_ms, resolved_retry_attempts


def get_genai_clients(
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retry_attempts: int | None = DEFAULT_RETRY_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Lazily initialize one Google GenAI client for each configured API key."""
    api_keys = get_aistudio_api_keys()
    if not api_keys:
        raise EnvironmentError(
            "No AI Studio API keys are set. Expected GOOGLE_API_KEY, "
            "GOOGLE_API_KEY_2, and/or GOOGLE_API_KEY_3."
        )

    http_options, timeout_ms, resolved_retry_attempts = build_http_options(
        request_timeout_seconds=request_timeout_seconds,
        retry_attempts=retry_attempts,
    )

    clients: list[dict[str, Any]] = []
    init_errors: list[str] = []
    for key_name, api_key in api_keys:
        cache_key = (key_name, timeout_ms, resolved_retry_attempts)
        try:
            if cache_key not in _client_cache:
                _client_cache[cache_key] = genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                )
            clients.append({"key_name": key_name, "client": _client_cache[cache_key]})
        except Exception as exc:
            init_errors.append(f"{key_name}: {exc}")

    if not clients:
        raise EnvironmentError(
            "No AI Studio clients could be initialized. Errors: "
            + " | ".join(init_errors)
        )

    return clients


def get_openrouter_client(request_timeout_seconds: float | None = None):
    """Lazily initialize and return an OpenRouter client."""
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
        timeout = (
            request_timeout_seconds
            if request_timeout_seconds and request_timeout_seconds > 0
            else 30.0
        )
        _openrouter_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=0,
        )
    return _openrouter_client

 
 
 
 
 
 
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
    system_prompt: str,
    user_content: str,
    schema: type,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> tuple[BaseModel, dict]:
    """
    Call the model with fallback:
    1. AI Studio (gemma-4-31b-it) across GOOGLE_API_KEY, GOOGLE_API_KEY_2,
       and GOOGLE_API_KEY_3 in order.
    2. OpenRouter (google/gemma-4-31b-it:free)
    3. OpenRouter (google/gemma-4-31b-it)
    """
    full_prompt = f"{system_prompt.strip()}\n\n{user_content.strip()}"
    errors = []

    # 1. AI Studio key rotation
    try:
        clients = get_genai_clients(
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
        )
    except Exception as e:
        clients = []
        error_msg = f"AI Studio client initialization failed: {e}"
        errors.append(error_msg)
        print(f"    [Fallback] {error_msg}")

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=get_response_schema(schema),
    )

    for client_info in clients:
        key_name = str(client_info.get("key_name") or "unknown_key")
        client = client_info["client"]
        try:
            print(f"    Calling AI Studio with {key_name}...")
            try:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=full_prompt,
                    config=generation_config,
                    request_options={"timeout": request_timeout_seconds},
                )
            except TypeError:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=full_prompt,
                    config=generation_config,
                )

            usage_meta = getattr(response, "usage_metadata", None)
            prompt_tokens      = getattr(usage_meta, "prompt_token_count",      0) or 0
            completion_tokens  = getattr(usage_meta, "candidates_token_count",  0) or 0
            total_tokens       = getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens) or 0

            usage = {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      total_tokens,
                "provider":          "aistudio",
                "key_name":          key_name,
                "model":             "gemma-4-31b-it",
            }

            raw_text = getattr(response, "text", "") or ""
            parsed = parse_llm_json(raw_text, schema)
            return parsed, usage
        except Exception as e:
            error_msg = f"AI Studio ({key_name}) failed: {e}"
            errors.append(error_msg)
            print(f"    [Fallback] {error_msg}")

    # Fallbacks via OpenRouter
    or_client = get_openrouter_client(request_timeout_seconds)
    
    fallback_models = [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-31b-it"
    ]
    
    for or_model in fallback_models:
        try:
            response = or_client.chat.completions.create(
                model=or_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                timeout=request_timeout_seconds,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = parse_llm_json(raw_text, schema)
            
            usage_obj = getattr(response, "usage", None)
            prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider": "openrouter",
                "model": or_model,
            }
            return parsed, usage
        except Exception as e:
            error_msg = f"OpenRouter ({or_model}) failed: {e}"
            errors.append(error_msg)
            print(f"    [Fallback] {error_msg}")

    raise ValueError(
        f"All models failed to parse {schema.__name__}.\nErrors: {errors}"
    )
 
 
# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================
def process_caselaw_file(
    input_filepath: str,
    output_filepath: str,
    skip_existing: bool = True,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> tuple[float, str]:
    """
    Processes a single stage2 JSON file, extracts data using Gemma 4 31B via
    Google AI Studio, saves the result to stage3, and returns the cost in USD.

    If skip_existing is True (default) and the output file already exists,
    the file is skipped and 0.0 is returned.
    """
    # --- Skip already-processed files ---
    if skip_existing and os.path.isfile(output_filepath):
        print(f"Skipped (already exists): {os.path.basename(input_filepath)}")
        return 0.0, "skipped"

    with open(input_filepath, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)
 
    filename = os.path.basename(input_filepath)
    file_cost = 0.0
    total_prompt_tokens     = 0
    total_completion_tokens = 0
    total_tokens            = 0
    usage_calls: list       = []
 
    def add_usage_call(call_name: str, usage: dict):
        nonlocal file_cost, total_prompt_tokens, total_completion_tokens, total_tokens
 
        prompt_tokens     = usage.get("prompt_tokens",     0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tok         = usage.get("total_tokens",      prompt_tokens + completion_tokens)
 
        model_used = usage.get("model", "")
        if model_used == "google/gemma-4-31b-it":
            in_price = 0.02
            out_price = 1.25
        else:
            in_price = 0.0
            out_price = 0.0
 
        prompt_cost     = (prompt_tokens     / 1_000_000) * in_price
        completion_cost = (completion_tokens / 1_000_000) * out_price
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
            "provider":           usage.get("provider", "unknown"),
            "key_name":           usage.get("key_name"),
            "model":              model_used,
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
        "NHAN_DINH_CUA_TOA_AN":  stage2_data.get("Nhan_dinh_cua_toa_an", {}),
        "De_Nghi_Cua_Vien_Kiem_Sat": [],
        "PHAN_QUYET_CUA_TOA_SO_THAM": [],
    }
 
    llm_input_context = f"""
    --- DANH SÁCH BỊ CÁO VÀ NGƯỜI LIÊN QUAN ---
    {json.dumps(stage2_data.get('Danh_sach_bi_cao', []), ensure_ascii=False)}
 
    --- KẾT LUẬN CỦA CÁC BÊN (VIỆN KIỂM SÁT) ---
    {stage2_data.get('Noi_dung_vu_an', {}).get('Ket_luan_cac_ben', '')}

    --- NHẬN ĐỊNH CỦA TÒA ÁN ---
    {json.dumps(stage2_data.get('Nhan_dinh_cua_toa_an', {}), ensure_ascii=False, indent=2)}
 
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

CRITICAL INSTRUCTION FOR NHẬN ĐỊNH CỦA TÒA ÁN (Indexes):
In the "NHẬN ĐỊNH CỦA TÒA ÁN" section, there are numerical keys (e.g. "[1]", "[2]") with text describing particular matters (e.g., aggravating/mitigating factors, additional fine).
For fields requiring an index (Hinh_Phat_Bo_Sung_index, Tang_nang_index, Giam_nhe_index):
1. First, select the numerical index corresponding to the text that describes the matter.
2. Extract the integer from the key (e.g., 1 for "[1]").
3. Then process the text and fill the required text field in the schema (e.g., Tang_nang).
4. Save the integer index to the corresponding index field in the schema (e.g., Tang_nang_index).
5. If the matter required in the schema is not present in any index, leave the text field as an empty string ("") and the index as null.
 
Respond with ONLY a valid JSON object. Do NOT include any explanation or markdown fences.
 
JSON SCHEMA:
{build_json_schema_prompt(LLMExtractionOutput)}
"""
 
    # ------------------------------------------------------------------
    # Main LLM call
    # ------------------------------------------------------------------
    status = "success"
    provider = "unknown"
    try:
        print(f"  --> Generating {filename} (this may take 1-3 minutes)...")
        extracted_data, usage = call_model(
            system_prompt,
            llm_input_context,
            LLMExtractionOutput,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
        )
        provider = usage.get("provider", "unknown")
        add_usage_call("llm_extraction", usage)
 
        stage3_data["THONG_TIN_CHUNG"]["Thong_Tin_Bi_Cao"] = [
            d.model_dump(exclude_none=True) for d in extracted_data.Thong_Tin_Bi_Cao
        ]
        stage3_data["De_Nghi_Cua_Vien_Kiem_Sat"] = [
            d.model_dump(exclude_none=True) for d in extracted_data.De_Nghi_Cua_Vien_Kiem_Sat
        ]
        stage3_data["PHAN_QUYET_CUA_TOA_SO_THAM"] = [
            v.model_dump(exclude_none=True) for v in extracted_data.PHAN_QUYET_CUA_TOA_SO_THAM
        ]

        for item in stage3_data["De_Nghi_Cua_Vien_Kiem_Sat"]:
            if not isinstance(item, dict):
                continue
            pham_toi = item.get("Pham_Toi")
            if isinstance(pham_toi, str):
                text = pham_toi.strip()
                item["Pham_Toi"] = [text] if text else []

        missing_fields = False
        for v in extracted_data.PHAN_QUYET_CUA_TOA_SO_THAM:
            if not v.Tang_nang or v.Tang_nang_index is None:
                missing_fields = True
            if not v.Giam_nhe or v.Giam_nhe_index is None:
                missing_fields = True
            if not v.Hinh_Phat_Bo_Sung or v.Hinh_Phat_Bo_Sung_index is None:
                missing_fields = True
        
        if missing_fields:
            status = "success_missing_fields"
 
        # ------------------------------------------------------------------
        # Usage block
        # ------------------------------------------------------------------
        stage3_data["_usage"] = {
            "api":   "Fallback pipeline (AI Studio -> OpenRouter Free -> OpenRouter Paid)",
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
        status = "failed"
 
    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if status != "failed":
        os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(stage3_data, f, ensure_ascii=False, indent=2)
 
    print(f"Processed: {filename} | Provider: {provider} | Cost: ${file_cost:.6f} | Status: {status}")
    return file_cost, status
 
 
# ==========================================
# 4. BATCH PROCESSING & AGGREGATION
# ==========================================
def process_directory(
    input_dir: str,
    output_dir: str,
    skip_existing: bool = True,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
):
    """Process all JSON files in input_dir and write results to output_dir."""
    total_pipeline_cost = 0.0
    skipped_count       = 0
    failed_files        = []
    missing_fields_files = []
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
 
    print(f"Found {len(input_files)} files to process.")
    if skip_existing:
        print("Mode: skipping already-processed files (use --reprocess to override).")
    print("-" * 40)
 
    for input_file in input_files:
        filename    = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("stage2", "stage3"))
        cost, status = process_caselaw_file(
            input_file,
            output_file,
            skip_existing=skip_existing,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
        )
        if status == "skipped":
            skipped_count += 1
        elif status == "failed":
            failed_files.append(filename)
            with open(os.path.join(output_dir, "failed_files.txt"), "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        elif status == "success_missing_fields":
            missing_fields_files.append(filename)
            with open(os.path.join(output_dir, "missing_fields_files.txt"), "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        total_pipeline_cost += cost
        
        # Add 10s wait between processing files to prevent API rate limit stalling
        if status != "skipped":
            time.sleep(10)
 
    print("-" * 40)
    print("BATCH COMPLETE")
    print(f"Total Files Found:      {len(input_files)}")
    print(f"Skipped (exist):        {skipped_count}")
    print(f"Processed (new):        {len(input_files) - skipped_count}")
    print(f"Failed:                 {len(failed_files)}")
    print(f"Missing Fields:         {len(missing_fields_files)}")
    print(f"FINAL TOTAL COST: ${total_pipeline_cost:.4f}")
 
 
def process_selected_files(
    input_dir: str,
    output_dir: str,
    selected_filenames: List[str],
    skip_existing: bool = True,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
):
    """Process only the specified filenames from input_dir."""
    total_pipeline_cost = 0.0
    skipped_count       = 0
    failed_files        = []
    missing_fields_files = []
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
        cost, status = process_caselaw_file(
            input_file,
            output_file,
            skip_existing=skip_existing,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
        )
        if status == "skipped":
            skipped_count += 1
        elif status == "failed":
            failed_files.append(filename)
            with open(os.path.join(output_dir, "failed_files.txt"), "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        elif status == "success_missing_fields":
            missing_fields_files.append(filename)
            with open(os.path.join(output_dir, "missing_fields_files.txt"), "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        total_pipeline_cost += cost
        
        if status != "skipped":
            time.sleep(10)
 
    print("-" * 40)
    print("BATCH COMPLETE")
    print(f"Total Files Found:      {len(existing_files)}")
    print(f"Skipped (exist):        {skipped_count}")
    print(f"Processed (new):        {len(existing_files) - skipped_count}")
    print(f"Failed:                 {len(failed_files)}")
    print(f"Missing Fields:         {len(missing_fields_files)}")
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
        default="data_create/extracted_fields/2023",
        help="Directory containing input stage2 JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="data_create/full_aistudio/2023",
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
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout in seconds for each model request",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Max attempts per AI Studio request (1 disables retries; default: 1).",
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

        process_selected_files(
            args.input_dir,
            args.output_dir,
            deduped,
            skip_existing=skip_existing,
            request_timeout_seconds=args.request_timeout_seconds,
            retry_attempts=args.retry_attempts,
        )

    elif first_n_mode_enabled:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")

        all_files = sorted(
            f for f in glob.glob(os.path.join(args.input_dir, "*.json"))
            if not os.path.basename(f).lower() == "law_doc.json"
        )
        if not all_files:
            raise ValueError(f"No JSON files found in {args.input_dir}")

        selected = [os.path.basename(p) for p in all_files[: args.first_n]]
        process_selected_files(
            args.input_dir,
            args.output_dir,
            selected,
            skip_existing=skip_existing,
            request_timeout_seconds=args.request_timeout_seconds,
            retry_attempts=args.retry_attempts,
        )

    else:
        process_directory(
            args.input_dir,
            args.output_dir,
            skip_existing=skip_existing,
            request_timeout_seconds=args.request_timeout_seconds,
            retry_attempts=args.retry_attempts,
        )    
