import argparse
import glob
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import google.genai as genai
from google.genai import types


DEFAULT_MODEL_NAME = "gemma-4-31b-it"
DEFAULT_INPUT_DIR = "chunk/test/2023"
DEFAULT_INPUT_FIELD = "NOI_DUNG_VU_AN"
DEFAULT_OUTPUT_FIELD = "Synthetic_summary_2"
DEFAULT_SLEEP_SECONDS = 20
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_RETRY_ATTEMPTS = 1
DEFAULT_OPENROUTER_FALLBACK_MODELS = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-31b-it",
)

DEFAULT_INPUT_FIELD_ALIASES = (
    "NOI_DUNG_VU_AN",
    "Noi_dung_vu_an.Qua_trinh_dieu_tra",
    "Noi_dung_vu_an",
)

_client_cache: dict[tuple[str, int | None, int | None], Any] = {}
_openrouter_client = None

NOISY_MARKERS = [
    "role:",
    "audience:",
    "short/complete",
    "paragraph format",
    "no bullet points",
    "no bold text",
    "summarize the events",
    "defendant (",
    "drafting (",
    "refining for",
    "applying constraints",
    "vietnamese translation/polishing",
]

IMPORTANT_QUANTITY_INSTRUCTION = (
    "Giữ lại các số lượng quan trọng đúng như hồ sơ nếu có, đặc biệt là số tiền, "
    "giá trị tài sản, khối lượng ma túy/cocain, số lần phạm tội, số người liên quan, "
    "ngày tháng, tỷ lệ thương tích và các định lượng khác ảnh hưởng đến tội danh hoặc khung hình phạt. "
    "Không làm tròn, không ước lượng và không bỏ qua các con số này."
)


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


def get_models(
    model_name: str,
    request_timeout_seconds: float | None = None,
    retry_attempts: int | None = None,
) -> list[dict[str, Any]]:
    """Lazily initialize one Google GenAI client for each configured API key."""
    del model_name

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
        except Exception as exc:  # noqa: BLE001
            init_errors.append(f"{key_name}: {exc}")

    if not clients:
        raise EnvironmentError(
            "No AI Studio clients could be initialized. Errors: "
            + " | ".join(init_errors)
        )

    return clients


def get_openrouter_client(request_timeout_seconds: float | None = None):
    """Lazily initialize an OpenRouter client for fallback generation."""
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


def get_nested_value(data: dict[str, Any], field_path: str) -> Any:
    """Read a field by dot path, e.g. NOI_DUNG_VU_AN.Qua_trinh_dieu_tra."""
    current: Any = data
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_input_text(data: dict[str, Any], input_field: str) -> tuple[str, str | None]:
    """Return the first non-empty prompt text and the field path that produced it."""
    candidate_fields = [input_field]
    if input_field == DEFAULT_INPUT_FIELD:
        for field_path in DEFAULT_INPUT_FIELD_ALIASES:
            if field_path not in candidate_fields:
                candidate_fields.append(field_path)

    for field_path in candidate_fields:
        input_value = get_nested_value(data, field_path)
        input_text = to_text(input_value)
        if input_text:
            return input_text, field_path

    return "", None


def to_text(value: Any) -> str:
    """Convert nested JSON value into a plain string for prompting."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [to_text(v) for v in value]
        return "\n\n".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        parts = [to_text(v) for v in value.values()]
        return "\n\n".join([p for p in parts if p]).strip()
    return str(value).strip()


def build_prompt(input_text: str) -> str:
    return (
        "Dựa vào hồ sơ vụ án của toà án, đóng vai bị cáo trong vụ án sau và tường trình lại vụ việc "
        "cho một luật sư theo góc nhìn của bạn. Hãy kể lại thật ngắn gọn nhưng đầy đủ tình tiết để "
        "luật sư đánh giá các điều khoản mà bạn đã sai phạm. "
        f"{IMPORTANT_QUANTITY_INSTRUCTION} "
        "CHỈ trả về đúng 1 đoạn văn tiếng Việt duy nhất. "
        "KHÔNG trả về checklist, tiêu đề, phân tích, role labels, gạch đầu dòng, markdown, "
        "hoặc tự đánh giá định dạng. "
        "Bắt đầu trực tiếp bằng nội dung tường trình. "
        "Nội dung hồ sơ: "
        f"{input_text}"
    )


def build_defendant_prompt(
    case_text: str,
    defendant_name: str,
    tang_nang: str,
    giam_nhe: str,
) -> str:
    tang_nang_text = tang_nang or "Không có thông tin riêng về tình tiết tăng nặng."
    giam_nhe_text = giam_nhe or "Không có thông tin riêng về tình tiết giảm nhẹ."
    defendant_label = defendant_name or "bị cáo trong vụ án"

    return (
        "Dựa vào hồ sơ vụ án của toà án, đóng vai đúng bị cáo được nêu dưới đây và tường trình lại vụ việc "
        "cho một luật sư theo góc nhìn của bạn. Hãy kể lại thật ngắn gọn nhưng đầy đủ tình tiết để "
        "luật sư đánh giá các điều khoản mà bạn đã sai phạm. "
        f"{IMPORTANT_QUANTITY_INSTRUCTION} "
        "Chỉ viết phần liên quan đến bị cáo này, không nhập vai các bị cáo khác. "
        "Khi nhắc đến tình tiết tăng nặng và giảm nhẹ, hãy diễn đạt tự nhiên như một người đang nói với luật sư, "
        "không chép nguyên văn máy móc từ hồ sơ. "
        "CHỈ trả về đúng 1 đoạn văn tiếng Việt duy nhất. "
        "KHÔNG trả về checklist, tiêu đề, phân tích, role labels, gạch đầu dòng, markdown, "
        "hoặc tự đánh giá định dạng. "
        "Bắt đầu trực tiếp bằng nội dung tường trình. "
        f"Bị cáo cần nhập vai: {defendant_label}\n\n"
        f"Tình tiết tăng nặng của bị cáo này: {tang_nang_text}\n\n"
        f"Tình tiết giảm nhẹ của bị cáo này: {giam_nhe_text}\n\n"
        f"Nội dung hồ sơ vụ án: {case_text}"
    )


def extract_json_from_response(text: str) -> str:
    """Strip markdown code fences and return JSON-ish text."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_synthetic_summary_response(raw_text: str) -> str:
    raw = (raw_text or "").strip()
    if not raw:
        return ""

    cleaned_raw = extract_json_from_response(raw)
    try:
        parsed = json.loads(cleaned_raw)
        if isinstance(parsed, dict):
            return str(parsed.get("synthetic_summary", "")).strip()
    except Exception:
        pass

    return cleaned_raw


def generate_summary_aistudio(
    client,
    model_name: str,
    prompt: str,
    request_timeout_seconds: float,
    key_name: str | None = None,
) -> str:
    key_label = f" using {key_name}" if key_name else ""
    print(f"Calling generate_content{key_label} (timeout={request_timeout_seconds}s)...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "synthetic_summary": {"type": "string"},
                    },
                    "required": ["synthetic_summary"],
                },
            ),
            request_options={"timeout": request_timeout_seconds},
        )
    except TypeError:
        # Older/newer google.genai clients may not accept `request_options` kwarg.
        # Fall back to calling without it (may use default timeout or honor
        # environment-configured timeout in the client).
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "synthetic_summary": {"type": "string"},
                    },
                    "required": ["synthetic_summary"],
                },
            ),
        )
    print("generate_content returned.")

    output = parse_synthetic_summary_response(response.text or "")
    if not output:
        raise ValueError("Model returned an empty response")
    return output


def generate_summary_openrouter(model_name: str, prompt: str, request_timeout_seconds: float) -> str:
    print(
        f"Calling OpenRouter chat.completions model={model_name} "
        f"(timeout={request_timeout_seconds}s)..."
    )
    client = get_openrouter_client(request_timeout_seconds)
    system_prompt = (
        "Bạn là trợ lý viết tóm tắt pháp lý bằng tiếng Việt. "
        f"{IMPORTANT_QUANTITY_INSTRUCTION} "
        "Chỉ trả về JSON object hợp lệ với đúng một key: synthetic_summary. "
        "Giá trị synthetic_summary phải là đúng 1 đoạn văn tiếng Việt, không markdown, không bullet, "
        "không checklist, không giải thích thêm."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        timeout=request_timeout_seconds,
    )
    print("OpenRouter chat.completions returned.")

    output = parse_synthetic_summary_response(response.choices[0].message.content or "")
    if not output:
        raise ValueError("OpenRouter model returned an empty response")
    return output


def generate_summary(
    clients: list[dict[str, Any]] | None,
    model_name: str,
    prompt: str,
    request_timeout_seconds: float,
) -> str:
    errors: list[str] = []

    if clients:
        for client_info in clients:
            key_name = str(client_info.get("key_name") or "unknown_key")
            try:
                return generate_summary_aistudio(
                    client_info["client"],
                    model_name,
                    prompt,
                    request_timeout_seconds,
                    key_name=key_name,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"AI Studio ({key_name}) failed: {exc}"
                errors.append(error_msg)
                print(f"    [Fallback] {error_msg}")
    else:
        errors.append("AI Studio unavailable: no clients were initialized")
        print("    [Fallback] AI Studio unavailable: no clients were initialized")

    for fallback_model in DEFAULT_OPENROUTER_FALLBACK_MODELS:
        try:
            return generate_summary_openrouter(fallback_model, prompt, request_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"OpenRouter ({fallback_model}) failed: {exc}"
            errors.append(error_msg)
            print(f"    [Fallback] {error_msg}")

    raise ValueError("All summary generation providers failed. Errors: " + " | ".join(errors))


def extract_defendant_contexts(data: dict[str, Any]) -> list[dict[str, str]]:
    """Return per-defendant context from judgment entries, with metadata fallback."""
    contexts: list[dict[str, str]] = []
    seen_names: set[str] = set()

    phan_quyet = data.get("PHAN_QUYET_CUA_TOA_SO_THAM")
    if isinstance(phan_quyet, list):
        for entry in phan_quyet:
            if not isinstance(entry, dict):
                continue
            name = to_text(entry.get("Bi_Cao"))
            if name in seen_names:
                continue
            contexts.append(
                {
                    "name": name,
                    "tang_nang": to_text(entry.get("Tang_nang")),
                    "giam_nhe": to_text(entry.get("Giam_nhe")),
                }
            )
            if name:
                seen_names.add(name)

    if contexts:
        return contexts

    thong_tin_chung = data.get("THONG_TIN_CHUNG")
    defendants = None
    if isinstance(thong_tin_chung, dict):
        defendants = thong_tin_chung.get("Thong_Tin_Bi_Cao")

    if isinstance(defendants, list):
        for defendant in defendants:
            if isinstance(defendant, dict):
                contexts.append(
                    {
                        "name": to_text(defendant.get("Ho_Ten")),
                        "tang_nang": "",
                        "giam_nhe": "",
                    }
                )

    return contexts or [{"name": "", "tang_nang": "", "giam_nhe": ""}]


def cleanup_synthetic_summary(text: str) -> str:
    """Remove planning/checklist artifacts and keep one clean paragraph."""
    if not text:
        return ""

    cleaned = text.strip()

    # Remove markdown code fences when present.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Keep content from the last "Thưa luật sư" occurrence if model includes drafts.
    marker = "Thưa luật sư"
    idx = cleaned.rfind(marker)
    if idx != -1:
        cleaned = cleaned[idx:].strip()

    noisy_line_patterns = [
        r"^\s*[*-]\s+",
        r"^\s*Role\s*:",
        r"^\s*Audience\s*:",
        r"^\s*Short/Complete\s*\?",
        r"^\s*Paragraph format\s*\?",
        r"^\s*No bullet points\s*\?",
        r"^\s*No bold text\s*\?",
        r"^\s*Defendant\s*\(",
        r"^\s*Lawyer\.?$",
        r"^\s*Summarize the events",
        r"^\s*Tone\s*:",
        r"^\s*Drafting\s*\(",
        r"^\s*Refining for",
        r"^\s*Applying constraints",
        r"^\s*Vietnamese Translation/Polishing",
    ]

    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            kept_lines.append("")
            continue
        if any(re.match(pat, line_stripped, flags=re.IGNORECASE) for pat in noisy_line_patterns):
            continue
        kept_lines.append(line_stripped)

    cleaned = "\n".join(kept_lines).strip()

    # Collapse to a single paragraph while preserving sentence spacing.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def cleanup_synthetic_summary_list(value: Any) -> list[str]:
    """Clean existing/generated summary values into a list of paragraph strings."""
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []

    cleaned_items: list[str] = []
    for item in items:
        cleaned = cleanup_synthetic_summary(to_text(item))
        if cleaned:
            cleaned_items.append(cleaned)
    return cleaned_items


def normalize_summary_slots(value: Any, expected_count: int) -> list[str]:
    """Return one output slot per defendant, preserving empty slots for retries."""
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []

    slots: list[str] = []
    for index in range(expected_count):
        item = items[index] if index < len(items) else ""
        slots.append(cleanup_synthetic_summary(to_text(item)))
    return slots


def save_json_file(file_path: str, data: dict[str, Any]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_valid_synthetic_summary(text: str) -> bool:
    """Check whether a synthetic summary is already usable and clean."""
    if not text:
        return False

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) < 80:
        return False

    lower = normalized.lower()
    if any(marker in lower for marker in NOISY_MARKERS):
        return False

    if "*" in normalized or "```" in normalized:
        return False

    return True


def is_valid_synthetic_summary_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(is_valid_synthetic_summary(to_text(item)) for item in value)


def is_complete_synthetic_summary_list(value: Any, expected_count: int) -> bool:
    if expected_count < 1 or not isinstance(value, list) or len(value) < expected_count:
        return False
    return all(is_valid_synthetic_summary(to_text(value[index])) for index in range(expected_count))


def process_file(
    file_path: str,
    model_name: str,
    model_holder: dict[str, Any],
    input_field: str,
    output_field: str,
    overwrite: bool,
    clean_existing_only: bool,
    request_timeout_seconds: float,
    retry_attempts: int,
) -> tuple[bool, str]:
    """Return (processed, message)."""
    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    defendant_contexts = extract_defendant_contexts(data)
    defendant_count = len(defendant_contexts)

    if clean_existing_only:
        existing = normalize_summary_slots(data.get(output_field), defendant_count)
        if not any(existing):
            return False, f"Skipped (no existing {output_field}): {file_name}"

        if not any(is_valid_synthetic_summary(item) for item in existing):
            return False, f"Skipped (no valid cleaned {output_field} entries): {file_name}"

        if existing == data.get(output_field):
            return False, f"Skipped (already clean): {file_name}"

        data[output_field] = existing
        save_json_file(file_path, data)
        return True, f"Cleaned: {file_name}"

    existing_value = data.get(output_field)
    if (not overwrite) and existing_value:
        cleaned_existing = normalize_summary_slots(existing_value, defendant_count)
        if is_complete_synthetic_summary_list(cleaned_existing, defendant_count):
            # Save cleaned version (if needed) without regenerating.
            if cleaned_existing != existing_value:
                data[output_field] = cleaned_existing
                save_json_file(file_path, data)
                return True, f"Cleaned existing valid {output_field}: {file_name}"
            return False, f"Skipped (already has valid {output_field} for all defendants): {file_name}"

        if cleaned_existing != existing_value:
            data[output_field] = cleaned_existing
            save_json_file(file_path, data)
            print(f"Saved cleaned partial {output_field} for {file_name}")

    input_text, resolved_input_field = resolve_input_text(data, input_field)
    if not input_text:
        return False, f"Skipped (missing/empty input field '{input_field}'): {file_name}"

    if resolved_input_field and resolved_input_field != input_field:
        print(f"Using fallback input field '{resolved_input_field}' for {file_name}")

    if model_holder.get("models") is None and not model_holder.get("aistudio_unavailable"):
        print(f"Initializing AI Studio clients for generation ({model_name})...")
        try:
            model_holder["models"] = get_models(
                model_name,
                request_timeout_seconds=request_timeout_seconds,
                retry_attempts=retry_attempts,
            )
            key_names = ", ".join(item["key_name"] for item in model_holder["models"])
            print(f"Initialized AI Studio client(s): {key_names}")
        except Exception as exc:  # noqa: BLE001
            model_holder["aistudio_unavailable"] = True
            print(f"    [Fallback] AI Studio client initialization failed: {exc}")

    if overwrite:
        synthetic_summaries = [""] * defendant_count
    else:
        synthetic_summaries = normalize_summary_slots(data.get(output_field), defendant_count)

    generated_count = 0
    skipped_count = 0
    error_count = 0
    for defendant_index, context in enumerate(defendant_contexts, start=1):
        slot_index = defendant_index - 1
        defendant_name = context["name"] or f"bị cáo {defendant_index}"
        if not overwrite and is_valid_synthetic_summary(synthetic_summaries[slot_index]):
            print(
                f"Skipping defendant {defendant_index}/"
                f"{len(defendant_contexts)} (already processed): {defendant_name}"
            )
            skipped_count += 1
            continue

        time.sleep(10)
        print(
            f"Generating summary for defendant {defendant_index}/"
            f"{len(defendant_contexts)}: {defendant_name}"
        )
        prompt = build_defendant_prompt(
            case_text=input_text,
            defendant_name=context["name"],
            tang_nang=context["tang_nang"],
            giam_nhe=context["giam_nhe"],
        )
        try:
            synthetic_summary = cleanup_synthetic_summary(
                generate_summary(
                    model_holder.get("models"),
                    model_name,
                    prompt,
                    request_timeout_seconds,
                )
            )
            if not is_valid_synthetic_summary(synthetic_summary):
                raise ValueError("generated summary is empty or invalid after cleanup")
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            print(
                f"Error generating defendant {defendant_index}/"
                f"{len(defendant_contexts)} ({defendant_name}): {exc}"
            )
            continue

        synthetic_summaries[slot_index] = synthetic_summary
        data[output_field] = synthetic_summaries
        save_json_file(file_path, data)
        generated_count += 1
        print(f"Saved summary for defendant {defendant_index}: {defendant_name}")

    if data.get(output_field) != synthetic_summaries:
        data[output_field] = synthetic_summaries
        save_json_file(file_path, data)

    if generated_count:
        return (
            True,
            f"Processed: {file_name} "
            f"(generated={generated_count}, skipped_existing={skipped_count}, errors={error_count})",
        )

    if error_count:
        return False, f"Skipped incomplete: {file_name} (errors={error_count}, skipped_existing={skipped_count})"

    return False, f"Skipped (already processed per defendant): {file_name}"


def collect_files(input_dir: str, input_file: str | None, first_n: int | None) -> list[str]:
    if input_file:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        return [input_file]

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    files = [f for f in files if not f.endswith("law_doc.json")]

    if first_n is not None:
        if first_n < 1:
            raise ValueError("--first-n must be >= 1")
        files = files[:first_n]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Synthetic_summary using Google AI Studio Gemma and append it "
            "to original JSON files in place."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Folder containing JSON files to update in place",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional single JSON file path to process",
    )
    parser.add_argument(
        "--input-field",
        default=DEFAULT_INPUT_FIELD,
        help=(
            "Input field to use as prompt text. Supports dot path, e.g. "
            "NOI_DUNG_VU_AN or NOI_DUNG_VU_AN.Qua_trinh_dieu_tra"
        ),
    )
    parser.add_argument(
        "--output-field",
        default=DEFAULT_OUTPUT_FIELD,
        help="Output field to write generated summary into",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Google AI Studio model name",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Process only first N files (sorted by filename)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output field if it already exists",
    )
    parser.add_argument(
        "--clean-existing-only",
        action="store_true",
        default=False,
        help=(
            "Do not call model. Only clean existing output-field values in-place "
            "to remove checklist/thinking artifacts."
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between files (default: 15)",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout in seconds for each model generate_content call",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Max attempts per request (1 disables retries; default: 1).",
    )

    args = parser.parse_args()

    files = collect_files(args.input_dir, args.input_file, args.first_n)
    if not files:
        raise ValueError("No JSON files found to process")

    print(f"Found {len(files)} file(s) to process")
    print(f"Input field : {args.input_field}")
    print(f"Output field: {args.output_field}")
    print(f"Model       : {args.model}")
    print(f"Clean only  : {args.clean_existing_only}")
    print(f"Sleep secs  : {args.sleep_seconds}")
    print(f"Req timeout : {args.request_timeout_seconds}")
    print(f"Retries     : {args.retry_attempts}")
    print("-" * 50)

    model_holder: dict[str, Any] = {"models": None}

    processed = 0
    skipped = 0
    failed = 0

    total_files = len(files)
    for idx, file_path in enumerate(files):
        should_sleep = False
        try:
            ok, message = process_file(
                file_path=file_path,
                model_name=args.model,
                model_holder=model_holder,
                input_field=args.input_field,
                output_field=args.output_field,
                overwrite=args.overwrite,
                clean_existing_only=args.clean_existing_only,
                request_timeout_seconds=args.request_timeout_seconds,
                retry_attempts=args.retry_attempts,
            )
            print(message)
            if ok:
                processed += 1
                should_sleep = True
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            should_sleep = True
            print(f"Error: {os.path.basename(file_path)} -> {e}")

        # Sleep between files (not after the last file).
        if idx < total_files - 1 and args.sleep_seconds > 0 and should_sleep:
            print(f"Sleeping {args.sleep_seconds} second(s) before next file...")
            time.sleep(args.sleep_seconds)

    print("-" * 50)
    print("DONE")
    print(f"Processed: {processed}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")


if __name__ == "__main__":
    main()
