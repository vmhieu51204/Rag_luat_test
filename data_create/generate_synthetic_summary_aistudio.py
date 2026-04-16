import argparse
import glob
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


DEFAULT_MODEL_NAME = "gemma-4-31b-it"
DEFAULT_INPUT_DIR = "chunk/test"
DEFAULT_INPUT_FIELD = "NOI_DUNG_VU_AN"
DEFAULT_OUTPUT_FIELD = "Synthetic_summary"
DEFAULT_SLEEP_SECONDS = 20
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90

_model = None

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


def get_model(model_name: str):
    """Lazily initialize a Google AI Studio model client."""
    global _model
    if _model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(model_name=model_name)
    return _model


def get_nested_value(data: dict[str, Any], field_path: str) -> Any:
    """Read a field by dot path, e.g. NOI_DUNG_VU_AN.Qua_trinh_dieu_tra."""
    current: Any = data
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


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
        "CHỈ trả về đúng 1 đoạn văn tiếng Việt duy nhất. "
        "KHÔNG trả về checklist, tiêu đề, phân tích, role labels, gạch đầu dòng, markdown, "
        "hoặc tự đánh giá định dạng. "
        "Bắt đầu trực tiếp bằng nội dung tường trình. "
        "Nội dung hồ sơ: "
        f"{input_text}"
    )


def generate_summary(model, prompt: str, request_timeout_seconds: float) -> str:
    print(f"Calling generate_content (timeout={request_timeout_seconds}s)...")
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
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
    print("generate_content returned.")

    raw = (response.text or "").strip()
    output = ""
    if raw:
        try:
            parsed = json.loads(raw)
            output = str(parsed.get("synthetic_summary", "")).strip()
        except Exception:
            # Fallback if server returns plain text unexpectedly.
            output = raw

    if not output:
        raise ValueError("Model returned an empty response")
    return output


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


def process_file(
    file_path: str,
    model_name: str,
    model_holder: dict[str, Any],
    input_field: str,
    output_field: str,
    overwrite: bool,
    clean_existing_only: bool,
    request_timeout_seconds: float,
) -> tuple[bool, str]:
    """Return (processed, message)."""
    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if clean_existing_only:
        existing = to_text(data.get(output_field))
        if not existing:
            return False, f"Skipped (no existing {output_field}): {file_name}"

        cleaned_existing = cleanup_synthetic_summary(existing)
        if not is_valid_synthetic_summary(cleaned_existing):
            return False, f"Skipped (cleaned output empty): {file_name}"

        if cleaned_existing == existing:
            return False, f"Skipped (already clean): {file_name}"

        data[output_field] = cleaned_existing
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True, f"Cleaned: {file_name}"

    existing = to_text(data.get(output_field))
    if (not overwrite) and existing:
        cleaned_existing = cleanup_synthetic_summary(existing)
        if is_valid_synthetic_summary(cleaned_existing):
            # Save cleaned version (if needed) without regenerating.
            if cleaned_existing != existing:
                data[output_field] = cleaned_existing
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                return True, f"Cleaned existing valid {output_field}: {file_name}"
            return False, f"Skipped (already has valid {output_field}): {file_name}"
        # Existing output is invalid/noisy -> continue to regenerate.

    input_value = get_nested_value(data, input_field)
    input_text = to_text(input_value)
    if not input_text:
        return False, f"Skipped (missing/empty input field '{input_field}'): {file_name}"

    if model_holder.get("model") is None:
        print(f"Initializing model client for generation ({model_name})...")
        model_holder["model"] = get_model(model_name)

    prompt = build_prompt(input_text)
    synthetic_summary = cleanup_synthetic_summary(
        generate_summary(model_holder["model"], prompt, request_timeout_seconds)
    )
    if not is_valid_synthetic_summary(synthetic_summary):
        return False, f"Skipped (generated empty after cleanup): {file_name}"

    data[output_field] = synthetic_summary

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return True, f"Processed: {file_name}"


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
    print("-" * 50)

    model_holder: dict[str, Any] = {"model": None}

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
