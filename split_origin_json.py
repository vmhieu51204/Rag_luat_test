#!/usr/bin/env python3
"""Split a single origin.json (array of records) into one JSON file per Ma_Ban_An.

Example:
    python split_origin_json.py \
        --input ./chunk/Chuong_XXII_chunked/synth/origin.json \
        --output ./chunk/Chuong_XXII_chunked/synth/split
"""

import argparse
import json
import re
import sys
from pathlib import Path


ID_FIELD = "Ma_Ban_An"


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def load_records(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        records = data["data"]
    else:
        raise ValueError(
            "Unsupported JSON format. Expected a list of objects or {'data': [...]}"
        )

    if not records:
        raise ValueError("No records found in input JSON")

    return records


def split_to_files(records: list[dict], output_dir: Path, id_field: str) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    used_names: dict[str, int] = {}

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            skipped += 1
            continue

        raw_id = (record.get(id_field) or "").strip()
        if not raw_id:
            skipped += 1
            continue

        base = safe_filename(raw_id)
        count = used_names.get(base, 0)
        used_names[base] = count + 1

        stem = base if count == 0 else f"{base}__dup{count+1}"
        out_path = output_dir / f"{stem}.json"

        # Keep UTF-8 text as-is and pretty-print for readability.
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)

        written += 1

        if written % 200 == 0:
            print(f"  wrote {written} files...", file=sys.stderr)

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split origin.json into multiple JSON files named by Ma_Ban_An"
    )
    parser.add_argument(
        "--input",
        default="./chunk/Chuong_XXII_chunked/synth/origin.json",
        help="Path to origin JSON file",
    )
    parser.add_argument(
        "--output",
        default="./chunk/Chuong_XXII_chunked/synth/split",
        help="Output folder for split JSON files",
    )
    parser.add_argument(
        "--id_field",
        default=ID_FIELD,
        help="Field to use for output filename (default: Ma_Ban_An)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    try:
        records = load_records(input_path)
        written, skipped = split_to_files(records, output_dir, args.id_field)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"[DONE] Written: {written} file(s)")
    print(f"[DONE] Skipped: {skipped} record(s)")
    print(f"[DONE] Output : {output_dir}")


if __name__ == "__main__":
    main()

#/home/hieujayce/Downloads/complete_repo/.venv/bin/python evaluate.py --train_dir ./chunk/Chuong_XXII_chunked/train --test_dir ./chunk/Chuong_XXII_chunked/synth/split --train_db_dir ./output/chroma_db_train --test_db_dir ./output/chroma_db_test_split --top_k 10 --results_out ./output/eval_results_synth_split.json
