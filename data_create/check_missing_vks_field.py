#!/usr/bin/env python3
"""Find JSON files missing a required field.

Default target:
- Folder: data_create/filled_template_openai
- Field: De_Nghi_Cua_Vien_Kiem_Sat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check JSON files for a missing field.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(__file__).resolve().parent / "filled_template_openai",
        help="Folder containing JSON files (default: data_create/filled_template_openai).",
    )
    parser.add_argument(
        "--field",
        default="De_Nghi_Cua_Vien_Kiem_Sat",
        help="Field name to check (default: De_Nghi_Cua_Vien_Kiem_Sat).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder
    field = args.field

    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder not found or not a directory: {folder}")
        return 2

    pattern = "**/*.json" if args.recursive else "*.json"
    json_files = sorted(folder.glob(pattern))

    if not json_files:
        print(f"No JSON files found in {folder}")
        return 0

    missing_field: list[Path] = []
    invalid_json: list[tuple[Path, str]] = []

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001 - report all parse/read issues
            invalid_json.append((path, str(exc)))
            continue

        if not isinstance(data, dict) or field not in data:
            missing_field.append(path)

    print(f"Scanned files: {len(json_files)}")
    print(f"Missing field '{field}': {len(missing_field)}")

    if missing_field:
        print("\nFiles missing the field:")
        for path in missing_field:
            print(path)

    if invalid_json:
        print(f"\nInvalid JSON files: {len(invalid_json)}")
        for path, err in invalid_json:
            print(f"{path} -> {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
