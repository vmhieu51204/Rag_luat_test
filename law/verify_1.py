
import json
import re

def analyze_output(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = json.load(f)

    issues = {
        "possible_footnotes": [],
        "too_short": [],
        "numeric_only": [],
        "weird_symbols": [],
        "missing_structure": [],
        "suspicious_lowercase_start": []
    }

    footnote_pattern = re.compile(
        r'(?:Điều này|Khoản này|Điểm này|Theo quy định)', re.IGNORECASE
    )

    for i, line in enumerate(lines):
        line_strip = line.strip()

        # 1. Possible missed footnotes
        if footnote_pattern.search(line_strip):
            issues["possible_footnotes"].append((i, line))

        # 2. Too short lines
        if len(line_strip) <= 3:
            issues["too_short"].append((i, line))

        # 3. Numeric-only or near numeric
        if re.match(r'^\d+[\.\)]?$', line_strip):
            issues["numeric_only"].append((i, line))

        # 4. Weird symbol-heavy lines
        if re.match(r'^[^a-zA-ZĐđ]+$', line_strip):
            issues["weird_symbols"].append((i, line))

        # 5. Missing legal structure (should start with uppercase or keyword)
        if not re.match(r'^(Chương|Điều|Khoản|Điểm|[A-ZĐ])', line_strip):
            issues["missing_structure"].append((i, line))

        # 6. Suspicious lowercase start (common OCR error)
        if re.match(r'^[a-zđ]', line_strip):
            issues["suspicious_lowercase_start"].append((i, line))

    # Print report
    print("🔍 ANALYSIS REPORT")
    print("=" * 50)
    for key, vals in issues.items():
        print(f"{key}: {len(vals)} issues")

    # Optional: print examples
    print("\n📌 Sample problematic lines:\n")
    for key, vals in issues.items():
        if vals:
            print(f"--- {key} ---")
            for idx, line in vals[:5]:
                print(f"[{idx}] {line}")
            print()

    return issues


if __name__ == "__main__":
    json_file = r"C:\Users\hungn\Downloads\darkin\stage1_lines.json"
    analyze_output(json_file)