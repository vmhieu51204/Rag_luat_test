"""Helper utilities for loading test cases, processing data, and formatting UI elements."""

import json
from pathlib import Path
from rag.evaluation.eval_utils import _extract_gt_defendants

# Resolve path to test files
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "chunk" / "test"

def load_test_cases() -> list[str]:
    """Returns a sorted list of test case filenames from the test directory."""
    if not TEST_DIR.exists():
        return []
    return sorted([f.name for f in TEST_DIR.glob("*.json")])

def load_case_data(filename: str) -> dict:
    """Loads JSON data for a specific test case file."""
    file_path = TEST_DIR / filename
    if not file_path.exists():
        return {}
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def extract_defendants_and_summary(case_data: dict) -> dict:
    """Extracts string versions of Thong_Tin_Bi_Cao and Synthetic_summary_2."""
    # Defendant Info
    defendant_info = case_data.get("THONG_TIN_CHUNG", {}).get("Thong_Tin_Bi_Cao", "")
    if not isinstance(defendant_info, str):
        defendant_info = json.dumps(defendant_info, ensure_ascii=False, indent=2)
    
    # Synthetic Summary
    synth_summary = case_data.get("Synthetic_summary_2", "")
    if isinstance(synth_summary, list):
        synth_summary = "\n".join(synth_summary)
    
    return {
        "defendant_info": defendant_info,
        "synthetic_summary_2": synth_summary
    }

def get_ground_truth(case_data: dict) -> list[dict]:
    """Extracts ground-truth defendants and their verdicts for comparison."""
    try:
        return _extract_gt_defendants(case_data, only_blhs=True)
    except Exception:
        return []
