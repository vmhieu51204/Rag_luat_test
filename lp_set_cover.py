#!/usr/bin/env python3
"""LP-based set cover for law-clause matching.

For each test case:
- Universe U: clause signatures from that test case's Cac_Dieu_Quyet_Dinh
- Candidate subsets: clause signatures from every train case
- Selection: solve a weighted set cover LP relaxation, then apply
  randomized rounding to pick binary case assignments, ranked to top-k.

LP formulation
--------------
Variables : x_i ∈ [0, 1]  for each train case i
Objective : minimise  Σ x_i          (fewest cases selected)
Constraints:
  for each clause c ∈ U:  Σ_{i: c ∈ S_i} x_i ≥ 1   (every clause covered)
  x_i ≥ 0

Randomized rounding
-------------------
Each fractional x_i* is the probability of selecting case i in one draw.
We repeat MAX_ROUNDS independent Bernoulli trials and keep the union that
achieves best coverage.  The top-k cases are then chosen by descending
LP weight, breaking ties by coverage contribution.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

ID_FIELD = "Ma_Ban_An"
ARTICLES_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM"

DEFAULT_TRAIN_DIR = "chunk/train"
DEFAULT_TEST_DIR = "chunk/test"
DEFAULT_OUTPUT = "/home/hieujayce/Downloads/complete_repo/output/lp_set_cover_results.json"

# Randomized rounding: number of independent Bernoulli trials per test case
MAX_ROUNDS = 200
# Minimum LP weight to be eligible for selection (filters noise)
LP_WEIGHT_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Shared helpers (unchanged from greedy version)
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: set[str], ground_truth: set[str]
) -> tuple[float, float, float]:
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def extract_clause_signatures(articles: Any) -> set[str]:
    """Extract unique clause signatures from PHAN_QUYET_CUA_TOA_SO_THAM."""
    signatures: set[str] = set()

    def _normalize_part(value: Any, default: str = "0") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _add_from_rule(rule: dict[str, Any]) -> None:
        dieu = _normalize_part(rule.get("Dieu"), default="")
        if not dieu:
            return
        khoan = _normalize_part(rule.get("Khoan"), default="0")
        diem = _normalize_part(rule.get("Diem"), default="0").lower()
        signatures.add(f"{dieu}-{khoan}-{diem}")

    def _add_item(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            can_cu = item.get("Can_Cu_Dieu_Luat")
            if isinstance(can_cu, list):
                for rule in can_cu:
                    if isinstance(rule, dict):
                        _add_from_rule(rule)
                return
            if "Dieu" in item:
                _add_from_rule(item)
                return
            signatures.add(str(item))
            return
        if isinstance(item, list):
            signatures.add("-".join(str(part) for part in item))
        else:
            signatures.add(str(item))

    if isinstance(articles, dict):
        for value in articles.values():
            if isinstance(value, list):
                for item in value:
                    _add_item(item)
            else:
                _add_item(value)
    elif isinstance(articles, list):
        for item in articles:
            _add_item(item)
    elif articles:
        _add_item(articles)

    return signatures


def load_case_clause_sets(
    folder: Path,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Load case_id -> clause-signature set from all JSON files in folder."""
    case_to_clauses: dict[str, set[str]] = {}
    stats = {
        "files_seen": 0,
        "loaded_cases": 0,
        "skipped_missing_articles": 0,
        "skipped_empty_clause_set": 0,
        "json_errors": 0,
        "duplicate_case_ids": 0,
    }

    for json_file in sorted(folder.glob("*.json")):
        stats["files_seen"] += 1
        try:
            with open(json_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            stats["json_errors"] += 1
            print(f"  [WARN] Could not parse JSON: {json_file}")
            continue

        case_id = data.get(ID_FIELD, json_file.stem)
        articles = data.get(ARTICLES_FIELD)
        if not articles:
            stats["skipped_missing_articles"] += 1
            continue

        clauses = extract_clause_signatures(articles)
        if not clauses:
            stats["skipped_empty_clause_set"] += 1
            continue

        if case_id in case_to_clauses:
            stats["duplicate_case_ids"] += 1
            case_to_clauses[case_id] |= clauses
        else:
            case_to_clauses[case_id] = clauses

    stats["loaded_cases"] = len(case_to_clauses)
    return case_to_clauses, stats


# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

def solve_lp_set_cover(
    query_clauses: set[str],
    train_case_clauses: dict[str, set[str]],
) -> dict[str, float]:
    """Solve the fractional weighted set cover LP.

    Returns a dict mapping train case_id -> LP weight x_i* ∈ [0, 1].
    Only cases whose weight exceeds LP_WEIGHT_THRESHOLD are included.

    The LP is:
        min  1ᵀ x
        s.t. A x ≥ 1   (coverage constraints, one row per query clause)
             0 ≤ x ≤ 1

    We restrict columns to train cases that cover at least one query clause
    so the LP stays small even with a large train set.
    """
    if not query_clauses:
        return {}

    clause_list = sorted(query_clauses)
    clause_idx = {c: i for i, c in enumerate(clause_list)}
    n_clauses = len(clause_list)

    # Filter to relevant train cases only
    relevant = {
        cid: clauses
        for cid, clauses in train_case_clauses.items()
        if clauses & query_clauses
    }
    if not relevant:
        return {}

    case_ids = sorted(relevant.keys())
    n_cases = len(case_ids)

    # Build sparse coverage matrix A  (n_clauses × n_cases)
    rows, cols, data = [], [], []
    for j, cid in enumerate(case_ids):
        for clause in relevant[cid]:
            if clause in clause_idx:
                rows.append(clause_idx[clause])
                cols.append(j)
                data.append(1.0)

    A_sparse = csc_matrix(
        (data, (rows, cols)), shape=(n_clauses, n_cases), dtype=np.float64
    )

    # linprog uses  A_ub x ≤ b_ub, so negate coverage constraints
    c_obj = np.array([float(len(train_case_clauses[cid])) for cid in case_ids])
    A_ub = -A_sparse          # shape (n_clauses, n_cases)
    b_ub = -np.ones(n_clauses)
    bounds = [(0.0, 1.0)] * n_cases

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

    if result.status != 0:
        # LP infeasible or failed — fall back: assign weight 1 to all relevant
        return {cid: 1.0 for cid in case_ids}

    weights = {}
    for j, cid in enumerate(case_ids):
        w = float(result.x[j])
        if w > LP_WEIGHT_THRESHOLD:
            weights[cid] = min(w, 1.0)
    return weights


# ---------------------------------------------------------------------------
# Randomized rounding + top-k selection
# ---------------------------------------------------------------------------

def lp_select_top_k(
    query_clauses: set[str],
    train_case_clauses: dict[str, set[str]],
    top_k: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Select top-k train cases via LP relaxation + randomized rounding.

    Algorithm
    ---------
    1. Solve the LP to get fractional weights x_i*.
    2. Perform MAX_ROUNDS independent Bernoulli trials:
       each case i is included with probability x_i*.
    3. Keep the rounding that maximises coverage of query_clauses.
    4. If no rounding achieves full coverage, supplement with the highest-
       weight uncovered cases until coverage is maximised or k is reached.
    5. Finally rank all selected cases by LP weight (desc) and truncate to k.
    """
    if not query_clauses:
        return {
            "selected_train_case_ids": [],
            "lp_weights": {},
            "lp_status": "empty_universe",
            "final_covered_clauses": [],
            "final_uncovered_clauses": [],
            "final_coverage": 1.0,
            "rounding_rounds_tried": 0,
        }

    lp_weights = solve_lp_set_cover(query_clauses, train_case_clauses)

    if not lp_weights:
        return {
            "selected_train_case_ids": [],
            "lp_weights": {},
            "lp_status": "no_relevant_cases",
            "final_covered_clauses": [],
            "final_uncovered_clauses": sorted(query_clauses),
            "final_coverage": 0.0,
            "rounding_rounds_tried": 0,
        }

    candidate_ids = sorted(lp_weights.keys())
    probs = np.array([lp_weights[cid] for cid in candidate_ids])

    # Randomized rounding: find the rounding with best coverage
    best_covered: set[str] = set()
    best_selected: set[str] = set()

    for _ in range(MAX_ROUNDS):
        mask = rng.random(len(candidate_ids)) < probs
        selected = {cid for cid, sel in zip(candidate_ids, mask) if sel}
        covered: set[str] = set()
        for cid in selected:
            covered |= train_case_clauses[cid] & query_clauses
        if len(covered) > len(best_covered):
            best_covered = covered
            best_selected = selected
            if best_covered == query_clauses:
                break  # full coverage found early

    # Supplement with deterministic high-weight cases if coverage is incomplete
    uncovered = query_clauses - best_covered
    if uncovered:
        remaining = [
            cid for cid in sorted(candidate_ids, key=lambda c: -lp_weights[c])
            if cid not in best_selected
        ]
        for cid in remaining:
            gain = uncovered & train_case_clauses[cid]
            if gain:
                best_selected.add(cid)
                best_covered |= gain
                uncovered -= gain
            if not uncovered:
                break

    # Rank selected cases by LP weight (desc), tie-break by case_id
    ranked = sorted(
        best_selected,
        key=lambda cid: (-lp_weights.get(cid, 0.0), cid),
    )[:top_k]

    covered_final: set[str] = set()
    for cid in ranked:
        covered_final |= train_case_clauses[cid] & query_clauses
    uncovered_final = query_clauses - covered_final
    coverage = len(covered_final) / len(query_clauses)

    return {
        "selected_train_case_ids": ranked,
        "lp_weights": {cid: round(lp_weights[cid], 6) for cid in ranked},
        "lp_status": "optimal",
        "final_covered_clauses": sorted(covered_final),
        "final_uncovered_clauses": sorted(uncovered_final),
        "final_coverage": round(coverage, 6),
        "rounding_rounds_tried": MAX_ROUNDS,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_lp_set_cover(
    train_dir: Path,
    test_dir: Path,
    top_k: int,
    output_path: Path,
    seed: int = 42,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    rng = np.random.default_rng(seed)

    print("Loading train cases...")
    train_case_clauses, train_stats = load_case_clause_sets(train_dir)
    print(f"  Loaded {len(train_case_clauses)} train case(s)")

    print("Loading test cases...")
    test_case_clauses, test_stats = load_case_clause_sets(test_dir)
    print(f"  Loaded {len(test_case_clauses)} test case(s)")

    if not train_case_clauses:
        raise RuntimeError("No train cases with valid clause sets were loaded")
    if not test_case_clauses:
        raise RuntimeError("No test cases with valid clause sets were loaded")

    per_test: list[dict[str, Any]] = []
    total_query_clauses = 0
    total_covered_clauses = 0
    sum_precision = sum_recall = sum_f1 = 0.0

    print(f"Running LP set cover for {len(test_case_clauses)} test case(s)...")
    for idx, test_case_id in enumerate(sorted(test_case_clauses.keys()), start=1):
        query_clauses = test_case_clauses[test_case_id]
        result = lp_select_top_k(query_clauses, train_case_clauses, top_k=top_k, rng=rng)

        covered_count = len(result["final_covered_clauses"])
        total_query_clauses += len(query_clauses)
        total_covered_clauses += covered_count

        predicted_clauses: set[str] = set()
        for cid in result["selected_train_case_ids"]:
            predicted_clauses |= train_case_clauses[cid]

        p, r, f1 = precision_recall_f1(predicted_clauses, query_clauses)
        sum_precision += p
        sum_recall += r
        sum_f1 += f1

        per_test.append(
            {
                "test_case_id": test_case_id,
                "query_clause_count": len(query_clauses),
                "query_clauses": sorted(query_clauses),
                "selected_train_case_ids": result["selected_train_case_ids"],
                "lp_weights": result["lp_weights"],
                "lp_status": result["lp_status"],
                "selected_count": len(result["selected_train_case_ids"]),
                "predicted_clause_count": len(predicted_clauses),
                "predicted_clauses": sorted(predicted_clauses),
                "final_covered_count": covered_count,
                "final_uncovered_count": len(result["final_uncovered_clauses"]),
                "final_covered_clauses": result["final_covered_clauses"],
                "final_uncovered_clauses": result["final_uncovered_clauses"],
                "final_coverage": result["final_coverage"],
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "rounding_rounds_tried": result["rounding_rounds_tried"],
            }
        )

        if idx % 10 == 0 or idx == len(test_case_clauses):
            print(f"  Processed {idx}/{len(test_case_clauses)} test case(s)")

    n = len(per_test)
    macro_coverage = sum(item["final_coverage"] for item in per_test) / n
    macro_precision = sum_precision / n
    macro_recall = sum_recall / n
    macro_f1 = sum_f1 / n
    micro_coverage = total_covered_clauses / total_query_clauses if total_query_clauses else 1.0

    output = {
        "config": {
            "algorithm": "lp_set_cover",
            "clause_granularity": "full_signature",
            "selection_rule": "lp_relaxation_randomized_rounding",
            "top_k": top_k,
            "max_rounding_rounds": MAX_ROUNDS,
            "lp_weight_threshold": LP_WEIGHT_THRESHOLD,
            "random_seed": seed,
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
        },
        "train_stats": train_stats,
        "test_stats": test_stats,
        "summary": {
            "n_train_cases": len(train_case_clauses),
            "n_test_cases": len(test_case_clauses),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_final_coverage": round(macro_coverage, 6),
            "micro_final_coverage": round(micro_coverage, 6),
            "total_query_clauses": total_query_clauses,
            "total_covered_clauses": total_covered_clauses,
            "full_coverage_test_cases": sum(
                1 for item in per_test if item["final_coverage"] >= 1.0
            ),
        },
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_test": per_test,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LP-based set cover where each test case defines a universe "
            "of law clause signatures and each train case defines a subset. "
            "Solves the fractional weighted set cover LP, then applies "
            "randomized rounding for binary selection."
        )
    )
    parser.add_argument(
        "--train_dir",
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing train case JSON files",
    )
    parser.add_argument(
        "--test_dir",
        default=DEFAULT_TEST_DIR,
        help="Directory containing test case JSON files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Maximum number of train cases to select per test case",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible rounding",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_lp_set_cover(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        top_k=args.top_k,
        output_path=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
