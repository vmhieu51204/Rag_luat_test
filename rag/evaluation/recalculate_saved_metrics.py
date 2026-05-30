"""Recalculate metrics in a saved evaluation report without rerunning generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.core.sentencing import extract_imprisonment_months
from rag.evaluation.eval_utils import _aggregate, _macro_mean, _safe_float, _set_prf, _to_dieu_set


def _recalculate_defendant_metrics(defendant: dict[str, Any]) -> None:
    ground_truth = defendant.get("ground_truth") or {}
    prediction = defendant.get("prediction") or {}
    gt_clauses = _to_dieu_set(set(ground_truth.get("Applied_Law_Clauses") or []))
    pred_clauses = _to_dieu_set(set(prediction.get("Applied_Law_Clauses") or []))
    gt_months = extract_imprisonment_months(ground_truth.get("Phat_Tu"))
    pred_months = extract_imprisonment_months(prediction.get("Phat_Tu"))
    squared_error = float((pred_months - gt_months) ** 2)

    defendant["metrics"] = {
        "law_clause_prf": _set_prf(pred_clauses, gt_clauses),
        "phat_tu_months": {
            "ground_truth": gt_months,
            "prediction": pred_months,
            "squared_error": _safe_float(squared_error),
        },
    }


def _recalculate_doc_metrics(doc: dict[str, Any]) -> None:
    defendants = doc.get("defendants")
    if doc.get("status") != "processed" or not isinstance(defendants, list):
        return

    clause_precision: list[float] = []
    clause_recall: list[float] = []
    clause_f1: list[float] = []
    sentence_squared_errors: list[float] = []
    for defendant in defendants:
        if not isinstance(defendant, dict):
            continue
        _recalculate_defendant_metrics(defendant)
        metrics = defendant["metrics"]
        clause_metrics = metrics["law_clause_prf"]
        sentence_metrics = metrics["phat_tu_months"]
        clause_precision.append(float(clause_metrics["precision"]))
        clause_recall.append(float(clause_metrics["recall"]))
        clause_f1.append(float(clause_metrics["f1"]))
        sentence_squared_errors.append(float(sentence_metrics["squared_error"]))

    rmse_months = (
        _safe_float((sum(sentence_squared_errors) / len(sentence_squared_errors)) ** 0.5)
        if sentence_squared_errors
        else 0.0
    )
    doc["doc_metrics"] = {
        "law_clause_precision_macro": _macro_mean(clause_precision),
        "law_clause_recall_macro": _macro_mean(clause_recall),
        "law_clause_f1_macro": _macro_mean(clause_f1),
        "phat_tu_rmse_months": rmse_months,
        "n_defendants_scored": len(sentence_squared_errors),
    }


def recalculate_saved_metrics(report: dict[str, Any]) -> dict[str, Any]:
    per_doc = report.get("per_doc")
    if not isinstance(per_doc, list):
        raise ValueError("Saved report must contain a per_doc list")

    for doc in per_doc:
        if isinstance(doc, dict):
            _recalculate_doc_metrics(doc)
    report["summary"] = _aggregate(per_doc)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="act_full.json", help="Saved evaluation JSON report")
    parser.add_argument("--output", default=None, help="Output path; defaults to overwriting input")
    parser.add_argument("--no-backup", action="store_true", help="Do not save <input>.bak before overwriting input")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    with open(input_path, encoding="utf-8") as fh:
        report = json.load(fh)

    recalculate_saved_metrics(report)
    if output_path == input_path and not args.no_backup:
        backup_path = input_path.with_name(f"{input_path.name}.bak")
        backup_path.write_bytes(input_path.read_bytes())
        print(f"Backup: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Recalculated metrics: {output_path}")


if __name__ == "__main__":
    main()
