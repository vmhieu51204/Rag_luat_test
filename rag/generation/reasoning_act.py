"""ReAct-style judgment prediction using exact law lookup plus past-case RAG."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag.config import VERDICT_FIELD
from rag.core.law_retriever import LawClauseRetriever, parse_clause_signature
from rag.evaluation.eval_utils import load_articles_index
from rag.generation.schemas import (
    AdditionalLawQuery,
    CandidateOffence,
    ReasonActAnalysisOutput,
    ReasonActFinalOutput,
    ReasonActLegalAnalysis,
    ReasonActTrace,
    RetrievedLawArticle,
    SentencingCalibrationCase,
    SimilarCaseSummary,
    SupportingArticleAssessment,
    build_output_schema_instruction,
)
from rag.llm.providers import (
    LLMProvider,
    generate_structured_output,
    generate_structured_output_with_fallback,
)
from rag.parse_penalty import parse_penalty_to_months
from rag.runtime.retrieval import RetrievalRuntime

MANDATORY_SUPPORTING_DIEU = ("38", "50", "51", "52", "53", "54", "55", "56", "57", "58", "65", "47")
DEFAULT_REASON_ACT_TRAIN_FIELDS = ["NHAN_DINH_CUA_TOA_AN.[2]"]
MITIGATION_EMBED_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe"
AGGRAVATION_EMBED_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang"
DEFAULT_SENTENCING_CALIBRATION_FIELDS = [AGGRAVATION_EMBED_FIELD, MITIGATION_EMBED_FIELD]
MAX_FINAL_CASES_PER_ISSUE = 5
SYNTHETIC_SUMMARY_FORMAT_NOTE = (
    "case_fields.Synthetic_summary may be a JSON array encoded as a string. "
    "Each array item is a separate first-person story for one defendant. "
    "Map each story to the defendant named inside that story and keep per-defendant facts separate."
)


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _tokenize(text: str) -> set[str]:
    folded = _strip_accents(text).lower()
    return {tok for tok in re.findall(r"[a-z0-9]+", folded) if len(tok) >= 3}


def _name_key(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _strip_accents(str(text or "")).lower()))


def _unique_text_items(items: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = _normalize_space(item)
        key = _strip_accents(text).lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _resolve_field_value(data: dict[str, Any], field: str) -> Any:
    if field in {"Defendant_info", "defendant_info", "Thong_Tin_Bi_Cao"}:
        info = data.get("THONG_TIN_CHUNG")
        return info.get("Thong_Tin_Bi_Cao") if isinstance(info, dict) else None
    if "." not in field:
        return data.get(field)
    cur: Any = data
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract_input_payload(data: dict[str, Any], fields: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in fields:
        value = _resolve_field_value(data, field)
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if text.strip():
            payload[field] = text.strip()
    return payload


def build_query_text(data: dict[str, Any], fields: list[str]) -> str:
    payload = extract_input_payload(data, fields)
    return "\n\n".join(f"[{key}]\n{value}" for key, value in payload.items()).strip()


def doc_id_from_case(data: dict[str, Any], fallback: str = "") -> str:
    info = data.get("THONG_TIN_CHUNG")
    if isinstance(info, dict):
        doc_id = info.get("Ma_Ban_An")
        if doc_id:
            return str(doc_id).strip()
    return str(data.get("Ma_Ban_An") or fallback).strip()


def candidate_signature(candidate: CandidateOffence) -> str | None:
    if not candidate.Dieu:
        return None
    signature = str(candidate.Dieu).strip()
    if candidate.Khoan:
        signature += f"-{str(candidate.Khoan).strip()}"
    if candidate.Diem:
        signature += f"-{str(candidate.Diem).strip().lower()}"
    return signature


def _retrieved_article_from_result(signature: str, result: dict[str, Any]) -> RetrievedLawArticle:
    return RetrievedLawArticle(
        signature=signature,
        found=bool(result.get("found")),
        level=result.get("level"),
        text=result.get("text") if result.get("found") else None,
        missing_reason=None if result.get("found") else str(result.get("reason") or "not_found"),
    )


def _canonical_law_signature(signature: Any) -> str | None:
    raw = str(signature or "").strip()
    if not raw:
        return None
    try:
        parsed = parse_clause_signature(raw)
    except ValueError:
        return raw
    out = parsed.dieu
    if parsed.khoan is not None:
        out += f"-{parsed.khoan}"
    if parsed.diem is not None:
        out += f"-{parsed.diem}"
    return out


def _signature_dieu_and_is_full(signature: Any) -> tuple[str, bool] | None:
    canonical = _canonical_law_signature(signature)
    if not canonical:
        return None
    try:
        parsed = parse_clause_signature(canonical)
    except ValueError:
        return None
    return parsed.dieu, parsed.khoan is None


def compact_law_signatures(signatures: list[str]) -> list[str]:
    parsed_items: list[tuple[str, str | None, bool]] = []
    full_dieu: set[str] = set()
    for raw_sig in signatures:
        signature = str(raw_sig or "").strip()
        if not signature:
            continue
        canonical = _canonical_law_signature(signature)
        is_full = False
        dieu: str | None = None
        if canonical:
            parsed = _signature_dieu_and_is_full(canonical)
            if parsed is not None:
                dieu, is_full = parsed
                if is_full:
                    full_dieu.add(dieu)
        parsed_items.append((signature, dieu, is_full))

    compacted: list[str] = []
    seen: set[str] = set()
    for signature, dieu, is_full in parsed_items:
        if dieu in full_dieu and not is_full:
            continue
        key = _canonical_law_signature(signature) or signature
        if key in seen:
            continue
        seen.add(key)
        compacted.append(signature)
    return compacted


def _existing_law_coverage(articles: list[RetrievedLawArticle]) -> tuple[set[str], set[str]]:
    full_dieu: set[str] = set()
    signatures: set[str] = set()
    for item in articles:
        canonical = _canonical_law_signature(item.signature)
        if not canonical:
            continue
        signatures.add(canonical)
        parsed = _signature_dieu_and_is_full(canonical)
        if item.found and item.level == "dieu" and parsed is not None and parsed[1]:
            full_dieu.add(parsed[0])
    return full_dieu, signatures


def _filter_new_law_signatures(
    signatures: list[str],
    existing_articles: list[RetrievedLawArticle],
) -> list[str]:
    full_dieu, existing_signatures = _existing_law_coverage(existing_articles)
    new_signatures: list[str] = []
    for signature in compact_law_signatures(signatures):
        canonical = _canonical_law_signature(signature)
        if not canonical or canonical in existing_signatures:
            continue
        parsed = _signature_dieu_and_is_full(canonical)
        if parsed is not None and not parsed[1] and parsed[0] in full_dieu:
            continue
        new_signatures.append(signature)
    return new_signatures


def _additional_law_query_signatures(queries: list[AdditionalLawQuery]) -> list[str]:
    signatures: list[str] = []
    for query in queries:
        signature = _normalize_space(query.signature)
        if signature:
            signatures.append(signature)
    return signatures


def retrieve_law_articles(signatures: list[str], law_retriever: LawClauseRetriever) -> list[RetrievedLawArticle]:
    articles: list[RetrievedLawArticle] = []
    seen: set[str] = set()
    for raw_sig in compact_law_signatures(signatures):
        signature = str(raw_sig or "").strip()
        key = _canonical_law_signature(signature) or signature
        if not signature or key in seen:
            continue
        seen.add(key)
        result = law_retriever.retrieve(signature)
        articles.append(_retrieved_article_from_result(signature, result))
    return articles


def retrieve_candidate_articles(
    candidates: list[CandidateOffence],
    law_retriever: LawClauseRetriever,
) -> list[RetrievedLawArticle]:
    signatures: list[str] = []
    for candidate in candidates:
        signature = candidate_signature(candidate)
        if signature:
            signatures.append(signature)
        if candidate.Dieu and str(candidate.Dieu).strip() != signature:
            signatures.append(str(candidate.Dieu).strip())
    return retrieve_law_articles(signatures, law_retriever)


def detect_offence_specific_supporting_articles(case_text: str, offence_text: str) -> list[str]:
    text = _strip_accents(f"{case_text}\n{offence_text}").lower()
    out: set[str] = set()
    if any(term in text for term in ["tra lai", "boi thuong", "trach nhiem dan su", "khac phuc", "hoan tra"]):
        out.update({"46", "48"})
    if any(term in text for term in ["vat chung", "tich thu", "sung quy", "cong cu", "phuong tien"]):
        out.update({"46", "47"})
    if any(term in text for term in ["phap nhan thuong mai", "cong ty", "doanh nghiep"]):
        out.add("76")
    if any(term in text for term in ["duoi 18", "chua du 18", "nguoi duoi 18", "vi thanh nien"]):
        out.update({"91", "101"})
    return sorted(out, key=lambda item: int(item) if item.isdigit() else item)


def retrieve_supporting_articles(
    *,
    case_text: str,
    selected_offence_text: str,
    law_retriever: LawClauseRetriever,
) -> list[RetrievedLawArticle]:
    optional = detect_offence_specific_supporting_articles(case_text, selected_offence_text)
    signatures = list(MANDATORY_SUPPORTING_DIEU) + optional
    return retrieve_law_articles(signatures, law_retriever)


def classify_supporting_article_by_facts(
    *,
    article: str,
    retrieved: RetrievedLawArticle | None,
    case_text: str,
) -> SupportingArticleAssessment:
    if retrieved is None or not retrieved.found:
        return SupportingArticleAssessment(
            article=article,
            status="not_retrieved",
            factual_trigger=None,
            explanation="Article was checked but not retrieved from raw_law.json.",
        )

    folded = _strip_accents(case_text).lower()
    if article == "50":
        return SupportingArticleAssessment(
            article=article,
            status="applicable",
            factual_trigger="A penalty must be decided if the defendant is convicted.",
            explanation="Article 50 is the general basis for deciding penalties.",
        )
    if article == "51":
        has_mitigation = any(
            term in folded
            for term in ["thanh khan", "an nan", "boi thuong", "khac phuc", "dau thu", "tu thu", "nhan than tot"]
        )
        return SupportingArticleAssessment(
            article=article,
            status="applicable" if has_mitigation else "fact_dependent",
            factual_trigger="Input mentions potential mitigating facts." if has_mitigation else None,
            explanation="Article 51 applies only when mitigating circumstances are factually established.",
        )
    if article == "52":
        has_aggravation = any(
            term in folded
            for term in ["pham toi 02 lan", "pham toi nhieu lan", "co to chuc", "tai pham", "con do", "loi dung"]
        )
        return SupportingArticleAssessment(
            article=article,
            status="applicable" if has_aggravation else "fact_dependent",
            factual_trigger="Input mentions potential aggravating facts." if has_aggravation else None,
            explanation="Article 52 applies only when aggravating circumstances are factually established.",
        )
    if article == "53":
        has_recidivism = any(term in folded for term in ["tien an", "tien_an", "tai pham", "tai pham nguy hiem"])
        clean_record = any(
            term in folded
            for term in ["tien an\": \"khong", "tien_an\": \"khong", "tien an: khong", "tien an khong"]
        )
        return SupportingArticleAssessment(
            article=article,
            status="fact_dependent" if has_recidivism and not clean_record else "not_applicable",
            factual_trigger="Input mentions prior convictions or recidivism." if has_recidivism and not clean_record else None,
            explanation="Article 53 is relevant only for recidivism or dangerous recidivism.",
        )
    if article == "47":
        has_property = any(
            term in folded
            for term in ["vat chung", "tich thu", "sung quy", "cong cu", "phuong tien", "tra lai", "tai san", "tien"]
        )
        return SupportingArticleAssessment(
            article=article,
            status="fact_dependent" if has_property else "not_applicable",
            factual_trigger="Input mentions money/property/evidence handling." if has_property else None,
            explanation="Article 47 is relevant only when money/items directly related to the crime must be handled.",
        )
    if article == "58":
        has_accomplice = any(term in folded for term in ["dong pham", "cung thuc hien", "giup suc", "chu muu", "nhieu bi cao"])
        return SupportingArticleAssessment(
            article=article,
            status="fact_dependent" if has_accomplice else "not_applicable",
            factual_trigger="Input mentions possible accomplice participation." if has_accomplice else None,
            explanation="Article 58 applies in accomplice cases.",
        )
    if article in {"55", "56", "57", "54", "65", "38"}:
        return SupportingArticleAssessment(
            article=article,
            status="fact_dependent",
            factual_trigger=None,
            explanation="Applicability depends on the selected sentence, procedural posture, or additional facts.",
        )
    return SupportingArticleAssessment(
        article=article,
        status="fact_dependent",
        factual_trigger=None,
        explanation="Offence-specific supporting article was retrieved and requires fact-specific assessment.",
    )


def ensure_mandatory_supporting_assessments(
    assessments: list[SupportingArticleAssessment],
    retrieved_supporting: list[RetrievedLawArticle],
    *,
    case_text: str = "",
) -> list[SupportingArticleAssessment]:
    by_article = {item.article: item for item in assessments if item.article}
    retrieved_by_sig = {item.signature: item for item in retrieved_supporting}
    for article in MANDATORY_SUPPORTING_DIEU:
        if article in by_article:
            continue
        retrieved = retrieved_by_sig.get(article)
        by_article[article] = classify_supporting_article_by_facts(
            article=article,
            retrieved=retrieved,
            case_text=case_text,
        )
    return [by_article[key] for key in sorted(by_article, key=lambda value: int(value) if value.isdigit() else 9999)]


def _verdict_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get(VERDICT_FIELD)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _prosecution_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("De_Nghi_Cua_Vien_Kiem_Sat")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _with_parsed_prosecution_sentence(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["Phat_Tu_month_range"] = parse_penalty_to_months(_normalize_space(item.get("Phat_Tu")))
    return out


def _prosecution_item_for_defendant(data: dict[str, Any], defendant_name: str | None) -> dict[str, Any] | None:
    items = _prosecution_items(data)
    if not items:
        return None
    if defendant_name:
        defendant_key = _name_key(defendant_name)
        for item in items:
            if _name_key(item.get("Bi_Cao")) == defendant_key:
                return _with_parsed_prosecution_sentence(item)
    return _with_parsed_prosecution_sentence(items[0]) if len(items) == 1 else None


def _verdict_item_from_metadata(data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    verdicts = _verdict_items(data)
    record_index = metadata.get("record_index")
    if isinstance(record_index, int) and 0 <= record_index < len(verdicts):
        return verdicts[record_index]
    if isinstance(record_index, str) and record_index.isdigit():
        idx = int(record_index)
        if 0 <= idx < len(verdicts):
            return verdicts[idx]

    defendant_name = metadata.get("defendant_name")
    if defendant_name:
        defendant_key = _name_key(defendant_name)
        for item in verdicts:
            if _name_key(item.get("Bi_Cao")) == defendant_key:
                return item
    return verdicts[0] if len(verdicts) == 1 else None


def _case_labels_contain_dieu(labels: dict[str, set[str]] | None, dieu: str | None) -> bool:
    return bool(dieu and labels and str(dieu) in labels.get("dieu_only", set()))


def _load_train_doc_map(train_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for path in sorted(train_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        docs[doc_id_from_case(data, path.stem)] = data
    return docs


def _sentence_for_selected_dieu(data: dict[str, Any], selected_dieu: str | None) -> str | None:
    for item in _verdict_items(data):
        basis = item.get("Can_Cu_Dieu_Luat")
        if isinstance(basis, list):
            for clause in basis:
                if not isinstance(clause, dict):
                    continue
                if selected_dieu and str(clause.get("Dieu") or "").strip() == str(selected_dieu):
                    return _normalize_space(item.get("Phat_Tu"))
    for item in _verdict_items(data):
        sentence = _normalize_space(item.get("Phat_Tu"))
        if sentence:
            return sentence
    return None


def _case_profile_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("Synthetic_summary", "Summary", "NOI_DUNG_VU_AN"):
        value = data.get(field)
        if isinstance(value, str):
            parts.append(value[:2500])
        elif value:
            parts.append(json.dumps(value, ensure_ascii=False)[:2500])
    verdict_profiles: list[str] = []
    for item in _verdict_items(data):
        for field in ("Tang_nang", "Giam_nhe", "Trach_Nhiem_Dan_Su", "Pham_Toi"):
            value = item.get(field)
            if value:
                verdict_profiles.append(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
    parts.extend(verdict_profiles)
    return "\n".join(parts)


def _summarize_similar_case(data: dict[str, Any], doc_id: str, selected_dieu: str | None) -> SimilarCaseSummary:
    profile = _case_profile_text(data)
    mitigation_aggravation: list[str] = []
    offences: list[str] = []
    civil: list[str] = []
    for item in _verdict_items(data):
        if item.get("Pham_Toi"):
            offences.append(json.dumps(item.get("Pham_Toi"), ensure_ascii=False))
        if item.get("Tang_nang"):
            mitigation_aggravation.append(f"Tăng nặng: {_normalize_space(item.get('Tang_nang'))}")
        if item.get("Giam_nhe"):
            mitigation_aggravation.append(f"Giảm nhẹ: {_normalize_space(item.get('Giam_nhe'))}")
        if item.get("Trach_Nhiem_Dan_Su"):
            civil.append(_normalize_space(item.get("Trach_Nhiem_Dan_Su")))
    return SimilarCaseSummary(
        doc_id=doc_id,
        matched_offence_article=selected_dieu,
        matched_factual_profile=_normalize_space(profile[:900]),
        mitigation_aggravation_profile=_normalize_space("; ".join(mitigation_aggravation)[:900]) or None,
        sentence=_sentence_for_selected_dieu(data, selected_dieu),
        notable_reasoning=_normalize_space("; ".join(offences + civil)[:900]) or None,
    )


def retrieve_similar_cases(
    *,
    runtime: RetrievalRuntime,
    train_dir: Path,
    train_articles_index: dict[str, dict[str, set[str]]],
    query_text: str,
    selected_dieu: str | None,
    exclude_doc_id: str | None,
    broad_top_k: int = 64,
    top_k: int = 5,
) -> list[SimilarCaseSummary]:
    top_k = min(top_k, MAX_FINAL_CASES_PER_ISSUE)
    if not query_text or not selected_dieu or top_k <= 0:
        return []

    results = runtime.query_train(
        query_text=query_text,
        top_k=broad_top_k,
        exclude_doc_id=exclude_doc_id,
        include=["metadatas", "distances"],
    )
    doc_scores: dict[str, float] = {}
    for meta, distance in zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]):
        if not isinstance(meta, dict):
            continue
        if str(meta.get("source_type", "")).lower() == "law":
            continue
        rid = str(meta.get("doc_id") or "").strip()
        if not rid or not _case_labels_contain_dieu(train_articles_index.get(rid), selected_dieu):
            continue
        score = 1.0 - float(distance or 0.0)
        doc_scores[rid] = max(score, doc_scores.get(rid, -999.0))

    if not doc_scores:
        return []

    train_docs = _load_train_doc_map(train_dir)
    query_tokens = _tokenize(query_text)
    ranked: list[tuple[float, str]] = []
    for rid, base_score in doc_scores.items():
        data = train_docs.get(rid)
        if not data:
            continue
        profile = _case_profile_text(data)
        tokens = _tokenize(profile)
        overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
        has_sentence = 0.15 if _sentence_for_selected_dieu(data, selected_dieu) else 0.0
        ranked.append((base_score + overlap + has_sentence, rid))

    ranked.sort(reverse=True)
    return [_summarize_similar_case(train_docs[rid], rid, selected_dieu) for _, rid in ranked[:top_k]]


def retrieve_sentencing_calibration_cases(
    *,
    runtime: RetrievalRuntime,
    train_dir: Path,
    train_articles_index: dict[str, dict[str, set[str]]],
    mitigation_factors: list[str],
    aggravation_factors: list[str],
    selected_dieu: str | None,
    exclude_doc_id: str | None,
    top_k_per_factor: int = 3,
    broad_top_k: int = 64,
    max_cases_per_issue: int = MAX_FINAL_CASES_PER_ISSUE,
) -> list[SentencingCalibrationCase]:
    if top_k_per_factor <= 0 or max_cases_per_issue <= 0 or not selected_dieu:
        return []

    train_docs = _load_train_doc_map(train_dir)
    queries = [
        ("mitigation", MITIGATION_EMBED_FIELD, factor)
        for factor in _unique_text_items(mitigation_factors)
    ] + [
        ("aggravation", AGGRAVATION_EMBED_FIELD, factor)
        for factor in _unique_text_items(aggravation_factors)
    ]
    calibration_cases: list[SentencingCalibrationCase] = []
    issue_counts: dict[str, int] = {"mitigation": 0, "aggravation": 0}
    issue_seen: dict[str, set[tuple[str, str | None, str | None]]] = {
        "mitigation": set(),
        "aggravation": set(),
    }

    for factor_type, target_field, factor in queries:
        if issue_counts[factor_type] >= max_cases_per_issue:
            continue
        n_fetch = max(broad_top_k, top_k_per_factor * 16)
        cap = max(n_fetch, top_k_per_factor * 80)
        factor_cases: list[SentencingCalibrationCase] = []
        seen: set[tuple[str, str | None, str | None]] = set()

        while True:
            results = runtime.query_train(
                query_text=factor,
                top_k=n_fetch,
                exclude_doc_id=exclude_doc_id,
                include=["metadatas", "distances"],
            )
            rows = zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0])
            for meta, distance in rows:
                if not isinstance(meta, dict):
                    continue
                if meta.get("field") != target_field:
                    continue
                rid = str(meta.get("doc_id") or "").strip()
                if not rid:
                    continue
                if not _case_labels_contain_dieu(train_articles_index.get(rid), selected_dieu):
                    continue
                data = train_docs.get(rid)
                if not data:
                    continue
                verdict = _verdict_item_from_metadata(data, meta)
                if not verdict:
                    continue
                if issue_counts[factor_type] + len(factor_cases) >= max_cases_per_issue:
                    break
                defendant_name = _normalize_space(verdict.get("Bi_Cao")) or None
                seen_key = (rid, defendant_name, target_field)
                if seen_key in seen or seen_key in issue_seen[factor_type]:
                    continue
                seen.add(seen_key)
                score = 1.0 - float(distance or 0.0)
                factor_cases.append(
                    SentencingCalibrationCase(
                        factor_type=factor_type,
                        query_factor=factor,
                        doc_id=rid,
                        similarity_score=score,
                        matched_field=target_field,
                        defendant_name=defendant_name,
                        prosecution_proposal=_prosecution_item_for_defendant(data, defendant_name),
                        court_aggravation=_normalize_space(verdict.get("Tang_nang")) or None,
                        court_mitigation=_normalize_space(verdict.get("Giam_nhe")) or None,
                        court_sentence=_normalize_space(verdict.get("Phat_Tu")) or None,
                    )
                )
                if len(factor_cases) >= top_k_per_factor:
                    break

            if (
                len(factor_cases) >= top_k_per_factor
                or issue_counts[factor_type] + len(factor_cases) >= max_cases_per_issue
                or n_fetch >= cap
            ):
                break
            n_fetch = min(n_fetch * 2, cap)

        remaining = max_cases_per_issue - issue_counts[factor_type]
        selected_factor_cases = factor_cases[: min(top_k_per_factor, remaining)]
        calibration_cases.extend(selected_factor_cases)
        issue_counts[factor_type] += len(selected_factor_cases)
        for item in selected_factor_cases:
            issue_seen[factor_type].add((item.doc_id, item.defendant_name, item.matched_field))

    return calibration_cases


def _call_llm(
    *,
    provider: LLMProvider | str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[Any],
    use_provider_fallback: bool,
) -> tuple[Any, dict[str, Any]]:
    if use_provider_fallback:
        return generate_structured_output_with_fallback(
            preferred_provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )
    return generate_structured_output(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=output_model,
    )


def _candidate_prompt(doc_id: str, case_payload: dict[str, str]) -> tuple[str, str]:
    system = "You are a Vietnamese criminal law analysis assistant. Return only valid JSON."
    payload = {
        "doc_id": doc_id,
        "case_fields": case_payload,
        "input_format": {
            "Synthetic_summary": SYNTHETIC_SUMMARY_FORMAT_NOTE,
            "per_defendant_requirement": (
                "Extract defendant names and facts from each separate story. "
                "Do not merge one defendant's mitigating/aggravating facts, prior convictions, conduct, or sentence-relevant details into another defendant."
            ),
        },
        "task": [
            "Extract structured facts. Distinguish stated facts, conservative inferred facts, and missing facts.",
            "When Synthetic_summary is a list, extract facts per story and preserve which facts belong to which defendant.",
            "List mitigating factors as atomic statements in mitigation_factors; use one standalone fact per string.",
            "List aggravating factors as atomic statements in aggravation_factors; use one standalone fact per string.",
            "For multi-defendant cases, include the defendant name in each mitigation/aggravation factor string.",
            "Only include mitigation/aggravation factors supported by stated facts; do not invent factors.",
            "Generate 2 to 5 plausible BLHS offence candidates.",
            "Do not assume unknown facts are true.",
            "Preserve defendant names exactly as provided.",
        ],
        "output_schema": build_output_schema_instruction(ReasonActAnalysisOutput),
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def _legal_analysis_prompt(
    *,
    doc_id: str,
    facts_and_candidates: ReasonActAnalysisOutput,
    offence_articles: list[RetrievedLawArticle],
    additional_articles: list[RetrievedLawArticle],
    supporting_articles: list[RetrievedLawArticle],
    additional_law_round: int = 0,
) -> tuple[str, str]:
    system = "You are a Vietnamese criminal law judgment assistant. Return only valid JSON."
    payload = {
        "doc_id": doc_id,
        "facts": facts_and_candidates.facts.model_dump(),
        "candidates": [item.model_dump() for item in facts_and_candidates.candidates],
        "retrieved_offence_articles": [item.model_dump() for item in offence_articles],
        "retrieved_additional_articles": [item.model_dump() for item in additional_articles],
        "retrieved_supporting_articles": [item.model_dump() for item in supporting_articles],
        "additional_law_round": additional_law_round,
        "rules": [
            "Statutory law controls charge and sentencing-frame selection.",
            (
                "Select the likely offence and sentencing bracket from retrieved law only when the retrieved law covers "
                "the case facts."
            ),
            (
                "If the retrieved offence/supporting law appears incorrect, incomplete, or does not fully cover the "
                "case facts, "
                "put exact BLHS signatures in additional_law_queries with concise reasons."
            ),
            (
                "Prefer Dieu-level additional_law_queries when a whole article is needed; do not request smaller "
                "clauses inside an already retrieved full article."
            ),
            "If additional law is needed, still provide the best provisional selected_offence from the available law.",
            "Reject or downgrade every non-selected candidate with a concise reason.",
            "Classify every supporting article as applicable, fact_dependent, not_applicable, or not_retrieved.",
            "For every supporting article status, explain the factual trigger.",
            "Do not cite/apply a supporting article merely because it was retrieved.",
            "Do not infer unknown facts as true.",
        ],
        "mandatory_supporting_articles": list(MANDATORY_SUPPORTING_DIEU),
        "output_schema": build_output_schema_instruction(ReasonActLegalAnalysis),
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def _final_prompt(
    *,
    doc_id: str,
    case_payload: dict[str, str],
    legal_analysis: ReasonActLegalAnalysis,
    offence_articles: list[RetrievedLawArticle],
    additional_articles: list[RetrievedLawArticle],
    supporting_articles: list[RetrievedLawArticle],
    similar_cases: list[SimilarCaseSummary],
    sentencing_calibration_cases: list[SentencingCalibrationCase],
) -> tuple[str, str]:
    system = "You are a Vietnamese criminal judgment prediction assistant. Return only valid JSON."
    payload = {
        "doc_id": doc_id,
        "case_fields": case_payload,
        "input_format": {
            "Synthetic_summary": SYNTHETIC_SUMMARY_FORMAT_NOTE,
            "per_defendant_requirement": (
                "Generate one defendants entry for each defendant story and use only that defendant's own story plus shared case facts for individual sentencing factors."
            ),
        },
        "legal_analysis": legal_analysis.model_dump(),
        "retrieved_offence_articles": [item.model_dump() for item in offence_articles if item.found],
        "retrieved_additional_articles": [item.model_dump() for item in additional_articles if item.found],
        "retrieved_supporting_articles": [item.model_dump() for item in supporting_articles],
        "similar_cases_for_analogy_only": [item.model_dump() for item in similar_cases],
        "sentencing_calibration_cases": [item.model_dump() for item in sentencing_calibration_cases],
        "rules": [
            "Produce final prediction in GenerationOutput.",
            "When Synthetic_summary is a list, generate one matching defendants item per story and preserve the defendant name from that story.",
            (
                "Use retrieved_offence_articles and retrieved_additional_articles as the statutory "
                "offence/sentencing-frame context."
            ),
            "Include only actually applicable clauses in Applied_Law_Clauses.",
            "Do not include checked-but-not-applicable or fact-dependent articles in Applied_Law_Clauses.",
            "Use similar cases only for analogy and sentencing calibration, never to override statutory law.",
            (
                "For sentencing_calibration_cases, read De_Nghi_Cua_Vien_Kiem_Sat.Phat_Tu as the prosecution's requested "
                "prison range, De_Nghi_Cua_Vien_Kiem_Sat.Phat_Tu_month_range as its parsed [minimum, maximum] months, "
                "and PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang, Giam_nhe, and Phat_Tu as the court's applied "
                "aggravating factors, mitigating factors, and final prison term. Use these past cases only to calibrate "
                "the current prison term by comparing how similar the current mitigation/aggravation factors are to the "
                "retrieved cases."
            ),
            "Predict concrete Phat_Tu, additional fine, civil/property consequences, and Xu_Ly_Vat_Chung when supported.",
            "For mitigation advice, only mention lawful cooperation, restitution, documentation, and procedural steps.",
        ],
        "output_schema": build_output_schema_instruction(ReasonActFinalOutput),
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def run_reasoning_act(
    *,
    data: dict[str, Any],
    doc_id: str,
    law_retriever: LawClauseRetriever,
    case_runtime: RetrievalRuntime,
    train_dir: Path,
    train_articles_index: dict[str, dict[str, set[str]]] | None,
    provider: LLMProvider | str,
    model_name: str,
    use_provider_fallback: bool = True,
    input_fields: list[str] | None = None,
    query_fields: list[str] | None = None,
    broad_top_k_case: int = 64,
    top_k_case: int = 5,
    max_additional_law_rounds: int = 1,
) -> dict[str, Any]:
    input_fields = input_fields or ["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", "Synthetic_summary_2"]
    query_fields = query_fields or ["Synthetic_summary_2", "THONG_TIN_CHUNG.Thong_Tin_Bi_Cao"]
    train_articles_index = train_articles_index if train_articles_index is not None else load_articles_index(train_dir)[0]

    case_payload = extract_input_payload(data, input_fields)
    query_text = build_query_text(data, query_fields)
    case_text = "\n\n".join(case_payload.values())
    usage: dict[str, Any] = {"calls": []}

    system, user = _candidate_prompt(doc_id, case_payload)
    facts_and_candidates, call_usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActAnalysisOutput,
        use_provider_fallback=use_provider_fallback,
    )
    usage["calls"].append({"name": "facts_and_candidates", **call_usage})

    offence_articles = retrieve_candidate_articles(facts_and_candidates.candidates, law_retriever)
    found_offence_text = "\n\n".join(item.text or "" for item in offence_articles if item.found)
    supporting_articles = retrieve_supporting_articles(
        case_text="\n\n".join(case_payload.values()),
        selected_offence_text=found_offence_text,
        law_retriever=law_retriever,
    )
    additional_articles: list[RetrievedLawArticle] = []

    system, user = _legal_analysis_prompt(
        doc_id=doc_id,
        facts_and_candidates=facts_and_candidates,
        offence_articles=offence_articles,
        additional_articles=additional_articles,
        supporting_articles=supporting_articles,
    )
    legal_analysis, call_usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActLegalAnalysis,
        use_provider_fallback=use_provider_fallback,
    )
    legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
        legal_analysis.supporting_article_assessments,
        supporting_articles,
        case_text=case_text,
    )
    usage["calls"].append({"name": "legal_analysis", **call_usage})

    for round_idx in range(max(max_additional_law_rounds, 0)):
        requested_signatures = _additional_law_query_signatures(legal_analysis.additional_law_queries)
        new_signatures = _filter_new_law_signatures(
            requested_signatures,
            offence_articles + supporting_articles + additional_articles,
        )
        if not new_signatures:
            break
        additional_articles.extend(retrieve_law_articles(new_signatures, law_retriever))
        system, user = _legal_analysis_prompt(
            doc_id=doc_id,
            facts_and_candidates=facts_and_candidates,
            offence_articles=offence_articles,
            additional_articles=additional_articles,
            supporting_articles=supporting_articles,
            additional_law_round=round_idx + 1,
        )
        legal_analysis, call_usage = _call_llm(
            provider=provider,
            model_name=model_name,
            system_prompt=system,
            user_prompt=user,
            output_model=ReasonActLegalAnalysis,
            use_provider_fallback=use_provider_fallback,
        )
        legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
            legal_analysis.supporting_article_assessments,
            supporting_articles,
            case_text=case_text,
        )
        usage["calls"].append({"name": "legal_analysis_additional_law", "round": round_idx + 1, **call_usage})

    selected_dieu = legal_analysis.selected_offence.Dieu
    selected_key = _canonical_law_signature(selected_dieu)
    all_retrieved_articles = offence_articles + additional_articles
    _, all_retrieved_signatures = _existing_law_coverage(all_retrieved_articles)
    if selected_key and selected_key not in all_retrieved_signatures:
        offence_articles.extend(retrieve_law_articles([selected_dieu], law_retriever))

    similar_cases = retrieve_similar_cases(
        runtime=case_runtime,
        train_dir=train_dir,
        train_articles_index=train_articles_index,
        query_text=query_text,
        selected_dieu=selected_dieu,
        exclude_doc_id=doc_id,
        broad_top_k=broad_top_k_case,
        top_k=top_k_case,
    )
    sentencing_calibration_cases = retrieve_sentencing_calibration_cases(
        runtime=case_runtime,
        train_dir=train_dir,
        train_articles_index=train_articles_index,
        mitigation_factors=facts_and_candidates.mitigation_factors,
        aggravation_factors=facts_and_candidates.aggravation_factors,
        selected_dieu=selected_dieu,
        exclude_doc_id=doc_id,
        top_k_per_factor=3,
        broad_top_k=broad_top_k_case,
    )

    system, user = _final_prompt(
        doc_id=doc_id,
        case_payload=case_payload,
        legal_analysis=legal_analysis,
        offence_articles=offence_articles,
        additional_articles=additional_articles,
        supporting_articles=supporting_articles,
        similar_cases=similar_cases,
        sentencing_calibration_cases=sentencing_calibration_cases,
    )
    final_output, call_usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActFinalOutput,
        use_provider_fallback=use_provider_fallback,
    )
    usage["calls"].append({"name": "final_prediction", **call_usage})

    trace = ReasonActTrace(
        facts=facts_and_candidates.facts,
        candidates=facts_and_candidates.candidates,
        mitigation_factors=facts_and_candidates.mitigation_factors,
        aggravation_factors=facts_and_candidates.aggravation_factors,
        selected_offence=final_output.legal_analysis.selected_offence,
        rejected_candidates=final_output.legal_analysis.rejected_candidates,
        additional_law_queries=final_output.legal_analysis.additional_law_queries,
        retrieved_offence_articles=offence_articles,
        retrieved_additional_articles=additional_articles,
        retrieved_supporting_articles=supporting_articles,
        supporting_article_assessments=ensure_mandatory_supporting_assessments(
            final_output.legal_analysis.supporting_article_assessments,
            supporting_articles,
            case_text=case_text,
        ),
        similar_cases=similar_cases,
        sentencing_calibration_cases=sentencing_calibration_cases,
        sentencing_bracket=final_output.legal_analysis.sentencing_bracket,
        confidence=final_output.legal_analysis.confidence,
        missing_facts=final_output.legal_analysis.missing_facts,
    )
    return {
        "prediction": final_output.prediction,
        "trace": trace,
        "usage": usage,
        "llm_input_payload": case_payload,
    }


def safe_run_reasoning_act(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return run_reasoning_act(**kwargs), None
    except (ValidationError, json.JSONDecodeError) as exc:
        return None, f"parse_error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"generation_error: {exc}"
