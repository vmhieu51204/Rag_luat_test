"""Shared Pydantic schemas for generation and generation evaluation flows."""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, Field


class PredictedLawClause(BaseModel):
    Dieu: str | None = Field(default=None, description="Law article number (Dieu), e.g. '173'.")
    Khoan: str | None = Field(default=None, description="Law clause number (Khoan) within the Dieu.")
    Diem: str | None = Field(default=None, description="Law point letter (Diem), typically lowercase letter like 'a'.")
    Tinh_tiet_ap_dung: str | None = Field(
        default=None,
        description=(
            "Concise explanation of the specific case details that trigger the application of this clause, based on the provided case facts."
        ),
    )
    Bo_Luat_Va_Van_Ban_Khac: str | None = Field(
        default=None,
        description="Legal source name/code (e.g. BLHS) for the cited clause.",
    )


class PredictedDefendant(BaseModel):
    Bi_Cao: str = Field(description="Defendant name exactly matching the case context.")
    Phan_Tich_Phap_Ly: str | None = Field(
        default=None,
        description=(
            "Concise legal reasoning based only on provided case facts. Identify mitigating and aggravating factors based on the provided article 51 and article 51."
        ),
    )
    Toi_Danh: str | None = Field(default=None, description="Offense/crime name for the defendant.")
    Applied_Law_Clauses: list[PredictedLawClause] = Field(
        default_factory=list,
        description="List of applicable legal clauses supporting the verdict for this defendant.",
    )
    Phat_Tu: str | None = Field(
        default=None,
        description="Final concrete imprisonment verdict text, combine all charges or past imprisonment sentences if available; not a sentencing range.",
    )
    Phat_Tien: str | None = Field(
        default=None,
        description="Monetary fine verdict text for the defendant when applicable; keep null if not imposed.",
    )
    Trach_Nhiem_Dan_Su: str | None = Field(
        default=None,
        description="Civil liability decision for the defendant when applicable.",
    )


class PredictedDefendantRange(BaseModel):
    Bi_Cao: str = Field(description="Defendant name exactly matching the case context.")
    Phan_Tich_Phap_Ly: str | None = Field(
        default=None,
        description=(
            "Concise legal reasoning based only on provided case facts. Identify mitigating and aggravating factors based on the provided articles 51 and 52."
        ),
    )
    Toi_Danh: str | None = Field(default=None, description="Offense/crime name for the defendant.")
    Applied_Law_Clauses: list[PredictedLawClause] = Field(
        default_factory=list,
        description="List of applicable legal clauses supporting the verdict range for this defendant.",
    )
    Phat_Tu_Range: str | None = Field(
        default=None,
        description="Predicted imprisonment range based on the law clauses and case facts, e.g. 'từ 02 năm đến 03 năm tù'. Must be a range.",
    )
    Phat_Tien: str | None = Field(
        default=None,
        description="Monetary fine verdict text for the defendant when applicable; keep null if not imposed.",
    )
    Trach_Nhiem_Dan_Su: str | None = Field(
        default=None,
        description="Civil liability decision for the defendant when applicable.",
    )


class GenerationOutput(BaseModel):
    defendants: list[PredictedDefendant] = Field(description="Per-defendant structured predictions.")
    Xu_Ly_Vat_Chung: str | None = Field(
        default=None,
        description="Decision on handling/seizure/disposal of physical evidence related to the case based on the provided article 47",
    )


class GenerationRangeOutput(BaseModel):
    defendants: list[PredictedDefendantRange] = Field(description="Per-defendant structured range predictions.")
    Xu_Ly_Vat_Chung: str | None = Field(
        default=None,
        description="Decision on handling/seizure/disposal of physical evidence related to the case based on the provided article 47",
    )


class ExtractedFacts(BaseModel):
    defendants: list[str] = Field(default_factory=list, description="Defendant names exactly as provided.")
    conduct: str | None = Field(default=None, description="Concise factual conduct alleged or admitted.")
    dates: list[str] = Field(default_factory=list, description="Relevant dates or periods stated in the facts.")
    victims: list[str] = Field(default_factory=list, description="Victims or harmed parties stated in the facts.")
    property_value: str | None = Field(default=None, description="Property, benefit, damage, or value amount if stated.")
    harm: str | None = Field(default=None, description="Stated harm or consequence.")
    admissions: str | None = Field(default=None, description="Confession, admission, cooperation, or denial facts.")
    prior_convictions: str | None = Field(default=None, description="Criminal record / prior conviction facts.")
    mitigation_signals: list[str] = Field(default_factory=list, description="Potential mitigating facts stated by the user.")
    aggravation_signals: list[str] = Field(default_factory=list, description="Potential aggravating facts stated by the user.")
    stated_facts: list[str] = Field(default_factory=list, description="Important facts directly stated in the input.")
    inferred_facts: list[str] = Field(default_factory=list, description="Facts inferred from stated facts; keep conservative.")
    missing_facts: list[str] = Field(default_factory=list, description="Missing facts that could change the legal result.")

class CandidateOffence(BaseModel):
    Dieu: str | None = Field(default=None, description="Candidate BLHS article number.")
    Khoan: str | None = Field(default=None, description="Likely clause number if known, otherwise keep null if unsure of exact clause.")
    Diem: str | None = Field(default=None, description="Likely point letter if known, otherwise keep null if unsure of exact point.")
    offence_name: str | None = Field(default=None, description="Candidate offence name.")
    search_query: str | None = Field(default=None, description="Short search/retrieval query for this candidate.")
    supporting_facts: str | None = Field(default=None, description="Facts supporting this candidate.")
    rejection_or_downgrade_reason: str | None = Field(
        default=None,
        description="Reason this candidate is rejected or downgraded; null for the selected candidate.",
    )

class RetrievedLawArticle(BaseModel):
    signature: str = Field(description="Requested law signature, e.g. 174-4-a or 51.")
    found: bool = Field(description="Whether exact retrieval found text in raw_law.json.")
    level: str | None = Field(default=None, description="Retrieved level: dieu, khoan, diem, or null.")
    text: str | None = Field(default=None, description="Retrieved statutory text.")
    missing_reason: str | None = Field(default=None, description="Retriever reason when not found.")

class SupportingArticleAssessment(BaseModel):
    article: str = Field(description="BLHS article number checked.")
    status: Literal["applicable", "fact_dependent", "not_applicable", "not_retrieved"] = Field(
        description="Applicability classification for the case facts."
    )
    factual_trigger: str | None = Field(default=None, description="Facts triggering this classification.")
    explanation: str | None = Field(default=None, description="Concise legal explanation.")


class AdditionalLawQuery(BaseModel):
    signature: str = Field(description="Exact BLHS signature to retrieve, e.g. 174, 174-2, or 174-2-a.")
    reason: str | None = Field(
        default=None,
        description="Why the currently retrieved law is incorrect, incomplete, or insufficient for the case facts.",
    )


class SimilarCaseSummary(BaseModel):
    doc_id: str = Field(description="Similar train-case document id.")
    matched_offence_article: str | None = Field(default=None, description="Selected offence article matched in this case.")
    matched_factual_profile: str | None = Field(default=None, description="Similar fact pattern summary.")
    mitigation_aggravation_profile: str | None = Field(default=None, description="Mitigating/aggravating profile.")
    sentence: str | None = Field(default=None, description="Ground-truth sentence in the similar case.")
    notable_reasoning: str | None = Field(default=None, description="Notable reasoning or property/civil handling details.")

class SentencingCalibrationCase(BaseModel):
    factor_type: Literal["mitigation", "aggravation"] = Field(
        description="Whether the retrieved past case matched a mitigating or aggravating factor."
    )
    query_factor: str = Field(description="Atomic current-case factor used as the retrieval query.")
    doc_id: str = Field(description="Retrieved train-case document id.")
    similarity_score: float | None = Field(default=None, description="Vector similarity score, if available.")
    matched_field: str | None = Field(default=None, description="Embedded verdict field matched by vector retrieval.")
    defendant_name: str | None = Field(default=None, description="Defendant in the retrieved verdict item, if available.")
    prosecution_proposal: Any | None = Field(
        default=None,
        description="Matching De_Nghi_Cua_Vien_Kiem_Sat item for this defendant, including Phat_Tu range if available.",
    )
    court_aggravation: str | None = Field(
        default=None,
        description="PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang for the matched defendant.",
    )
    court_mitigation: str | None = Field(
        default=None,
        description="PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe for the matched defendant.",
    )
    court_sentence: str | None = Field(
        default=None,
        description="PHAN_QUYET_CUA_TOA_SO_THAM.Phat_Tu for the matched defendant.",
    )

class ReasonActAnalysisOutput(BaseModel):
    facts: ExtractedFacts = Field(description="Structured facts extracted from the input.")
    candidates: list[CandidateOffence] = Field(description="Two to five candidate BLHS offences.")
    mitigation_factors: list[str] = Field(
        default_factory=list,
        description="Atomic mitigating-factor statements extracted from the case facts; one fact per string and include the defendant name when relevant.",
    )
    aggravation_factors: list[str] = Field(
        default_factory=list,
        description="Atomic aggravating-factor statements extracted from the case facts; one fact per string and include the defendant name when relevant.",
    )

class ReasonActLegalAnalysis(BaseModel):
    selected_offence: CandidateOffence = Field(description="Likely offence and sentencing bracket selected from retrieved law.")
    rejected_candidates: list[CandidateOffence] = Field(default_factory=list, description="Rejected or downgraded candidates.")
    additional_law_queries: list[AdditionalLawQuery] = Field(
        default_factory=list,
        description=(
            "Additional exact BLHS law signatures that should be retrieved when the provided law text appears "
            "incorrect, "
            "incomplete, or does not fully cover the case facts. Leave empty when no additional law is needed."
        ),
    )
    supporting_article_assessments: list[SupportingArticleAssessment] = Field(
        default_factory=list,
        description="Classification for every mandatory and offence-specific supporting article.",
    )
    sentencing_bracket: str | None = Field(default=None, description="Selected statutory sentencing frame.")
    missing_facts: list[str] = Field(default_factory=list, description="Missing facts that could change the result.")
    confidence: str | None = Field(default=None, description="Low/medium/high confidence with short explanation.")


class ReasonActFinalOutput(BaseModel):
    legal_analysis: ReasonActLegalAnalysis = Field(description="Final legal analysis trace.")
    prediction: GenerationOutput = Field(description="Final scored prediction output.")


class ReasonActTrace(BaseModel):
    facts: ExtractedFacts | None = Field(default=None, description="Structured facts from step 1.")
    candidates: list[CandidateOffence] = Field(default_factory=list, description="Candidate offences from step 2.")
    mitigation_factors: list[str] = Field(default_factory=list)
    aggravation_factors: list[str] = Field(default_factory=list)
    selected_offence: CandidateOffence | None = Field(default=None, description="Selected offence from statutory matching.")
    rejected_candidates: list[CandidateOffence] = Field(default_factory=list, description="Rejected/downgraded candidates.")
    additional_law_queries: list[AdditionalLawQuery] = Field(default_factory=list)
    retrieved_offence_articles: list[RetrievedLawArticle] = Field(default_factory=list)
    retrieved_additional_articles: list[RetrievedLawArticle] = Field(default_factory=list)
    retrieved_supporting_articles: list[RetrievedLawArticle] = Field(default_factory=list)
    supporting_article_assessments: list[SupportingArticleAssessment] = Field(default_factory=list)
    similar_cases: list[SimilarCaseSummary] = Field(default_factory=list)
    sentencing_calibration_cases: list[SentencingCalibrationCase] = Field(default_factory=list)
    sentencing_bracket: str | None = None
    confidence: str | None = None
    missing_facts: list[str] = Field(default_factory=list)


def _is_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _annotation_label(annotation: Any) -> str | dict[str, Any] | list[Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and args:
        return [_annotation_label(args[0])]

    if args and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            inner = _annotation_label(non_none[0])
            if isinstance(inner, str):
                return f"{inner}|null"
            return inner

    if _is_model_type(annotation):
        return build_output_schema_instruction(annotation)

    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"

    return "any"


def build_output_schema_instruction(model: type[BaseModel]) -> dict[str, Any]:
    """Build a prompt-friendly schema skeleton from a Pydantic model.

    Each field is rendered with its expected type and optional instruction text from
    the Field description, so the prompt schema and runtime parser stay synchronized.
    """

    output: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        value = _annotation_label(field_info.annotation)
        desc = (field_info.description or "").strip()

        if isinstance(value, str):
            output[field_name] = f"{value} ({desc})" if desc else value
            continue

        if isinstance(value, list):
            if desc:
                    # For lists, return a simple structure: [item_type] with instruction as suffix.
                    # This avoids wrapping list items in metadata dicts that confuse LLMs.
                    output[field_name] = f"[{value[0]}] ({desc})" if desc else value
            else:
                output[field_name] = value
            continue

        if isinstance(value, dict):
            if desc:
                value = {"_instruction": desc, **value}
            output[field_name] = value
            continue

        output[field_name] = value

    return output
