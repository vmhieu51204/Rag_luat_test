"""Step-by-step execution pipeline for the Multistage Legal Reasoning Demo."""

import json
from pathlib import Path
from typing import Any, Tuple

from rag.generation.reasoning_act import (
    _candidate_prompt,
    _legal_analysis_prompt,
    _final_prompt,
    _call_llm,
    retrieve_candidate_articles,
    retrieve_supporting_articles,
    ensure_mandatory_supporting_assessments,
    _additional_law_query_signatures,
    _filter_new_law_signatures,
    retrieve_law_articles,
    _canonical_law_signature,
    _existing_law_coverage,
    retrieve_similar_cases,
    retrieve_sentencing_calibration_cases,
)
from rag.generation.schemas import (
    ReasonActAnalysisOutput,
    ReasonActLegalAnalysis,
    ReasonActFinalOutput,
    RetrievedLawArticle,
)
from rag.core.law_retriever import LawClauseRetriever
from rag.runtime.retrieval import RetrievalRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "chunk" / "train"

def run_stage_1(
    doc_id: str,
    defendant_info: str,
    synthetic_summary_2: str,
    provider: str,
    model_name: str
) -> Tuple[ReasonActAnalysisOutput, dict, dict]:
    """Runs Stage 1: Facts and candidate offences extraction."""
    case_payload = {
        "THONG_TIN_CHUNG.Thong_Tin_Bi_Cao": defendant_info,
        "Synthetic_summary_2": synthetic_summary_2
    }
    system, user = _candidate_prompt(doc_id, case_payload)
    
    facts_and_candidates, usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActAnalysisOutput,
        use_provider_fallback=True,
    )
    return facts_and_candidates, case_payload, usage

def run_law_retrieval(
    facts_and_candidates: ReasonActAnalysisOutput,
    case_payload: dict,
    law_retriever: LawClauseRetriever
) -> Tuple[list[RetrievedLawArticle], list[RetrievedLawArticle]]:
    """Retrieves candidate offence and supporting law articles."""
    case_text = "\n\n".join(case_payload.values())
    
    # Retrieve offence articles from proposed candidates
    offence_articles = retrieve_candidate_articles(
        facts_and_candidates.candidates, law_retriever
    )
    
    # Retrieve supporting articles
    found_offence_text = "\n\n".join(
        a.text or "" for a in offence_articles if a.found
    )
    supporting_articles = retrieve_supporting_articles(
        case_text=case_text,
        selected_offence_text=found_offence_text,
        law_retriever=law_retriever,
    )
    return offence_articles, supporting_articles

def run_stage_2(
    doc_id: str,
    case_payload: dict,
    facts_and_candidates: ReasonActAnalysisOutput,
    offence_articles: list[RetrievedLawArticle],
    supporting_articles: list[RetrievedLawArticle],
    law_retriever: LawClauseRetriever,
    provider: str,
    model_name: str
) -> Tuple[ReasonActLegalAnalysis, list[RetrievedLawArticle], bool, list[str], dict]:
    """Runs Stage 2: Legal analysis and optional additional law round."""
    additional_articles = []
    case_text = "\n\n".join(case_payload.values())
    
    system, user = _legal_analysis_prompt(
        doc_id=doc_id,
        facts_and_candidates=facts_and_candidates,
        offence_articles=offence_articles,
        additional_articles=additional_articles,
        supporting_articles=supporting_articles,
    )
    
    legal_analysis, usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActLegalAnalysis,
        use_provider_fallback=True,
    )
    
    legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
        legal_analysis.supporting_article_assessments,
        supporting_articles,
        case_text=case_text,
    )
    
    # Check if additional law was requested
    requested_sigs = _additional_law_query_signatures(legal_analysis.additional_law_queries)
    new_sigs = _filter_new_law_signatures(
        requested_sigs, offence_articles + supporting_articles + additional_articles
    )
    
    re_ran = False
    if new_sigs:
        additional_articles.extend(retrieve_law_articles(new_sigs, law_retriever))
        # Re-run legal analysis with the extra articles
        system_2, user_2 = _legal_analysis_prompt(
            doc_id=doc_id,
            facts_and_candidates=facts_and_candidates,
            offence_articles=offence_articles,
            additional_articles=additional_articles,
            supporting_articles=supporting_articles,
            additional_law_round=1,
        )
        legal_analysis, usage_2 = _call_llm(
            provider=provider,
            model_name=model_name,
            system_prompt=system_2,
            user_prompt=user_2,
            output_model=ReasonActLegalAnalysis,
            use_provider_fallback=True,
        )
        legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
            legal_analysis.supporting_article_assessments,
            supporting_articles,
            case_text=case_text,
        )
        usage = {"round_1": usage, "round_2": usage_2}
        re_ran = True
        
    return legal_analysis, additional_articles, re_ran, new_sigs, usage

def run_case_retrieval(
    case_payload: dict,
    legal_analysis: ReasonActLegalAnalysis,
    facts_and_candidates: ReasonActAnalysisOutput,
    offence_articles: list[RetrievedLawArticle],
    additional_articles: list[RetrievedLawArticle],
    law_retriever: LawClauseRetriever,
    case_runtime: RetrievalRuntime,
    train_articles_index: dict,
    doc_id: str
) -> Tuple[list, list]:
    """Retrieves similar cases and sentencing calibration cases."""
    selected_dieu = legal_analysis.selected_offence.Dieu
    
    # Ensure selected offence article is retrieved
    selected_key = _canonical_law_signature(selected_dieu)
    _, all_sigs = _existing_law_coverage(offence_articles + additional_articles)
    if selected_key and selected_key not in all_sigs:
        offence_articles.extend(retrieve_law_articles([selected_dieu], law_retriever))
        
    # Reconstruct query text
    query_text = f"[Synthetic_summary_2]\n{case_payload.get('Synthetic_summary_2', '')}\n\n[THONG_TIN_CHUNG.Thong_Tin_Bi_Cao]\n{case_payload.get('THONG_TIN_CHUNG.Thong_Tin_Bi_Cao', '')}"
    
    # Retrieve similar past cases (by factual profile)
    similar_cases = retrieve_similar_cases(
        runtime=case_runtime,
        train_dir=TRAIN_DIR,
        train_articles_index=train_articles_index,
        query_text=query_text,
        selected_dieu=selected_dieu,
        exclude_doc_id=doc_id,
        broad_top_k=64,
        top_k=5,
    )
    
    # Retrieve calibration cases (by mitigation/aggravation factors)
    sentencing_calibration_cases = retrieve_sentencing_calibration_cases(
        runtime=case_runtime,
        train_dir=TRAIN_DIR,
        train_articles_index=train_articles_index,
        mitigation_factors=facts_and_candidates.mitigation_factors,
        aggravation_factors=facts_and_candidates.aggravation_factors,
        selected_dieu=selected_dieu,
        exclude_doc_id=doc_id,
        top_k_per_factor=3,
        broad_top_k=64,
    )
    
    return similar_cases, sentencing_calibration_cases

def run_stage_3(
    doc_id: str,
    case_payload: dict,
    facts_and_candidates: ReasonActAnalysisOutput,
    legal_analysis: ReasonActLegalAnalysis,
    offence_articles: list[RetrievedLawArticle],
    additional_articles: list[RetrievedLawArticle],
    supporting_articles: list[RetrievedLawArticle],
    similar_cases: list,
    sentencing_calibration_cases: list,
    provider: str,
    model_name: str
) -> Tuple[ReasonActFinalOutput, dict]:
    """Runs Stage 3: Final verdict prediction and sentencing."""
    system, user = _final_prompt(
        doc_id=doc_id,
        case_payload=case_payload,
        facts_and_candidates=facts_and_candidates,
        legal_analysis=legal_analysis,
        offence_articles=offence_articles,
        additional_articles=additional_articles,
        supporting_articles=supporting_articles,
        similar_cases=similar_cases,
        sentencing_calibration_cases=sentencing_calibration_cases,
    )
    
    final_output, usage = _call_llm(
        provider=provider,
        model_name=model_name,
        system_prompt=system,
        user_prompt=user,
        output_model=ReasonActFinalOutput,
        use_provider_fallback=True,
    )
    return final_output, usage
