# %% [markdown]
# # 🔍 Multistage ReAct Reasoning Pipeline — Demo
# 
# This notebook walks through the **3-stage LLM reasoning pipeline** defined in
# `rag/generation/reasoning_act.py`, step by step:
# 
# | Stage | LLM Call | Purpose |
# |-------|----------|--------|
# | 1 | `ReasonActAnalysisOutput` | Extract facts, candidates, mitigation/aggravation factors |
# | 2 | `ReasonActLegalAnalysis` | Select offence, assess supporting articles, request additional law |
# | 3 | `ReasonActFinalOutput` | Produce final verdict prediction with sentencing |
# 
# Between each LLM call, **retrieval steps** fetch law articles and similar past cases.

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 0 — Imports, environment, initialization
# ═══════════════════════════════════════════════════════════════════════
import json, os, sys
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv

# Ensure the repo root is on sys.path
REPO_ROOT = Path("/home/hieujayce/Downloads/complete_repo")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

# ── Core imports ──────────────────────────────────────────────────────
from rag.config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, DEFAULT_COLLECTION_NAME
from rag.core.law_retriever import LawClauseRetriever
from rag.evaluation.eval_utils import load_articles_index, _extract_gt_defendants
from rag.generation.reasoning_act import (
    extract_input_payload,
    build_query_text,
    doc_id_from_case,
    retrieve_candidate_articles,
    retrieve_supporting_articles,
    retrieve_similar_cases,
    retrieve_sentencing_calibration_cases,
    ensure_mandatory_supporting_assessments,
    _candidate_prompt,
    _legal_analysis_prompt,
    _final_prompt,
    _call_llm,
    _filter_new_law_signatures,
    _additional_law_query_signatures,
    retrieve_law_articles,
    _canonical_law_signature,
    _existing_law_coverage,
    MANDATORY_SUPPORTING_DIEU,
    DEFAULT_REASON_ACT_TRAIN_FIELDS,
    DEFAULT_SENTENCING_CALIBRATION_FIELDS,
)
from rag.generation.schemas import (
    ReasonActAnalysisOutput,
    ReasonActLegalAnalysis,
    ReasonActFinalOutput,
)
from rag.llm.providers import LLMProvider
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig

# ── Paths ───────────────────────────────────────────────────────────
TRAIN_DIR   = REPO_ROOT / "chunk" / "train"
TEST_DIR    = REPO_ROOT / "chunk" / "test"
LAW_JSON    = REPO_ROOT / "raw_law.json"
CASE_DB_DIR = REPO_ROOT / "output" / "reasoning_act_eval" / "case_db"

INPUT_FIELDS = ["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", "Synthetic_summary_2"]
QUERY_FIELDS = ["Synthetic_summary_2", "THONG_TIN_CHUNG.Thong_Tin_Bi_Cao"]

# ── LLM provider config ──────────────────────────────────────────────
PROVIDER   = LLMProvider.AISTUDIO
MODEL_NAME = "gemma-4-31b-it"   # change as needed

# ── Initialize heavy components once ─────────────────────────────────
print("Loading law retriever …")
law_retriever = LawClauseRetriever(LAW_JSON)

print("Building train article index …")
train_articles_index, train_skipped = load_articles_index(TRAIN_DIR)
print(f"  Index covers {len(train_articles_index)} docs  (skipped {len(train_skipped)})")

print("Connecting to case vector DB …")
case_runtime = RetrievalRuntime(
    RetrievalRuntimeConfig(
        model_name=DEFAULT_MODEL_NAME,
        device=DEFAULT_DEVICE,
        train_db_dir=str(CASE_DB_DIR),
        collection_name=DEFAULT_COLLECTION_NAME,
    )
)
print(f"  DB doc count = {case_runtime.train_doc_count()}")

# ── Load a sample test case ──────────────────────────────────────────
SAMPLE_FILE = sorted(TEST_DIR.glob("*.json"))[0]   # first file alphabetically
with open(SAMPLE_FILE, encoding="utf-8") as fh:
    case_data = json.load(fh)

doc_id = doc_id_from_case(case_data, SAMPLE_FILE.stem)
print(f"\n✅ Loaded sample: {SAMPLE_FILE.name}  (doc_id = {doc_id})")

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 1 — Show the input: defendant info + synthetic summary
# ═══════════════════════════════════════════════════════════════════════
case_payload = extract_input_payload(case_data, INPUT_FIELDS)
query_text   = build_query_text(case_data, QUERY_FIELDS)
case_text    = "\n\n".join(case_payload.values())

print("── Input fields sent to the LLM ──────────────────────────")
for field, value in case_payload.items():
    print(f"\n🔹 [{field}]")
    # Pretty-print JSON-like fields, plain-print text
    try:
        parsed = json.loads(value)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(value[:2000])

print("\n── Ground-truth verdict (for later comparison) ──────────")
gt_defendants = _extract_gt_defendants(case_data, only_blhs=True)
for d in gt_defendants:
    print(f"  Bị cáo: {d['Bi_Cao']}")
    print(f"  Tội danh: {d['Toi_Danh']}")
    print(f"  Phạt tù: {d['Phat_Tu']}")
    print(f"  Điều luật: {d['Applied_Law_Clauses']}")
    print()

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 2 — Stage 1: Extract facts & candidate offences (LLM call 1)
# ═══════════════════════════════════════════════════════════════════════
system_1, user_1 = _candidate_prompt(doc_id, case_payload)

print("📤 Calling LLM — Stage 1 (fact extraction + candidate offences) …")
facts_and_candidates, usage_1 = _call_llm(
    provider=PROVIDER,
    model_name=MODEL_NAME,
    system_prompt=system_1,
    user_prompt=user_1,
    output_model=ReasonActAnalysisOutput,
    use_provider_fallback=True,
)

print("\n✅ Stage 1 complete.  Structured output:")
print(json.dumps(facts_and_candidates.model_dump(), ensure_ascii=False, indent=2))

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 3 — Retrieve law articles from raw_law.json
# ═══════════════════════════════════════════════════════════════════════

# 3a — Offence articles (from LLM-proposed candidates)
offence_articles = retrieve_candidate_articles(
    facts_and_candidates.candidates, law_retriever
)
found_offence_text = "\n\n".join(
    a.text or "" for a in offence_articles if a.found
)

# 3b — Supporting articles (mandatory + offence-specific)
supporting_articles = retrieve_supporting_articles(
    case_text=case_text,
    selected_offence_text=found_offence_text,
    law_retriever=law_retriever,
)

# ── Display offence articles ─────────────────────────────────────────
print("── Retrieved OFFENCE articles ────────────────────────────")
for a in offence_articles:
    status = "✅ found" if a.found else "❌ not found"
    print(f"  {a.signature:12s}  [{a.level or '—':5s}]  {status}")
    if a.found and a.text:
        # Show first 300 chars of the law text
        print(f"    {a.text[:300]}…" if len(a.text) > 300 else f"    {a.text}")
    print()

# ── Display supporting articles (skip mandatory for brevity) ────────
print("── Retrieved SUPPORTING articles (non-mandatory sample) ──")
non_mandatory = [a for a in supporting_articles if a.signature not in MANDATORY_SUPPORTING_DIEU]
display_list = non_mandatory if non_mandatory else supporting_articles[:3]
for a in display_list:
    status = "✅ found" if a.found else "❌ not found"
    print(f"  Điều {a.signature:5s}  [{a.level or '—':5s}]  {status}")
    if a.found and a.text:
        print(f"    {a.text[:300]}…" if len(a.text) > 300 else f"    {a.text}")
    print()

print(f"Total offence articles: {len(offence_articles)}")
print(f"Total supporting articles: {len(supporting_articles)}")
print(f"Mandatory supporting Điều: {list(MANDATORY_SUPPORTING_DIEU)}")

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 4 — Stage 2: Legal analysis (LLM call 2)
# ═══════════════════════════════════════════════════════════════════════
additional_articles = []   # starts empty; may grow if LLM requests more law

system_2, user_2 = _legal_analysis_prompt(
    doc_id=doc_id,
    facts_and_candidates=facts_and_candidates,
    offence_articles=offence_articles,
    additional_articles=additional_articles,
    supporting_articles=supporting_articles,
)

print("📤 Calling LLM — Stage 2 (legal analysis) …")
legal_analysis, usage_2 = _call_llm(
    provider=PROVIDER,
    model_name=MODEL_NAME,
    system_prompt=system_2,
    user_prompt=user_2,
    output_model=ReasonActLegalAnalysis,
    use_provider_fallback=True,
)

# Ensure all mandatory supporting assessments are present
legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
    legal_analysis.supporting_article_assessments,
    supporting_articles,
    case_text=case_text,
)

# ── Handle additional law round (if the LLM requested more articles) ─
requested_sigs = _additional_law_query_signatures(legal_analysis.additional_law_queries)
new_sigs = _filter_new_law_signatures(
    requested_sigs, offence_articles + supporting_articles + additional_articles
)
if new_sigs:
    print(f"  LLM requested additional law: {new_sigs}")
    additional_articles.extend(retrieve_law_articles(new_sigs, law_retriever))
    # Re-run stage 2 with the extra articles
    system_2b, user_2b = _legal_analysis_prompt(
        doc_id=doc_id,
        facts_and_candidates=facts_and_candidates,
        offence_articles=offence_articles,
        additional_articles=additional_articles,
        supporting_articles=supporting_articles,
        additional_law_round=1,
    )
    legal_analysis, usage_2b = _call_llm(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        system_prompt=system_2b,
        user_prompt=user_2b,
        output_model=ReasonActLegalAnalysis,
        use_provider_fallback=True,
    )
    legal_analysis.supporting_article_assessments = ensure_mandatory_supporting_assessments(
        legal_analysis.supporting_article_assessments,
        supporting_articles,
        case_text=case_text,
    )
    print("  ✅ Re-ran Stage 2 with additional law.")
else:
    print("  (No additional law requested by the LLM.)")

print("\n✅ Stage 2 complete.  Structured output:")
print(json.dumps(legal_analysis.model_dump(), ensure_ascii=False, indent=2))

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 5 — Retrieve similar past cases & sentencing calibration
# ═══════════════════════════════════════════════════════════════════════
selected_dieu = legal_analysis.selected_offence.Dieu

# Ensure the selected offence article is fully retrieved
selected_key = _canonical_law_signature(selected_dieu)
_, all_sigs = _existing_law_coverage(offence_articles + additional_articles)
if selected_key and selected_key not in all_sigs:
    offence_articles.extend(retrieve_law_articles([selected_dieu], law_retriever))

# 5a — Similar past cases (by factual profile)
print(f"🔎 Retrieving similar cases for Điều {selected_dieu} …")
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
print(f"  Found {len(similar_cases)} similar cases.")
if similar_cases:
    sc = similar_cases[0]
    print(f"\n── Sample similar case ───────────────────────────────────")
    print(json.dumps(sc.model_dump(), ensure_ascii=False, indent=2))

# 5b — Sentencing calibration cases (by mitigation/aggravation factors)
print(f"\n🔎 Retrieving sentencing calibration cases …")
print(f"  Mitigation factors: {facts_and_candidates.mitigation_factors}")
print(f"  Aggravation factors: {facts_and_candidates.aggravation_factors}")

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
print(f"  Found {len(sentencing_calibration_cases)} calibration cases.")
if sentencing_calibration_cases:
    cc = sentencing_calibration_cases[0]
    print(f"\n── Sample calibration case ──────────────────────────────")
    print(json.dumps(cc.model_dump(), ensure_ascii=False, indent=2))

# %%
# ═══════════════════════════════════════════════════════════════════════
# Cell 6 — Stage 3: Final prediction (LLM call 3) + ground-truth comparison
# ═══════════════════════════════════════════════════════════════════════
system_3, user_3 = _final_prompt(
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

print("📤 Calling LLM — Stage 3 (final verdict prediction) …")
final_output, usage_3 = _call_llm(
    provider=PROVIDER,
    model_name=MODEL_NAME,
    system_prompt=system_3,
    user_prompt=user_3,
    output_model=ReasonActFinalOutput,
    use_provider_fallback=True,
)

print("\n✅ Stage 3 complete.  Final prediction:")
print(json.dumps(final_output.prediction.model_dump(), ensure_ascii=False, indent=2))

# ── Compare with ground truth ─────────────────────────────────────────
print("\n" + "═" * 70)
print("📊 COMPARISON: Prediction vs Ground Truth")
print("═" * 70)

for pred_def in final_output.prediction.defendants:
    name = pred_def.Bi_Cao
    # Find matching GT defendant
    gt_match = None
    for gt in gt_defendants:
        if gt["Bi_Cao"].strip().lower() == name.strip().lower():
            gt_match = gt
            break
    if gt_match is None and len(gt_defendants) == 1:
        gt_match = gt_defendants[0]

    print(f"\n── Bị cáo: {name} ──────────────────────────────────────")
    print(f"  {'':30s} {'PREDICTION':30s} {'GROUND TRUTH':30s}")
    print(f"  {'Tội danh':30s} {(pred_def.Toi_Danh or '—'):30s} {(gt_match or {}).get('Toi_Danh', '—'):30s}")
    print(f"  {'Phạt tù':30s} {(pred_def.Phat_Tu or '—'):30s} {(gt_match or {}).get('Phat_Tu', '—'):30s}")

    pred_clauses = sorted({f"{c.Dieu}-{c.Khoan or ''}" for c in pred_def.Applied_Law_Clauses if c.Dieu})
    gt_clauses = sorted((gt_match or {}).get("Applied_Law_Clauses", []))
    print(f"  {'Điều luật (pred)':30s} {', '.join(pred_clauses)}")
    print(f"  {'Điều luật (GT)':30s} {', '.join(gt_clauses)}")

    if pred_def.Phan_Tich_Phap_Ly:
        print(f"\n  📝 Legal reasoning (excerpt):")
        print(f"    {pred_def.Phan_Tich_Phap_Ly[:500]}")

if final_output.prediction.Xu_Ly_Vat_Chung:
    print(f"\n  Xử lý vật chứng: {final_output.prediction.Xu_Ly_Vat_Chung}")

print("\n" + "═" * 70)
print("Demo complete. All 3 LLM stages executed successfully.")


