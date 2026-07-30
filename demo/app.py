"""Main Streamlit application for the Multistage Legal Reasoning Demo."""

import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Set page config first before any other streamlit commands
st.set_page_config(
    page_title="Multistage Legal Reasoning Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables
load_dotenv(REPO_ROOT / ".env")

import os
import json
from llm import get_available_providers, get_default_model, configure_api_keys
from retrieval import get_law_retriever, get_train_articles_index, get_case_runtime
from utils import load_test_cases, load_case_data, extract_defendants_and_summary, get_ground_truth
from pipeline import run_stage_1, run_law_retrieval, run_stage_2, run_case_retrieval, run_stage_3
from components import (
    render_styling,
    render_header,
    render_step_progress,
    render_facts_and_candidates,
    render_retrieved_laws,
    render_legal_analysis,
    render_retrieved_cases,
    render_final_prediction_and_comparison
)

# 1. Custom styling
render_styling()

# 2. Sidebar Configuration
st.sidebar.markdown("### ⚙️ Engine Settings")

# Provider and model selector
providers = get_available_providers()
provider = st.sidebar.selectbox("LLM Provider", providers, index=0)
default_model = get_default_model(provider)
model_name = st.sidebar.text_input("LLM Model Name", value=default_model)

# API Key inputs with fallback checks
env_key_name = {
    "aistudio": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY"
}[provider]

has_env_key = os.environ.get(env_key_name) is not None
key_placeholder = "Key detected in .env" if has_env_key else f"Enter {provider.upper()} API Key"
api_key = st.sidebar.text_input(f"{provider.upper()} API Key", type="password", placeholder=key_placeholder)

# Case loading mechanism
st.sidebar.markdown("### 📁 Test Case Loader")
test_cases = load_test_cases()

def handle_case_load():
    """Handles case loading from dropdown and clears previous session state predictions."""
    filename = st.session_state.case_selector
    if filename != "-- Select a test case --":
        case_data = load_case_data(filename)
        payload = extract_defendants_and_summary(case_data)
        st.session_state.def_text = payload["defendant_info"]
        st.session_state.sum_text = payload["synthetic_summary_2"]
        st.session_state.defendant_input = payload["defendant_info"]
        st.session_state.summary_input = payload["synthetic_summary_2"]
        st.session_state.ground_truth = get_ground_truth(case_data)
        st.session_state.doc_id = case_data.get("THONG_TIN_CHUNG", {}).get("Ma_Ban_An", filename.replace(".json", ""))
        
        # Reset previous run results
        for k in ["stage1_result", "law_retrieved", "stage2_result", "rag_retrieved", "stage3_result", "pipeline_run", "current_step"]:
            if k in st.session_state:
                del st.session_state[k]

case_selector = st.sidebar.selectbox(
    "Load Sample Case File",
    ["-- Select a test case --"] + test_cases,
    key="case_selector",
    on_change=handle_case_load
)

# 3. Main Page Layout
render_header()

# Text Area Inputs
st.markdown("### 📝 Case Input Data")
col_input_1, col_input_2 = st.columns(2)

with col_input_1:
    defendant_input_text = st.text_area(
        "Defendant Information (THONG_TIN_CHUNG.Thong_Tin_Bi_Cao)",
        value=st.session_state.get("defendant_input", ""),
        placeholder="Paste defendant information JSON array or text profile here...",
        height=200,
        key="def_text"
    )
    st.session_state.defendant_input = defendant_input_text

with col_input_2:
    summary_input_text = st.text_area(
        "Synthetic Case Summary (Synthetic_summary_2)",
        value=st.session_state.get("summary_input", ""),
        placeholder="Paste the defendant's synthetic story or summary here...",
        height=200,
        key="sum_text"
    )
    st.session_state.summary_input = summary_input_text

# Keep track of custom doc_id if not loaded from file
if "doc_id" not in st.session_state:
    st.session_state.doc_id = "custom_demo_case"

# Trigger run button
run_pipeline = st.button("🚀 Run Multistage Analysis")

# Check validation and trigger pipeline execution
if run_pipeline:
    # Key validation
    if not api_key and not has_env_key:
        st.error(f"Please provide an API Key for {provider.upper()} in the sidebar or configured in .env.")
    elif not st.session_state.defendant_input.strip() or not st.session_state.summary_input.strip():
        st.error("Please enter/paste both Defendant Information and Case Summary before running.")
    else:
        # Load keys into environment variables
        if api_key:
            configure_api_keys(api_key, provider)
            
        # Run Step Progress visual
        st.session_state.pipeline_run = True
        
        # Load heavy dependencies
        law_retriever = get_law_retriever()
        train_articles_index = get_train_articles_index()
        case_runtime = get_case_runtime()
        
        # Step 1 execution
        st.session_state.current_step = 1
        progress_bar = st.progress(0, text="Initializing Stage 1: Facts & Candidate Extraction...")
        
        try:
            facts_and_candidates, case_payload, usage_1 = run_stage_1(
                doc_id=st.session_state.doc_id,
                defendant_info=st.session_state.defendant_input,
                synthetic_summary_2=st.session_state.summary_input,
                provider=provider,
                model_name=model_name
            )
            st.session_state.stage1_result = facts_and_candidates
            st.session_state.case_payload = case_payload
            
            # Step 2 execution
            st.session_state.current_step = 2
            progress_bar.progress(25, text="Stage 1 complete. Retrieving Law Articles from Database...")
            offence_articles, supporting_articles = run_law_retrieval(
                facts_and_candidates=facts_and_candidates,
                case_payload=case_payload,
                law_retriever=law_retriever
            )
            st.session_state.law_retrieved = {
                "offence_articles": offence_articles,
                "supporting_articles": supporting_articles
            }
            
            # Step 3 execution
            st.session_state.current_step = 3
            progress_bar.progress(50, text="Law articles fetched. Starting Stage 2: Charge & Legal Analysis...")
            legal_analysis, additional_articles, re_ran, new_sigs, usage_2 = run_stage_2(
                doc_id=st.session_state.doc_id,
                case_payload=case_payload,
                facts_and_candidates=facts_and_candidates,
                offence_articles=offence_articles,
                supporting_articles=supporting_articles,
                law_retriever=law_retriever,
                provider=provider,
                model_name=model_name
            )
            st.session_state.stage2_result = {
                "legal_analysis": legal_analysis,
                "additional_articles": additional_articles,
                "re_ran": re_ran,
                "new_sigs": new_sigs
            }
            
            # Step 4 execution
            st.session_state.current_step = 4
            progress_bar.progress(75, text="Stage 2 complete. Querying Vector Database for Past Case RAG...")
            similar_cases, calibration_cases = run_case_retrieval(
                case_payload=case_payload,
                legal_analysis=legal_analysis,
                facts_and_candidates=facts_and_candidates,
                offence_articles=offence_articles,
                additional_articles=additional_articles,
                law_retriever=law_retriever,
                case_runtime=case_runtime,
                train_articles_index=train_articles_index,
                doc_id=st.session_state.doc_id
            )
            st.session_state.rag_retrieved = {
                "similar_cases": similar_cases,
                "calibration_cases": calibration_cases
            }
            
            # Step 5 execution
            st.session_state.current_step = 5
            progress_bar.progress(90, text="Retrieval completed. Running Stage 3: Verdict Prediction & Calibration...")
            final_output, usage_3 = run_stage_3(
                doc_id=st.session_state.doc_id,
                case_payload=case_payload,
                facts_and_candidates=facts_and_candidates,
                legal_analysis=legal_analysis,
                offence_articles=offence_articles,
                additional_articles=additional_articles,
                supporting_articles=supporting_articles,
                similar_cases=similar_cases,
                sentencing_calibration_cases=calibration_cases,
                provider=provider,
                model_name=model_name
            )
            st.session_state.stage3_result = final_output
            
            # Finish
            progress_bar.progress(100, text="Analysis Complete!")
            st.success("All pipeline stages executed successfully!")
            
        except Exception as e:
            st.error(f"Pipeline Execution Failed: {str(e)}")
            st.exception(e)

# Render Step Indicator based on state
if st.session_state.get("pipeline_run"):
    render_step_progress(st.session_state.get("current_step", 5))

# Render outputs if available in session state
if "stage3_result" in st.session_state:
    st.markdown("### 🏁 Pipeline Results")
    
    # Create tabbed visual layout
    tab_pred, tab_stage1, tab_laws, tab_stage2, tab_rag = st.tabs([
        "🏆 Final Verdict & Comparison",
        "Stage 1: Facts Profile",
        "Stage 2: Laws",
        "Stage 2: Legal Analysis",
        "RAG: Analogies & Calibration"
    ])
    
    with tab_pred:
        render_final_prediction_and_comparison(
            st.session_state.stage3_result.prediction,
            st.session_state.get("ground_truth")
        )
        
    with tab_stage1:
        render_facts_and_candidates(st.session_state.stage1_result)
        
    with tab_laws:
        render_retrieved_laws(
            st.session_state.law_retrieved["offence_articles"],
            st.session_state.law_retrieved["supporting_articles"],
            st.session_state.stage2_result["additional_articles"]
        )
        
    with tab_stage2:
        render_legal_analysis(
            st.session_state.stage2_result["legal_analysis"],
            st.session_state.stage2_result["new_sigs"]
        )
        
    with tab_rag:
        render_retrieved_cases(
            st.session_state.rag_retrieved["similar_cases"],
            st.session_state.rag_retrieved["calibration_cases"]
        )
