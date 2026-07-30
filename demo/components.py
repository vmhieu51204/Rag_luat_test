"""Reusable UI widgets and custom CSS components for the Streamlit demo."""

import json
import streamlit as st

def render_styling():
    """Injects custom CSS to style the application with a premium dark-themed aesthetic."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        /* Set App Font and Background */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #080c14 !important;
            color: #e2e8f0;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Titles and Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        /* Main gradient title */
        .app-title {
            background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-bottom: 5px;
        }

        .app-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-top: 0px;
            margin-bottom: 30px;
            font-weight: 400;
        }

        /* Glassmorphism Card Containers */
        .glass-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }

        .glass-card-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 18px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Step Process indicators */
        .step-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            margin-bottom: 30px;
        }

        .step-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            position: relative;
        }

        .step-circle {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #1e293b;
            border: 2px solid #475569;
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 8px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .step-item.active .step-circle {
            background: #3b82f6;
            border-color: #60a5fa;
            color: #ffffff;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }

        .step-item.completed .step-circle {
            background: #10b981;
            border-color: #34d399;
            color: #ffffff;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }

        .step-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .step-item.active .step-label {
            color: #60a5fa;
        }

        .step-item.completed .step-label {
            color: #34d399;
        }

        /* Color Badges for Supporting Laws */
        .badge {
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .badge-applicable {
            background-color: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-dependent {
            background-color: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .badge-not_applicable {
            background-color: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-not_retrieved {
            background-color: rgba(107, 114, 128, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(107, 114, 128, 0.3);
        }

        /* Fact items and lists */
        .fact-item {
            padding: 8px 12px;
            background: rgba(30, 41, 59, 0.4);
            border-left: 3px solid #60a5fa;
            border-radius: 0 8px 8px 0;
            margin-bottom: 8px;
            font-size: 0.92rem;
        }

        .fact-mitigation {
            border-left-color: #10b981;
        }

        .fact-aggravation {
            border-left-color: #ef4444;
        }

        /* Custom buttons styling */
        div.stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }

        div.stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(59, 130, 246, 0.4) !important;
        }
        
        /* Table enhancements */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.95rem;
        }

        .comparison-table th {
            background-color: #0f172a;
            color: #94a3b8;
            font-weight: 700;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }

        .comparison-table td {
            padding: 14px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            vertical-align: top;
        }

        .comparison-table tr:hover {
            background-color: rgba(255, 255, 255, 0.02);
        }
        
        .label-cell {
            font-weight: 600;
            color: #cbd5e1;
            width: 25%;
        }

        .pred-cell {
            color: #60a5fa;
            width: 37.5%;
        }

        .gt-cell {
            color: #34d399;
            width: 37.5%;
        }

        /* Fix text color in input fields */
        .stTextArea textarea, .stTextInput input {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the dashboard header section."""
    st.markdown('<div class="app-title">⚖️ Multistage Legal Reasoning Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">A multi-stage ReAct-style LLM analysis and past-case RAG calibration demo.</div>', unsafe_allow_html=True)

def render_step_progress(current_step: int):
    """Renders a progress tracker detailing the multistage pipeline's state."""
    steps = [
        ("1", "Extract Facts"),
        ("2", "Retrieve Laws"),
        ("3", "Analyze Charge"),
        ("4", "Calibrate"),
        ("5", "Final Verdict")
    ]
    
    html = '<div class="step-container">'
    for idx, (num, label) in enumerate(steps):
        step_idx = idx + 1
        status_class = ""
        if step_idx == current_step:
            status_class = "active"
        elif step_idx < current_step:
            status_class = "completed"
            
        html += f"""<div class="step-item {status_class}">
    <div class="step-circle">{"✓" if step_idx < current_step else num}</div>
    <div class="step-label">{label}</div>
</div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_facts_and_candidates(facts_and_candidates):
    """Displays Stage 1 facts and candidates output inside glass cards."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📊 Stage 1: Extracted Case Profile</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Key Facts")
        facts = facts_and_candidates.facts
        
        # Display defendants
        if hasattr(facts, "defendants") and facts.defendants:
            st.markdown(f"**Defendants:** {', '.join(facts.defendants)}")
        if hasattr(facts, "property_value") and facts.property_value:
            st.markdown(f"**Property Value:** {facts.property_value}")
        if hasattr(facts, "harm") and facts.harm:
            st.markdown(f"**Harm:** {facts.harm}")
            
        # Display general facts
        if hasattr(facts, "conduct") and facts.conduct:
            st.markdown(f"**Criminal Conduct:** {facts.conduct}")
            
    with col2:
        st.subheader("⚖️ Plausible Offence Candidates")
        for idx, candidate in enumerate(facts_and_candidates.candidates):
            title = f"{idx+1}. {candidate.offence_name or 'Unknown Offence'}"
            sections = f"Điều {candidate.Dieu or '?'}"
            if candidate.Khoan:
                sections += f", Khoan {candidate.Khoan}"
            if candidate.Diem:
                sections += f", Điểm {candidate.Diem}"
            st.markdown(f'<div class="fact-item"><strong>{title}</strong><br><small>{sections}</small></div>', unsafe_allow_html=True)
            
    st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🟢 Mitigating Circumstances (Tình tiết giảm nhẹ)")
        if facts_and_candidates.mitigation_factors:
            for factor in facts_and_candidates.mitigation_factors:
                st.markdown(f'<div class="fact-item fact-mitigation">{factor}</div>', unsafe_allow_html=True)
        else:
            st.info("No mitigating circumstances identified.")
            
    with col4:
        st.subheader("🔴 Aggravating Circumstances (Tình tiết tăng nặng)")
        if facts_and_candidates.aggravation_factors:
            for factor in facts_and_candidates.aggravation_factors:
                st.markdown(f'<div class="fact-item fact-aggravation">{factor}</div>', unsafe_allow_html=True)
        else:
            st.info("No aggravating circumstances identified.")
            
    st.markdown('</div>', unsafe_allow_html=True)

def render_retrieved_laws(offence_articles, supporting_articles, additional_articles=None):
    """Displays retrieved law articles."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📚 Retrieved Legal Basis & Articles</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔒 Offence Articles", "⚙️ Supporting Articles", "➕ Additional Articles"])
    
    with tab1:
        st.markdown(f"**Found {len(offence_articles)} potential offence articles:**")
        for article in offence_articles:
            status_text = "Found" if article.found else f"Not Found ({article.missing_reason or 'missing'})"
            with st.expander(f"Điều {article.signature} - [{article.level or '—'}] ({status_text})"):
                if article.found and article.text:
                    st.write(article.text)
                else:
                    st.warning(f"This clause could not be retrieved. Reason: {article.missing_reason}")
                    
    with tab2:
        st.markdown(f"**Retrieved {len(supporting_articles)} supporting general articles (Articles 46-58, etc.):**")
        for article in supporting_articles:
            status_text = "Found" if article.found else "Not Found"
            with st.expander(f"Điều {article.signature} - [{article.level or '—'}] ({status_text})"):
                if article.found and article.text:
                    st.write(article.text)
                else:
                    st.warning("Article text unavailable.")
                    
    with tab3:
        if additional_articles:
            st.markdown(f"**Retrieved {len(additional_articles)} extra requested articles:**")
            for article in additional_articles:
                status_text = "Found" if article.found else "Not Found"
                with st.expander(f"Điều {article.signature} - [{article.level or '—'}] ({status_text})"):
                    if article.found and article.text:
                        st.write(article.text)
                    else:
                        st.warning("Article text unavailable.")
        else:
            st.info("No additional law queries triggered.")
            
    st.markdown('</div>', unsafe_allow_html=True)

def render_legal_analysis(legal_analysis, additional_sigs=None):
    """Displays selected offence and supporting article assessments."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">🔍 Stage 2: Legal Analysis Result</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Primary Offence")
        offence = legal_analysis.selected_offence
        st.markdown(f"**Offence:** {offence.offence_name or 'None Selected'}")
        st.markdown(f"**Section:** Điều {offence.Dieu or '?'}, Khoan {offence.Khoan or '?'}, Điểm {offence.Diem or '—'}")
        st.markdown(f"**Sentencing Bracket:** {legal_analysis.sentencing_bracket or '—'}")
        st.markdown(f"**Confidence Level:** {legal_analysis.confidence or '—'}/5")
        
    with col2:
        st.subheader("Requested Law Updates")
        if additional_sigs:
            st.warning(f"LLM requested additional law query signatures: {additional_sigs}")
            st.info("Pipeline auto-retrieved these articles and re-ran Stage 2 analysis.")
        else:
            st.success("No additional law updates requested by LLM. Current context is sufficient.")
            
    if hasattr(legal_analysis, "rejected_candidates") and legal_analysis.rejected_candidates:
        st.subheader("❌ Rejected or Downgraded Candidates")
        for rejected in legal_analysis.rejected_candidates:
            sections = f"Điều {rejected.Dieu or '?'}"
            if rejected.Khoan: sections += f", Khoan {rejected.Khoan}"
            if rejected.Diem: sections += f", Điểm {rejected.Diem}"
            st.markdown(f"""<div class="fact-item" style="border-left-color: #ef4444; background: rgba(30, 41, 59, 0.2); margin-bottom: 8px;">
    <div style="font-weight: 700; font-size: 1.05rem; color: #f1f5f9;">{rejected.offence_name or 'Unknown Offence'} <span style="font-size: 0.9rem; color: #94a3b8; font-weight: normal;">({sections})</span></div>
    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 4px;"><strong>Reason:</strong> {rejected.rejection_or_downgrade_reason or 'None provided'}</div>
</div>""", unsafe_allow_html=True)

    st.subheader("📝 Supporting Article Assessments")
    for item in legal_analysis.supporting_article_assessments:
        status = item.status or "not_retrieved"
        badge_class = f"badge-{status}"
        
        # Format display name
        title = f"Điều {item.article}"
        trigger_text = f"**Trigger:** {item.factual_trigger}" if item.factual_trigger else "*No factual trigger activated*"
        
        st.markdown(f"""<div class="fact-item" style="border-left-color: var(--light-blue); background: rgba(30, 41, 59, 0.2); margin-bottom: 12px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
        <span style="font-weight: 700; font-size: 1.05rem; color: #f1f5f9;">{title}</span>
        <span class="badge {badge_class}">{status.replace('_', ' ')}</span>
    </div>
    <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 4px;">{trigger_text}</div>
    <div style="font-size: 0.9rem; color: #cbd5e1;">{item.explanation or ''}</div>
</div>""", unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

def render_retrieved_cases(similar_cases, calibration_cases):
    """Displays similar cases and calibration cases in a clean visual layout."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">🔎 RAG: Case Database Retrieval (Analogies & Calibration)</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🏘️ Similar Cases (Fact Match)", "⚖️ Calibration Cases (Mitigation/Aggravation Match)"])
    
    with tab1:
        st.markdown(f"**Found {len(similar_cases)} cases matching the factual profile and selected offence:**")
        for idx, case in enumerate(similar_cases):
            with st.expander(f"Case {case.doc_id} (Điều {case.matched_offence_article}) - Sentence: {case.sentence or 'Unknown'}"):
                st.markdown(f"**Factual Profile Excerpt:**")
                st.write(case.matched_factual_profile)
                if case.mitigation_aggravation_profile:
                    st.markdown(f"**Mitigation/Aggravation Profile:**")
                    st.write(case.mitigation_aggravation_profile)
                if case.notable_reasoning:
                    st.markdown(f"**Notable Court Reasoning:**")
                    st.write(case.notable_reasoning)
                    
    with tab2:
        st.markdown(f"**Found {len(calibration_cases)} cases matching specific mitigation/aggravation factors:**")
        for idx, case in enumerate(calibration_cases):
            title = f"{case.factor_type.capitalize()} Match: '{case.query_factor}' -> Case {case.doc_id} ({case.defendant_name or 'Defendant'})"
            with st.expander(title):
                st.markdown(f"**Similarity Score:** `{case.similarity_score:.4f}`")
                st.markdown(f"**Court Sentence Applied:** `{case.court_sentence or 'None'}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Court Mitigating Factors:**")
                    st.write(case.court_mitigation or "*None stated*")
                with col2:
                    st.markdown("**Court Aggravating Factors:**")
                    st.write(case.court_aggravation or "*None stated*")
                    
                if case.prosecution_proposal:
                    st.markdown("**Prosecution Proposal:**")
                    st.json(case.prosecution_proposal)
                    
    st.markdown('</div>', unsafe_allow_html=True)

def render_final_prediction_and_comparison(prediction, ground_truth=None):
    """Displays predicted verdict and compares it with ground truth if available."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📊 Stage 3: Predicted Verdict & Comparison</div>', unsafe_allow_html=True)
    
    # Display physical evidence handling if present
    if prediction.Xu_Ly_Vat_Chung:
        st.subheader("📦 Handling of Physical Evidence (Xử lý vật chứng)")
        st.info(prediction.Xu_Ly_Vat_Chung)
        
    st.markdown('</div>', unsafe_allow_html=True)
        
    for pred_def in prediction.defendants:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        name = pred_def.Bi_Cao
        st.markdown(f"### 👤 Defendant: {name}")
        
        # Match ground truth
        gt_match = None
        if ground_truth:
            for gt in ground_truth:
                if gt.get("Bi_Cao", "").strip().lower() == name.strip().lower():
                    gt_match = gt
                    break
            # Fallback if single defendant
            if gt_match is None and len(ground_truth) == 1:
                gt_match = ground_truth[0]
                
        # Comparison Table
        pred_clauses = ", ".join(sorted({f"{c.Dieu}-{c.Khoan or ''}" for c in pred_def.Applied_Law_Clauses if c.Dieu}))
        
        gt_clauses = ""
        gt_crime = "—"
        gt_sentence = "—"
        if gt_match:
            gt_clauses = ", ".join(sorted(gt_match.get("Applied_Law_Clauses", [])))
            gt_crime = gt_match.get("Toi_Danh", "—")
            gt_sentence = gt_match.get("Phat_Tu", "—")
            
        html_table = f"""<table class="comparison-table">
    <thead>
        <tr>
            <th>Parameter</th>
            <th style="color: #60a5fa;">Model Prediction</th>
            <th style="color: #34d399;">Ground Truth Verdict</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="label-cell">Crime / Charge (Tội danh)</td>
            <td class="pred-cell">{pred_def.Toi_Danh or '—'}</td>
            <td class="gt-cell">{gt_crime}</td>
        </tr>
        <tr>
            <td class="label-cell">Prison Term (Phạt tù)</td>
            <td class="pred-cell">{pred_def.Phat_Tu or '—'}</td>
            <td class="gt-cell">{gt_sentence}</td>
        </tr>
        <tr>
            <td class="label-cell">Applied Laws (Điều luật áp dụng)</td>
            <td class="pred-cell">{pred_clauses or '—'}</td>
            <td class="gt-cell">{gt_clauses or '—'}</td>
        </tr>
    </tbody>
</table>"""
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Legal reasoning
        if pred_def.Phan_Tich_Phap_Ly:
            st.markdown("**📝 Predicted Legal Analysis (Phân tích pháp lý):**")
            st.write(pred_def.Phan_Tich_Phap_Ly)
            
        if hasattr(pred_def, "Tu_Van_Giam_Nhe") and pred_def.Tu_Van_Giam_Nhe:
            st.markdown("**🟢 Recommended Mitigation Steps (Tư vấn giảm nhẹ):**")
            st.success(pred_def.Tu_Van_Giam_Nhe)
            
        st.markdown('</div>', unsafe_allow_html=True)
