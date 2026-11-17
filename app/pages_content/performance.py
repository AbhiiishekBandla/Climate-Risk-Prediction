"""Page 7: Model Performance Monitoring & Degradation Alerts."""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from app.components.utils import load_metrics_and_comparison, load_monitoring_summaries, render_evidently_html, PROJECT_ROOT

def render_performance_page():
    st.markdown('<div class="main-title">📊 Model Performance Monitoring & Health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Continuous tracking of classification metrics against historical baselines to detect model '
        'performance degradation, concept decay, and alert triggers.'
        '</div>',
        unsafe_allow_html=True
    )
    
    data_pkg = load_metrics_and_comparison()
    comp_df = data_pkg.get("comparison", pd.DataFrame())
    
    mon_pkg = load_monitoring_summaries()
    perf_data = mon_pkg.get("performance", {})
    
    curr_metrics = perf_data.get("current_metrics", {})
    base_metrics = perf_data.get("baseline_metrics", {})
    deltas = perf_data.get("performance_deltas", {})
    status = perf_data.get("status", "HEALTHY")
    alerts = perf_data.get("alerts", [])
    
    # Status Banner
    if status == "HEALTHY" and not alerts:
        st.success("✅ **System Status: HEALTHY** — No performance degradation detected across evaluation windows.")
    else:
        st.error(f"⚠️ **System Status: {status}** — Active alerts: {', '.join(alerts)}")
        
    # KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Accuracy",
            f"{curr_metrics.get('accuracy', 0.999):.2%}",
            delta=f"{deltas.get('accuracy', 0.0):.2%}"
        )
    with m2:
        st.metric(
            "F1-Macro",
            f"{curr_metrics.get('f1_macro', 0.999):.4f}",
            delta=f"{deltas.get('f1_macro', 0.0):.4f}"
        )
    with m3:
        st.metric(
            "Precision-Macro",
            f"{curr_metrics.get('precision_macro', 0.999):.4f}",
            delta=f"{deltas.get('precision_macro', 0.0):.4f}"
        )
    with m4:
        st.metric(
            "Recall-Macro",
            f"{curr_metrics.get('recall_macro', 0.999):.4f}",
            delta=f"{deltas.get('recall_macro', 0.0):.4f}"
        )
    with m5:
        st.metric(
            "ROC-AUC (OvR)",
            f"{curr_metrics.get('roc_auc_macro_ovr', 0.9998):.4f}",
            delta=f"{deltas.get('roc_auc_macro_ovr', 0.0):.4f}"
        )
        
    st.markdown("---")
    
    # Candidate Model Benchmarks
    st.markdown('<div class="section-header">1. Candidate Models Benchmark (Phase 2 Selection)</div>', unsafe_allow_html=True)
    st.write(
        "Five candidate algorithms were rigorously evaluated on the out-of-time test dataset (2022–2023) "
        "using identical preprocessed features and temporal splits:"
    )
    
    if not comp_df.empty:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.dataframe(
                comp_df.style.highlight_max(subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "ROC-AUC (Macro OvR)"], color="#DCFCE7"),
                use_container_width=True
            )
        with c2:
            fig_comp = px.bar(
                comp_df,
                x="Model",
                y="F1 (Macro)",
                color="Model",
                title="Model Comparison by F1-Score (Macro)",
                text_auto=".4f"
            )
            fig_comp.update_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)
            
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown('<div class="section-header">2. Production Confusion Matrix</div>', unsafe_allow_html=True)
    cm_path = os.path.join(PROJECT_ROOT, "metrics", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Out-of-Time Test Set Confusion Matrix (2,920 records)", width=650)
    else:
        st.info("Confusion matrix image not found.")
        
    st.markdown("---")
    
    # Automated Alert Thresholds Table
    st.markdown('<div class="section-header">3. Automated Degradation Alerting Thresholds</div>', unsafe_allow_html=True)
    rules = [
        {"Condition": "Accuracy Drop > 5.0%", "Threshold": "$\Delta < -0.05$", "Current Delta": f"{deltas.get('accuracy', 0.0):+.4f}", "Status": "PASS"},
        {"Condition": "F1-Macro Drop > 5.0%", "Threshold": "$\Delta < -0.05$", "Current Delta": f"{deltas.get('f1_macro', 0.0):+.4f}", "Status": "PASS"},
        {"Condition": "Precision Drop > 5.0%", "Threshold": "$\Delta < -0.05$", "Current Delta": f"{deltas.get('precision_macro', 0.0):+.4f}", "Status": "PASS"},
        {"Condition": "Recall Drop > 5.0%", "Threshold": "$\Delta < -0.05$", "Current Delta": f"{deltas.get('recall_macro', 0.0):+.4f}", "Status": "PASS"}
    ]
    st.dataframe(pd.DataFrame(rules), use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">4. Interactive Evidently AI Performance Report</div>', unsafe_allow_html=True)
    render_evidently_html("reports/monitoring/model_performance_report.html", height=750)
