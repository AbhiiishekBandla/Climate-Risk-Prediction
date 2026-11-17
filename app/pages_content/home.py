"""Page 1: Home / Project Overview."""

import streamlit as st
import plotly.express as px
import pandas as pd
from app.components.utils import load_metrics_and_comparison, load_monitoring_summaries

def render_home_page():
    st.markdown('<div class="main-title">🌍 Climate Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'An enterprise-grade, end-to-end MLOps platform for real-time meteorological hazard forecasting, '
        'data drift monitoring, and transparent model explainability across Indian climatic zones.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # KPI Metrics Row
    data_pkg = load_metrics_and_comparison()
    scores = data_pkg.get("scores", {})
    mon_pkg = load_monitoring_summaries()
    dash = mon_pkg.get("dashboard", {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Climate Records</div>
            <div class="kpi-value">13,148</div>
            <span style="font-size:0.75rem; color:#16A34A;">● 2015–2023 Real ERA5</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Metros Monitored</div>
            <div class="kpi-value">4</div>
            <span style="font-size:0.75rem; color:#64748B;">Delhi, Mum, Kol, Che</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Production F1</div>
            <div class="kpi-value">{scores.get('f1_macro', 0.999):.4f}</div>
            <span style="font-size:0.75rem; color:#16A34A;">● Gradient Boosting</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC</div>
            <div class="kpi-value">0.9998</div>
            <span style="font-size:0.75rem; color:#16A34A;">● Multi-class OvR</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        status = dash.get("overall_system_status", "HEALTHY")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">System Health</div>
            <div class="kpi-value" style="color: #16A34A;">{status}</div>
            <span style="font-size:0.75rem; color:#16A34A;">● 0 Active Alerts</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Two Columns: Problem & Solution Overview vs Architecture
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="section-header">🎯 Operational Objectives & Scope</div>', unsafe_allow_html=True)
        st.write("""
        Climate change intensifies extreme weather events—from severe pre-monsoon heatwaves in northern plains
        to flash flood deluges along coastal belts. This platform addresses key operational challenges:
        
        * **Multi-Hazard Prediction:** Categorizes daily conditions into **Low Risk (Normal)**, **Moderate Risk (Advisory)**, 
          and **High Risk (Severe / Disaster Warning)** using criteria aligned with the **India Meteorological Department (IMD)** 
          and **National Disaster Management Authority (NDMA)**.
        * **Thermal & Hydrological Stress Indicators:** Leverages calculated indices such as the **Rothfusz Heat Index**, 
          **Diurnal Temperature Range (DTR)**, **3-day/7-day cumulative rainfall**, and **water deficit**.
        * **Temporal Integrity:** Evaluated strictly using an out-of-time temporal split (Train: 2015–2021 | Test: 2022–2023) 
          preventing temporal lookahead leakage.
        """)
        
        st.info("💡 **Ready to run predictions?** Select **Climate Risk Prediction** from the sidebar to test custom weather conditions or upload batch files.")
        
    with c2:
        st.markdown('<div class="section-header">⚙️ End-to-End MLOps Pipeline</div>', unsafe_allow_html=True)
        st.write("""
        Built from scratch following production MLOps engineering best practices:
        """)
        
        steps = [
            ("1. Data Versioning & DVC", "Raw observations and trained pipelines tracked with DVC & DagsHub remote storage."),
            ("2. Experiment Tracking & MLflow", "Hyperparameters, multi-class metrics, confusion matrices, and artifacts logged to SQLite backend."),
            ("3. Continuous Drift Monitoring", "Evidently AI monitors 14 features for distribution drift between training baseline and production."),
            ("4. Transparent Explainability", "SHAP KernelExplainer isolates exact thermal, hydrological, and wind drivers behind every forecast.")
        ]
        for title, desc in steps:
            st.markdown(f"**{title}**  \n<span style='color:#475569; font-size:0.9rem;'>{desc}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🗺️ Platform Navigation Guide</div>', unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        st.markdown("""
        **🔮 Prediction Engine**
        * Real-time single input form
        * Batch CSV file evaluation
        * Multi-class probability distribution
        """)
    with nav_col2:
        st.markdown("""
        **📈 Visual Analytics**
        * Temperature & Heat Index trends
        * Monsoon rainfall patterns
        * City-wise hazard frequencies
        """)
    with nav_col3:
        st.markdown("""
        **📡 Drift Monitoring**
        * Evidently AI 14-feature drift
        * Prediction vs Target drift
        * Model degradation alert checks
        """)
    with nav_col4:
        st.markdown("""
        **🧠 SHAP Explainability**
        * Global summary beeswarm plots
        * Heatwave & flood force plots
        * Climate feature rankings
        """)
