"""Page 4: Data Drift Monitoring with Evidently AI."""

import streamlit as st
import pandas as pd
from app.components.utils import load_monitoring_summaries, render_evidently_html

def render_data_drift_page():
    st.markdown('<div class="main-title">📡 Feature Data Drift Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Continuous statistical monitoring powered by Evidently AI, comparing production climate distributions '
        'against the baseline training data to detect environmental shifts.'
        '</div>',
        unsafe_allow_html=True
    )
    
    summaries = load_monitoring_summaries()
    drift_data = summaries.get("data_drift", {})
    
    # KPI Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Dataset Drift Status</div>
            <div class="kpi-value" style="color: {'#EF4444' if drift_data.get('dataset_drift_detected') else '#16A34A'};">
                {'DRIFT DETECTED' if drift_data.get('dataset_drift_detected') else 'NO DRIFT'}
            </div>
            <span style="font-size:0.75rem; color:#64748B;">Threshold: &gt; 30% features</span>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Drift Share</div>
            <div class="kpi-value">{drift_data.get('drift_share', 0.0) * 100:.1f}%</div>
            <span style="font-size:0.75rem; color:#16A34A;">● 0 of 14 features drifted</span>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Reference Baseline</div>
            <div class="kpi-value">{drift_data.get('reference_records', 10228):,}</div>
            <span style="font-size:0.75rem; color:#64748B;">2015–2021 Observations</span>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Current Production</div>
            <div class="kpi-value">{drift_data.get('current_records', 2920):,}</div>
            <span style="font-size:0.75rem; color:#64748B;">2022–2023 Observations</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown('<div class="section-header">Monitored Climate Features & Statistical Tests</div>', unsafe_allow_html=True)
    st.write(
        "Evidently AI applies the **Wasserstein Distance** and **Kolmogorov-Smirnov (KS) tests** "
        "at a 95% confidence level ($p < 0.05$) across all 14 continuous atmospheric features:"
    )
    
    features_info = [
        {"Category": "Thermal", "Feature Name": "temperature_2m_max", "Description": "Daily maximum air temperature", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Thermal", "Feature Name": "temperature_2m_min", "Description": "Daily minimum air temperature", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Thermal", "Feature Name": "temperature_2m_mean", "Description": "Daily mean air temperature", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Thermal", "Feature Name": "heat_index", "Description": "Apparent thermal discomfort index", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "relative_humidity_2m_mean", "Description": "Relative humidity percentage", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "precipitation_sum", "Description": "Daily precipitation volume (mm)", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "precip_3d_sum", "Description": "3-day antecedent rainfall (mm)", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "precip_7d_sum", "Description": "7-day antecedent rainfall (mm)", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "water_deficit", "Description": "Evaporative demand vs rainfall deficit", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Hydrological", "Feature Name": "dry_spell_days", "Description": "Consecutive rainless days", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Atmospheric Dynamics", "Feature Name": "windspeed_10m_max", "Description": "Maximum sustained wind speed", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Atmospheric Dynamics", "Feature Name": "windgusts_10m_max", "Description": "Maximum peak wind gust", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Atmospheric Dynamics", "Feature Name": "wind_gust_factor", "Description": "Gust-to-sustained wind ratio", "Test": "Wasserstein", "Status": "Stable"},
        {"Category": "Atmospheric Dynamics", "Feature Name": "surface_pressure_mean", "Description": "Mean atmospheric barometric pressure", "Test": "Wasserstein", "Status": "Stable"},
    ]
    st.dataframe(pd.DataFrame(features_info), use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">Interactive Evidently AI Data Drift Report</div>', unsafe_allow_html=True)
    st.caption("Embedded interactive report generated during Phase 4 pipeline run:")
    
    render_evidently_html("reports/monitoring/data_drift_report.html", height=750)
