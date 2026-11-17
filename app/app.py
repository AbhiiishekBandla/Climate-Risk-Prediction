"""Main Streamlit Application Entrypoint for Climate Risk Prediction System.

Provides an enterprise-grade multi-page dashboard featuring real-time prediction,
visual analytics, Evidently AI drift monitoring, SHAP explainability, and MLOps metrics.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.utils import get_custom_css
from app.pages_content.home import render_home_page
from app.pages_content.prediction import render_prediction_page
from app.pages_content.visualization import render_visualization_page
from app.pages_content.data_drift import render_data_drift_page
from app.pages_content.target_drift import render_target_drift_page
from app.pages_content.explainability import render_explainability_page
from app.pages_content.performance import render_performance_page
from app.pages_content.project_info import render_project_info_page

def main():
    st.set_page_config(
        page_title="Climate Risk Prediction System | MLOps Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject Custom Styling
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Sidebar Branding & Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 12px 0 16px 0;">
            <div style="font-size: 2.2rem;">🌍</div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-top: 4px;">
                Climate Risk MLOps
            </div>
            <div style="font-size: 0.8rem; color: #64748B; font-weight: 500;">
                Indian Metropolitan System
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Active Model</div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #166534;">Gradient Boosting (99.9%)</div>
            <div style="font-size: 0.75rem; color: #16A34A;">● DVC & MLflow Logged</div>
        </div>
        """, unsafe_allow_html=True)
        
        selected_page = st.radio(
            "Navigation Menu:",
            [
                "🏠 Home / Overview",
                "🔮 Climate Risk Prediction",
                "📈 Data Visualization & EDA",
                "📡 Data Drift Monitoring",
                "🎯 Target Drift Monitoring",
                "🧠 SHAP Explainability",
                "📊 Model Performance & Alerts",
                "ℹ️ Project Information"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.4;">
            <strong>System:</strong> v1.0.0<br/>
            <strong>Dataset:</strong> ERA5 / Open-Meteo<br/>
            <strong>Storage:</strong> DVC + DagsHub<br/>
            <strong>Backend:</strong> Python 3.11
        </div>
        """, unsafe_allow_html=True)

    # Route to Selected Page
    if selected_page == "🏠 Home / Overview":
        render_home_page()
    elif selected_page == "🔮 Climate Risk Prediction":
        render_prediction_page()
    elif selected_page == "📈 Data Visualization & EDA":
        render_visualization_page()
    elif selected_page == "📡 Data Drift Monitoring":
        render_data_drift_page()
    elif selected_page == "🎯 Target Drift Monitoring":
        render_target_drift_page()
    elif selected_page == "🧠 SHAP Explainability":
        render_explainability_page()
    elif selected_page == "📊 Model Performance & Alerts":
        render_performance_page()
    elif selected_page == "ℹ️ Project Information":
        render_project_info_page()

if __name__ == "__main__":
    main()
