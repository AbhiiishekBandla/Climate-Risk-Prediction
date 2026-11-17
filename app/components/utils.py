"""Utility functions, data loaders, and cached resources for Streamlit dashboard."""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from src.models.predict import ClimateRiskPredictor
from src.features.build_features import calculate_heat_index, assign_imd_season

@st.cache_resource(show_spinner="Loading Climate Risk Model Pipeline...")
def get_predictor() -> ClimateRiskPredictor:
    """Returns singleton cached instance of ClimateRiskPredictor."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
    return ClimateRiskPredictor(model_path=model_path)

@st.cache_data(show_spinner="Loading climate dataset...")
def load_processed_data() -> pd.DataFrame:
    """Loads a slice of the processed test dataset for visualization."""
    path = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_metrics_and_comparison() -> Dict[str, Any]:
    """Loads benchmark scores and comparison data."""
    scores_path = os.path.join(PROJECT_ROOT, "metrics", "scores.json")
    comp_path = os.path.join(PROJECT_ROOT, "metrics", "model_comparison.csv")
    
    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
            
    comp_df = pd.DataFrame()
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        
    return {"scores": scores, "comparison": comp_df}

@st.cache_data
def load_monitoring_summaries() -> Dict[str, Any]:
    """Loads all Phase 4 monitoring JSON summaries."""
    mon_dir = os.path.join(PROJECT_ROOT, "reports", "monitoring")
    exp_dir = os.path.join(PROJECT_ROOT, "reports", "explainability")
    
    def read_json_safe(fpath):
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    return {
        "dashboard": read_json_safe(os.path.join(mon_dir, "latest_monitoring_dashboard.json")),
        "data_drift": read_json_safe(os.path.join(mon_dir, "data_drift_summary.json")),
        "target_drift": read_json_safe(os.path.join(mon_dir, "target_drift_summary.json")),
        "performance": read_json_safe(os.path.join(mon_dir, "model_performance_summary.json")),
        "shap": read_json_safe(os.path.join(exp_dir, "shap_summary.json"))
    }

def render_evidently_html(rel_path: str, height: int = 700):
    """Safely renders an Evidently HTML report inside a Streamlit component."""
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=height, scrolling=True)
    else:
        st.warning(f"Report not found at {rel_path}. Run the monitoring pipeline first.")

def get_custom_css() -> str:
    """Returns modern CSS styling for Streamlit application."""
    return """
    <style>
    /* Global Container Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    /* Metric Card */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 0.2rem;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    /* Risk Badges */
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #BBF7D0;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #FECACA;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #2563EB;
        padding-left: 0.6rem;
    }
    </style>
    """
