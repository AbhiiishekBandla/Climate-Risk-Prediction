"""Page 6: SHAP Model Explainability."""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from app.components.utils import load_monitoring_summaries, PROJECT_ROOT

def render_explainability_page():
    st.markdown('<div class="main-title">🧠 SHAP Model Explainability & Interpretability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Transparent decision attribution powered by SHAP (SHapley Additive exPlanations), '
        'quantifying how each meteorological feature influences multi-class risk predictions.'
        '</div>',
        unsafe_allow_html=True
    )
    
    summaries = load_monitoring_summaries()
    shap_data = summaries.get("shap", {})
    
    st.markdown("""
    The system utilizes a model-agnostic **`shap.KernelExplainer`** with $k$-Means ($k=5$) background clustering,
    specifically adapted to support multi-class Gradient Boosting pipelines with zero bias.
    """)
    
    # Global Feature Importance
    st.markdown('<div class="section-header">1. Global Climate Risk Drivers</div>', unsafe_allow_html=True)
    
    top_drivers = shap_data.get("top_climate_risk_drivers", [])
    if top_drivers:
        driver_df = pd.DataFrame(top_drivers)
        driver_df.columns = ["Climate Feature", "Mean |SHAP Value|"]
        
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig_bar = px.bar(
                driver_df.head(10).iloc[::-1],
                x="Mean |SHAP Value|",
                y="Climate Feature",
                orientation="h",
                color="Mean |SHAP Value|",
                color_continuous_scale="Viridis",
                title="Top 10 Atmospheric Drivers of Climate Risk"
            )
            fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.markdown("**Key Meteorological Insights from SHAP Analysis:**")
            for insight in shap_data.get("key_insights", []):
                st.markdown(f"🔹 {insight}")
                
            st.write("""
            * **Heat Index Dominance:** `heat_index` is the single strongest predictor of extreme thermal risk, 
              capturing the synergistic danger of simultaneous high ambient heat and high humidity.
            * **Severe Convection:** `windgusts_10m_max` acts as the primary differentiator for squalls, dust storms, and cyclonic systems.
            * **Hydrological Inundation:** `precip_3d_sum` (3-day cumulative precipitation) outweighs single-day rain in predicting 
              prolonged urban waterlogging and flash floods.
            """)
            
    st.markdown("---")
    st.markdown('<div class="section-header">2. Global Distribution Visualizations</div>', unsafe_allow_html=True)
    
    tab_beeswarm, tab_bar = st.tabs(["🐝 Beeswarm Feature Impact Summary", "📊 Global Feature Importance Chart"])
    
    exp_dir = os.path.join(PROJECT_ROOT, "reports", "explainability")
    
    with tab_beeswarm:
        beeswarm_path = os.path.join(exp_dir, "shap_summary_beeswarm.png")
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, caption="SHAP Summary Beeswarm Plot across Climate Hazard Classes", use_container_width=True)
        else:
            st.info("Beeswarm plot not found. Run Phase 4 pipeline to generate.")
            
    with tab_bar:
        bar_path = os.path.join(exp_dir, "shap_feature_importance_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="Mean Absolute SHAP Feature Importance Ranking", use_container_width=True)
        else:
            st.info("Feature importance plot not found.")
            
    st.markdown("---")
    st.markdown('<div class="section-header">3. Localized Extreme Weather Case Studies</div>', unsafe_allow_html=True)
    st.write("SHAP explains why the model triggered specific warnings for concrete real-world events:")
    
    case_tab1, case_tab2, case_tab3 = st.tabs([
        "☀️ Case 1: Pre-Monsoon Heatwave (Delhi)",
        "🌧️ Case 2: Extreme Monsoon Deluge (Mumbai)",
        "⛅ Case 3: Benign Winter Day (Kolkata)"
    ])
    
    with case_tab1:
        st.write("**Scenario:** New Delhi in late May with ambient temperature $44.5^\\circ\\text{C}$ and Heat Index $> 46^\\circ\\text{C}$.")
        hw_path = os.path.join(exp_dir, "shap_local_heatwave_explanation.png")
        if os.path.exists(hw_path):
            st.image(hw_path, caption="SHAP Decision Force for Extreme Heatwave Prediction", use_container_width=True)
        st.success("Decision Attribution: Model decisively pushed to High Risk (Level 2) primarily driven by `heat_index` and `temperature_2m_max`.")
        
    with case_tab2:
        st.write("**Scenario:** Mumbai in mid-July with $125\\text{ mm}$ daily deluge and $240\\text{ mm}$ 3-day accumulated rainfall.")
        fl_path = os.path.join(exp_dir, "shap_local_flood_explanation.png")
        if os.path.exists(fl_path):
            st.image(fl_path, caption="SHAP Decision Force for Extreme Flood Prediction", use_container_width=True)
        st.success("Decision Attribution: Model classified as High Risk (Level 2) dominated by `precip_3d_sum` and `precipitation_sum`.")
        
    with case_tab3:
        st.write("**Scenario:** Kolkata in January with mild $24.5^\\circ\\text{C}$ temperature, zero precipitation, and light breeze.")
        norm_path = os.path.join(exp_dir, "shap_local_normal_explanation.png")
        if os.path.exists(norm_path):
            st.image(norm_path, caption="SHAP Decision Force for Normal Observation", use_container_width=True)
        st.info("Decision Attribution: All hazard indicators remain within safe baselines; classified as Low Risk (Level 0).")
