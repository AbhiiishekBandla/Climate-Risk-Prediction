"""Page 2: Climate Risk Prediction (Single-Instance & Batch Interface)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from typing import Dict, Any

from app.components.utils import get_predictor, calculate_heat_index, assign_imd_season

PRESETS = {
    "Select a Scenario Preset...": None,
    "☀️ Severe Pre-Monsoon Heatwave (Delhi)": {
        "city": "New Delhi",
        "date": date(2023, 5, 25),
        "t_max": 44.5,
        "t_min": 29.5,
        "t_mean": 37.0,
        "rh": 38.0,
        "precip": 0.0,
        "precip_3d": 0.0,
        "precip_7d": 0.0,
        "wind_speed": 16.0,
        "wind_gusts": 32.0,
        "pressure": 997.5,
        "radiation": 28.5,
        "et0": 8.2,
        "dry_spell": 35
    },
    "🌧️ Monsoon Extreme Deluge / Flood (Mumbai)": {
        "city": "Mumbai",
        "date": date(2023, 7, 18),
        "t_max": 29.5,
        "t_min": 25.0,
        "t_mean": 27.2,
        "rh": 94.0,
        "precip": 125.0,
        "precip_3d": 240.0,
        "precip_7d": 380.0,
        "wind_speed": 28.0,
        "wind_gusts": 55.0,
        "pressure": 1002.0,
        "radiation": 12.0,
        "et0": 2.5,
        "dry_spell": 0
    },
    "🌀 Cyclone / Squall Windstorm (Chennai)": {
        "city": "Chennai",
        "date": date(2023, 11, 28),
        "t_max": 28.0,
        "t_min": 23.5,
        "t_mean": 25.5,
        "rh": 90.0,
        "precip": 85.0,
        "precip_3d": 140.0,
        "precip_7d": 190.0,
        "wind_speed": 38.0,
        "wind_gusts": 68.0,
        "pressure": 992.0,
        "radiation": 10.5,
        "et0": 2.1,
        "dry_spell": 0
    },
    "⛅ Benign Winter Day (Kolkata)": {
        "city": "Kolkata",
        "date": date(2023, 1, 15),
        "t_max": 24.5,
        "t_min": 13.5,
        "t_mean": 19.0,
        "rh": 55.0,
        "precip": 0.0,
        "precip_3d": 0.0,
        "precip_7d": 0.0,
        "wind_speed": 9.5,
        "wind_gusts": 18.0,
        "pressure": 1014.5,
        "radiation": 18.0,
        "et0": 3.2,
        "dry_spell": 20
    }
}

def render_prediction_page():
    st.markdown('<div class="main-title">🔮 Real-Time Climate Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Forecast multi-hazard meteorological severity levels using the trained, verified Gradient Boosting pipeline.'
        '</div>',
        unsafe_allow_html=True
    )
    
    predictor = get_predictor()
    
    tab_single, tab_batch = st.tabs(["🎯 Single Observation Forecast", "📂 Batch CSV File Processing"])
    
    with tab_single:
        st.markdown('<div class="section-header">1. Meteorological Scenario Selector</div>', unsafe_allow_html=True)
        st.write("Load a realistic extreme weather benchmark or manually input live meteorological observations:")
        
        selected_preset_key = st.selectbox("Quick-Load Weather Scenario Preset:", list(PRESETS.keys()))
        preset_values = PRESETS.get(selected_preset_key)
        
        st.markdown('<div class="section-header">2. Primary Meteorological Inputs</div>', unsafe_allow_html=True)
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            city = st.selectbox(
                "Metropolitan Region:",
                ["New Delhi", "Mumbai", "Kolkata", "Chennai"],
                index=["New Delhi", "Mumbai", "Kolkata", "Chennai"].index(preset_values["city"]) if preset_values else 0
            )
        with col_meta2:
            obs_date = st.date_input(
                "Observation Date:",
                value=preset_values["date"] if preset_values else date.today()
            )
            
        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            t_max = st.number_input(
                "Max Temperature (°C):",
                min_value=-10.0, max_value=60.0,
                value=float(preset_values["t_max"]) if preset_values else 36.5,
                step=0.5,
                help="Daily maximum 2m air temperature"
            )
        with c_t2:
            t_min = st.number_input(
                "Min Temperature (°C):",
                min_value=-15.0, max_value=45.0,
                value=float(preset_values["t_min"]) if preset_values else 25.0,
                step=0.5,
                help="Daily minimum 2m air temperature"
            )
        with c_t3:
            rh_mean = st.slider(
                "Mean Relative Humidity (%):",
                min_value=5.0, max_value=100.0,
                value=float(preset_values["rh"]) if preset_values else 55.0,
                step=1.0,
                help="Mean 2m relative humidity"
            )
            
        c_h1, c_h2, c_h3 = st.columns(3)
        with c_h1:
            precip = st.number_input(
                "Daily Rainfall (mm):",
                min_value=0.0, max_value=500.0,
                value=float(preset_values["precip"]) if preset_values else 0.0,
                step=1.0,
                help="Total 24-hour precipitation accumulation"
            )
        with c_h2:
            wind_speed = st.number_input(
                "Max Wind Speed (km/h):",
                min_value=0.0, max_value=200.0,
                value=float(preset_values["wind_speed"]) if preset_values else 15.0,
                step=1.0,
                help="Maximum sustained 10m wind speed"
            )
        with c_h3:
            wind_gusts = st.number_input(
                "Max Wind Gusts (km/h):",
                min_value=0.0, max_value=250.0,
                value=float(preset_values["wind_gusts"]) if preset_values else 28.0,
                step=1.0,
                help="Peak 10m wind gust"
            )
            
        # Live Calculated Indicators
        derived_hi = float(calculate_heat_index(pd.Series([t_max]), pd.Series([rh_mean])).iloc[0])
        derived_dtr = max(0.0, t_max - t_min)
        derived_season = assign_imd_season(obs_date.month)
        
        st.markdown(f"""
        <div style="background-color:#F1F5F9; border-radius:8px; padding:10px 16px; margin: 10px 0 15px 0;">
            <span style="font-weight:600; color:#334155;">Live Meteorological Indicators:</span> &nbsp;
            <span>Calculated Heat Index: <strong>{derived_hi:.1f}°C</strong></span> &nbsp;|&nbsp; 
            <span>Diurnal Temp Range: <strong>{derived_dtr:.1f}°C</strong></span> &nbsp;|&nbsp; 
            <span>IMD Season: <strong>{derived_season}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🛠️ Advanced Meteorological Parameters (Optional Fine-Tuning)", expanded=False):
            st.caption("These parameters are automatically populated with sound regional defaults, but can be customized:")
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            with adv_col1:
                precip_3d = st.number_input(
                    "3-Day Rolling Rain (mm):",
                    min_value=0.0, max_value=1000.0,
                    value=float(preset_values["precip_3d"]) if preset_values else float(precip * 1.5),
                    step=5.0
                )
                precip_7d = st.number_input(
                    "7-Day Rolling Rain (mm):",
                    min_value=0.0, max_value=1500.0,
                    value=float(preset_values["precip_7d"]) if preset_values else float(precip * 2.5),
                    step=5.0
                )
            with adv_col2:
                surface_pressure = st.number_input(
                    "Surface Pressure (hPa):",
                    min_value=900.0, max_value=1060.0,
                    value=float(preset_values["pressure"]) if preset_values else 1005.0,
                    step=1.0
                )
                radiation = st.number_input(
                    "Shortwave Radiation (MJ/m²):",
                    min_value=0.0, max_value=40.0,
                    value=float(preset_values["radiation"]) if preset_values else 20.0,
                    step=0.5
                )
            with adv_col3:
                et0 = st.number_input(
                    "Reference Evapotranspiration (mm):",
                    min_value=0.0, max_value=20.0,
                    value=float(preset_values["et0"]) if preset_values else 5.0,
                    step=0.2
                )
                dry_spell = st.number_input(
                    "Consecutive Dry Days:",
                    min_value=0, max_value=150,
                    value=int(preset_values["dry_spell"]) if preset_values else (0 if precip > 1.0 else 10),
                    step=1
                )
                
        # Validation Checks
        input_errors = []
        if t_max < t_min:
            input_errors.append("Maximum Temperature cannot be lower than Minimum Temperature.")
        if wind_gusts < wind_speed:
            input_errors.append("Peak Wind Gusts cannot be lower than sustained Wind Speed.")
            
        if input_errors:
            for err in input_errors:
                st.error(f"⚠️ Validation Error: {err}")
            st.stop()
            
        if st.button("🚀 Run Climate Risk Forecast", type="primary", use_container_width=True):
            with st.spinner("Analyzing atmospheric indicators with Gradient Boosting model..."):
                t_mean = round((t_max + t_min) / 2.0, 2)
                wind_gust_factor = round(wind_gusts / max(wind_speed, 1.0), 3)
                water_deficit = round(max(0.0, et0 - precip), 2)
                
                input_sample = {
                    "city": city,
                    "season": derived_season,
                    "temperature_2m_max": t_max,
                    "temperature_2m_min": t_min,
                    "temperature_2m_mean": t_mean,
                    "apparent_temperature_max": round(derived_hi, 2),
                    "precipitation_sum": precip,
                    "windspeed_10m_max": wind_speed,
                    "windgusts_10m_max": wind_gusts,
                    "surface_pressure_mean": surface_pressure,
                    "relative_humidity_2m_mean": rh_mean,
                    "shortwave_radiation_sum": radiation,
                    "et0_fao_evapotranspiration": et0,
                    "diurnal_temp_range": derived_dtr,
                    "heat_index": derived_hi,
                    "precip_3d_sum": precip_3d,
                    "precip_7d_sum": precip_7d,
                    "dry_spell_days": dry_spell,
                    "wind_gust_factor": wind_gust_factor,
                    "water_deficit": water_deficit,
                    "month": obs_date.month,
                    "day_of_year": obs_date.timetuple().tm_yday
                }
                
                prediction_result = predictor.predict_single(input_sample)
                pred_level = prediction_result["predicted_risk_level"]
                probs = prediction_result["probabilities"]
                confidence = prediction_result["confidence"]
                
                st.markdown("---")
                st.markdown('<div class="section-header">3. Forecast Results & Decision Advisory</div>', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1.2, 1])
                with res_col1:
                    if pred_level == 0:
                        badge_html = '<div class="badge-low" style="font-size:1.3rem;">🟢 LEVEL 0: LOW RISK (NORMAL)</div>'
                        advisory_text = (
                            "**NDMA Advisory:** General atmospheric conditions are benign. Standard civic and agricultural operations "
                            "can proceed without weather-related disruption."
                        )
                    elif pred_level == 1:
                        badge_html = '<div class="badge-moderate" style="font-size:1.3rem;">🟡 LEVEL 1: MODERATE RISK (ADVISORY)</div>'
                        advisory_text = (
                            "**NDMA Advisory:** Elevated weather stress detected. Vulnerable populations (elderly, outdoor laborers) "
                            "should avoid peak heat hours or localized water-logging zones. Maintain standard regional preparedness."
                        )
                    else:
                        badge_html = '<div class="badge-high" style="font-size:1.3rem;">🔴 LEVEL 2: HIGH RISK (SEVERE / DISASTER ALERT)</div>'
                        advisory_text = (
                            "**NDMA Advisory:** Severe meteorological emergency hazard triggered! Extreme heatwave, heavy deluge, or "
                            "destructive squall winds present high danger to life and urban infrastructure. Issue prompt citizen advisories."
                        )
                        
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:12px;'>{advisory_text}</div>", unsafe_allow_html=True)
                    st.metric("Model Confidence Score", f"{confidence * 100:.1f}%")
                    
                with res_col2:
                    st.markdown("**Class Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Risk Class": ["Low (Normal)", "Moderate (Advisory)", "High (Severe)"],
                        "Probability": [probs["Low"], probs["Moderate"], probs["High"]]
                    })
                    fig = px.bar(
                        prob_df,
                        x="Risk Class",
                        y="Probability",
                        color="Risk Class",
                        color_discrete_map={
                            "Low (Normal)": "#22C55E",
                            "Moderate (Advisory)": "#F59E0B",
                            "High (Severe)": "#EF4444"
                        },
                        range_y=[0, 1.0],
                        text_auto=".2%"
                    )
                    fig.update_layout(showlegend=False, height=240, margin=dict(l=20, r=20, t=10, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    with tab_batch:
        st.markdown('<div class="section-header">Batch CSV Inference Engine</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file containing climate observations to generate high-throughput risk predictions:")
        
        # Download template option
        sample_df = pd.DataFrame([
            {
                "city": "New Delhi", "season": "Summer/Pre-Monsoon", "temperature_2m_max": 44.0,
                "temperature_2m_min": 29.0, "temperature_2m_mean": 36.5, "apparent_temperature_max": 46.0,
                "precipitation_sum": 0.0, "windspeed_10m_max": 18.0, "windgusts_10m_max": 34.0,
                "surface_pressure_mean": 998.0, "relative_humidity_2m_mean": 38.0,
                "shortwave_radiation_sum": 28.0, "et0_fao_evapotranspiration": 8.0,
                "diurnal_temp_range": 15.0, "heat_index": 46.0, "precip_3d_sum": 0.0,
                "precip_7d_sum": 0.0, "dry_spell_days": 30, "wind_gust_factor": 1.89,
                "water_deficit": 8.0, "month": 5, "day_of_year": 140
            },
            {
                "city": "Mumbai", "season": "Monsoon", "temperature_2m_max": 29.0,
                "temperature_2m_min": 25.0, "temperature_2m_mean": 27.0, "apparent_temperature_max": 34.0,
                "precipitation_sum": 110.0, "windspeed_10m_max": 26.0, "windgusts_10m_max": 52.0,
                "surface_pressure_mean": 1002.0, "relative_humidity_2m_mean": 92.0,
                "shortwave_radiation_sum": 12.0, "et0_fao_evapotranspiration": 2.5,
                "diurnal_temp_range": 4.0, "heat_index": 34.0, "precip_3d_sum": 210.0,
                "precip_7d_sum": 330.0, "dry_spell_days": 0, "wind_gust_factor": 2.0,
                "water_deficit": 0.0, "month": 7, "day_of_year": 200
            }
        ])
        
        csv_sample = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=csv_sample,
            file_name="climate_risk_batch_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload CSV observations file:", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_input_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_input_df):,} rows from uploaded file.")
                
                with st.spinner("Executing model pipeline inference..."):
                    batch_results = predictor.predict(batch_input_df)
                    
                st.markdown('<div class="section-header">Batch Predictions Summary</div>', unsafe_allow_html=True)
                
                total_n = len(batch_results)
                low_pct = (batch_results["predicted_risk_level"] == 0).mean() * 100
                mod_pct = (batch_results["predicted_risk_level"] == 1).mean() * 100
                high_pct = (batch_results["predicted_risk_level"] == 2).mean() * 100
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Records Evaluated", f"{total_n:,}")
                m2.metric("Low Risk Proportion", f"{low_pct:.1f}%")
                m3.metric("Moderate Risk Proportion", f"{mod_pct:.1f}%")
                m4.metric("High Risk Proportion", f"{high_pct:.1f}%")
                
                st.dataframe(
                    batch_results[[
                        "city", "season", "temperature_2m_max", "precipitation_sum", 
                        "heat_index", "predicted_risk_level", "predicted_risk_label", "confidence"
                    ]].head(50),
                    use_container_width=True
                )
                
                # Download predictions
                out_csv = batch_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Complete Predicted Results (CSV)",
                    data=out_csv,
                    file_name="predicted_climate_risks.csv",
                    mime="text/csv",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"❌ Error processing batch CSV: {str(e)}")
