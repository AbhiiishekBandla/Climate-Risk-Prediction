"""Page 3: Climate Data Visualization and Exploratory Analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.components.utils import load_processed_data

def render_visualization_page():
    st.markdown('<div class="main-title">📈 Climate Data Visual Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Interactive exploratory data analysis revealing multi-variate climate relationships, '
        'extreme hazard distributions, and regional variations across India.'
        '</div>',
        unsafe_allow_html=True
    )
    
    df = load_processed_data()
    if df.empty:
        st.warning("Processed dataset not available. Please ensure data pipeline has executed.")
        return
        
    st.markdown('<div class="section-header">1. Interactive Regional Filters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cities = ["All Metros"] + sorted(list(df["city"].unique()))
        selected_city = st.selectbox("Filter by City:", cities)
    with c2:
        seasons = ["All Seasons"] + sorted(list(df["season"].unique()))
        selected_season = st.selectbox("Filter by Season:", seasons)
        
    filtered_df = df.copy()
    if selected_city != "All Metros":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]
    if selected_season != "All Seasons":
        filtered_df = filtered_df[filtered_df["season"] == selected_season]
        
    st.caption(f"Displaying **{len(filtered_df):,}** daily observations based on current selection.")
    
    st.markdown("---")
    
    # 1. Thermal & Heat Index Distribution
    st.markdown('<div class="section-header">2. Thermal Dynamics & Heat Index Stress</div>', unsafe_allow_html=True)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        fig_temp = px.histogram(
            filtered_df,
            x="temperature_2m_max",
            color="season",
            nbins=35,
            title="Max Temperature Distribution (°C)",
            labels={"temperature_2m_max": "Maximum Temperature (°C)", "season": "IMD Season"},
            opacity=0.75
        )
        fig_temp.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        st.plotly_chart(fig_temp, use_container_width=True)
        
    with t_col2:
        fig_hi = px.scatter(
            filtered_df,
            x="temperature_2m_max",
            y="relative_humidity_2m_mean",
            color="heat_index",
            color_continuous_scale="Turbo",
            title="Heat Index as a Function of Temp & Humidity",
            labels={
                "temperature_2m_max": "Max Temperature (°C)",
                "relative_humidity_2m_mean": "Relative Humidity (%)",
                "heat_index": "Heat Index (°C)"
            }
        )
        fig_hi.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        st.plotly_chart(fig_hi, use_container_width=True)
        
    st.markdown("---")
    
    # 2. Precipitation Dynamics & Hydrological Hazards
    st.markdown('<div class="section-header">3. Precipitation Dynamics & Flooding Indicators</div>', unsafe_allow_html=True)
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fig_rain = px.box(
            filtered_df,
            x="city" if selected_city == "All Metros" else "season",
            y="precipitation_sum",
            color="city" if selected_city == "All Metros" else "season",
            title="Daily Precipitation Accumulation (mm)",
            points=False
        )
        fig_rain.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_rain, use_container_width=True)
        
    with p_col2:
        fig_precip_roll = px.scatter(
            filtered_df,
            x="precipitation_sum",
            y="precip_3d_sum",
            color="season",
            size="windgusts_10m_max",
            title="Single-Day Rain vs 3-Day Cumulative Deluge (mm)",
            labels={
                "precipitation_sum": "Daily Rain (mm)",
                "precip_3d_sum": "3-Day Cumulative Rain (mm)",
                "windgusts_10m_max": "Peak Gusts (km/h)"
            }
        )
        fig_precip_roll.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        st.plotly_chart(fig_precip_roll, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Risk Level Breakdown
    st.markdown('<div class="section-header">4. Meteorological Risk Level Breakdown</div>', unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        risk_counts = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        risk_counts["label"] = risk_counts["risk_level"].map({0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"})
        
        fig_pie = px.pie(
            risk_counts,
            names="label",
            values="count",
            color="label",
            color_discrete_map={
                "Low Risk": "#22C55E",
                "Moderate Risk": "#F59E0B",
                "High Risk": "#EF4444"
            },
            hole=0.45,
            title="Overall Climate Risk Category Distribution"
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with r_col2:
        risk_by_group = filtered_df.groupby(["city", "risk_level"]).size().reset_index(name="count")
        risk_by_group["risk_label"] = risk_by_group["risk_level"].map({0: "Low", 1: "Moderate", 2: "High"})
        
        fig_bar = px.bar(
            risk_by_group,
            x="city",
            y="count",
            color="risk_label",
            barmode="group",
            color_discrete_map={"Low": "#22C55E", "Moderate": "#F59E0B", "High": "#EF4444"},
            title="Risk Breakdown by Metropolitan Zone"
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
