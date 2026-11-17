"""Integration and Component Tests for Phase 5: Streamlit Web Application."""

import os
import pytest
import pandas as pd
from app.components.utils import (
    get_predictor,
    load_processed_data,
    load_metrics_and_comparison,
    load_monitoring_summaries,
    calculate_heat_index,
    assign_imd_season
)
from app.pages_content.prediction import PRESETS

def test_app_structure_and_files():
    assert os.path.exists("app/app.py"), "app/app.py entrypoint is missing"
    assert os.path.exists(".streamlit/config.toml"), ".streamlit/config.toml missing"
    
    pages = [
        "app/pages_content/home.py",
        "app/pages_content/prediction.py",
        "app/pages_content/visualization.py",
        "app/pages_content/data_drift.py",
        "app/pages_content/target_drift.py",
        "app/pages_content/explainability.py",
        "app/pages_content/performance.py",
        "app/pages_content/project_info.py"
    ]
    for page in pages:
        assert os.path.exists(page), f"Page module {page} missing"

def test_app_predictor_loading():
    predictor = get_predictor()
    assert predictor is not None
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "predict_single")

def test_app_data_loaders():
    df = load_processed_data()
    assert not df.empty, "load_processed_data returned empty DataFrame"
    assert "risk_level" in df.columns
    assert "city" in df.columns
    
    metrics_pkg = load_metrics_and_comparison()
    assert "scores" in metrics_pkg
    assert "comparison" in metrics_pkg
    assert not metrics_pkg["comparison"].empty
    
    mon_pkg = load_monitoring_summaries()
    assert "dashboard" in mon_pkg
    assert "data_drift" in mon_pkg
    assert "target_drift" in mon_pkg
    assert "performance" in mon_pkg
    assert "shap" in mon_pkg

def test_app_prediction_presets():
    assert len(PRESETS) >= 4
    for name, p in PRESETS.items():
        if p is None:
            continue
        assert "city" in p
        assert "t_max" in p
        assert "t_min" in p
        assert "rh" in p
        assert "precip" in p
        assert "wind_speed" in p
        assert "wind_gusts" in p
        assert p["t_max"] >= p["t_min"]
        assert p["wind_gusts"] >= p["wind_speed"]

def test_app_single_prediction_flow():
    predictor = get_predictor()
    sample = {
        "city": "Mumbai",
        "season": "Monsoon",
        "temperature_2m_max": 30.0,
        "temperature_2m_min": 25.5,
        "temperature_2m_mean": 27.7,
        "apparent_temperature_max": 38.0,
        "precipitation_sum": 115.0,
        "windspeed_10m_max": 24.0,
        "windgusts_10m_max": 48.0,
        "surface_pressure_mean": 1001.0,
        "relative_humidity_2m_mean": 92.0,
        "shortwave_radiation_sum": 14.0,
        "et0_fao_evapotranspiration": 3.0,
        "diurnal_temp_range": 4.5,
        "heat_index": 38.0,
        "precip_3d_sum": 210.0,
        "precip_7d_sum": 320.0,
        "dry_spell_days": 0,
        "wind_gust_factor": 2.0,
        "water_deficit": 0.0,
        "month": 7,
        "day_of_year": 195
    }
    res = predictor.predict_single(sample)
    assert res["predicted_risk_level"] in [0, 1, 2]
    assert 0.0 <= res["confidence"] <= 1.0
    probs = res["probabilities"]
    assert "Low" in probs and "Moderate" in probs and "High" in probs
    assert abs(sum(probs.values()) - 1.0) < 0.05
    # For 115 mm rainfall in monsoon, should be High Risk (Level 2)
    assert res["predicted_risk_level"] == 2

def test_app_batch_prediction_flow():
    predictor = get_predictor()
    batch_df = pd.DataFrame([
        {
            "city": "New Delhi", "season": "Summer/Pre-Monsoon", "temperature_2m_max": 44.5,
            "temperature_2m_min": 29.5, "temperature_2m_mean": 37.0, "apparent_temperature_max": 47.0,
            "precipitation_sum": 0.0, "windspeed_10m_max": 18.0, "windgusts_10m_max": 32.0,
            "surface_pressure_mean": 997.0, "relative_humidity_2m_mean": 38.0,
            "shortwave_radiation_sum": 28.0, "et0_fao_evapotranspiration": 8.0,
            "diurnal_temp_range": 15.0, "heat_index": 46.5, "precip_3d_sum": 0.0,
            "precip_7d_sum": 0.0, "dry_spell_days": 35, "wind_gust_factor": 1.78,
            "water_deficit": 8.0, "month": 5, "day_of_year": 145
        },
        {
            "city": "Kolkata", "season": "Winter", "temperature_2m_max": 24.0,
            "temperature_2m_min": 14.0, "temperature_2m_mean": 19.0, "apparent_temperature_max": 24.0,
            "precipitation_sum": 0.0, "windspeed_10m_max": 10.0, "windgusts_10m_max": 18.0,
            "surface_pressure_mean": 1014.0, "relative_humidity_2m_mean": 55.0,
            "shortwave_radiation_sum": 18.0, "et0_fao_evapotranspiration": 3.0,
            "diurnal_temp_range": 10.0, "heat_index": 24.0, "precip_3d_sum": 0.0,
            "precip_7d_sum": 0.0, "dry_spell_days": 20, "wind_gust_factor": 1.8,
            "water_deficit": 3.0, "month": 1, "day_of_year": 15
        }
    ])
    res_df = predictor.predict(batch_df)
    assert len(res_df) == 2
    assert "predicted_risk_level" in res_df.columns
    assert "predicted_risk_label" in res_df.columns
    assert "confidence" in res_df.columns
    # Row 0: Heatwave -> High Risk (2)
    assert res_df.iloc[0]["predicted_risk_level"] == 2
    # Row 1: Winter benign -> Low Risk (0)
    assert res_df.iloc[1]["predicted_risk_level"] == 0

def test_evidently_html_reports_exist_for_app():
    reports = [
        "reports/monitoring/data_drift_report.html",
        "reports/monitoring/target_drift_report.html",
        "reports/monitoring/model_performance_report.html"
    ]
    for r in reports:
        assert os.path.exists(r), f"Evidently report {r} missing"
        assert os.path.getsize(r) > 100_000, f"Report {r} too small"

def test_all_pages_render_without_exception():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    pages = [
        "🏠 Home / Overview",
        "🔮 Climate Risk Prediction",
        "📈 Data Visualization & EDA",
        "📡 Data Drift Monitoring",
        "🎯 Target Drift Monitoring",
        "🧠 SHAP Explainability",
        "📊 Model Performance & Alerts",
        "ℹ️ Project Information"
    ]
    app_path = Path(__file__).resolve().parent.parent / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception
    for page in pages:
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, f"Exception rendering {page}: {at.exception}"
