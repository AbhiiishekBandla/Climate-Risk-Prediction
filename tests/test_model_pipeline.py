"""Unit and Integration Tests for Model Training, Evaluation, and Prediction Pipeline."""

import os
import pytest
import joblib
import pandas as pd
import numpy as np
from src.models.predict import ClimateRiskPredictor
from src.models.train import evaluate_predictions, get_candidate_models

def test_model_and_preprocessor_artifacts_exist():
    assert os.path.exists("models/best_model.joblib"), "models/best_model.joblib is missing"
    assert os.path.exists("models/preprocessor.joblib"), "models/preprocessor.joblib is missing"
    assert os.path.exists("metrics/model_comparison.csv"), "metrics/model_comparison.csv is missing"
    assert os.path.exists("metrics/scores.json"), "metrics/scores.json is missing"
    assert os.path.exists("metrics/confusion_matrix.png"), "metrics/confusion_matrix.png is missing"

def test_model_comparison_file():
    df = pd.read_csv("metrics/model_comparison.csv")
    assert len(df) == 5, f"Expected 5 candidate models, found {len(df)}"
    assert "accuracy" in df.columns
    assert "f1_macro" in df.columns
    assert "roc_auc_macro_ovr" in df.columns
    assert "model_name" in df.columns
    # Check that best model has top score
    assert df["f1_macro"].iloc[0] >= df["f1_macro"].iloc[-1]

def test_candidate_models_instantiation():
    models = get_candidate_models()
    assert "Logistic_Regression" in models
    assert "Decision_Tree" in models
    assert "Random_Forest" in models
    assert "Gradient_Boosting" in models
    assert "XGBoost" in models

def test_evaluate_predictions_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.6, 0.3]
    ])
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["roc_auc_macro_ovr"] <= 1.0

def test_predictor_batch_and_single_inference():
    predictor = ClimateRiskPredictor("models/best_model.joblib")
    
    # Single sample
    sample = {
        "city": "Mumbai",
        "season": "Monsoon",
        "temperature_2m_max": 29.5,
        "temperature_2m_min": 24.0,
        "temperature_2m_mean": 26.5,
        "apparent_temperature_max": 34.0,
        "precipitation_sum": 120.0,  # Extreme heavy monsoon rain
        "windspeed_10m_max": 35.0,
        "windgusts_10m_max": 65.0,
        "surface_pressure_mean": 1002.0,
        "relative_humidity_2m_mean": 92.0,
        "shortwave_radiation_sum": 10.0,
        "et0_fao_evapotranspiration": 2.5,
        "diurnal_temp_range": 5.5,
        "heat_index": 35.0,
        "precip_3d_sum": 210.0,
        "precip_7d_sum": 350.0,
        "dry_spell_days": 0,
        "wind_gust_factor": 1.85,
        "water_deficit": -117.5,
        "month": 7,
        "day_of_year": 200
    }
    
    single_res = predictor.predict_single(sample)
    assert single_res["predicted_risk_level"] in [0, 1, 2]
    assert single_res["predicted_risk_level"] == 2  # Extreme rain and winds -> High Risk
    assert sum(single_res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    
    # Batch inference
    batch_df = pd.DataFrame([sample, sample])
    batch_res = predictor.predict(batch_df)
    assert len(batch_res) == 2
    assert "predicted_risk_level" in batch_res.columns
    assert "predicted_risk_label" in batch_res.columns
