"""Unit and Integration Tests for Data Collection, Preprocessing & Feature Engineering."""

import os
import pytest
import pandas as pd
import numpy as np
from src.features.build_features import calculate_heat_index, assign_imd_season, derive_risk_labels, engineer_features

def test_raw_data_exists_and_valid():
    raw_path = "data/raw/climate_risk_data.csv"
    assert os.path.exists(raw_path), "Raw data file does not exist."
    df = pd.read_csv(raw_path)
    assert len(df) > 10000, f"Expected >10k rows, got {len(df)}"
    assert "time" in df.columns
    assert "city" in df.columns
    assert "temperature_2m_max" in df.columns
    assert "precipitation_sum" in df.columns
    assert "windgusts_10m_max" in df.columns

def test_processed_splits_exist_and_temporal_integrity():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    assert os.path.exists(train_path), "Train CSV missing"
    assert os.path.exists(test_path), "Test CSV missing"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Check temporal split (strictly no data leakage)
    assert train_df["year"].max() < 2022, "Train set contains post-2021 data (leakage!)"
    assert test_df["year"].min() >= 2022, "Test set contains pre-2022 data"
    assert set(train_df["city"].unique()) == set(test_df["city"].unique())

def test_heat_index_calculation():
    # Warm tropical conditions (35 C, 70% RH)
    temp = pd.Series([35.0, 20.0])
    rh = pd.Series([70.0, 50.0])
    hi = calculate_heat_index(temp, rh)
    
    assert len(hi) == 2
    # At 35C with 70% humidity, apparent heat index should be higher than 40C
    assert hi.iloc[0] > 40.0
    # At mild 20C, heat index should be close to ambient
    assert abs(hi.iloc[1] - 20.0) < 5.0

def test_imd_seasons():
    assert assign_imd_season(1) == "Winter"
    assert assign_imd_season(4) == "Summer/Pre-Monsoon"
    assert assign_imd_season(7) == "Monsoon"
    assert assign_imd_season(10) == "Post-Monsoon"

def test_proxy_risk_label_logic():
    mock_data = pd.DataFrame({
        "temperature_2m_max": [43.0, 25.0, 39.5],
        "heat_index": [46.0, 25.0, 40.0],
        "precipitation_sum": [0.0, 5.0, 0.0],
        "precip_3d_sum": [0.0, 10.0, 0.0],
        "windgusts_10m_max": [30.0, 20.0, 35.0]
    })
    risks = derive_risk_labels(mock_data)
    assert risks.iloc[0] == 2, "Extreme 43C / 46C HI must be High Risk (2)"
    assert risks.iloc[1] == 0, "Mild 25C / 5mm must be Low Risk (0)"
    assert risks.iloc[2] == 1, "39.5C / 40C HI must be Moderate Risk (1)"
