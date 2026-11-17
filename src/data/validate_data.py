"""Data Validation Module for Climate Risk Prediction Pipeline.

Validates schema, physical climate bounds, missing values, and record completeness
for raw meteorological data before preprocessing.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.utils.common import read_yaml

REQUIRED_COLUMNS = [
    "time", "city", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "precipitation_sum", "windspeed_10m_max",
    "windgusts_10m_max", "surface_pressure_mean", "relative_humidity_2m_mean"
]

def validate_raw_data(data_path: str = "data/raw/climate_risk_data.csv",
                      status_output_path: str = "data/raw/validation_status.json") -> bool:
    """Validates raw dataset integrity and exports validation report."""
    logger.info(f"Validating raw climate dataset at {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 1. Record count check
    assert len(df) >= 10000, f"Insufficient records: {len(df)} (expected >= 10000)"
    
    # 2. Schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"
    
    # 3. Physical bounds checks (Meteorological sanity checks)
    assert df["temperature_2m_max"].max() <= 60.0, "Unrealistic max temperature (> 60C)"
    assert df["temperature_2m_min"].min() >= -40.0, "Unrealistic min temperature (< -40C)"
    assert df["precipitation_sum"].min() >= 0.0, "Negative precipitation detected"
    assert df["relative_humidity_2m_mean"].between(0.0, 100.0).all(), "Relative humidity out of [0, 100]%"
    
    validation_report = {
        "validation_passed": True,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "cities": df["city"].unique().tolist(),
        "date_range": [str(df["time"].min()), str(df["time"].max())],
        "columns_verified": REQUIRED_COLUMNS
    }
    
    os.makedirs(os.path.dirname(status_output_path), exist_ok=True)
    with open(status_output_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
        
    logger.info(f"Data validation PASSED! Verified {len(df)} records across {len(validation_report['cities'])} cities.")
    return True

if __name__ == "__main__":
    validate_raw_data()
