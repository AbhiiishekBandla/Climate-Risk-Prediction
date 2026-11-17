"""Inference and Prediction Module for Climate Risk Prediction System.

Loads serialized model pipeline and provides batch and single-record prediction functions.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from src.utils.logger import logger

DEFAULT_MODEL_PATH = "models/best_model.joblib"
RISK_LEVEL_MAP = {
    0: "Low Risk (Normal)",
    1: "Moderate Risk (Advisory)",
    2: "High Risk (Severe / Extreme)"
}

class ClimateRiskPredictor:
    """Wrapper class for loading serialized pipeline and computing predictions."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        logger.info(f"Loading climate risk model pipeline from {self.model_path}...")
        return joblib.load(self.model_path)
        
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Computes predictions and probabilities for input dataframe."""
        df = input_df.copy()
        preds = self.pipeline.predict(df)
        probs = self.pipeline.predict_proba(df)
        
        result_df = df.copy()
        result_df["predicted_risk_level"] = preds
        result_df["predicted_risk_label"] = [RISK_LEVEL_MAP.get(p, "Unknown") for p in preds]
        result_df["prob_low_risk"] = probs[:, 0].round(4)
        result_df["prob_moderate_risk"] = probs[:, 1].round(4)
        result_df["prob_high_risk"] = probs[:, 2].round(4)
        result_df["confidence"] = np.max(probs, axis=1).round(4)
        
        return result_df
        
    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method for single-observation prediction."""
        df = pd.DataFrame([input_dict])
        res = self.predict(df).iloc[0]
        return {
            "predicted_risk_level": int(res["predicted_risk_level"]),
            "predicted_risk_label": res["predicted_risk_label"],
            "confidence": float(res["confidence"]),
            "probabilities": {
                "Low": float(res["prob_low_risk"]),
                "Moderate": float(res["prob_moderate_risk"]),
                "High": float(res["prob_high_risk"])
            }
        }

if __name__ == "__main__":
    predictor = ClimateRiskPredictor()
    test_sample = {
        "city": "New Delhi",
        "season": "Summer/Pre-Monsoon",
        "temperature_2m_max": 43.5,
        "temperature_2m_min": 29.0,
        "temperature_2m_mean": 36.2,
        "apparent_temperature_max": 46.8,
        "precipitation_sum": 0.0,
        "windspeed_10m_max": 18.5,
        "windgusts_10m_max": 35.0,
        "surface_pressure_mean": 998.2,
        "relative_humidity_2m_mean": 42.0,
        "shortwave_radiation_sum": 27.5,
        "et0_fao_evapotranspiration": 7.8,
        "diurnal_temp_range": 14.5,
        "heat_index": 45.2,
        "precip_3d_sum": 0.0,
        "precip_7d_sum": 0.0,
        "dry_spell_days": 25,
        "wind_gust_factor": 1.89,
        "water_deficit": 7.8,
        "month": 5,
        "day_of_year": 145
    }
    pred = predictor.predict_single(test_sample)
    print("Test Sample Prediction Result:")
    print(pred)
