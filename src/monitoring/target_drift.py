"""Target and Prediction Drift Monitoring Module.

Compares target distributions between reference and current climate periods.
Explicitly distinguishes between:
1. Actual Target Drift (requires ground-truth hazard labels)
2. Prediction Drift (monitors changes in model predicted risk distributions)
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils.logger import logger
from src.utils.common import read_yaml

DEFAULT_MODEL_PATH = "models/best_model.joblib"

def calculate_distribution_proportions(series: pd.Series) -> dict:
    """Calculates class percentage proportions for risk levels 0, 1, 2."""
    counts = series.value_counts(normalize=True).to_dict()
    return {
        "Low Risk (0)": float(round(counts.get(0, 0.0), 4)),
        "Moderate Risk (1)": float(round(counts.get(1, 0.0), 4)),
        "High Risk (2)": float(round(counts.get(2, 0.0), 4))
    }

def run_target_drift_monitoring(reference_path: str = "data/reference/reference_data.csv",
                                current_path: str = "data/current/current_data.csv",
                                model_path: str = DEFAULT_MODEL_PATH,
                                output_dir: str = "reports/monitoring") -> dict:
    """Executes target and prediction drift monitoring."""
    logger.info("Starting Target & Prediction Drift Analysis...")
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load model to generate model predictions for current data
    pipeline = joblib.load(model_path)
    ref_preds = pipeline.predict(ref_df)
    curr_preds = pipeline.predict(curr_df)
    
    ref_eval = pd.DataFrame({"prediction": ref_preds})
    curr_eval = pd.DataFrame({"prediction": curr_preds})
    
    has_ground_truth = "risk_level" in curr_df.columns and not curr_df["risk_level"].isna().all()
    
    if has_ground_truth:
        ref_eval["target"] = ref_df["risk_level"]
        curr_eval["target"] = curr_df["risk_level"]
        logger.info("Ground-truth target labels identified. Evaluating both Actual Target Drift and Prediction Drift.")
    else:
        logger.info("No ground-truth target available in current data. Monitoring Prediction Drift only.")
        
    # Run Evidently Report on target/prediction columns
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=ref_eval, current_data=curr_eval)
    
    html_path = os.path.join(output_dir, "target_drift_report.html")
    snapshot.save_html(html_path)
    logger.info(f"Saved Target Drift HTML Report to: {html_path}")
    
    # Statistical tests on distributions
    # Prediction drift test (Chi-Square)
    pred_ref_dist = ref_eval["prediction"].value_counts().sort_index()
    pred_curr_dist = curr_eval["prediction"].value_counts().sort_index()
    # Normalize frequencies for chi-square comparison
    f_exp = (pred_ref_dist / pred_ref_dist.sum()) * pred_curr_dist.sum()
    chi2_stat, pred_p_val = chisquare(f_obs=pred_curr_dist, f_exp=f_exp)
    
    prediction_drift_detected = bool(pred_p_val < 0.05)
    
    summary = {
        "status": "success",
        "has_ground_truth_target": bool(has_ground_truth),
        "prediction_drift": {
            "drift_detected": prediction_drift_detected,
            "p_value": float(round(pred_p_val, 4)),
            "test_type": "Chi-Square Goodness of Fit",
            "reference_predicted_distribution": calculate_distribution_proportions(ref_eval["prediction"]),
            "current_predicted_distribution": calculate_distribution_proportions(curr_eval["prediction"])
        }
    }
    
    if has_ground_truth:
        target_ref_dist = ref_eval["target"].value_counts().sort_index()
        target_curr_dist = curr_eval["target"].value_counts().sort_index()
        f_exp_target = (target_ref_dist / target_ref_dist.sum()) * target_curr_dist.sum()
        _, target_p_val = chisquare(f_obs=target_curr_dist, f_exp=f_exp_target)
        actual_target_drift_detected = bool(target_p_val < 0.05)
        
        summary["actual_target_drift"] = {
            "drift_detected": actual_target_drift_detected,
            "p_value": float(round(target_p_val, 4)),
            "test_type": "Chi-Square Goodness of Fit",
            "reference_actual_distribution": calculate_distribution_proportions(ref_eval["target"]),
            "current_actual_distribution": calculate_distribution_proportions(curr_eval["target"]),
            "distinction_note": "Target drift measures shifts in verified ground-truth risk events; Prediction drift measures shifts in model outputs."
        }
    else:
        summary["actual_target_drift"] = {
            "status": "UNAVAILABLE",
            "reason": "Ground-truth hazard labels not provided for incoming current dataset. Only prediction drift is measured."
        }
        
    json_path = os.path.join(output_dir, "target_drift_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Target Drift summary JSON to: {json_path}")
    
    return summary

if __name__ == "__main__":
    run_target_drift_monitoring()
