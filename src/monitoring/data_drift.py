"""Data Drift Monitoring Module using Evidently AI.

Compares incoming/current climate observations against baseline reference data.
Detects feature-level statistical drift across thermal, precipitation, moisture,
and pressure variables. Generates interactive HTML and JSON drift reports.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.utils.logger import logger
from src.utils.common import read_yaml

MONITORED_FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "precip_3d_sum",
    "precip_7d_sum",
    "relative_humidity_2m_mean",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "surface_pressure_mean",
    "heat_index",
    "diurnal_temp_range",
    "dry_spell_days"
]

def run_data_drift_monitoring(reference_path: str = "data/reference/reference_data.csv",
                              current_path: str = "data/current/current_data.csv",
                              output_dir: str = "reports/monitoring") -> dict:
    """Executes Evidently AI data drift detection report on meteorological features."""
    logger.info(f"Loading reference data ({reference_path}) and current data ({current_path})...")
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to monitored features for clean drift detection
    ref_sub = ref_df[MONITORED_FEATURES]
    curr_sub = curr_df[MONITORED_FEATURES]
    
    logger.info(f"Running Evidently DataDriftPreset on {len(MONITORED_FEATURES)} climate features...")
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=ref_sub, current_data=curr_sub)
    
    # Save interactive HTML report
    html_path = os.path.join(output_dir, "data_drift_report.html")
    snapshot.save_html(html_path)
    logger.info(f"Saved interactive HTML data drift report to: {html_path}")
    
    # Extract JSON metrics dictionary
    dict_results = snapshot.dict()
    
    # Parse high-level drift summary
    drift_summary = {
        "status": "success",
        "reference_records": len(ref_df),
        "current_records": len(curr_df),
        "monitored_features_count": len(MONITORED_FEATURES),
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "drifted_features": [],
        "feature_metrics": {}
    }
    
    try:
        # Search for DataDriftTable or DatasetDriftMetric in metric results
        for item in dict_results.get("metrics", []):
            result = item.get("result", {})
            if "dataset_drift" in result:
                drift_summary["dataset_drift_detected"] = bool(result.get("dataset_drift", False))
                drift_summary["drift_share"] = float(round(result.get("drift_share", 0.0), 4))
                drift_summary["number_of_drifted_columns"] = int(result.get("number_of_drifted_columns", 0))
            if "drift_by_columns" in result:
                for col, col_data in result["drift_by_columns"].items():
                    is_drift = col_data.get("drift_detected", False)
                    p_val = col_data.get("p_value", 1.0)
                    stat_test = col_data.get("stat_test_name", "KS")
                    drift_summary["feature_metrics"][col] = {
                        "drift_detected": bool(is_drift),
                        "p_value": float(round(p_val, 4)) if p_val is not None else None,
                        "stat_test": stat_test
                    }
                    if is_drift:
                        drift_summary["drifted_features"].append(col)
    except Exception as e:
        logger.warning(f"Could not parse all Evidently details: {e}")
        
    json_path = os.path.join(output_dir, "data_drift_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_summary, f, indent=2)
    logger.info(f"Saved data drift summary JSON to: {json_path}")
    
    logger.info(f"Data Drift Results: Dataset Drift={drift_summary['dataset_drift_detected']}, "
                f"Drift Share={drift_summary['drift_share']}, "
                f"Drifted Features={drift_summary['drifted_features']}")
    return drift_summary

if __name__ == "__main__":
    run_data_drift_monitoring()
