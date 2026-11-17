"""Unified Monitoring and Explainability Orchestrator Pipeline.

Executes:
1. Feature-level Data Drift Detection (Evidently AI)
2. Target & Prediction Drift Analysis (Evidently AI & Chi-Square)
3. Model Performance & Degradation Monitoring
4. SHAP Model Explainability & Local Incident Explanations
5. Timestamped Historical Logging for continuous monitoring audits
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import logger
from src.monitoring.data_drift import run_data_drift_monitoring
from src.monitoring.target_drift import run_target_drift_monitoring
from src.monitoring.performance_monitor import run_performance_monitoring
from src.monitoring.shap_explainer import generate_shap_explanations

def execute_monitoring_pipeline(reports_dir: str = "reports") -> dict:
    """Orchestrates end-to-end monitoring and explainability workflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== Starting Unified Climate Risk Monitoring Pipeline [{timestamp}] ===")
    
    monitoring_dir = os.path.join(reports_dir, "monitoring")
    explainability_dir = os.path.join(reports_dir, "explainability")
    history_dir = os.path.join(monitoring_dir, "history", timestamp)
    
    Path(monitoring_dir).mkdir(parents=True, exist_ok=True)
    Path(explainability_dir).mkdir(parents=True, exist_ok=True)
    Path(history_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Data Drift
    logger.info("\n--- Stage 1: Data Drift Detection ---")
    data_drift_res = run_data_drift_monitoring(output_dir=monitoring_dir)
    
    # 2. Target Drift
    logger.info("\n--- Stage 2: Target & Prediction Drift Detection ---")
    target_drift_res = run_target_drift_monitoring(output_dir=monitoring_dir)
    
    # 3. Model Performance Monitoring
    logger.info("\n--- Stage 3: Ground-Truth Performance Monitoring ---")
    perf_res = run_performance_monitoring(output_dir=monitoring_dir)
    
    # 4. SHAP Explainability
    logger.info("\n--- Stage 4: SHAP Model Explainability ---")
    shap_res = generate_shap_explanations(output_dir=explainability_dir)
    
    # Compile Master Dashboard Summary
    dashboard_summary = {
        "pipeline_run_timestamp": timestamp,
        "overall_system_status": "HEALTHY" if perf_res["status"] == "HEALTHY" and not data_drift_res["dataset_drift_detected"] else "ALERT",
        "data_drift": {
            "dataset_drift_detected": data_drift_res["dataset_drift_detected"],
            "drift_share": data_drift_res["drift_share"],
            "drifted_features": data_drift_res["drifted_features"]
        },
        "target_drift": {
            "prediction_drift_detected": target_drift_res["prediction_drift"]["drift_detected"],
            "actual_target_drift_detected": target_drift_res.get("actual_target_drift", {}).get("drift_detected", "UNAVAILABLE")
        },
        "performance": {
            "status": perf_res["status"],
            "current_f1_macro": perf_res["current_metrics"]["f1_macro"],
            "baseline_f1_macro": perf_res["baseline_metrics"]["f1_macro"],
            "alerts": perf_res["alerts"]
        },
        "explainability": {
            "top_drivers": shap_res["top_climate_risk_drivers"][:5],
            "insights": shap_res["key_insights"]
        },
        "generated_artifacts": [
            os.path.join(monitoring_dir, "data_drift_report.html"),
            os.path.join(monitoring_dir, "target_drift_report.html"),
            os.path.join(monitoring_dir, "model_performance_report.html"),
            os.path.join(explainability_dir, "shap_summary_beeswarm.png"),
            os.path.join(explainability_dir, "shap_feature_importance_bar.png"),
            os.path.join(explainability_dir, "shap_local_heatwave_explanation.png"),
            os.path.join(explainability_dir, "shap_local_flood_explanation.png")
        ]
    }
    
    dashboard_path = os.path.join(monitoring_dir, "latest_monitoring_dashboard.json")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_summary, f, indent=2)
    logger.info(f"Saved master monitoring dashboard JSON to: {dashboard_path}")
    
    # Save timestamped audit copy
    history_dashboard_path = os.path.join(history_dir, "monitoring_summary.json")
    with open(history_dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_summary, f, indent=2)
    logger.info(f"Saved historical audit snapshot to: {history_dir}")
    
    logger.info(f"=== Unified Monitoring Pipeline Completed Successfully! Status: {dashboard_summary['overall_system_status']} ===")
    return dashboard_summary

if __name__ == "__main__":
    execute_monitoring_pipeline()
