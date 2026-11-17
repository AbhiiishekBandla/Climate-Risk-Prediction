"""Integration and Functional Tests for Phase 4: Monitoring and Explainability."""

import os
import json
import pytest

def test_data_drift_reports_exist():
    html_path = "reports/monitoring/data_drift_report.html"
    summary_path = "reports/monitoring/data_drift_summary.json"
    
    assert os.path.exists(html_path), "Data drift HTML report missing"
    assert os.path.getsize(html_path) > 100_000, "Data drift HTML report is unexpectedly small"
    
    assert os.path.exists(summary_path), "Data drift summary JSON missing"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary.get("status") == "success"
    assert summary.get("monitored_features_count") == 14
    assert summary.get("reference_records") == 10228
    assert summary.get("current_records") == 2920
    assert "dataset_drift_detected" in summary

def test_target_drift_reports_and_distinction():
    html_path = "reports/monitoring/target_drift_report.html"
    summary_path = "reports/monitoring/target_drift_summary.json"
    
    assert os.path.exists(html_path), "Target drift HTML report missing"
    assert os.path.getsize(html_path) > 100_000, "Target drift HTML report is unexpectedly small"
    
    assert os.path.exists(summary_path), "Target drift summary JSON missing"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary.get("status") == "success"
    assert "prediction_drift" in summary
    assert "actual_target_drift" in summary
    
    # Verify distinction note is present
    actual_drift = summary["actual_target_drift"]
    assert "distinction_note" in actual_drift
    assert "Target drift" in actual_drift["distinction_note"]
    assert "Prediction drift" in actual_drift["distinction_note"]
    
    # Verify distribution keys
    pred_drift = summary["prediction_drift"]
    assert "reference_predicted_distribution" in pred_drift
    assert "current_predicted_distribution" in pred_drift

def test_model_performance_monitor():
    html_path = "reports/monitoring/model_performance_report.html"
    summary_path = "reports/monitoring/model_performance_summary.json"
    
    assert os.path.exists(html_path), "Model performance HTML report missing"
    assert os.path.getsize(html_path) > 100_000, "Model performance HTML report is unexpectedly small"
    
    assert os.path.exists(summary_path), "Model performance summary JSON missing"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary.get("status") in ["HEALTHY", "DEGRADED"]
    assert summary.get("sample_size") == 2920
    
    metrics = summary.get("current_metrics", {})
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert metrics["f1_macro"] >= 0.95
    
    # Verify deltas against baseline
    assert "performance_deltas" in summary
    assert "alerts" in summary

def test_shap_explainability_artifacts():
    summary_path = "reports/explainability/shap_summary.json"
    assert os.path.exists(summary_path), "SHAP summary JSON missing"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert "explainer_type" in summary
    assert "KernelExplainer" in summary["explainer_type"]
    assert len(summary.get("top_climate_risk_drivers", [])) >= 5
    assert len(summary.get("key_insights", [])) >= 1
    
    # Verify all expected plots exist
    plots = [
        "reports/explainability/shap_summary_beeswarm.png",
        "reports/explainability/shap_feature_importance_bar.png",
        "reports/explainability/shap_local_heatwave_explanation.png",
        "reports/explainability/shap_local_flood_explanation.png",
        "reports/explainability/shap_local_normal_explanation.png"
    ]
    for plot_path in plots:
        assert os.path.exists(plot_path), f"Explainability plot {plot_path} missing"
        assert os.path.getsize(plot_path) > 10_000, f"Plot {plot_path} is unexpectedly small"

def test_master_monitoring_dashboard():
    dashboard_path = "reports/monitoring/latest_monitoring_dashboard.json"
    assert os.path.exists(dashboard_path), "Latest monitoring dashboard JSON missing"
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        dash = json.load(f)
        
    assert "pipeline_run_timestamp" in dash
    assert dash.get("overall_system_status") in ["HEALTHY", "DEGRADED"]
    assert "data_drift" in dash
    assert "target_drift" in dash
    assert "performance" in dash
    assert "explainability" in dash
    assert len(dash.get("generated_artifacts", [])) >= 5
