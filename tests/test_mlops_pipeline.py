"""Integration Tests for MLOps Pipeline (DVC, MLflow, and DagsHub configuration)."""

import os
import json
import pytest
import yaml
import mlflow
from mlflow.tracking import MlflowClient

def test_dvc_artifacts_exist():
    assert os.path.exists(".dvc"), ".dvc folder missing"
    assert os.path.exists("dvc.yaml"), "dvc.yaml pipeline definition missing"
    assert os.path.exists("dvc.lock"), "dvc.lock lockfile missing"
    assert os.path.exists("data/raw/climate_risk_data.csv.dvc"), "Raw data .dvc file missing"

def test_dvc_yaml_structure():
    with open("dvc.yaml", "r", encoding="utf-8") as f:
        dvc_config = yaml.safe_load(f)
        
    stages = dvc_config.get("stages", {})
    expected_stages = [
        "data_validation",
        "feature_engineering",
        "eda",
        "model_training_and_evaluation",
        "mlflow_tracking"
    ]
    for s in expected_stages:
        assert s in stages, f"Stage {s} not found in dvc.yaml"

def test_data_validation_status():
    status_path = "data/raw/validation_status.json"
    assert os.path.exists(status_path), "validation_status.json is missing"
    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)
    assert status.get("validation_passed") is True
    assert status.get("total_records") >= 10000

def test_mlflow_local_tracking_and_metrics():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("Climate_Risk_Prediction")
    assert experiment is not None, "Experiment Climate_Risk_Prediction not found in MLflow"
    
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) >= 1, "No MLflow runs found"
    
    latest_run = runs[0]
    assert latest_run.info.status == "FINISHED", f"Run status is {latest_run.info.status}"
    
    metrics = latest_run.data.metrics
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc_macro_ovr" in metrics
    assert metrics["f1_macro"] > 0.95

def test_env_example_exists():
    assert os.path.exists(".env.example"), ".env.example template missing"
    with open(".env.example", "r", encoding="utf-8") as f:
        content = f.read()
    assert "DAGSHUB_USER_TOKEN" in content
    assert "MLFLOW_TRACKING_URI" in content
