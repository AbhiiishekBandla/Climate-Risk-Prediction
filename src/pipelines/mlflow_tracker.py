"""MLflow and DagsHub Experiment Tracking Pipeline for Climate Risk Prediction.

Logs hyperparameters, dataset versions, evaluation metrics, confusion matrices,
and model artifacts to MLflow (locally in mlruns or remotely on DagsHub).
"""

import os
import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import logger
from src.utils.common import read_yaml

load_dotenv()

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

EXPERIMENT_NAME = "Climate_Risk_Prediction"

def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a Git repo."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"

def get_dvc_hash(dvc_file_path: str) -> str:
    """Reads md5 hash from a .dvc file."""
    try:
        with open(dvc_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    except Exception:
        pass
    return "unknown"

def setup_tracking_uri() -> str:
    """Configures MLflow tracking URI based on environment variables or defaults to local mlruns.
    
    If DAGSHUB_USER_TOKEN and DAGSHUB_USERNAME / REPO are set, integrates with DagsHub MLflow server.
    Otherwise defaults to local mlruns directory.
    """
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    dagshub_user = os.getenv("DAGSHUB_USERNAME", "abhishekbandla07")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME", "Climate-Risk-Prediction")
    explicit_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if explicit_uri:
        logger.info(f"Using explicitly configured MLFLOW_TRACKING_URI: {explicit_uri}")
        mlflow.set_tracking_uri(explicit_uri)
        return explicit_uri
        
    if dagshub_token:
        remote_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        logger.info(f"Connecting to DagsHub MLflow remote server: {remote_uri}")
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(remote_uri)
        return remote_uri
        
    # Local database backend (recommended for MLflow 3.x)
    local_uri = "sqlite:///mlflow.db"
    logger.info(f"Using local SQLite MLflow tracking backend: {local_uri}")
    mlflow.set_tracking_uri(local_uri)
    return local_uri

def log_experiment_to_mlflow(config_path: str = "config/config.yaml",
                             params_path: str = "config/params.yaml",
                             scores_path: str = "metrics/scores.json",
                             model_path: str = "models/best_model.joblib") -> str:
    """Runs complete MLflow logging workflow."""
    tracking_uri = setup_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
        
    best_model_name = scores["best_model"]
    test_metrics = scores["test_metrics"]
    git_hash = get_git_commit_hash()
    data_dvc_hash = get_dvc_hash("data/raw/climate_risk_data.csv.dvc")
    
    run_name = f"{best_model_name}_Pipeline_Run"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run: {run_id} ({run_name})")
        
        # 1. Tags
        mlflow.set_tags({
            "project": "Climate-Risk-Prediction",
            "model_type": best_model_name,
            "git_commit": git_hash,
            "data_dvc_md5": data_dvc_hash,
            "tracking_uri": tracking_uri,
            "framework": "scikit-learn"
        })
        
        # 2. Parameters
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("test_split_strategy", "temporal_2015_2021_train__2022_2023_test")
        mlflow.log_param("target_variable", "risk_level")
        mlflow.log_param("num_features", len(config["features"]["numerical"]))
        mlflow.log_param("cat_features", len(config["features"]["categorical"]))
        
        # Model specific hyperparameters
        if "gradient_boosting" in params.get("models", {}):
            for k, v in params["models"]["gradient_boosting"].items():
                mlflow.log_param(f"gb_{k}", v)
                
        # 3. Metrics
        for metric_name, val in test_metrics.items():
            if val is not None:
                mlflow.log_metric(metric_name, float(val))
                logger.info(f"Logged metric {metric_name} = {val}")
                
        # 4. Artifacts
        artifacts_to_log = [
            scores_path,
            "metrics/model_comparison.csv",
            "metrics/confusion_matrix.png",
            "reports/figures/correlation_heatmap.png",
            "reports/figures/temperature_trends.png",
            "reports/figures/precipitation_patterns.png",
            "reports/figures/risk_distribution_by_city.png",
            "reports/figures/diurnal_and_heat_index.png",
            "reports/eda_summary.json",
            "config/config.yaml",
            "config/params.yaml"
        ]
        
        for art in artifacts_to_log:
            if os.path.exists(art):
                mlflow.log_artifact(art, artifact_path="evaluation_reports")
                logger.info(f"Logged artifact: {art}")
                
        # Log model artifact with MLflow
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")
            logger.info(f"Logged model binary: {model_path}")
            
        logger.info(f"MLflow Run completed successfully! Run ID: {run_id}")
        return run_id

if __name__ == "__main__":
    log_experiment_to_mlflow()
