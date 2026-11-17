"""Model Performance Monitoring and Degradation Detection Module.

Evaluates model on current production batches with genuine ground-truth labels.
Compares performance against the baseline reference metrics.
Triggers automated degradation alerts when accuracy or F1-macro drop below thresholds.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from evidently import Report, DataDefinition, Dataset, MulticlassClassification
from evidently.presets import ClassificationPreset

from src.utils.logger import logger
from src.utils.common import read_yaml

DEFAULT_MODEL_PATH = "models/best_model.joblib"
BASELINE_SCORES_PATH = "metrics/scores.json"
CURRENT_DATA_PATH = "data/current/current_data.csv"
OUTPUT_DIR = "reports/monitoring"

# Alert thresholds
MAX_F1_DROP_TOLERANCE = 0.05
MIN_ACCEPTABLE_F1 = 0.90

def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculates ground-truth performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
        
    return {
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec, 4)),
        "recall_macro": float(round(rec, 4)),
        "f1_macro": float(round(f1, 4)),
        "f1_weighted": float(round(f1_w, 4)),
        "roc_auc_macro_ovr": float(round(auc, 4))
    }

def run_performance_monitoring(model_path: str = DEFAULT_MODEL_PATH,
                               current_data_path: str = CURRENT_DATA_PATH,
                               baseline_scores_path: str = BASELINE_SCORES_PATH,
                               output_dir: str = OUTPUT_DIR) -> dict:
    """Runs performance monitoring comparing current ground-truth metrics against baseline."""
    logger.info("Executing Model Performance Monitoring...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pipeline = joblib.load(model_path)
    curr_df = pd.read_csv(current_data_path)
    
    # Ground truth validation check
    if "risk_level" not in curr_df.columns or curr_df["risk_level"].isna().all():
        raise ValueError("Cannot calculate genuine performance metrics: Ground-truth target 'risk_level' is missing.")
        
    y_true = curr_df["risk_level"].values
    y_pred = pipeline.predict(curr_df)
    y_prob = pipeline.predict_proba(curr_df)
    
    # Calculate current metrics
    current_metrics = evaluate_performance(y_true, y_pred, y_prob)
    
    # Load baseline metrics
    with open(baseline_scores_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    baseline_metrics = baseline_data["test_metrics"]
    
    # Compute performance deltas
    deltas = {}
    for metric_name, curr_val in current_metrics.items():
        base_val = baseline_metrics.get(metric_name)
        if base_val is not None and not np.isnan(curr_val):
            deltas[metric_name] = float(round(curr_val - base_val, 4))
            
    # Check degradation triggers
    f1_drop = deltas.get("f1_macro", 0.0)
    alerts = []
    
    if f1_drop < -MAX_F1_DROP_TOLERANCE:
        alerts.append({
            "severity": "CRITICAL",
            "message": f"F1-Macro dropped by {abs(f1_drop):.4f} (exceeds tolerance of {MAX_F1_DROP_TOLERANCE})"
        })
    if current_metrics["f1_macro"] < MIN_ACCEPTABLE_F1:
        alerts.append({
            "severity": "WARNING",
            "message": f"Current F1-Macro ({current_metrics['f1_macro']:.4f}) is below acceptable floor ({MIN_ACCEPTABLE_F1})"
        })
        
    performance_status = "HEALTHY" if len(alerts) == 0 else "DEGRADED"
    
    # Generate Evidently Classification Quality Report
    eval_df = curr_df.copy()
    eval_df["target"] = y_true
    eval_df["prediction"] = y_pred
    
    ref_df = pd.read_csv("data/reference/reference_data.csv")
    ref_df["target"] = ref_df["risk_level"]
    ref_df["prediction"] = pipeline.predict(ref_df)
    
    classification_schema = DataDefinition(
        classification=[
            MulticlassClassification(
                target="target",
                prediction_labels="prediction"
            )
        ]
    )
    
    ref_dataset = Dataset.from_pandas(ref_df[["target", "prediction"]], data_definition=classification_schema)
    curr_dataset = Dataset.from_pandas(eval_df[["target", "prediction"]], data_definition=classification_schema)
    
    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)
    
    html_path = os.path.join(output_dir, "model_performance_report.html")
    snapshot.save_html(html_path)
    logger.info(f"Saved interactive HTML model performance report to: {html_path}")
    
    monitoring_summary = {
        "status": performance_status,
        "sample_size": len(curr_df),
        "current_metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
        "performance_deltas": deltas,
        "alerts": alerts,
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    json_path = os.path.join(output_dir, "model_performance_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(monitoring_summary, f, indent=2)
    logger.info(f"Saved model performance summary JSON to: {json_path}")
    
    logger.info(f"Performance Monitoring Result: Status={performance_status}, "
                f"Current F1={current_metrics['f1_macro']}, Delta={deltas.get('f1_macro', 0.0)}")
    return monitoring_summary

if __name__ == "__main__":
    run_performance_monitoring()
