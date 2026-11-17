"""Model Training, Comparison, and Selection Pipeline.

Trains and evaluates 5 classification models on out-of-time test data:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

Selects best model based on test F1-macro and ROC-AUC.
Saves preprocessor and best model artifacts for deployment and DVC tracking.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.utils.logger import logger
from src.utils.common import read_yaml

FEATURE_COLS_NUM = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "surface_pressure_mean",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "diurnal_temp_range",
    "heat_index",
    "precip_3d_sum",
    "precip_7d_sum",
    "dry_spell_days",
    "wind_gust_factor",
    "water_deficit",
    "month",
    "day_of_year"
]

FEATURE_COLS_CAT = [
    "city",
    "season"
]

TARGET_COL = "risk_level"

def build_preprocessor() -> ColumnTransformer:
    """Creates scikit-learn ColumnTransformer for numerical scaling and categorical encoding."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, FEATURE_COLS_NUM),
            ("cat", cat_pipeline, FEATURE_COLS_CAT)
        ],
        remainder="drop"
    )
    return preprocessor

def get_candidate_models(random_state: int = 42) -> dict:
    """Returns dictionary of candidate models with specified hyperparameters."""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    }

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = np.nan
        
    return {
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec_macro, 4)),
        "recall_macro": float(round(rec_macro, 4)),
        "f1_macro": float(round(f1_macro, 4)),
        "f1_weighted": float(round(f1_weighted, 4)),
        "roc_auc_macro_ovr": float(round(roc_auc, 4))
    }

def train_and_compare_models(train_path: str = "data/processed/train.csv",
                             test_path: str = "data/processed/test.csv",
                             models_dir: str = "models",
                             metrics_dir: str = "metrics") -> dict:
    """Executes end-to-end model training, evaluation, comparison, and best model selection."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading processed train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_raw = train_df[FEATURE_COLS_NUM + FEATURE_COLS_CAT]
    y_train = train_df[TARGET_COL].values
    
    X_test_raw = test_df[FEATURE_COLS_NUM + FEATURE_COLS_CAT]
    y_test = test_df[TARGET_COL].values
    
    logger.info(f"Training set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")
    
    # 1. Fit preprocessor
    logger.info("Fitting feature preprocessor on training data...")
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")
    
    # 2. Train and evaluate all candidates
    models = get_candidate_models()
    results = []
    trained_estimators = {}
    test_predictions = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_estimators[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        test_predictions[name] = (y_pred, y_prob)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["model_name"] = name
        results.append(metrics)
        logger.info(f"{name} Results: Acc={metrics['accuracy']:.4f}, F1-Macro={metrics['f1_macro']:.4f}, ROC-AUC={metrics['roc_auc_macro_ovr']:.4f}")
        
    results_df = pd.DataFrame(results).sort_values(
        by=["f1_macro", "roc_auc_macro_ovr"], ascending=[False, False]
    ).reset_index(drop=True)
    comparison_csv_path = os.path.join(metrics_dir, "model_comparison.csv")
    results_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Model comparison table saved to {comparison_csv_path}")
    
    # 3. Select Best Model
    best_row = results_df.iloc[0]
    best_model_name = best_row["model_name"]
    best_estimator = trained_estimators[best_model_name]
    best_pred, best_prob = test_predictions[best_model_name]
    
    logger.info(f"BEST MODEL SELECTED: {best_model_name} (F1-Macro: {best_row['f1_macro']:.4f})")
    
    # Build complete inference pipeline with preprocessor + best estimator
    inference_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_estimator)
    ])
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(inference_pipeline, best_model_path)
    logger.info(f"Saved complete inference pipeline to {best_model_path}")
    
    # 4. Generate Confusion Matrix for Best Model
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low (0)", "Moderate (1)", "High (2)"],
                yticklabels=["Low (0)", "Moderate (1)", "High (2)"])
    plt.title(f"Confusion Matrix: Best Model ({best_model_name})", fontsize=13, weight="bold")
    plt.xlabel("Predicted Risk Level", fontsize=11)
    plt.ylabel("Actual Risk Level", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(metrics_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    
    # 5. Save detailed scores JSON
    scores = {
        "best_model": best_model_name,
        "selection_metric": "f1_macro",
        "test_metrics": {
            "accuracy": best_row["accuracy"],
            "precision_macro": best_row["precision_macro"],
            "recall_macro": best_row["recall_macro"],
            "f1_macro": best_row["f1_macro"],
            "f1_weighted": best_row["f1_weighted"],
            "roc_auc_macro_ovr": best_row["roc_auc_macro_ovr"]
        },
        "all_models_comparison": results_df.to_dict(orient="records"),
        "classification_report": classification_report(
            y_test, best_pred, target_names=["Low Risk (0)", "Moderate Risk (1)", "High Risk (2)"], output_dict=True
        ),
        "confusion_matrix": cm.tolist()
    }
    scores_path = os.path.join(metrics_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Evaluation metrics JSON saved to {scores_path}")
    
    return scores

if __name__ == "__main__":
    train_and_compare_models()
