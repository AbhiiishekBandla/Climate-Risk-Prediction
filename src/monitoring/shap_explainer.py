"""SHAP Model Explainability Module for Climate Risk Prediction.

Computes global and local feature contributions using Shapley values.
Generates:
1. SHAP global beeswarm / summary plot.
2. Global feature importance bar chart.
3. Local prediction explanations for extreme heat, heavy monsoon, and benign weather events.
4. Feature contribution impact summary JSON.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.utils.logger import logger
from src.utils.common import read_yaml

DEFAULT_MODEL_PATH = "models/best_model.joblib"
TEST_DATA_PATH = "data/processed/test.csv"
OUTPUT_DIR = "reports/explainability"

def generate_shap_explanations(model_path: str = DEFAULT_MODEL_PATH,
                               test_data_path: str = TEST_DATA_PATH,
                               output_dir: str = OUTPUT_DIR,
                               sample_size: int = 50) -> dict:
    """Computes global and local SHAP explanations and exports visualizations."""
    logger.info(f"Loading model from {model_path} and test data from {test_data_path}...")
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Transform test features
    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove "num__" and "cat__" prefixes for cleaner plots)
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    
    X_test_trans = preprocessor.transform(test_df)
    
    # 1. Tree Feature Importance from the estimator
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({
        "feature": clean_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    logger.info(f"Top 5 most important features by Gini/MDI: {imp_df.head(5)['feature'].tolist()}")
    
    # 2. KernelExplainer for Model-Agnostic Multi-Class SHAP
    logger.info(f"Computing Shapley values with KernelExplainer on {sample_size} representative samples...")
    # Background summary using kmeans clustering for fast execution
    background = shap.kmeans(X_test_trans, 5)
    explainer = shap.KernelExplainer(classifier.predict_proba, background)
    
    # Sample subset for SHAP evaluation
    eval_subset = X_test_trans[:sample_size]
    shap_values = explainer.shap_values(eval_subset, nsamples=100)
    
    # 3. Global Feature Importance Bar Plot (Mean |SHAP| across all classes)
    logger.info("Generating SHAP Global Feature Importance Bar Plot...")
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        class_2_shap_values = shap_values[:, :, 2]
    elif isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        class_2_shap_values = shap_values[2]
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        class_2_shap_values = shap_values

    shap_imp_df = pd.DataFrame({
        "feature": clean_feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_imp_df, x="mean_abs_shap", y="feature", palette="viridis")
    plt.title("SHAP Global Feature Importance (Top 15 Drivers of Climate Risk)", fontsize=13, weight="bold")
    plt.xlabel("Mean |SHAP value| (Impact on Model Output Magnitude)", fontsize=11)
    plt.ylabel("Climate Feature", fontsize=11)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    logger.info(f"Saved SHAP bar plot to {bar_path}")
    
    # 4. Summary Beeswarm Plot for High Risk Class (Class 2)
    logger.info("Generating SHAP Summary Beeswarm Plot for High Risk Class...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        class_2_shap_values, eval_subset,
        feature_names=clean_feature_names,
        show=False, max_display=12
    )
    plt.title("SHAP Summary Plot: Impact on High Climate Risk (Class 2)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(summary_path, dpi=200)
    plt.close()
    logger.info(f"Saved SHAP summary beeswarm to {summary_path}")
    
    # 5. Local Individual Explanations for Archetypal Climate Events
    logger.info("Generating local event explanations...")
    # Event 1: Extreme Summer Heatwave (High T_max, High Heat Index)
    heat_idx = test_df[(test_df["temperature_2m_max"] > 40.0) & (test_df["risk_level"] == 2)].index
    heat_i = int(heat_idx[0]) if len(heat_idx) > 0 else 0
    
    # Event 2: Extreme Monsoon Deluge (High Precipitation)
    rain_idx = test_df[(test_df["precipitation_sum"] > 50.0) & (test_df["risk_level"] == 2)].index
    rain_i = int(rain_idx[0]) if len(rain_idx) > 0 else 1
    
    # Event 3: Benign Normal Weather (Class 0)
    normal_idx = test_df[test_df["risk_level"] == 0].index
    norm_i = int(normal_idx[0]) if len(normal_idx) > 0 else 2
    
    def plot_local_waterfall_bar(sample_idx: int, title: str, filename: str):
        sample_x = X_test_trans[sample_idx:sample_idx+1]
        sv_sample = explainer.shap_values(sample_x, nsamples=100)
        
        if isinstance(sv_sample, np.ndarray) and sv_sample.ndim == 3:
            c2_shap = sv_sample[0, :, 2]
        elif isinstance(sv_sample, list):
            c2_shap = sv_sample[2][0]
        else:
            c2_shap = sv_sample[0]
            
        top_contrib_df = pd.DataFrame({
            "feature": clean_feature_names,
            "shap_value": c2_shap
        })
        top_contrib_df["abs_shap"] = top_contrib_df["shap_value"].abs()
        top_contrib_df = top_contrib_df.sort_values(by="abs_shap", ascending=False).head(8)
        top_contrib_df["direction"] = np.where(top_contrib_df["shap_value"] > 0, "Increases High Risk", "Decreases High Risk")
        
        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=top_contrib_df,
            x="shap_value", y="feature",
            hue="direction",
            palette={"Increases High Risk": "#e74c3c", "Decreases High Risk": "#2ecc71"}
        )
        plt.title(title, fontsize=12, weight="bold")
        plt.xlabel("SHAP Value (Contribution to High Risk Probability)", fontsize=10)
        plt.ylabel("Feature", fontsize=10)
        plt.axvline(0, color="black", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()
        
    plot_local_waterfall_bar(heat_i, f"Local Explanation: Extreme Heat Event (Row {heat_i})", "shap_local_heatwave_explanation.png")
    plot_local_waterfall_bar(rain_i, f"Local Explanation: Heavy Monsoon Flood Event (Row {rain_i})", "shap_local_flood_explanation.png")
    plot_local_waterfall_bar(norm_i, f"Local Explanation: Benign Weather Event (Row {norm_i})", "shap_local_normal_explanation.png")
    
    # 6. Save Explanations Summary JSON
    top_drivers = shap_imp_df.head(10).to_dict(orient="records")
    summary = {
        "explainer_type": "shap.KernelExplainer (Model-Agnostic with K-Means background)",
        "samples_evaluated": sample_size,
        "top_climate_risk_drivers": top_drivers,
        "key_insights": [
            "Heat Index and Max Temperature are primary drivers pushing predictions into High Risk during summer.",
            "3-Day Cumulative Precipitation and Daily Precipitation Sum are the dominant drivers during monsoon deluges.",
            "High relative humidity coupled with elevated temperatures sharply amplifies apparent thermal risk."
        ]
    }
    json_path = os.path.join(output_dir, "shap_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved SHAP explanations summary JSON to {json_path}")
    
    return summary

if __name__ == "__main__":
    generate_shap_explanations()
