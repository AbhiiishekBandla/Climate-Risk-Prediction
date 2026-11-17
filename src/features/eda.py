"""Exploratory Data Analysis (EDA) Module for Climate Risk Prediction.

Generates:
1. Summary descriptive statistics.
2. Inter-feature correlation matrix & heatmap.
3. Temperature & Heat Index seasonal trends.
4. Monsoon precipitation & extreme rainfall patterns.
5. Climate risk class distributions across geographical zones.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import logger

def run_eda(input_path: str = "data/processed/train.csv", reports_dir: str = "reports") -> None:
    """Performs comprehensive EDA and exports figures and statistical summaries."""
    logger.info(f"Running EDA on {input_path}...")
    df = pd.read_csv(input_path)
    
    figures_dir = Path(reports_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descriptive Statistics
    logger.info("Generating descriptive statistics...")
    numeric_cols = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_mean", "precipitation_sum", "precip_3d_sum",
        "windspeed_10m_max", "windgusts_10m_max", "surface_pressure_mean",
        "heat_index", "diurnal_temp_range", "dry_spell_days"
    ]
    desc_stats = df[numeric_cols].describe().round(2)
    stats_csv_path = Path(reports_dir) / "eda_statistics.csv"
    desc_stats.to_csv(stats_csv_path)
    logger.info(f"Descriptive statistics saved to {stats_csv_path}")
    
    # Summary JSON
    eda_summary = {
        "total_training_records": len(df),
        "period": f"{df['year'].min()} to {df['year'].max()}",
        "cities": df["city"].unique().tolist(),
        "temperature_mean_c": float(df["temperature_2m_mean"].mean().round(2)),
        "temperature_max_c": float(df["temperature_2m_max"].max().round(2)),
        "temperature_min_c": float(df["temperature_2m_min"].min().round(2)),
        "annual_avg_precip_mm": float((df.groupby(["city", "year"])["precipitation_sum"].sum().mean()).round(2)),
        "max_single_day_rain_mm": float(df["precipitation_sum"].max().round(2)),
        "max_wind_gust_kmh": float(df["windgusts_10m_max"].max().round(2)),
        "risk_class_proportions": df["risk_level"].value_counts(normalize=True).round(3).to_dict()
    }
    with open(Path(reports_dir) / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
        
    # Styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    
    # 2. Correlation Heatmap
    logger.info("Plotting correlation matrix...")
    corr_features = numeric_cols + ["risk_level"]
    corr_matrix = df[corr_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Meteorological Features Correlation Matrix", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    corr_path = figures_dir / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=200)
    plt.close()
    
    # 3. Temperature Trends across Indian Cities
    logger.info("Plotting temperature patterns...")
    plt.figure(figsize=(14, 6))
    df["time"] = pd.to_datetime(df["time"])
    monthly_temp = df.groupby(["city", df["time"].dt.month])["temperature_2m_max"].mean().unstack(level=0)
    monthly_temp.plot(marker="o", linewidth=2.5, ax=plt.gca())
    plt.title("Monthly Mean Maximum Temperature Pattern by Indian Metro Zone", fontsize=14, weight="bold")
    plt.xlabel("Month of Year", fontsize=12)
    plt.ylabel("Mean Max Temperature (°C)", fontsize=12)
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.legend(title="City Zone")
    plt.tight_layout()
    temp_path = figures_dir / "temperature_trends.png"
    plt.savefig(temp_path, dpi=200)
    plt.close()
    
    # 4. Precipitation Patterns by Season and City
    logger.info("Plotting rainfall patterns...")
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="season", y="precipitation_sum", hue="city", showfliers=False)
    plt.title("Seasonal Rainfall Distribution (Excluding Extreme Outliers) by Metro Hub", fontsize=14, weight="bold")
    plt.xlabel("IMD Season", fontsize=12)
    plt.ylabel("Daily Precipitation (mm)", fontsize=12)
    plt.legend(title="City Zone")
    plt.tight_layout()
    rain_path = figures_dir / "precipitation_patterns.png"
    plt.savefig(rain_path, dpi=200)
    plt.close()
    
    # 5. Risk Distribution by City
    logger.info("Plotting risk distributions...")
    plt.figure(figsize=(10, 6))
    risk_ct = pd.crosstab(df["city"], df["risk_level"], normalize="index") * 100
    risk_ct.columns = ["Low (0)", "Moderate (1)", "High (2)"]
    risk_ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#f39c12", "#e74c3c"], ax=plt.gca())
    plt.title("Climate Risk Level Distribution by City (%)", fontsize=14, weight="bold")
    plt.xlabel("Metropolitan Hub", fontsize=12)
    plt.ylabel("Percentage of Total Days (%)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Risk Level", frameon=True)
    plt.tight_layout()
    risk_path = figures_dir / "risk_distribution_by_city.png"
    plt.savefig(risk_path, dpi=200)
    plt.close()
    
    # 6. Heat Index vs Diurnal Temperature Range
    logger.info("Plotting thermal dynamics...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df.sample(min(2000, len(df)), random_state=42),
        x="diurnal_temp_range", y="heat_index",
        hue="risk_level", palette=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.6, s=40
    )
    plt.title("Heat Index vs. Diurnal Temperature Range by Climate Risk Level", fontsize=14, weight="bold")
    plt.xlabel("Diurnal Temperature Range (°C)", fontsize=12)
    plt.ylabel("Apparent Heat Index (°C)", fontsize=12)
    plt.axhline(42.0, color="red", linestyle="--", alpha=0.7, label="IMD Heatwave Threshold (42°C)")
    plt.legend(title="Risk Level")
    plt.tight_layout()
    thermal_path = figures_dir / "diurnal_and_heat_index.png"
    plt.savefig(thermal_path, dpi=200)
    plt.close()
    
    logger.info(f"EDA successfully executed! All figures exported to {figures_dir}")

if __name__ == "__main__":
    run_eda()
