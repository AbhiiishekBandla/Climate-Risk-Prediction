# 🌍 Climate Risk Prediction System

An end-to-end production-grade Machine Learning Operations (MLOps) system designed to predict climate-related environmental risks (extreme heatwaves, flooding rainfalls, severe windstorms, and droughts) using real historical multi-variate climate observations across Indian climate zones.

---

## 📌 Project Architecture

```
Climate-Risk-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yaml               # GitHub Actions CI workflow
├── .venv/                        # Dedicated Python 3.11 virtual environment
├── config/
│   ├── config.yaml               # Central paths, features & drift thresholds
│   └── params.yaml               # Model hyperparameters & training configs
├── data/
│   ├── raw/                      # Raw historical climate dataset (DVC tracked)
│   ├── processed/                # Scaled & train/test splits (DVC tracked)
│   ├── reference/                # Baseline data for drift detection
│   └── current/                  # Production inference data for monitoring
├── models/                       # Trained models & scalers (DVC tracked)
├── metrics/                      # Evaluation scores, confusion matrices, comparison tables
├── notebooks/                    # Exploratory Data Analysis & experiments
├── reports/                      # EDA reports, statistics & figures
│   ├── figures/                  # Correlation, thermal, rainfall, and risk plots
│   └── eda_statistics.csv        # Multi-variate descriptive statistics
├── src/                          # Modular production source code
│   ├── data/
│   │   └── load_data.py          # Data ingestion and provenance tracking
│   ├── features/
│   │   ├── build_features.py     # Meteorological feature engineering & temporal split
│   │   └── eda.py                # EDA and visualization generation
│   ├── models/
│   │   ├── train.py              # Candidate training, evaluation & model selection
│   │   └── predict.py            # Standalone single & batch inference pipeline
│   ├── monitoring/               # Evidently AI data drift & SHAP explainability
│   ├── pipelines/                # Automated orchestrator pipelines
│   └── utils/
│       ├── common.py             # YAML reader and directory utilities
│       └── logger.py             # Centralized application logging
├── app/                          # Interactive Streamlit dashboard
├── tests/                        # Automated unit & integration test suite
├── .gitignore                    # Git exclusions
├── .dvcignore                    # DVC exclusions
├── requirements.txt              # Core project dependencies
└── README.md                     # Documentation
```

---

## 📊 Dataset Information & Provenance

* **Dataset Name:** Indian Historical Multi-Zone Meteorological Dataset
* **Source:** Open-Meteo Historical Weather Archive API
* **Underlying Observation Sources:** European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 Reanalysis & IMD/WMO Surface Station Network.
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Temporal Coverage:** January 1, 2015 to December 31, 2023 (9 full consecutive years).
* **Sample Size:** 13,148 daily multi-variate observations across 4 major Indian climate zones:
  1. **New Delhi** (North: Composite / Semi-Arid)
  2. **Mumbai** (West: Coastal / Tropical Wet)
  3. **Kolkata** (East: Gangetic Plain / Humid Subtropical)
  4. **Chennai** (South: Coromandel Coast / Tropical Maritime)

### Features Extracted & Engineered
* **Thermal:** Mean, Max, Min Temperatures (°C), Diurnal Temperature Range ($T_{max} - T_{min}$), Apparent Temperature (°C), Heat Index (Rothfusz equation).
* **Hydrological:** Daily Precipitation Sum (mm), 3-Day & 7-Day Cumulative Rainfall (mm), Evapotranspiration ($ET_0$), Water Deficit ($ET_0 - Rain$), Dry Spell Consecutive Days.
* **Dynamic & Atmospheric:** Surface Pressure (hPa), Max Wind Speed (km/h), Max Wind Gusts (km/h), Wind Gust Factor ($Gust / Speed$).
* **Calendar & Regional:** Month, Day of Year, City Zone, IMD Meteorological Seasons (Winter, Summer/Pre-Monsoon, Monsoon, Post-Monsoon).

### Target Variable: Meteorological Proxy Risk Level
> **Note on Risk Labeling Methodology:** In accordance with rigorous scientific practices, risk categories are derived directly from published operational meteorological thresholds of the **India Meteorological Department (IMD)** and **National Disaster Management Authority (NDMA)**. These represent objective **meteorological hazard severity proxies**:
* **Class 0 (Low Risk / Normal):** Standard benign weather conditions.
* **Class 1 (Moderate Risk / Advisory):** Moderate thermal stress (Heat Index $\ge 38^\circ\text{C}$ or $T_{max} \ge 39^\circ\text{C}$), moderate rainfall ($30\text{--}64\text{ mm/day}$ or 3-day sum $\ge 60\text{ mm}$), or elevated wind gusts ($\ge 40\text{ km/h}$).
* **Class 2 (High Risk / Severe Weather):** Extreme Heatwave ($T_{max} \ge 42^\circ\text{C}$ or Heat Index $\ge 45^\circ\text{C}$), Heavy Flood-Triggering Rainfall ($Rain \ge 64.5\text{ mm/day}$ [IMD Heavy Rainfall criterion] or 3-day sum $\ge 120\text{ mm}$), or Gale/Severe Storm Gusts ($\ge 60\text{ km/h}$).

---

## ⚙️ Data Preprocessing & Train/Test Strategy

1. **Missing Values & Imputation:** Forward and backward fill per city time-series, combined with Median/Mode imputation in the production scikit-learn `ColumnTransformer`.
2. **Feature Transformation:** Numerical features scaled with `StandardScaler()`; geographical and seasonal categorical features encoded with `OneHotEncoder(handle_unknown='ignore')`.
3. **Temporal Split (Zero Data Leakage):**
   * **Training Set:** 2015-01-01 to 2021-12-31 (10,228 rows, 7 historical years).
   * **Test Set (Out-of-Time Evaluation):** 2022-01-01 to 2023-12-31 (2,920 rows, 2 future years).
   * Strict avoidance of random splitting prevents time-series data leakage.

---

## 📈 Model Comparison & Benchmark Results

All models were evaluated on the held-out out-of-time test dataset (`2022–2023`):

| Model Name | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | ROC-AUC (OVR Macro) |
|---|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | **99.90%** | **0.9990** | **0.9990** | **0.9990** | **0.9990** | **0.9998** |
| Decision Tree | 99.90% | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9992 |
| Random Forest | 99.45% | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9999 |
| XGBoost | 99.25% | 0.9925 | 0.9923 | 0.9924 | 0.9925 | 0.9999 |
| Logistic Regression | 88.39% | 0.8834 | 0.8843 | 0.8835 | 0.8834 | 0.9740 |

### Best Model Selection & Justification: **Gradient Boosting Classifier**
* **Primary Metric:** Achieved an **F1-Macro of 0.9990** and **Accuracy of 99.90%** on future unseen climate data.
* **ROC-AUC & Calibration:** Outperforms single Decision Trees with an ROC-AUC of **0.9998**, providing smooth, calibrated class probabilities essential for probabilistic risk thresholds.
* **Generalization:** Ensemble boosting mitigates single-tree variance and step-function overfitting on edge cases.
* **Serialized Artifact:** Saved as a complete deployable pipeline inside `models/best_model.joblib`.

---

## 🧪 Testing & Verification

15 automated unit and integration tests passing in pytest:
```bash
pytest tests/ -v
```
* `test_data_pipeline.py`: Validates dataset schema, temporal boundaries, heat index calculations, IMD seasonal mapping, and proxy risk labeling.
* `test_model_pipeline.py`: Validates model comparison tables, metric calculations, single & batch inference, and artifact loading.
* `test_setup.py`: Validates configuration files, paths, and directory structures.
* `test_imports.py`: Verifies dependencies (NumPy, Pandas, Scikit-learn, XGBoost, SHAP, MLflow, DVC, DagsHub, Evidently, Streamlit).

---

## 📋 Project Phases Roadmap

- [x] **Phase 1: Project Setup**
- [x] **Phase 2: Data Science**
  - [x] Real Indian multi-zone climate data collection (Open-Meteo ERA5 / IMD stations)
  - [x] Clean data preprocessing & temporal train/test split
  - [x] Exploratory Data Analysis & visual reporting (`reports/figures/`)
  - [x] Meteorological feature engineering (Heat Index, multi-day rainfall, dry spells)
  - [x] Candidate model training & benchmark comparison (5 models)
  - [x] Best model selection (`Gradient Boosting`) & pipeline serialization (`models/best_model.joblib`)
  - [x] Comprehensive pytest test suite & documentation
- [ ] **Phase 3: MLOps Pipeline**
  - DVC data & model versioning
  - MLflow remote experiment tracking with DagsHub
  - Automated repeatable training pipeline (`dvc.yaml`)
- [ ] **Phase 4: Monitoring & Explainability**
  - Evidently AI data drift detection report & alerts
  - SHAP global and local feature importance
  - Target drift & model decay monitoring
- [ ] **Phase 5: Deployment & Dashboard**
  - Streamlit climate risk prediction web application
  - Real-time batch and single-instance prediction interface
  - Drift & performance metrics visualizer
