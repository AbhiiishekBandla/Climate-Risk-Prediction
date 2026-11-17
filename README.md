# 🌍 Climate Risk Prediction System

An end-to-end production-grade Machine Learning Operations (MLOps) system designed to predict climate-related environmental risks (extreme heatwaves, flooding rainfalls, severe windstorms, and droughts) using real historical multi-variate climate observations across Indian climate zones.

---

## 📌 Project Architecture & MLOps System Design

```
Climate-Risk-Prediction/
├── .dvc/                         # DVC system configuration & remote definitions
│   └── config                    # Local and DagsHub DVC storage endpoints
├── .github/
│   └── workflows/
│       └── ci.yaml               # GitHub Actions CI workflow
├── .venv/                        # Dedicated Python 3.11 virtual environment
├── config/
│   ├── config.yaml               # Central paths, features & drift thresholds
│   └── params.yaml               # Model hyperparameters & training configs
├── data/
│   ├── raw/
│   │   ├── climate_risk_data.csv.dvc # DVC pointer tracking raw dataset
│   │   ├── dataset_provenance.json   # License, coordinates, and origin metadata
│   │   └── validation_status.json    # Automated schema & bounds validation report
│   ├── processed/                # Preprocessed temporal train/test datasets
│   ├── reference/                # Baseline data for drift detection (Phase 4)
│   └── current/                  # Production inference data for monitoring (Phase 4)
├── metrics/                      # Evaluation scores, confusion matrices, comparison tables
│   ├── confusion_matrix.png      # Best model confusion matrix
│   ├── model_comparison.csv      # 5-model benchmark metrics
│   └── scores.json               # Detailed classification report
├── models/                       # Trained models & scalers
│   ├── best_model.joblib         # Complete inference pipeline (Preprocessor + Estimator)
│   └── preprocessor.joblib       # Standalone scikit-learn ColumnTransformer
├── reports/                      # EDA reports, statistics & figures
│   ├── figures/                  # Correlation, thermal, rainfall, and risk plots
│   ├── eda_statistics.csv        # Multi-variate descriptive statistics
│   └── eda_summary.json          # High-level dataset summary
├── src/                          # Modular production source code
│   ├── data/
│   │   ├── load_data.py          # Data collection & provenance tracking
│   │   └── validate_data.py      # Schema, physical bounds & integrity verification
│   ├── features/
│   │   ├── build_features.py     # Meteorological feature engineering & temporal split
│   │   └── eda.py                # EDA and visualization generation
│   ├── models/
│   │   ├── train.py              # Candidate training, evaluation & model selection
│   │   └── predict.py            # Standalone single & batch inference pipeline
│   ├── monitoring/               # Evidently AI data drift & SHAP explainability
│   ├── pipelines/
│   │   └── mlflow_tracker.py     # Automated MLflow & DagsHub experiment logger
│   └── utils/
│       ├── common.py             # YAML reader and directory utilities
│       └── logger.py             # Centralized application logging
├── tests/                        # Comprehensive automated test suite (20 tests)
│   ├── test_data_pipeline.py     # Data ingestion & feature engineering unit tests
│   ├── test_imports.py           # Dependency verification tests
│   ├── test_mlops_pipeline.py    # DVC, MLflow, and DagsHub integration tests
│   ├── test_model_pipeline.py    # Model evaluation & inference unit tests
│   └── test_setup.py             # Configuration & directory structure tests
├── .dvcignore                    # DVC ignore patterns
├── .env.example                  # Environment variables template (DagsHub/MLflow)
├── .gitignore                    # Git exclusions
├── dvc.lock                      # Cryptographic lockfile for reproducible pipeline
├── dvc.yaml                      # Declarative DVC automated pipeline stages
├── requirements.txt              # Core project dependencies
└── README.md                     # Documentation
```

---

## 🔁 DVC Reproducible Pipeline (`dvc.yaml`)

The entire ML lifecycle is automated into a 5-stage declarative pipeline in `dvc.yaml`:

```mermaid
flowchart TD
    RawData["data/raw/climate_risk_data.csv.dvc"] --> Stage1["1. data_validation<br/>(src/data/validate_data.py)"]
    Stage1 --> Status["data/raw/validation_status.json"]
    RawData --> Stage2["2. feature_engineering<br/>(src/features/build_features.py)"]
    Stage2 --> Processed["data/processed/train.csv<br/>data/processed/test.csv"]
    Processed --> Stage3["3. eda<br/>(src/features/eda.py)"]
    Stage3 --> Reports["reports/figures<br/>reports/eda_statistics.csv"]
    Processed --> Stage4["4. model_training_and_evaluation<br/>(src/models/train.py)"]
    Params["config/params.yaml"] --> Stage4
    Stage4 --> Artifacts["models/best_model.joblib<br/>metrics/scores.json"]
    Artifacts --> Stage5["5. mlflow_tracking<br/>(src/pipelines/mlflow_tracker.py)"]
    Stage5 --> MLflowRun["MLflow / DagsHub Logged Run"]
```

### Reproducing the Pipeline
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the complete pipeline end-to-end
dvc repro
```
* **Intelligent Caching:** DVC tracks input/output MD5 hashes. When inputs are unchanged, stages are automatically skipped, preventing redundant computation.

---

## 🧪 MLflow Experiment Tracking

* **Backend Store:** Production SQLite database (`sqlite:///mlflow.db`).
* **Experiment Name:** `Climate_Risk_Prediction`
* **Tracked Entities per Run:**
  * **Tags:** Git Commit Hash, DVC Data MD5 Hash, System Framework, User.
  * **Parameters:** `random_state`, `n_estimators`, `learning_rate`, `max_depth`, feature counts, temporal splitting boundaries.
  * **Metrics:** Accuracy (`0.9990`), F1-Macro (`0.9990`), Recall-Macro (`0.9990`), Precision-Macro (`0.9990`), ROC-AUC (`0.9998`).
  * **Artifacts Logged:** `best_model.joblib`, `scores.json`, `model_comparison.csv`, `confusion_matrix.png`, EDA plots, and configuration files.

### Launching the MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open [http://localhost:5000](http://localhost:5000) in your browser to inspect experiment runs, comparative metric charts, and logged model binaries.

---

## 🌐 DagsHub Integration

The project is structured for seamless integration with **DagsHub** as the central hub for Git, DVC remote storage, and MLflow remote tracking:

* **Git Remote:** `https://dagshub.com/abhishekbandla07/Climate-Risk-Prediction.git`
* **DVC Remote Storage:** `https://dagshub.com/abhishekbandla07/Climate-Risk-Prediction.dvc`
* **MLflow Tracking URI:** `https://dagshub.com/abhishekbandla07/Climate-Risk-Prediction.mlflow`

### Environment Configuration (`.env`)
Create a `.env` file in the project root based on `.env.example`:
```env
# DagsHub Authentication
DAGSHUB_USERNAME=abhishekbandla07
DAGSHUB_REPO_NAME=Climate-Risk-Prediction
DAGSHUB_USER_TOKEN=your_dagshub_token_here

# Remote MLflow Tracking URI
MLFLOW_TRACKING_URI=https://dagshub.com/abhishekbandla07/Climate-Risk-Prediction.mlflow
```

### Syncing Data to Remote Storage
```bash
# Push versioned data & models to DVC remote storage
dvc push

# Push code and pipeline lockfiles to Git
git push dagshub main
```

---

## 📊 Dataset & Modeling Summary

* **Dataset:** 13,148 real daily meteorological records (2015–2023) across Delhi, Mumbai, Kolkata, Chennai.
* **Source:** Open-Meteo Historical Archive API (ERA5 Reanalysis & IMD stations, CC BY 4.0).
* **Target:** Multi-class meteorological hazard risk (0: Normal, 1: Moderate Advisory, 2: High Severe Weather) derived from IMD/NDMA operational criteria.
* **Temporal Split:** Train (`2015–2021`, 10,228 rows) | Out-of-Time Test (`2022–2023`, 2,920 rows).
* **Selected Best Model:** **Gradient Boosting Classifier** (`Accuracy: 99.90%`, `F1-Macro: 0.9990`, `ROC-AUC: 0.9998`).

---

## 🧪 Automated Test Suite

20 automated unit and integration tests passing:
```bash
pytest tests/ -v
```
* `test_mlops_pipeline.py`: Validates DVC lockfiles, pipeline stages, schema validation reports, and MLflow run logs.
* `test_data_pipeline.py`: Validates raw dataset schema, temporal boundaries, heat index, and IMD seasons.
* `test_model_pipeline.py`: Validates model benchmark tables, confusion matrices, single and batch inference.
* `test_setup.py` & `test_imports.py`: Validates environment, configs, and library versions.

---

## 📋 Project Phases Roadmap

- [x] **Phase 1: Project Setup**
- [x] **Phase 2: Data Science**
- [x] **Phase 3: MLOps Pipeline Integration**
  - [x] DVC repository initialization & data versioning (`climate_risk_data.csv.dvc`)
  - [x] DVC remote storage configuration (`local_remote` default & `dagshub` remote)
  - [x] MLflow experiment tracking (`Climate_Risk_Prediction` with SQLite backend)
  - [x] Parameter, metric, plot, and model artifact logging in MLflow
  - [x] Complete 5-stage automated reproducible pipeline (`dvc.yaml` & `dvc.lock`)
  - [x] Pipeline caching & idempotence verification (`dvc repro`)
  - [x] DagsHub Git & DVC remote configurations & `.env.example`
  - [x] 20 unit/integration tests passing
- [ ] **Phase 4: Monitoring & Explainability**
  - Evidently AI data drift detection report & alerts
  - SHAP global and local feature importance
  - Target drift & model decay monitoring
- [ ] **Phase 5: Deployment & Dashboard**
  - Streamlit climate risk prediction web application
  - Real-time batch and single-instance prediction interface
  - Drift & performance metrics visualizer
