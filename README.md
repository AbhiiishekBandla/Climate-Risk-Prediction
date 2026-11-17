# 🌍 Climate Risk Prediction System

An end-to-end production-grade Machine Learning Operations (MLOps) system designed to predict climate-related environmental risks (e.g., floods, severe storms, droughts, wildfires) using historical multi-variate climate parameters.

---

## 📌 Project Architecture

```
Climate-Risk-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yaml               # GitHub Actions CI workflow
├── config/
│   ├── config.yaml               # Project paths, features & monitoring settings
│   └── params.yaml               # Model hyperparameters & training configs
├── data/
│   ├── raw/                      # Raw historical climate dataset (DVC tracked)
│   ├── processed/                # Scaled & train/test splits (DVC tracked)
│   ├── reference/                # Baseline data for drift detection
│   └── current/                  # Production inference/recent data for monitoring
├── models/                       # Trained models & scalers (DVC tracked)
├── metrics/                      # Evaluation scores, confusion matrices, drift reports
├── notebooks/                    # Exploratory Data Analysis & experiments
├── src/                          # Modular source code
│   ├── data/                     # Ingestion and train/test splitting
│   ├── features/                 # Feature engineering & scaling pipelines
│   ├── models/                   # Training, evaluation & inference logic
│   ├── monitoring/               # Evidently AI data drift & SHAP explainability
│   ├── pipelines/                # Automated orchestrator pipelines
│   └── utils/                    # Common helpers, YAML reader & central logger
├── app/                          # Interactive Streamlit dashboard
├── tests/                        # Automated unit & integration tests
├── .gitignore                    # Git exclusions
├── .dvcignore                    # DVC exclusions
├── requirements.txt              # Core project dependencies
└── README.md                     # Documentation
```

---

## 🛠️ Technology Stack

| Layer | Tools & Frameworks |
|---|---|
| **Core Environment** | Python 3.11, NumPy, Pandas |
| **Machine Learning** | Scikit-learn, SciPy |
| **Model Explainability** | SHAP |
| **Data & Model Versioning** | DVC (Data Version Control) |
| **Experiment Tracking** | MLflow, DagsHub |
| **Continuous Monitoring** | Evidently AI (Data & Target Drift) |
| **Deployment / UI** | Streamlit |
| **Version Control & CI/CD** | Git, GitHub, GitHub Actions |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Git & Git Bash / PowerShell

### 2. Virtual Environment Setup
```bash
# Navigate to the project directory
cd Climate-Risk-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Tests
```bash
pytest tests/ -v
```

---

## 📋 Project Phases Roadmap

- [x] **Phase 1: Project Setup**
  - Modular project architecture & folder structure
  - Virtual environment & dependency management
  - Configuration files (`config.yaml`, `params.yaml`, `.gitignore`, `.dvcignore`)
  - CI pipeline & Unit test skeleton
- [ ] **Phase 2: Data Science**
  - Multi-variate climate data collection & cleaning
  - Exploratory Data Analysis (EDA)
  - Feature engineering & preprocessing pipeline
  - Baseline vs. ensemble model training (Random Forest, Gradient Boosting)
  - Model evaluation & selection
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
