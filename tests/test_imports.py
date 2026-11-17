"""Test script to verify all core dependencies import cleanly."""
import sys

def test_imports():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    import numpy as np
    print(f"✓ numpy: {np.__version__}")

    import pyarrow as pa
    print(f"✓ pyarrow: {pa.__version__}")

    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")

    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")

    import scipy
    print(f"✓ scipy: {scipy.__version__}")

    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")

    import seaborn as sns
    print(f"✓ seaborn: {sns.__version__}")

    import shap
    print(f"✓ shap: {shap.__version__}")

    import mlflow
    print(f"✓ mlflow: {mlflow.__version__}")

    import dvc
    print(f"✓ dvc: {dvc.__version__}")

    import dagshub
    print(f"✓ dagshub: {dagshub.__version__}")

    import evidently
    print(f"✓ evidently: {evidently.__version__}")

    import streamlit as st
    print(f"✓ streamlit: {st.__version__}")

    import yaml
    print(f"✓ pyyaml: {yaml.__version__}")

    import pytest
    print(f"✓ pytest: {pytest.__version__}")

    print("\nALL IMPORTS SUCCESSFUL! No issues detected.")

if __name__ == "__main__":
    test_imports()
