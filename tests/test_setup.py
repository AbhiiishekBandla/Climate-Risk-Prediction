import os
import pytest
import yaml

def test_config_files_exist():
    assert os.path.exists("config/config.yaml"), "config/config.yaml is missing"
    assert os.path.exists("config/params.yaml"), "config/params.yaml is missing"

def test_config_yaml_validity():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "project" in config
    assert "paths" in config
    assert "features" in config

def test_params_yaml_validity():
    with open("config/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    assert "base" in params
    assert "models" in params

def test_directories_structure():
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/reference",
        "data/current",
        "models",
        "metrics",
        "src/data",
        "src/features",
        "src/models",
        "src/monitoring",
        "src/pipelines",
        "src/utils",
        "app",
        "tests"
    ]
    for directory in expected_dirs:
        assert os.path.isdir(directory), f"Directory {directory} does not exist"
