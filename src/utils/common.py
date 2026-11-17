import os
import yaml
from typing import Any, Dict, List
from pathlib import Path
from src.utils.logger import logger

def read_yaml(path_to_yaml: str) -> Dict[str, Any]:
    """Reads a YAML file safely and returns a dictionary.
    
    Args:
        path_to_yaml (str): Path to YAML file.
        
    Returns:
        dict: Parsed YAML contents.
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Loaded YAML file successfully from: {path_to_yaml}")
            return content
    except Exception as e:
        logger.error(f"Error reading YAML file at {path_to_yaml}: {e}")
        raise e

def create_directories(paths: List[str]) -> None:
    """Creates a list of directories if they do not exist.
    
    Args:
        paths (list): List of directory paths.
    """
    for p in paths:
        os.makedirs(p, exist_ok=True)
        logger.info(f"Ensured directory exists: {p}")
