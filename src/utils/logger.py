import os
import sys
import logging

LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")

def setup_logger(name: str = "climate_risk_logger") -> logging.Logger:
    """Configures and returns a centralized application logger."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILEPATH)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
