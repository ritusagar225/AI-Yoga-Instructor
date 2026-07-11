import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name="yoga_instructor", log_file="yoga_instructor.log", level=logging.INFO):
    """
    Configures and returns a multi-handler logger that outputs to both console and rolling file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger # Already configured

    logger.setLevel(level)
    
    # Format pattern
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (Rolling log up to 5MB, keeping 3 backups)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    
    try:
        file_handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file logger at {file_path}. Reason: {e}")
        
    return logger
