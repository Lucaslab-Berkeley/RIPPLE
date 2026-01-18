"""Logger utility for RIPPLE."""

import logging
from pathlib import Path


def setup_logger(log_file: str | Path = "log.txt", clear_existing: bool = True) -> logging.Logger:
    """Set up a logger that writes to a file.

    Parameters
    ----------
    log_file : str | Path
        Path to the log file. Default is "log.txt".
    clear_existing : bool
        If True, clear the existing log file. Default is True.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("ripple")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear existing log file if requested
    if clear_existing and log_path.exists():
        log_path.unlink()
    
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the RIPPLE logger instance.

    Returns
    -------
    logging.Logger
        Logger instance. If not set up, returns a basic logger.
    """
    logger = logging.getLogger("ripple")
    if not logger.handlers:
        # If logger not set up, create a basic one
        logger = setup_logger()
    return logger

