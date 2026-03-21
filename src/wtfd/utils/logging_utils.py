"""
logging_utils.py

Purpose
-------
Provides a reusable logger configuration utility for the WTDF project.

This module standardizes logging across all components (data processing,
model training, evaluation, etc.) while keeping configuration lightweight.

Features
--------
- Consistent formatting across modules
- Console logging by default
- Optional file logging
- Prevents duplicate handlers (important in notebooks and repeated imports)

Usage
-----
from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)

logger.info("Pipeline started")
logger.warning("Missing features detected")
logger.error("Failed to load file")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module).
    log_file : str or Path or None, optional
        If provided, logs will also be written to this file.
    level : int, default=logging.INFO
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers (important for notebooks / re-runs)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Optional file handler
    # ------------------------------------------------------------------
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent logs from propagating to root logger (avoids duplicates)
    logger.propagate = False

    return logger