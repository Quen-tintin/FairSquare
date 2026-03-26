"""
Centralised logger factory.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

from config import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
            if hasattr(sys.stdout, "fileno") else sys.stdout
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(settings.log_level.upper())
    return logger
