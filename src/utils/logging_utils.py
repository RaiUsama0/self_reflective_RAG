"""Consistent logging across scripts."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "selfrag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
