#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities: seeding, logging, and timing."""

from __future__ import annotations

import logging
import os
import random
import sys
import time

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility.

    Args:
        seed: Integer seed applied to all available RNG backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        logging.getLogger(__name__).warning('torch unavailable; torch seeds skipped')


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a concise stream handler attached once.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level for the handler.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Timer:
    """Context manager measuring elapsed wall-clock seconds.

    Attributes:
        elapsed: Seconds elapsed after the context exits; 0.0 before that.
    """

    def __init__(self) -> None:
        """Initialize the timer with a zeroed elapsed counter."""
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> 'Timer':
        """Start the timer.

        Returns:
            The timer instance.
        """
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the timer and record elapsed seconds.

        Args:
            exc_type: Exception type if the body raised.
            exc_value: Exception value if the body raised.
            traceback: Traceback if the body raised.
        """
        self.elapsed = time.time() - self._start
