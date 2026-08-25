#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal structured logging with rich fallback to stdlib.

Every named logger gets the stdout handler and INFO level -- attaching
to only the first caller silently swallowed all other modules' logs
(propagate=False left them handler-less).
"""

from __future__ import annotations

import logging
import sys

_FORMATTER = logging.Formatter(
  '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s', '%H:%M:%S'
)


def get_logger(name: str = 'knee') -> logging.Logger:
  """Return ``name`` wired to stdout at INFO level.

  Args:
      name: Logger namespace (usually the module or component).

  Returns:
      Configured logger; safe to call repeatedly and from forked
      DataLoader workers.
  """
  logger = logging.getLogger(name)
  if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)
  if logger.level == logging.NOTSET or logger.level > logging.INFO:
    logger.setLevel(logging.INFO)
  logger.propagate = False
  return logger
