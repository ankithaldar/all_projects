#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal structured logging with rich fallback to stdlib."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = 'knee') -> logging.Logger:
  global _CONFIGURED
  logger = logging.getLogger(name)
  if not _CONFIGURED:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
      logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s', '%H:%M:%S'
      )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _CONFIGURED = True
  return logger
