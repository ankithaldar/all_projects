#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structured logging.

the framework's de facto logging was ``print``, used at several
hundred call sites, and the ``dictConfig`` that would have replaced it required
an environment variable that was never set anywhere. Two *governance* decisions —
which metric the promotion gate selected, and whether a promotion was skipped —
were signalled exclusively through stdout, so an operator auditing a promotion
had to find the right ephemeral container log.

This module provides a small logging facade that:

* is enabled by default, with no environment variable required;
* carries a run identifier on every record so framework messages can be
  correlated with a scheduler run;
* supports a verbosity switch, so the genuinely valuable distribution reports
  survive while routine debug output is suppressed in production.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

#: Environment variable that raises the root framework logger to ``DEBUG``.
VERBOSITY_ENV_VAR = 'FORECASTING_ML_LOG_LEVEL'

#: Environment variable that carries the scheduler run identifier.
RUN_NAME_ENV_VAR = 'FORECASTING_ML_RUN_NAME'

_DEFAULT_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
_CONFIGURED = False


def configure_logging(level: int | None = None) -> None:
  """Install the framework's logging configuration.

  The configuration is idempotent: calling it repeatedly does not add duplicate
  handlers.

  Args:
    level: An explicit level. When omitted the level is read from the
      ``FORECASTING_ML_LOG_LEVEL`` environment variable and defaults to
      ``INFO``.
  """
  global _CONFIGURED  # noqa: PLW0603 - module-level singleton guard
  if _CONFIGURED:
    return

  resolved = level if level is not None else _level_from_env()
  handler = logging.StreamHandler(stream=sys.stdout)
  handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
  handler.addFilter(_RunContextFilter())

  root = logging.getLogger('forecasting_ml_framework')
  root.setLevel(resolved)
  root.handlers.clear()
  root.addHandler(handler)
  # Propagation is deliberately left enabled. A framework that silences its own
  # records from the root logger is invisible to the host application's logging
  # configuration, to an aggregator that installs a root handler, and to any
  # test that captures through the root -- so a Kedro `dictConfig` supplied via
  # KEDRO_LOGGING_CONFIG, which is exactly what this framework is designed to
  # accept, would not see a single framework record.
  root.propagate = True
  _CONFIGURED = True


def _level_from_env() -> int:
  """Resolve the configured verbosity from the environment.

  Returns:
    A standard-library logging level.
  """
  raw = os.getenv(VERBOSITY_ENV_VAR, 'INFO').upper()
  return getattr(logging, raw, logging.INFO) if raw.isalpha() else logging.INFO


class _RunContextFilter(logging.Filter):
  """Prefix every record with the scheduler run identifier when one is known."""

  def filter(self, record: logging.LogRecord) -> bool:
    """Attach the run name to a record.

    Args:
      record: The record being emitted.

    Returns:
      ``True``, so the record is always emitted.
    """
    record.run_name = os.getenv(RUN_NAME_ENV_VAR, 'local')  # type: ignore[attr-defined]
    return True


def get_logger(name: str) -> logging.Logger:
  """Return a framework logger.

  Args:
    name: Usually ``__name__``.

  Returns:
    A configured logger.
  """
  configure_logging()
  return logging.getLogger(name)


def log_distribution(logger: logging.Logger, label: str, values: dict[str, Any]) -> None:
  """Emit a structured distribution report.

  The distribution reports are the framework's most valuable observability
  artefact: a per-column five-number summary bracketing each stateful transform
  immediately reveals a broken fill value. They are kept at ``INFO`` and routed
  through the logger rather than ``print`` so they become queryable and levelled.

  Args:
    logger: The emitting logger.
    label: A human-readable stage label.
    values: A mapping of column name to summary statistic.
  """
  if not logger.isEnabledFor(logging.INFO):
    return
  rendered = ', '.join(f'{key}={value}' for key, value in sorted(values.items()))
  logger.info('%s | %s', label, rendered)
