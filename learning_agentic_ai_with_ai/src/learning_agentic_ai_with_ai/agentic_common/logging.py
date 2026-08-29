#!/usr/bin/env python
# -- coding: utf-8 --

'''Structured JSON logging.

Production agents are debugged from logs. Human-readable single-line logs are
hard to grep and analyze, so every record is emitted as one JSON object on
stderr (stdout is reserved for stdio transport protocol traffic - never mix
them, see chapter01_mcp/server/stdio.py).
'''


from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
  '''Format log records as single-line JSON objects.'''

  def format(self, record: logging.LogRecord) -> str:
    '''Convert a record into a JSON line.

    Args:
      record: The logging record.

    Returns:
      JSON string ending in a newline.
    '''
    payload: Dict[str, Any] = {
      'ts': datetime.now(timezone.utc).isoformat(),
      'level': record.levelname,
      'logger': record.name,
      'msg': record.getMessage(),
    }

    extra = getattr(record, 'extra_fields', None)
    if isinstance(extra, dict):
      payload.update(extra)

    if record.exc_info:
      payload['exc'] = self.formatException(record.exc_info)

    return json.dumps(payload, default=str)


_configured = False


def setup_logging(level: str = 'INFO') -> None:
  '''Configure the root logger with JSON formatting, once per process.

  Args:
    level: Log level name (DEBUG, INFO, WARNING, ERROR).
  '''
  global _configured
  if _configured:
    return

  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(JsonFormatter())

  root = logging.getLogger()
  root.handlers = [handler]
  root.setLevel(level.upper())
  _configured = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
  '''Return a named logger, ensuring JSON setup ran.

  Args:
    name: Logger name, usually `__name__`.
    level: Optional level override.

  Returns:
    Configured logger.
  '''
  setup_logging(level or 'INFO')
  return logging.getLogger(name)


def log_event(
  logger: logging.Logger,
  level: int,
  msg: str,
  **fields: Any,
) -> None:
  '''Emit a structured event with extra fields attached.

  Args:
    logger: Logger to emit with.
    level: Standard logging level constant.
    msg: Human readable message.
    **fields: Structured fields merged into the JSON payload.
  '''
  logger.log(level, msg, extra={'extra_fields': fields})
