#!/usr/bin/env python
# -- coding: utf-8 --

'''Logging setup with run/node context propagation.'''


from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

run_id_var: ContextVar[Optional[str]] = ContextVar('run_id', default=None)
node_var: ContextVar[Optional[str]] = ContextVar('node', default=None)


class JsonFormatter(logging.Formatter):
  '''Format records as single-line JSON enriched with run context.'''

  def format(self, record: logging.LogRecord) -> str:
    '''Render one record.

    Args:
      record: Log record.

    Returns:
      JSON line.
    '''
    payload: Dict[str, Any] = {
      'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z'),
      'level': record.levelname,
      'logger': record.name,
      'message': record.getMessage(),
      'run_id': run_id_var.get(),
      'node': node_var.get(),
    }
    if record.exc_info:
      payload['exc'] = self.formatException(record.exc_info)
    return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = 'INFO', log_dir: Optional[Path] = None) -> None:
    '''Configure root logging with console + rotating JSON file handlers.

    Args:
      level: Log level name.
      log_dir: Optional directory for app.log; console only when None.
    '''
    root = logging.getLogger()
    if getattr(root, '_job_hunter_configured', False):
      return
    root.setLevel(level.upper())
    formatter = JsonFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_dir is not None:
      from logging.handlers import RotatingFileHandler
      log_dir.mkdir(parents=True, exist_ok=True)
      file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
      )
      file_handler.setFormatter(formatter)
      root.addHandler(file_handler)
    setattr(root, '_job_hunter_configured', True)
