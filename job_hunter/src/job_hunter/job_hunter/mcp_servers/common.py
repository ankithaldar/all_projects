#!/usr/bin/env python
# -- coding: utf-8 --

'''Shared helpers for Job Hunter MCP servers.'''


from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def app_root() -> Path:
  '''Locate the job_hunter project root from this file's location.

  Returns:
    Absolute path to the project root.
  '''
  return Path(__file__).resolve().parents[4]


def config_path() -> str:
  '''Resolve the app config path from env or default location.

  Returns:
    Config file path string.
  '''
  return os.getenv('JH_APP_CONFIG', str(app_root() / 'config' / 'app.yaml'))


def ensure_sys_path() -> None:
  '''Make top-level packages (job_hunter, llm_gateway) importable.'''
  src = str(app_root() / 'src' / 'job_hunter')
  if src not in sys.path:
    sys.path.insert(0, src)


def dumps(payload: object) -> str:
  '''Serialize a tool result as compact JSON.

  Args:
    payload: Any JSON-safe value.

  Returns:
    JSON text.
  '''
  return json.dumps(payload, ensure_ascii=False, default=str)
