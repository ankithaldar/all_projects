#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared helpers for Job Hunter MCP servers.'''


from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from job_hunter.core.config import AppSettings


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


# NOTE (docs-free sys.path): MCP servers import `job_hunter.*` at module
# load. The launcher (host MCP config) must either run them with
# PYTHONPATH=src/job_hunter (installed layout) or from the source tree
# where ensure_sys_path() can find <root>/src/job_hunter by file
# location (app_root uses this file's path). ensure_sys_path() cannot
# rescue a launcher that copies server files elsewhere with no
# PYTHONPATH set; the full launcher requirement lives in docs/architecture.md MCP section.
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


_settings: AppSettings | None = None
'''Shared lazy settings cache; see get_settings_obj().'''

_settings_bootstrap_fn: object | None = None
'''Bootstrap function that created the cached _settings (cache key).'''

_settings_lock = threading.Lock()
'''Guards shared settings creation against concurrent races.'''


def get_settings_obj() -> 'AppSettings':
  '''Return lazily-bootstrapped settings.

  Returns:
    Application settings.
  '''
  global _settings, _settings_bootstrap_fn
  # Legacy test hooks (mirror the _resolve_http pattern in
  # sources_server): existing tests monkeypatch _settings/_settings_obj/
  # bootstrap directly on store_server/resume_server. Honour a
  # module-level _settings override before checking the shared cache so
  # those tests keep passing after the dedup.
  for _modname in (
    'job_hunter.mcp_servers.store_server',
    'job_hunter.mcp_servers.resume_server',
  ):
    _mod = sys.modules.get(_modname)
    if _mod is not None:
      _patched = getattr(_mod, '_settings', None)
      if _patched is not None and _patched is not _settings:
        return _patched
  from job_hunter.core.bootstrap import bootstrap as _core_now

  effective = _core_now
  candidates: list = []
  _common_mod = sys.modules.get('job_hunter.mcp_servers.common')
  if _common_mod is not None:
    candidates.append(getattr(_common_mod, 'bootstrap', None))
  for _modname in (
    'job_hunter.mcp_servers.store_server',
    'job_hunter.mcp_servers.resume_server',
  ):
    _mod = sys.modules.get(_modname)
    if _mod is not None:
      candidates.append(getattr(_mod, 'bootstrap', None))
  for _candidate in candidates:
    if _candidate is not None and _candidate is not _core_now:
      effective = _candidate
      break
  if _settings is not None and _settings_bootstrap_fn is effective:
    return _settings
  with _settings_lock:
    if _settings is not None and _settings_bootstrap_fn is effective:
      return _settings
    _settings = effective(config_path())
    _settings_bootstrap_fn = effective
    return _settings
