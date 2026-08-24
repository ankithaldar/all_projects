#!/usr/bin/env python
# -- coding: utf-8 --

'''Application settings loaded from config/app.yaml plus .env overrides.'''


from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from job_hunter.core.errors import ConfigError


class AppSettings:
  '''Typed view over config/app.yaml with path resolution helpers.

  Attributes:
    config_path: Absolute path to the YAML file.
    app_root: Job Hunter project root (folder containing pyproject).
    data_dir: Directory holding databases, resumes, inbox, logs.
    raw: Full parsed configuration mapping.
  '''

  def __init__(self, config_path: str | Path) -> None:
    '''Load and validate configuration.

    Args:
      config_path: Path to app.yaml.

    Raises:
      ConfigError: If the file is missing or not a mapping.
    '''
    self.config_path = Path(config_path).resolve()
    if not self.config_path.exists():
      raise ConfigError(f'config file not found: {self.config_path}')
    loaded = yaml.safe_load(self.config_path.read_text(encoding='utf-8')) or {}
    if not isinstance(loaded, dict):
      raise ConfigError('app config must be a mapping')
    self.raw: Dict[str, Any] = loaded
    self.app_root = self.config_path.parent.parent
    data_dir = Path(
      os.getenv('APP_DATA_DIR', str(self.raw.get('app', {}).get('data_dir', '')))
      or (self.app_root / 'data'),
    )
    self.data_dir = data_dir if data_dir.is_absolute() else (self.app_root / data_dir)

  @property
  def seeds_dir(self) -> Path:
    '''Return the seeds directory.'''
    return self.app_root / 'seeds'

  @property
  def db_path(self) -> Path:
    '''Return the main application database path.'''
    return self.data_dir / 'app.db'

  @property
  def checkpoint_path(self) -> Path:
    '''Return the LangGraph checkpoint database path.'''
    return self.data_dir / 'checkpoints.sqlite'

  @property
  def gateway_root(self) -> Path:
    '''Return the llm_gateway package directory.'''
    return Path(__file__).resolve().parent.parent / 'llm_gateway'

  @property
  def gateway_config_path(self) -> Path:
    '''Return the gateway YAML config path.'''
    return self.gateway_root / 'config' / 'gateway.yaml'

  @property
  def gateway_env_path(self) -> Path:
    '''Return the gateway .env path.'''
    env = self.gateway_root / '.env'
    fallback = self.app_root / '.env'
    return env if env.exists() else fallback

  @property
  def gateway_db_path(self) -> Path:
    '''Return the gateway logging database path.'''
    return self.data_dir / 'gateway.db'

  @property
  def gateway_cache_path(self) -> Path:
    '''Return the gateway cache database path.'''
    return self.data_dir / 'cache.db'

  @property
  def host(self) -> str:
    '''Return API bind host.'''
    return str(os.getenv('APP_HOST', self.raw.get('app', {}).get('host', '127.0.0.1')))

  @property
  def port(self) -> int:
    '''Return API bind port.'''
    return int(os.getenv('APP_PORT', self.raw.get('app', {}).get('port', 8088)))

  @property
  def log_level(self) -> str:
    '''Return log level name.'''
    return str(os.getenv('APP_LOG_LEVEL', self.raw.get('app', {}).get('log_level', 'INFO'))).upper()

  @property
  def salary_floor_lpa(self) -> float:
    '''Return the hard salary floor in LPA.'''
    value = os.getenv(
      'SALARY_HARD_FLOOR_LPA',
      str(self.raw.get('salary_hard_floor_lpa', 45)),
    )
    return float(value)

  @property
  def schedule(self) -> Dict[str, Any]:
    '''Return schedule configuration mapping.'''
    return dict(self.raw.get('schedule', {}))

  @property
  def sources(self) -> Dict[str, Any]:
    '''Return per-source configuration mapping.'''
    return dict(self.raw.get('sources', {}))

  @property
  def discovery(self) -> Dict[str, Any]:
    '''Return discovery configuration mapping.'''
    return dict(self.raw.get('discovery', {}))

  @property
  def scoring_weights(self) -> Dict[str, float]:
    '''Return default scoring weights.'''
    return dict(self.raw.get('scoring_weights', {}))

  @property
  def embeddings(self) -> Dict[str, Any]:
    '''Return embedding configuration mapping.'''
    return dict(self.raw.get('embeddings', {}))

  @property
  def mcp(self) -> Dict[str, Any]:
    '''Return MCP server configuration mapping.'''
    return dict(self.raw.get('mcp', {}))

  def section(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '''Return a top-level section as a dict.

    Args:
      key: Section name.
      default: Value when absent.

    Returns:
      The section mapping.
    '''
    value = self.raw.get(key, default or {})
    return dict(value) if isinstance(value, dict) else {}
