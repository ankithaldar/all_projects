#!/usr/bin/env python
# -- coding: utf-8 --

'''Typed runtime settings loaded from environment variables.

Settings keep a single source of truth for tunables: LLM sampling params,
tool-execution policy limits, mock mode, and observability switches.
'''


from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentic_common import paths


def _env_int(name: str, default: int) -> int:
  '''Read an integer environment variable.

  Args:
    name: Environment variable name.
    default: Value used when unset or unparsable.

  Returns:
    Parsed integer value.
  '''
  raw = os.getenv(name)
  if not raw:
    return default
  try:
    return int(raw)
  except ValueError:
    return default


def _env_float(name: str, default: float) -> float:
  '''Read a float environment variable.

  Args:
    name: Environment variable name.
    default: Value used when unset or unparsable.

  Returns:
    Parsed float value.
  '''
  raw = os.getenv(name)
  if not raw:
    return default
  try:
    return float(raw)
  except ValueError:
    return default


def _env_bool(name: str, default: bool) -> bool:
  '''Read a boolean environment variable.

  Args:
    name: Environment variable name.
    default: Value used when unset.

  Returns:
    Parsed boolean value.
  '''
  raw = os.getenv(name)
  if raw is None or raw == '':
    return default
  return raw.strip().lower() in ('1', 'true', 'yes', 'on')


class Settings(BaseModel):
  '''Runtime settings for agents and demos.

  Attributes:
    mock_llm: When true, agents run with a scripted mock LLM (offline mode).
    llm_temperature: Default sampling temperature for gateway calls.
    llm_max_tokens: Default max output tokens for gateway calls.
    agent_max_iterations: Hard stop for the tool-use loop.
    agent_token_budget: Max total tokens per task run.
    tool_timeout_seconds: Per-tool-call timeout.
    tool_max_result_chars: Truncation limit for tool results.
    require_write_approval: Whether write tools need approval callback.
    max_restock_quantity: Safety cap used by the retail write tool policy.
    allowed_dispatch_priorities: Priorities the field-tech write tool accepts.
    log_level: Structured log verbosity.
    trace_enabled: Whether JSONL traces are written.
  '''

  model_config = ConfigDict(extra='ignore')

  mock_llm: bool = False
  llm_temperature: float = 0.2
  llm_max_tokens: int = 1024
  agent_max_iterations: int = 8
  agent_token_budget: int = 60000
  tool_timeout_seconds: float = 15.0
  tool_max_result_chars: int = 6000
  require_write_approval: bool = True
  max_restock_quantity: int = 500
  allowed_dispatch_priorities: tuple[str, ...] = ('low', 'medium', 'high')
  log_level: str = 'INFO'
  trace_enabled: bool = True
  gateway_config_path: Optional[str] = None
  gateway_env_path: Optional[str] = None


def load_settings() -> Settings:
  '''Build settings from environment variables with sensible defaults.

  Returns:
    A validated Settings instance.
  '''
  return Settings(
    mock_llm=_env_bool('AGENTIC_MOCK_LLM', False),
    llm_temperature=_env_float('AGENTIC_LLM_TEMPERATURE', 0.2),
    llm_max_tokens=_env_int('AGENTIC_LLM_MAX_TOKENS', 1024),
    agent_max_iterations=_env_int('AGENTIC_MAX_ITERATIONS', 8),
    agent_token_budget=_env_int('AGENTIC_TOKEN_BUDGET', 60000),
    tool_timeout_seconds=_env_float('AGENTIC_TOOL_TIMEOUT_S', 15.0),
    tool_max_result_chars=_env_int('AGENTIC_TOOL_MAX_RESULT_CHARS', 6000),
    require_write_approval=_env_bool('AGENTIC_REQUIRE_WRITE_APPROVAL', True),
    max_restock_quantity=_env_int('AGENTIC_MAX_RESTOCK_QTY', 500),
    log_level=os.getenv('AGENTIC_LOG_LEVEL', 'INFO'),
    trace_enabled=_env_bool('AGENTIC_TRACE_ENABLED', True),
    gateway_config_path=os.getenv('GATEWAY_CONFIG_PATH'),
    gateway_env_path=os.getenv('GATEWAY_ENV_PATH'),
  )


def default_settings() -> Settings:
  '''Return the shared default settings instance.

  Returns:
    Settings loaded from the environment.
  '''
  paths.ensure_data_dirs()
  return load_settings()
