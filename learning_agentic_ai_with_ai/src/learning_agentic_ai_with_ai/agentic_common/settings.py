#!/usr/bin/env python
# -- coding: utf-8 --

'''Typed runtime settings loaded from environment variables.

Settings keep a single source of truth for tunables: LLM sampling params,
tool-execution policy limits, mock mode, and observability switches.
'''


from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from agentic_common import paths


def env_int(name: str, default: int) -> int:
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


def env_float(name: str, default: float) -> float:
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


def env_bool(name: str, default: bool) -> bool:
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


def load_settings() -> Settings:
  '''Build settings from environment variables with sensible defaults.

  Returns:
    A validated Settings instance.
  '''
  return Settings(
    mock_llm=env_bool('AGENTIC_MOCK_LLM', False),
    llm_temperature=env_float('AGENTIC_LLM_TEMPERATURE', 0.2),
    llm_max_tokens=env_int('AGENTIC_LLM_MAX_TOKENS', 1024),
    agent_max_iterations=env_int('AGENTIC_MAX_ITERATIONS', 8),
    agent_token_budget=env_int('AGENTIC_TOKEN_BUDGET', 60000),
    tool_timeout_seconds=env_float('AGENTIC_TOOL_TIMEOUT_S', 15.0),
    tool_max_result_chars=env_int('AGENTIC_TOOL_MAX_RESULT_CHARS', 6000),
    require_write_approval=env_bool('AGENTIC_REQUIRE_WRITE_APPROVAL', True),
    max_restock_quantity=env_int('AGENTIC_MAX_RESTOCK_QTY', 500),
  )


def default_settings() -> Settings:
  '''Return the shared default settings instance.

  Returns:
    Settings loaded from the environment.
  '''
  paths.ensure_data_dirs()
  return load_settings()
