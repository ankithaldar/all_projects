#!/usr/bin/env python
# -- coding: utf-8 --

'''YAML-backed configuration models and loader.'''


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from llm_gateway.errors import ConfigError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
  '''Configuration for one model alias under a provider.'''

  model_config = ConfigDict(extra='ignore')

  model: str
  temperature: float = 0.2
  max_tokens: int = 2048
  cost_per_million_input: float = 0.0
  cost_per_million_output: float = 0.0
  supports_tools: bool = True


class ProviderConfig(BaseModel):
  '''Configuration for one LLM provider.'''

  model_config = ConfigDict(extra='ignore')

  enabled: bool = True
  base_url: str = ''
  api_key_env: str = ''
  rpm: int = 60
  timeout_seconds: float = 60.0
  models: Dict[str, ModelConfig] = Field(default_factory=dict)


class ExecutionStep(BaseModel):
  '''One provider/model alias step in the execution order.'''

  model_config = ConfigDict(extra='ignore')

  provider: str
  alias: str


class RetryConfig(BaseModel):
  '''Retry configuration.'''

  model_config = ConfigDict(extra='ignore')

  max_attempts: int = 3
  base_delay_seconds: float = 0.5
  max_delay_seconds: float = 8.0


class CacheConfig(BaseModel):
  '''Cache configuration.'''

  model_config = ConfigDict(extra='ignore')

  enabled: bool = True
  ttl_seconds: int = 86400
  include_temperature: bool = False


class GatewayConfig(BaseModel):
  '''Root gateway configuration.'''

  model_config = ConfigDict(extra='ignore')

  default_system_prompt: str = ''
  retry: RetryConfig = Field(default_factory=RetryConfig)
  cache: CacheConfig = Field(default_factory=CacheConfig)
  providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
  execution_order: List[ExecutionStep] = Field(default_factory=list)
  fallback: Optional[ExecutionStep] = None

  @model_validator(mode='after')
  def validate_steps(self) -> 'GatewayConfig':
    '''Validate execution order and fallback references.

    Returns:
      The validated configuration.

    Raises:
      ValueError: If a referenced provider or alias does not exist.
    '''
    steps = list(self.execution_order)
    if self.fallback:
      steps.append(self.fallback)

    for step in steps:
      provider = self.providers.get(step.provider)
      if provider is None:
        raise ValueError(f'unknown provider: {step.provider}')
      if step.alias not in provider.models:
        raise ValueError(
          f'unknown alias: {step.alias} for provider: {step.provider}'
        )

    return self


def load_gateway_config(path: str | Path) -> GatewayConfig:
  '''Load and validate gateway configuration from YAML.

  Args:
    path: Path to the YAML configuration file.

  Returns:
    Validated gateway configuration.

  Raises:
    ConfigError: If the file is missing or invalid.
  '''
  config_path = Path(path)
  if not config_path.exists():
    raise ConfigError(f'config file not found: {config_path}')

  raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
  if not isinstance(raw, dict):
    raise ConfigError('gateway config must be a mapping')

  return GatewayConfig.model_validate(raw)
