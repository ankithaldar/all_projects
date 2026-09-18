#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 3 runtime configuration.

Chapter-specific knobs (planner/codegen temperatures, adaptive retry caps,
sandbox limits, memory sizing) live here instead of in
`agentic_common.settings` so Chapters 1 and 2 remain untouched. Values load
from `AGENTIC_COG_*` environment variables with production-shaped defaults;
a shared `agentic_common.settings.Settings` instance seeds the LLM and
observability defaults.
'''


from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentic_common import paths
from agentic_common.settings import (
  Settings,
  env_bool,
  env_float,
  env_int,
)


class CognitionConfig(BaseModel):
  '''Tunables for the Chapter 3 cognitive stack.

  Attributes:
    planner_temperature: Sampling temperature for DAG planning.
    code_temperature: Sampling temperature for code generation.
    reason_temperature: Sampling temperature for reason/synthesis steps.
    max_tokens: Default max output tokens per LLM call.
    token_budget: Total tokens allowed per task run.
    max_steps: Default plan step cap.
    max_attempts_per_step: Default adaptive attempts per step.
    max_replans: Default whole-plan replans allowed.
    code_execution_enabled: Whether code plans may run at all.
    default_code_runner: `restricted` (in-process) or `subprocess`.
    code_timeout_seconds: Wall-clock budget for one code plan.
    code_call_budget: Tool calls allowed from one code plan.
    code_max_chars: Maximum generated code size accepted.
    require_write_approval: Whether write tools need approval.
    max_result_chars: Cap applied to sanitized tool observations.
    memory_db_path: SQLite path for episodes/facts/strategy stats.
    memory_recall_limit: Episodes recalled per task.
    llm_max_attempts: Attempts per LLM call (transient failures).
    retry_base_seconds: Base delay for exponential backoff with jitter.
    sanitize_observations: Wrap and sanitize untrusted tool output.
  '''

  model_config = ConfigDict(extra='ignore')

  planner_temperature: float = 0.0
  code_temperature: float = 0.0
  reason_temperature: float = 0.2
  max_tokens: int = 1024
  token_budget: int = 90000
  max_steps: int = 8
  max_attempts_per_step: int = 2
  max_replans: int = 1
  code_execution_enabled: bool = True
  default_code_runner: str = 'restricted'
  code_timeout_seconds: float = 5.0
  code_call_budget: int = 8
  code_max_chars: int = 12000
  require_write_approval: bool = True
  max_result_chars: int = 4000
  memory_db_path: str = str(paths.DATA_DIR / 'chapter03_memory.db')
  memory_recall_limit: int = 5
  llm_max_attempts: int = 3
  retry_base_seconds: float = 0.5
  sanitize_observations: bool = True

  def memory_path(self) -> Path:
    '''Return the memory database path.

    Returns:
      Path object for the memory SQLite file.
    '''
    return Path(self.memory_db_path)


def load_cognition_config(
  settings: Optional[Settings] = None,
) -> CognitionConfig:
  '''Build Chapter 3 config from environment variables.

  Args:
    settings: Optional shared settings whose LLM/observability knobs seed
      the chapter defaults.

  Returns:
    A validated CognitionConfig.
  '''
  default_max_tokens = settings.llm_max_tokens if settings else 1024
  default_budget = settings.agent_token_budget if settings else 90000
  default_steps = settings.agent_max_iterations if settings else 8
  default_result = settings.tool_max_result_chars if settings else 4000
  default_approval = settings.require_write_approval if settings else True

  return CognitionConfig(
    planner_temperature=env_float('AGENTIC_COG_PLANNER_TEMPERATURE', 0.0),
    code_temperature=env_float('AGENTIC_COG_CODE_TEMPERATURE', 0.0),
    reason_temperature=env_float(
      'AGENTIC_COG_REASON_TEMPERATURE', 0.2,
    ),
    max_tokens=env_int('AGENTIC_COG_MAX_TOKENS', default_max_tokens),
    token_budget=env_int('AGENTIC_COG_TOKEN_BUDGET', default_budget),
    max_steps=env_int('AGENTIC_COG_MAX_STEPS', default_steps),
    max_attempts_per_step=env_int('AGENTIC_COG_MAX_ATTEMPTS', 2),
    max_replans=env_int('AGENTIC_COG_MAX_REPLANS', 1),
    code_execution_enabled=env_bool('AGENTIC_COG_CODE_ENABLED', True),
    default_code_runner=os.getenv(
      'AGENTIC_COG_CODE_RUNNER', 'restricted',
    ),
    code_timeout_seconds=env_float('AGENTIC_COG_CODE_TIMEOUT_S', 5.0),
    code_call_budget=env_int('AGENTIC_COG_CODE_CALL_BUDGET', 8),
    code_max_chars=env_int('AGENTIC_COG_CODE_MAX_CHARS', 12000),
    require_write_approval=env_bool(
      'AGENTIC_COG_REQUIRE_WRITE_APPROVAL', default_approval,
    ),
    max_result_chars=env_int(
      'AGENTIC_COG_MAX_RESULT_CHARS', default_result,
    ),
    memory_db_path=os.getenv(
      'AGENTIC_COG_MEMORY_DB',
      str(paths.DATA_DIR / 'chapter03_memory.db'),
    ),
    memory_recall_limit=env_int('AGENTIC_COG_MEMORY_RECALL', 5),
    llm_max_attempts=env_int('AGENTIC_COG_LLM_ATTEMPTS', 3),
    retry_base_seconds=env_float('AGENTIC_COG_RETRY_BASE_S', 0.5),
    sanitize_observations=env_bool('AGENTIC_COG_SANITIZE', True),
  )
