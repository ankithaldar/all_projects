#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 2 runtime configuration.

Chapter-specific knobs (sampling temperatures per pattern, reasoning budget,
validation policy, retry policy) live here instead of in
`agentic_common.settings` so that Chapter 1 behavior is untouched. Values are
loaded from the environment with `AGENTIC_PLAN_*` names and fall back to
production-shaped defaults. A plain `agentic_common.settings.Settings`
instance can be supplied to inherit the shared LLM/observability knobs.
'''


from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentic_common.settings import (
  Settings,
  env_bool,
  env_float,
  env_int,
)


class PlanningConfig(BaseModel):
  '''Tunables for the Chapter 2 reasoning stack.

  Attributes:
    base_temperature: Default sampling temperature.
    cot_temperature: Temperature for chain-of-thought sampling (self-consistency
      benefits from some exploration).
    react_temperature: Temperature for ReAct turns (near-deterministic is
      safer when actions mutate systems).
    planner_temperature: Temperature for plan generation.
    critic_temperature: Temperature for the validation critic and revisions.
    max_tokens: Default max output tokens per LLM call.
    max_steps: Hard cap on ReAct steps and synthesis length.
    max_plan_nodes: Maximum number of nodes accepted in a plan.
    max_plan_waves: Maximum dependency generations accepted in a plan.
    token_budget: Total tokens allowed per task run.
    cot_samples: Default self-consistency samples for the CoT route.
    max_validation_rounds: Revision rounds the validator may run.
    critic_enabled: Whether the LLM critic runs after deterministic checks.
    critic_always: Run the critic even when deterministic checks pass.
    grounding_min_coverage: Minimum fraction of answer numbers that must be
      traceable to evidence before a grounding warning is raised.
    max_result_chars: Cap applied to sanitized tool observations.
    require_write_approval: Whether write tools require explicit approval.
    llm_max_attempts: Attempts per LLM call before giving up (transient only).
    retry_base_seconds: Base delay for exponential backoff with jitter.
    sanitize_observations: Wrap and sanitize untrusted tool output.
  '''

  model_config = ConfigDict(extra='ignore')

  base_temperature: float = 0.2
  cot_temperature: float = 0.3
  react_temperature: float = 0.0
  planner_temperature: float = 0.0
  critic_temperature: float = 0.0
  max_tokens: int = 1024
  max_steps: int = 8
  max_plan_nodes: int = 8
  max_plan_waves: int = 6
  token_budget: int = 90000
  cot_samples: int = 2
  max_validation_rounds: int = 1
  critic_enabled: bool = True
  critic_always: bool = False
  grounding_min_coverage: float = 0.6
  max_result_chars: int = 4000
  require_write_approval: bool = True
  llm_max_attempts: int = 3
  retry_base_seconds: float = 0.5
  sanitize_observations: bool = True


def load_planning_config(settings: Optional[Settings] = None) -> PlanningConfig:
  '''Build Chapter 2 config from environment variables.

  Args:
    settings: Optional shared settings instance whose LLM/observability
      knobs seed the chapter defaults.

  Returns:
    A validated PlanningConfig.
  '''
  default_temperature = settings.llm_temperature if settings else 0.2
  default_max_tokens = settings.llm_max_tokens if settings else 1024
  default_budget = settings.agent_token_budget if settings else 90000
  default_max_steps = settings.agent_max_iterations if settings else 8
  default_max_result = settings.tool_max_result_chars if settings else 4000
  default_approval = (
    settings.require_write_approval if settings else True
  )

  return PlanningConfig(
    base_temperature=env_float(
      'AGENTIC_PLAN_TEMPERATURE', default_temperature,
    ),
    cot_temperature=env_float('AGENTIC_PLAN_COT_TEMPERATURE', 0.3),
    react_temperature=env_float('AGENTIC_PLAN_REACT_TEMPERATURE', 0.0),
    planner_temperature=env_float('AGENTIC_PLAN_PLANNER_TEMPERATURE', 0.0),
    critic_temperature=env_float('AGENTIC_PLAN_CRITIC_TEMPERATURE', 0.0),
    max_tokens=env_int('AGENTIC_PLAN_MAX_TOKENS', default_max_tokens),
    max_steps=env_int('AGENTIC_PLAN_MAX_STEPS', default_max_steps),
    max_plan_nodes=env_int('AGENTIC_PLAN_MAX_NODES', 8),
    max_plan_waves=env_int('AGENTIC_PLAN_MAX_WAVES', 6),
    token_budget=env_int('AGENTIC_PLAN_TOKEN_BUDGET', default_budget),
    cot_samples=env_int('AGENTIC_PLAN_COT_SAMPLES', 2),
    max_validation_rounds=env_int('AGENTIC_PLAN_VALIDATION_ROUNDS', 1),
    critic_enabled=env_bool('AGENTIC_PLAN_CRITIC', True),
    critic_always=env_bool('AGENTIC_PLAN_CRITIC_ALWAYS', False),
    grounding_min_coverage=env_float(
      'AGENTIC_PLAN_GROUNDING_COVERAGE', 0.6,
    ),
    max_result_chars=env_int(
      'AGENTIC_PLAN_MAX_RESULT_CHARS', default_max_result,
    ),
    require_write_approval=env_bool(
      'AGENTIC_PLAN_REQUIRE_WRITE_APPROVAL', default_approval,
    ),
    llm_max_attempts=env_int('AGENTIC_PLAN_LLM_ATTEMPTS', 3),
    retry_base_seconds=env_float('AGENTIC_PLAN_RETRY_BASE_S', 0.5),
    sanitize_observations=env_bool('AGENTIC_PLAN_SANITIZE', True),
  )
