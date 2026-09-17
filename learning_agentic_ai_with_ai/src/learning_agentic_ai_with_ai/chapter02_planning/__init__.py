#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 2 package: planning, reasoning, and structured prompting.

Chapter 1 taught the *tools* (MCP, policy, audit). Chapter 2 teaches the
*thinking*: how tasks are routed to the right reasoning shape, how reasoning
interleaves with tool actions, how plans become dependency DAGs, and how
agents validate their own work before answering.

Public entry points:
  - `PlanningAgent`: orchestrates routing, reasoning, tools, validation.
  - `PlanningTaskInput` / `PlanningTaskOutput`: typed task boundary.
  - `ReasoningPolicy`: explainable route selection.
  - patterns: CoT, ReAct, self-validation, decomposition.
  - tools: provider-agnostic tool access (in-process or MCP).
'''


from __future__ import annotations

from chapter02_planning.agent import PlanningAgent
from chapter02_planning.config import PlanningConfig, load_planning_config
from chapter02_planning.policy import ReasoningPolicy, route_explanations
from chapter02_planning.schemas import (
  PlanningTaskInput,
  PlanningTaskOutput,
  PolicyDecision,
  ReasoningRoute,
)

__all__ = [
  'PlanningAgent',
  'PlanningConfig',
  'PlanningTaskInput',
  'PlanningTaskOutput',
  'PolicyDecision',
  'ReasoningPolicy',
  'ReasoningRoute',
  'load_planning_config',
  'route_explanations',
]
