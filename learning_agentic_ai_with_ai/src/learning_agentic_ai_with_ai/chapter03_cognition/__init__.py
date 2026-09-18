#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 3 package: cognitive architecture and adaptive planning.

Chapter 1 gave the agent tools (MCP). Chapter 2 gave it reasoning shapes.
Chapter 3 gives it an architecture: four typed layers - Perception, Memory,
Decision, Action - wired by pydantic models, with strategy profiles that
govern how boldly it plans and adapts, agent-written Python plans executed
in a capability-based sandbox, and memory that improves the next run.

Public entry points:
  - `CognitiveAgent`: the 4-layer orchestrator.
  - `CognitiveTaskInput` / `CognitiveTaskOutput`: typed task boundary.
  - `StrategySelector` / `STRATEGY_PROFILES`: conservative, exploratory,
    fallback operating policies.
  - `MemoryStore` / `MemoryLayer`: episodic, semantic, procedural memory.
  - `codegen`: sandboxed agent-written Python plans.
'''


from __future__ import annotations

from chapter03_cognition.agent import CognitiveAgent
from chapter03_cognition.config import CognitionConfig, load_cognition_config
from chapter03_cognition.memory import MemoryLayer, MemoryStore
from chapter03_cognition.schemas import (
  CognitiveTaskInput,
  CognitiveTaskOutput,
  Percept,
  StrategyName,
)
from chapter03_cognition.strategies import (
  STRATEGY_PROFILES,
  StrategySelector,
  profile_for,
)

__all__ = [
  'CognitiveAgent',
  'CognitiveTaskInput',
  'CognitiveTaskOutput',
  'CognitionConfig',
  'MemoryLayer',
  'MemoryStore',
  'Percept',
  'STRATEGY_PROFILES',
  'StrategyName',
  'StrategySelector',
  'load_cognition_config',
  'profile_for',
]
