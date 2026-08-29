#!/usr/bin/env python
# -- coding: utf-8 --

'''Central path resolution for the course project.

Every module resolves files relative to these constants so that code works
no matter which directory a script is launched from.
'''


from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
  '''Locate the project root directory.

  The root is the folder that contains `src/`, `tests/`, and `pyproject.toml`.

  Returns:
    Absolute path to the project root.
  '''
  env_root = os.getenv('AGENTIC_PROJECT_ROOT')
  if env_root:
    return Path(env_root).resolve()

  # agentic_common/paths.py -> parents[0]=agentic_common,
  # parents[1]=learning_agentic_ai_with_ai (pkg), parents[2]=src,
  # parents[3]=learning_agentic_ai_with_ai (project root)
  return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = _project_root()
SRC_DIR: Path = PROJECT_ROOT / 'src' / 'learning_agentic_ai_with_ai'
GATEWAY_DIR: Path = SRC_DIR / 'llm_gateway'
GATEWAY_CONFIG: Path = GATEWAY_DIR / 'config' / 'gateway.yaml'
GATEWAY_ENV: Path = GATEWAY_DIR / '.env'

DATA_DIR: Path = PROJECT_ROOT / 'data'
TRACES_DIR: Path = DATA_DIR / 'traces'
EVALS_DIR: Path = DATA_DIR / 'evals'

GATEWAY_DB_PATH: Path = DATA_DIR / 'gateway.db'
GATEWAY_CACHE_PATH: Path = DATA_DIR / 'cache.db'
AGENT_STATE_DB: Path = DATA_DIR / 'agent_state.db'
OPS_MOCK_DB: Path = DATA_DIR / 'ops_mock.db'

CHAPTER1_DIR: Path = SRC_DIR / 'chapter01_mcp'
CHAPTER1_TYPESCRIPT_DIR: Path = CHAPTER1_DIR / 'typescript'


def ensure_data_dirs() -> None:
  '''Create data directories if they are missing.'''
  for directory in (DATA_DIR, TRACES_DIR, EVALS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
