#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Structured logging for truck-carton RL system.

Provides hierarchical loggers for each subsystem
(env, packing, reward, training, routing, grid)
with configurable levels and optional JSON output
for dashboard consumption.
'''

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_LOG_FORMAT = '%(asctime)s [%(name)-24s] %(levelname)-5s %(message)s'
_DATE_FORMAT = '%H:%M:%S'

_LOGGERS_CONFIGURED = False

_TRUSTED_LOAD_ENV = 'TRUCK_CARTON_ALLOW_TRUSTED_LOAD'
_MODEL_DIRS_ENV = 'TRUCK_CARTON_MODEL_DIRS'
_BASE_DIR_ENV = 'TRUCK_CARTON_BASE_DIR'
_TRUSTED_VALUES = {'1', 'true', 'yes', 'on'}
_MODEL_SUFFIXES = {'', '.zip'}


def _effective_base(base_dir: str | Path | None) -> Path:
  '''Return the resolved base directory for containment checks.'''
  if base_dir is not None:
    return Path(base_dir).expanduser().resolve()
  env_base = os.environ.get(_BASE_DIR_ENV)
  if env_base:
    return Path(env_base).expanduser().resolve()
  return Path.cwd().resolve()


def resolve_contained_path(
  path: str | Path,
  base_dir: str | Path | None = None,
) -> Path:
  '''Resolve path, requiring it stay inside base_dir.

  Rejects empty paths, ``..`` escapes, absolute paths outside
  the base, and symlink escapes (resolution follows links).
  Normal relative paths such as ``./my_run`` are accepted.
  '''
  if path is None or str(path).strip() == '':
    raise ValueError('path must not be empty')
  base = _effective_base(base_dir)
  raw = Path(path).expanduser()
  candidate = raw if raw.is_absolute() else base / raw
  resolved = candidate.resolve(strict=False)
  if not resolved.is_relative_to(base):
    raise ValueError(
      f'path escapes base directory {base}: {path!r}'
    )
  return resolved


def _default_model_dirs() -> list[Path]:
  '''Return allowlisted model dirs (configurable via env).'''
  env_dirs = os.environ.get(_MODEL_DIRS_ENV)
  if env_dirs:
    return [
      Path(p).expanduser().resolve()
      for p in env_dirs.split(os.pathsep)
      if p.strip()
    ]
  cwd = Path.cwd().resolve()
  return [cwd / 'output', cwd / 'models']


def validate_model_path(
  model_path: str | Path,
  allow_dirs: Iterable[str | Path] | None = None,
) -> Path:
  '''Validate a model path for safe loading (S-01).

  Requires containment in an allowlisted model dir and an
  SB3-compatible suffix (``.zip`` or none, which SB3 treats
  as ``.zip``). Raw pickle/safetensors files are rejected:
  SB3 ``load`` unpickles and offers no weights_only or
  safetensors path, so loading stays trusted-only.
  '''
  if model_path is None or str(model_path).strip() == '':
    raise ValueError('model path must not be empty')
  suffix = Path(model_path).suffix.lower()
  if suffix not in _MODEL_SUFFIXES:
    raise ValueError(
      f'unsupported model suffix {suffix!r}: SB3 load only '
      'supports .zip models (no weights_only/safetensors); '
      'refusing arbitrary pickle files'
    )
  dirs = (
    [Path(d).expanduser().resolve() for d in allow_dirs]
    if allow_dirs is not None
    else _default_model_dirs()
  )
  # Relative model paths resolve against cwd, then the
  # containment check itself is delegated to the shared
  # resolve_contained_path core (single implementation).
  raw = Path(model_path).expanduser()
  candidate = (
    raw
    if raw.is_absolute()
    else Path.cwd().resolve() / raw
  ).resolve(strict=False)
  for allowed in dirs:
    try:
      return resolve_contained_path(candidate, base_dir=allowed)
    except ValueError:
      continue
  raise ValueError(
    f'model path outside allowlisted dirs {dirs}: {model_path!r}'
  )


def is_trusted_load_allowed(allow_trusted: bool = False) -> bool:
  '''Return True only on explicit trusted-load opt-in.'''
  if allow_trusted:
    return True
  return os.environ.get(_TRUSTED_LOAD_ENV, '').lower() in (
    _TRUSTED_VALUES
  )


def load_model_safely(
  model_path: str | Path,
  allow_dirs: Iterable[str | Path] | None = None,
  *,
  allow_trusted: bool = False,
  loader: Callable[[str], Any] | None = None,
) -> Any:
  '''Load a model only if contained and explicitly trusted.

  SB3 ``MaskablePPO.load`` unpickles checkpoint data, so
  this enforces trusted-only loading: path containment in
  an allowlisted dir plus an explicit opt-in (argument or
  ``TRUCK_CARTON_ALLOW_TRUSTED_LOAD=1``). Rejection happens
  before any loader call, so a malicious pickle is never
  executed. ``loader`` defaults to ``MaskablePPO.load``
  (imported lazily); tests inject a fake loader.
  '''
  resolved = validate_model_path(model_path, allow_dirs=allow_dirs)
  if not is_trusted_load_allowed(allow_trusted):
    raise ValueError(
      'refusing to load model without trusted opt-in: SB3 '
      'load unpickles checkpoints and has no '
      'weights_only/safetensors mode. Pass allow_trusted=True, '
      '--allow-trusted-load, or set '
      f'{_TRUSTED_LOAD_ENV}=1 for a model you fully trust'
    )
  if loader is None:
    # Lazy import: keeps this module usable without SB3.
    # pylint: disable=import-outside-toplevel
    from sb3_contrib import MaskablePPO

    # pylint: enable=import-outside-toplevel
    loader = MaskablePPO.load
  return loader(str(resolved))


def setup_logging(
  level: int = logging.INFO,
  log_file: str | None = None,
  json_log_file: str | None = None,
  base_dir: str | Path | None = None,
) -> None:
  '''Configure root + subsystem loggers.'''
  # Validate before the idempotent early return so a
  # traversal path is rejected on every call (S-02).
  if log_file:
    log_file = str(resolve_contained_path(log_file, base_dir=base_dir))
  if json_log_file:
    json_log_file = str(
      resolve_contained_path(json_log_file, base_dir=base_dir)
    )
  global _LOGGERS_CONFIGURED
  if _LOGGERS_CONFIGURED:
    return
  _LOGGERS_CONFIGURED = True

  root = logging.getLogger('truck_carton')
  root.setLevel(level)
  root.handlers.clear()

  console = logging.StreamHandler(sys.stdout)
  console.setLevel(level)
  console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
  root.addHandler(console)

  if log_file:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(fh)

  if json_log_file:
    Path(json_log_file).parent.mkdir(parents=True, exist_ok=True)
    jh = logging.FileHandler(json_log_file, mode='a')
    jh.setLevel(level)
    jh.setFormatter(_JsonFormatter())
    root.addHandler(jh)


class _JsonFormatter(logging.Formatter):
  '''Emits one JSON object per line for dashboard.'''

  def format(self, record: logging.LogRecord) -> str:
    '''Format.'''
    entry: dict[str, Any] = {
      'ts': record.created,
      'logger': record.name,
      'level': record.levelname,
      'msg': record.getMessage(),
    }
    if hasattr(record, 'data'):
      entry['data'] = record.data
    try:
      return json.dumps(entry, default=str)
    except (TypeError, ValueError):
      entry.pop('data', None)
      return json.dumps(entry, default=str)


def get_logger(subsystem: str) -> logging.Logger:
  '''Return a namespaced logger.

  Usage:
    log = get_logger('env')       # truck_carton.env
    log = get_logger('reward')    # truck_carton.reward
    log = get_logger('training')  # truck_carton.training
  '''
  return logging.getLogger(f'truck_carton.{subsystem}')


class EpisodeLogger:
  '''Accumulates structured episode data and
  emits it as a single JSON record at episode end.
  Designed for introspection and dashboard feeds.'''

  def __init__(self) -> None:
    '''Initialize the instance.'''
    self._log = get_logger('episode')
    self._steps: list[dict[str, Any]] = []
    self._start_time: float = 0.0
    self._episode_id: int = 0
    self._meta: dict[str, Any] = {}

  def begin_episode(
    self,
    episode_id: int,
    stage: int,
    num_trucks: int,
    num_cartons: int,
    grid_size: tuple[int, int],
  ) -> None:
    '''Begin episode.'''
    self._episode_id = episode_id
    self._start_time = time.monotonic()
    self._steps = []
    self._meta = {
      'episode_id': episode_id,
      'stage': stage,
      'num_trucks': num_trucks,
      'num_cartons': num_cartons,
      'grid_rows': grid_size[0],
      'grid_cols': grid_size[1],
    }
    self._log.info(
      'Episode %d started (stage=%d, trucks=%d, cartons=%d, grid=%dx%d)',
      episode_id,
      stage,
      num_trucks,
      num_cartons,
      *grid_size,
    )

  def log_step(
    self,
    step: int,
    action: int,
    action_type: str,
    reward: float,
    breakdown: dict[str, float],
    num_placed: int,
    num_delivered: int,
    active_truck: int,
    truck_states: list[str],
    truck_positions: list[tuple[int, int]],
  ) -> None:
    '''Log step.'''
    record = {
      'step': step,
      'action': action,
      'action_type': action_type,
      'reward': round(reward, 4),
      'breakdown': {k: round(v, 4) for k, v in breakdown.items()},
      'num_placed': num_placed,
      'num_delivered': num_delivered,
      'active_truck': active_truck,
      'truck_states': truck_states,
      'truck_positions': [list(p) for p in truck_positions],
    }
    self._steps.append(record)
    self._log.debug(
      'Step %d: action=%d (%s) reward=%.4f placed=%d delivered=%d',
      step,
      action,
      action_type,
      reward,
      num_placed,
      num_delivered,
    )

  def end_episode(
    self,
    total_reward: float,
    num_placed: int,
    num_delivered: int,
    total_cartons: int,
    terminated: bool,
  ) -> dict[str, Any]:
    '''End episode.'''
    elapsed = time.monotonic() - self._start_time
    summary = {
      **self._meta,
      'total_reward': round(total_reward, 4),
      'num_placed': num_placed,
      'num_delivered': num_delivered,
      'total_cartons': total_cartons,
      'completion_rate': round(num_placed / max(total_cartons, 1), 4),
      'delivery_rate': round(num_delivered / max(total_cartons, 1), 4),
      'terminated': terminated,
      'num_steps': len(self._steps),
      'elapsed_seconds': round(elapsed, 3),
      'steps': self._steps,
    }
    self._log.info(
      'Episode %d done: reward=%.2f placed=%d/%d'
      ' delivered=%d/%d steps=%d (%.1fs)',
      self._episode_id,
      total_reward,
      num_placed,
      total_cartons,
      num_delivered,
      total_cartons,
      len(self._steps),
      elapsed,
    )
    return summary

  def log_routing(
    self,
    truck_id: int,
    src: tuple[int, int],
    dst: tuple[int, int],
    distance: float,
    destination_type: str,
  ) -> None:
    '''Log routing.

    Public API: exercised by tests/test_logging.py and
    intended for env wiring; kept (not dead code).
    '''
    self._log.debug(
      'Truck %d routes %s->%s (dist=%.1f, type=%s)',
      truck_id,
      src,
      dst,
      distance,
      destination_type,
    )

  def log_packing(
    self,
    carton_id: int,
    truck_id: int,
    position: tuple[int, int, int],
    dims: tuple[int, int, int],
    weight: float,
  ) -> None:
    '''Log packing.

    Public API: exercised by tests/test_logging.py and
    intended for env wiring; kept (not dead code).
    '''
    self._log.debug(
      'Packed carton %d into truck %d at %s dims=%s weight=%.1f',
      carton_id,
      truck_id,
      position,
      dims,
      weight,
    )

  def log_delivery(
    self,
    truck_id: int,
    store_id: int,
    num_unloaded: int,
  ) -> None:
    '''Log delivery.

    Public API: exercised by tests/test_logging.py and
    intended for env wiring; kept (not dead code).
    '''
    self._log.debug(
      'Truck %d delivered %d cartons at store %d',
      truck_id,
      num_unloaded,
      store_id,
    )
