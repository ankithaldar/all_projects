"""Structured logging for truck-carton RL system.

Provides hierarchical loggers for each subsystem
(env, packing, reward, training, routing, grid)
with configurable levels and optional JSON output
for dashboard consumption.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


_LOG_FORMAT = (
  '%(asctime)s [%(name)-24s] %(levelname)-5s'
  ' %(message)s'
)
_DATE_FORMAT = '%H:%M:%S'

_LOGGERS_CONFIGURED = False


def setup_logging(
  level: int = logging.INFO,
  log_file: str | None = None,
  json_log_file: str | None = None,
) -> None:
  """Configure root + subsystem loggers."""
  global _LOGGERS_CONFIGURED
  if _LOGGERS_CONFIGURED:
    return
  _LOGGERS_CONFIGURED = True

  root = logging.getLogger('truck_carton')
  root.setLevel(level)
  root.handlers.clear()

  console = logging.StreamHandler(sys.stdout)
  console.setLevel(level)
  console.setFormatter(logging.Formatter(
    _LOG_FORMAT, datefmt=_DATE_FORMAT
  ))
  root.addHandler(console)

  if log_file:
    Path(log_file).parent.mkdir(
      parents=True, exist_ok=True
    )
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
      _LOG_FORMAT, datefmt=_DATE_FORMAT
    ))
    root.addHandler(fh)

  if json_log_file:
    Path(json_log_file).parent.mkdir(
      parents=True, exist_ok=True
    )
    jh = logging.FileHandler(
      json_log_file, mode='a'
    )
    jh.setLevel(level)
    jh.setFormatter(_JsonFormatter())
    root.addHandler(jh)


class _JsonFormatter(logging.Formatter):
  """Emits one JSON object per line for dashboard."""

  def format(self, record: logging.LogRecord) -> str:
    entry: dict[str, Any] = {
      'ts': record.created,
      'logger': record.name,
      'level': record.levelname,
      'msg': record.getMessage(),
    }
    if hasattr(record, 'data'):
      entry['data'] = record.data
    return json.dumps(entry, default=str)


def get_logger(subsystem: str) -> logging.Logger:
  """Return a namespaced logger.

  Usage:
    log = get_logger('env')       # truck_carton.env
    log = get_logger('reward')    # truck_carton.reward
    log = get_logger('training')  # truck_carton.training
  """
  return logging.getLogger(
    f'truck_carton.{subsystem}'
  )


class EpisodeLogger:
  """Accumulates structured episode data and
  emits it as a single JSON record at episode end.
  Designed for introspection and dashboard feeds."""

  def __init__(self) -> None:
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
      'Episode %d started (stage=%d,'
      ' trucks=%d, cartons=%d, grid=%dx%d)',
      episode_id, stage, num_trucks,
      num_cartons, *grid_size,
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
    record = {
      'step': step,
      'action': action,
      'action_type': action_type,
      'reward': round(reward, 4),
      'breakdown': {
        k: round(v, 4) for k, v in
        breakdown.items()
      },
      'num_placed': num_placed,
      'num_delivered': num_delivered,
      'active_truck': active_truck,
      'truck_states': truck_states,
      'truck_positions': [
        list(p) for p in truck_positions
      ],
    }
    self._steps.append(record)
    self._log.debug(
      'Step %d: action=%d (%s) reward=%.4f'
      ' placed=%d delivered=%d',
      step, action, action_type, reward,
      num_placed, num_delivered,
    )

  def end_episode(
    self,
    total_reward: float,
    num_placed: int,
    num_delivered: int,
    total_cartons: int,
    terminated: bool,
  ) -> dict[str, Any]:
    elapsed = time.monotonic() - self._start_time
    summary = {
      **self._meta,
      'total_reward': round(total_reward, 4),
      'num_placed': num_placed,
      'num_delivered': num_delivered,
      'total_cartons': total_cartons,
      'completion_rate': round(
        num_placed / max(total_cartons, 1), 4
      ),
      'delivery_rate': round(
        num_delivered / max(total_cartons, 1), 4
      ),
      'terminated': terminated,
      'num_steps': len(self._steps),
      'elapsed_seconds': round(elapsed, 3),
      'steps': self._steps,
    }
    self._log.info(
      'Episode %d done: reward=%.2f placed=%d/%d'
      ' delivered=%d/%d steps=%d (%.1fs)',
      self._episode_id, total_reward,
      num_placed, total_cartons,
      num_delivered, total_cartons,
      len(self._steps), elapsed,
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
    self._log.debug(
      'Truck %d routes %s->%s (dist=%.1f, type=%s)',
      truck_id, src, dst, distance, destination_type,
    )

  def log_packing(
    self,
    carton_id: int,
    truck_id: int,
    position: tuple[int, int, int],
    dims: tuple[int, int, int],
    weight: float,
  ) -> None:
    self._log.debug(
      'Packed carton %d into truck %d at %s'
      ' dims=%s weight=%.1f',
      carton_id, truck_id, position, dims, weight,
    )

  def log_delivery(
    self,
    truck_id: int,
    store_id: int,
    num_unloaded: int,
  ) -> None:
    self._log.debug(
      'Truck %d delivered %d cartons at store %d',
      truck_id, num_unloaded, store_id,
    )
