#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Structured inventory and event logging with gzip rotation.'''

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


class GzipRotatingFileHandler(RotatingFileHandler):
  '''Rotating file handler that gzips rotated logs.'''

  def rotation_filename(self, default_name: str) -> str:
    '''Return the gzipped name for a rotated log file.

    Args:
      default_name: Default rotated file name.

    Returns:
      Rotated file name with a gzip suffix.
    '''
    return default_name + '.gz'

  def rotate(self, source: str, dest: str) -> None:
    '''Gzip the source log file into the destination file.

    Args:
      source: Path of the log file to compress.
      dest: Path of the compressed log file.
    '''
    with open(source, 'rb') as f_in:
      with gzip.open(dest, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(source)


class InventoryLogger:
  '''Logs inventory decisions, transitions, and rewards as JSON.'''

  def __init__(
    self,
    log_dir: str = 'output/logs',
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    name: str = 'crafting_rl',
  ):
    '''Initialize the logger with a rotating gzip file handler.

    Args:
      log_dir: Directory that holds the log file.
      max_bytes: Maximum size of one log file segment.
      backup_count: Number of rotated segments to keep.
      name: Logger name and log file base name.
    '''
    os.makedirs(log_dir, exist_ok=True)
    self._logger = logging.getLogger(name)
    self._logger.setLevel(logging.DEBUG)
    self._logger.propagate = False

    if not self._logger.handlers:
      log_path = os.path.join(log_dir, f'{name}.log')
      handler = GzipRotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
      )
      fmt = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
      )
      handler.setFormatter(fmt)
      self._logger.addHandler(handler)

  def _ts(self) -> str:
    '''Return the current UTC timestamp as a string.

    Returns:
      ISO-like timestamp string with microsecond precision.
    '''
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')

  def _serialize(self, data: Dict[str, Any]) -> str:
    '''Serialize a mapping to compact JSON.

    Args:
      data: Mapping to serialize.

    Returns:
      Compact JSON string representation.
    '''
    def default(obj: Any) -> Any:
      '''Convert unsupported objects to serializable values.

      Args:
        obj: Object that JSON cannot serialize natively.

      Returns:
        List, name, or string form of the object.
      '''
      if hasattr(obj, 'tolist'):
        return obj.tolist()
      if hasattr(obj, 'name'):
        return obj.name
      return str(obj)
    return json.dumps(data, default=default, separators=(',', ':'))

  def log_decision(
    self, tick: int, action: Dict, mask_info: Optional[Dict] = None
  ) -> None:
    '''Log an agent decision at a tick.

    Args:
      tick: Tick index of the decision.
      action: Action taken at the tick.
      mask_info: Optional action mask information.
    '''
    msg = self._serialize({
      'event': 'decision',
      'tick': tick,
      'action': action,
      'mask_info': mask_info,
    })
    self._logger.info(msg)

  def log_transition(
    self, tick: int, obs_before: Dict, obs_after: Dict
  ) -> None:
    '''Log an observation transition at a tick.

    Args:
      tick: Tick index of the transition.
      obs_before: Observation before the transition.
      obs_after: Observation after the transition.
    '''
    msg = self._serialize({
      'event': 'transition',
      'tick': tick,
      'stash_before': obs_before.get('stash'),
      'stash_after': obs_after.get('stash'),
      'coins_before': obs_before.get('coins'),
      'coins_after': obs_after.get('coins'),
    })
    self._logger.debug(msg)

  def log_reward(
    self, tick: int, reward: float, components: Dict[str, float]
  ) -> None:
    '''Log a reward at a tick.

    Args:
      tick: Tick index of the reward.
      reward: Total reward value.
      components: Mapping of reward component names to values.
    '''
    msg = self._serialize({
      'event': 'reward',
      'tick': tick,
      'reward': reward,
      'components': components,
    })
    self._logger.info(msg)

  def log_warning(self, tick: int, message: str) -> None:
    '''Log a warning message at a tick.

    Args:
      tick: Tick index of the warning.
      message: Warning message text.
    '''
    msg = self._serialize({
      'event': 'warning',
      'tick': tick,
      'message': message,
    })
    self._logger.warning(msg)

  def log_slot_event(
    self,
    tick: int,
    event_type: str,
    item_name: str,
    details: Optional[Dict] = None,
  ) -> None:
    '''Log a manufacturing slot event at a tick.

    Args:
      tick: Tick index of the event.
      event_type: Type of slot event.
      item_name: Name of the item involved.
      details: Optional event details mapping.
    '''
    msg = self._serialize({
      'event': 'slot',
      'tick': tick,
      'type': event_type,
      'item': item_name,
      'details': details or {},
    })
    self._logger.info(msg)

  def log_ga_event(
    self, generation: int, event_type: str, details: Dict
  ) -> None:
    '''Log a genetic algorithm event for a generation.

    Args:
      generation: Generation index of the event.
      event_type: Type of genetic algorithm event.
      details: Event details mapping.
    '''
    msg = self._serialize({
      'event': 'ga',
      'generation': generation,
      'type': event_type,
      'details': details,
    })
    self._logger.info(msg)

  def log_frame_skip(self, tick: int, skip_count: int) -> None:
    '''Log a frame skip at a tick.

    Args:
      tick: Tick index of the skip.
      skip_count: Number of frames skipped.
    '''
    msg = self._serialize({
      'event': 'frame_skip',
      'tick': tick,
      'skip_count': skip_count,
    })
    self._logger.debug(msg)
