#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A/B and shadow testing router for serving.

Loaders are injected callables (e.g. FileModelLoader or
RegistryModelLoader) returning a model with ``predict()``,
so no MLflow access happens in this module. A loader may
also be a model itself or a callable taking features.
'''

import hashlib
import logging
import math
import os
import secrets
import time
from collections import deque

LOG = logging.getLogger(__name__)

_VALID_MODES = ('shadow', 'ab')

# Env var supplying the bucket salt when no explicit salt is
# given (e.g. in production, set a secret value per experiment).
_SALT_ENV_VAR = 'FORECAST_AB_TEST_SALT'

# Bounded in-memory route log (F-08): a long-running service
# must not grow ``logs`` without bound.
_MAX_LOGS = 10000

# Per-process random salt used when no env/explicit salt is
# set, so bucketing is not predictable from an empty salt.
_PROCESS_SALT = secrets.token_hex(16)


def _default_salt():
  '''Return the env salt or the per-process random salt.'''
  return os.environ.get(_SALT_ENV_VAR) or _PROCESS_SALT


def _scrub_request_id(request_id):
  '''Return a truncated hash of a request id for logging.

  The raw request id may carry PII or caller-chosen values,
  so log entries store only this one-way hash.
  '''
  return hashlib.sha256(
      str(request_id).encode('utf-8')).hexdigest()[:16]


def _bucket(request_id, salt=None) -> float:
  '''Return a stable salted hash bucket in [0.0, 1.0) for an id.

  The bucket is ``sha256(salt + request_id)`` so an attacker
  who does not know the salt cannot grind request ids to
  force a desired variant. The salt defaults to the
  ``FORECAST_AB_TEST_SALT`` environment variable.
  '''
  if salt is None:
    salt = _default_salt()
  material = str(salt) + str(request_id)
  digest = hashlib.sha256(material.encode('utf-8')).digest()
  return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def _normalize(prediction):
  '''Unwrap single-element model output to a float scalar.'''
  if hasattr(prediction, 'tolist'):
    prediction = prediction.tolist()
  if isinstance(prediction, (list, tuple)) and len(prediction) == 1:
    prediction = prediction[0]
  if isinstance(prediction, bool):
    return prediction
  if isinstance(prediction, (int, float)):
    return float(prediction)
  return prediction


def _resolve(loader):
  '''Resolve a loader to a predictor (model or callable).'''
  if hasattr(loader, 'predict'):
    return loader
  if callable(loader):
    try:
      candidate = loader()
    except TypeError:
      return loader
    if candidate is not None:
      return candidate
  return loader


def _predict(loader, features):
  '''Run one prediction, returning (value, latency_seconds).'''
  predictor = _resolve(loader)
  start = time.perf_counter()
  if hasattr(predictor, 'predict'):
    raw = predictor.predict(features)
  elif callable(predictor):
    raw = predictor(features)
  else:
    raise TypeError('loader must yield a predictor')
  latency = time.perf_counter() - start
  return _normalize(raw), latency


def _percentile(values, pct) -> float:
  '''Nearest-rank percentile of a list of numbers.'''
  if not values:
    return 0.0
  ordered = sorted(values)
  rank = math.ceil(pct / 100.0 * len(ordered))
  index = min(max(rank - 1, 0), len(ordered) - 1)
  return float(ordered[index])


def _aggregate(entries) -> dict:
  '''Aggregate prediction/latency entries for one variant.'''
  predictions = [e[0] for e in entries]
  latencies = [e[1] for e in entries]
  mean = sum(predictions) / len(predictions) if predictions else 0.0
  return {
    'mean_prediction': float(mean),
    'latency_p50': _percentile(latencies, 50),
    'latency_p95': _percentile(latencies, 95),
    'count': len(entries),
  }


def _is_numeric(value) -> bool:
  '''Return True for a finite real number (bools excluded).'''
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return False
  return math.isfinite(float(value))


def _latency(log, *keys) -> float:
  '''Return the first numeric latency among keys, else 0.0.'''
  for key in keys:
    value = log.get(key)
    if _is_numeric(value):
      return float(value)
  return 0.0


def compare_variants(logs) -> dict:
  '''Aggregate route logs per variant.

  Args:
    logs: Iterable of log dicts as written by ABRouter.route
      (or dicts with variant/prediction/latency keys).
      Entries that are not dicts, or whose prediction is
      None / non-numeric / malformed, are skipped.

  Returns:
    {'champion': {...}, 'challenger': {...}} each with
    mean_prediction, latency_p50, latency_p95, and count.
    Empty groups report 0.0 metrics and count 0.
  '''
  groups = {'champion': [], 'challenger': []}
  for log in logs or []:
    if not isinstance(log, dict):
      continue
    champion_pred = log.get('champion_prediction')
    challenger_pred = log.get('challenger_prediction')
    if 'champion_prediction' in log or 'challenger_prediction' in log:
      if _is_numeric(champion_pred):
        groups['champion'].append(
          (float(champion_pred),
           _latency(log, 'champion_latency', 'latency'))
        )
      if _is_numeric(challenger_pred):
        groups['challenger'].append(
          (float(challenger_pred),
           _latency(log, 'challenger_latency', 'latency'))
        )
    elif log.get('variant') in groups:
      prediction = log.get('prediction')
      if _is_numeric(prediction):
        groups[log['variant']].append(
          (float(prediction), _latency(log, 'latency'))
        )
  return {name: _aggregate(entries) for name, entries in groups.items()}


class ABRouter:
  '''Route requests between champion and challenger models.

  Args:
    champion_loader: Callable returning the champion model.
    challenger_loader: Callable returning the challenger model.
    traffic_split: Fraction of traffic sent to the challenger
      in ``ab`` mode, in [0.0, 1.0].
    mode: ``shadow`` (serve champion, also run challenger
      and log both) or ``ab`` (serve the bucketed variant).
    salt: Salt for the bucket hash (``sha256(salt + id)``).
      When None, read from the ``FORECAST_AB_TEST_SALT``
      environment variable ('' when unset); set a secret
      value in production per experiment.
  '''

  def __init__(
    self,
    champion_loader,
    challenger_loader,
    traffic_split: float = 0.1,
    mode: str = 'shadow',
    salt=None,
  ) -> None:
    '''Initialise the router with champion/challenger loaders.

    Args:
      champion_loader: Callable returning the champion model.
      challenger_loader: Callable returning the challenger model.
      traffic_split: Fraction of traffic sent to the
        challenger in ``ab`` mode, in [0.0, 1.0].
      mode: ``shadow`` or ``ab`` routing mode.
      salt: Bucket salt; None reads the
        ``FORECAST_AB_TEST_SALT`` environment variable.

    Raises:
      ValueError: If mode or traffic_split is invalid.
    '''
    if mode not in _VALID_MODES:
      raise ValueError('mode must be one of: shadow, ab')
    if not 0.0 <= float(traffic_split) <= 1.0:
      raise ValueError('traffic_split must be in [0.0, 1.0]')
    self.champion_loader = champion_loader
    self.challenger_loader = challenger_loader
    self.traffic_split = float(traffic_split)
    self.mode = mode
    if salt is None:
      env_salt = os.environ.get(_SALT_ENV_VAR)
      if env_salt:
        self.salt = env_salt
      else:
        self.salt = _PROCESS_SALT
        LOG.warning(
            'FORECAST_AB_TEST_SALT is not set; using per-process '
            'random salt for A/B bucketing. Set '
            'FORECAST_AB_TEST_SALT to a secret value in '
            'production.')
    elif str(salt) == '':
      self.salt = _PROCESS_SALT
      LOG.warning(
          'Empty A/B salt rejected; using per-process random '
          'salt for A/B bucketing. Set FORECAST_AB_TEST_SALT '
          'to a secret value in production.')
    else:
      self.salt = str(salt)
    self.logs: deque = deque(maxlen=_MAX_LOGS)

  def _variant(self, request_id) -> str:
    '''Bucket a request id deterministically (salted).'''
    if _bucket(request_id, self.salt) < self.traffic_split:
      return 'challenger'
    return 'champion'

  def route(self, request_id, features) -> dict:
    '''Route one request to a variant.

    Args:
      request_id: Stable identifier used for bucketing.
      features: Passed through to the model ``predict()``.

    Returns:
      {'prediction': ..., 'variant': 'champion'|'challenger'}.
      In shadow mode the served variant is always champion
      and both predictions/latencies are logged.
    '''
    if self.mode == 'shadow':
      champion_pred, champion_latency = _predict(self.champion_loader, features)
      try:
        challenger_pred, challenger_latency = _predict(
          self.challenger_loader, features
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Shadow must not lose the champion prediction.
        LOG.warning(
            'shadow challenger failed code=%s type=%s',
            'shadow_challenger_failed', type(exc).__name__)
        challenger_pred, challenger_latency = None, 0.0
      log = {
        'request_id_hash': _scrub_request_id(request_id),
        'variant': 'champion',
        'prediction': champion_pred,
        'champion_prediction': champion_pred,
        'challenger_prediction': challenger_pred,
        'champion_latency': champion_latency,
        'challenger_latency': challenger_latency,
        'latency': champion_latency,
      }
      self.logs.append(log)
      return {'prediction': champion_pred, 'variant': 'champion'}
    variant = self._variant(request_id)
    loader = (
      self.challenger_loader
      if variant == 'challenger'
      else self.champion_loader
    )
    prediction, latency = _predict(loader, features)
    log = {
      'request_id_hash': _scrub_request_id(request_id),
      'variant': variant,
      'prediction': prediction,
      'champion_prediction': (prediction if variant == 'champion' else None),
      'challenger_prediction': (
        prediction if variant == 'challenger' else None
      ),
      'champion_latency': latency if variant == 'champion' else None,
      'challenger_latency': (latency if variant == 'challenger' else None),
      'latency': latency,
    }
    self.logs.append(log)
    return {'prediction': prediction, 'variant': variant}

  @staticmethod
  def compare_variants(logs) -> dict:
    '''Aggregate logs; see :func:`compare_variants`.'''
    return compare_variants(logs)
