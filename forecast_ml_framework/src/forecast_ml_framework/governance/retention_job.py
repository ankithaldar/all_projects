#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Retention sweep and secure-channel erasure CLI.

The sweep calls the frozen retention enforcer read-only
and aggregates hashed audit entries. Erasure is an
operator action gated on ADR D-ERASURE-CLI-1: the clear
subject id comes only from the secure env channel named
below, never from argv and never from logs. Reports
carry the sole scrub hash only, via
:func:`governance.scrub.subject_hash`. Secret values are
read from the environment only; this module holds none.
'''
# pylint: disable=broad-exception-caught, import-outside-toplevel

import argparse
import os
from pathlib import Path

from forecast_ml_framework.governance.erasure import erase_subject
from forecast_ml_framework.governance.retention import (
  check_retention,
  enforce_retention,
  list_online_entities,
  list_snapshots,
)
from forecast_ml_framework.governance.scrub import subject_hash

__all__ = [
  'build_stores',
  'load_governance_params',
  'main',
  'run_erasure',
  'run_retention_sweep',
]

_DEFAULT_PARAMS = 'conf/base/platform.yml'
_ERASE_ENV = 'FORECAST_ERASE_SUBJECT_ID'
_ACTOR_ENVS = ('FORECAST_ACTOR', 'FORECAST_OPERATOR')
_DEFAULT_ACTOR = 'forecast-ml-retention-job'


def load_governance_params(path: str = 'conf/base/platform.yml'
                           ) -> dict:
  '''Load retention params from a platform params file.

  Args:
    path: Params file path. The sole source in P2 is
      ``conf/base/platform.yml`` under the key
      ``governance_params.retention_days``.

  Returns:
    Dict ``{'retention_days': int}``.

  Raises:
    ValueError: When the file is missing or unreadable,
      the key is missing, or the days value is not an
      int (bool and float are rejected) or is negative.
  '''
  try:
    text = Path(path).read_text(encoding='utf-8')
  except OSError as exc:
    raise ValueError('governance params file is unreadable') \
        from exc
  try:
    import yaml
  except ImportError as exc:
    raise ValueError('yaml loader is unavailable') from exc
  try:
    config = yaml.safe_load(text)
  except yaml.YAMLError as exc:
    raise ValueError('governance params file is invalid') from exc
  if not isinstance(config, dict):
    raise ValueError('governance params file is not a mapping')
  gov = config.get('governance_params')
  if not isinstance(gov, dict):
    raise ValueError('governance_params key is missing')
  days = gov.get('retention_days')
  if isinstance(days, bool) or not isinstance(days, int):
    raise ValueError('retention_days must be an int')
  if days < 0:
    raise ValueError('retention_days must be non-negative')
  return {'retention_days': days}


def build_stores(params: dict) -> list:
  '''Build store handles from params and the environment.

  Args:
    params: Mapping from :func:`load_governance_params`,
      optionally with ``artifact_base_path`` (or
      ``base_path``), ``stores`` (list of ``{'type': ...}``
      or plain type names), or ``store_type`` (``offline``,
      ``online``, or ``both``). Without a store spec both
      the offline and the online store are built. The
      offline root comes from the mapping, else the
      ``FORECAST_ARTIFACT_BASE_PATH`` env var, else
      ``./artifacts``. The online store uses the
      ``FORECAST_REDIS_URL`` env var when set.

  Returns:
    List of offline and/or online feature stores.

  Raises:
    ValueError: When params is not a mapping or names an
      unknown store type.
  '''
  if not isinstance(params, dict):
    raise ValueError('params must be a mapping')
  requested = _requested_kinds(params)
  base = (params.get('artifact_base_path')
          or params.get('base_path')
          or os.environ.get('FORECAST_ARTIFACT_BASE_PATH')
          or './artifacts')
  stores = []
  for kind in requested:
    if kind == 'offline':
      from forecast_ml_framework.core.artifact_store import (
        LocalArtifactStore,
      )
      from forecast_ml_framework.feature_store.offline import (
        OfflineFeatureStore,
      )
      stores.append(OfflineFeatureStore(
          LocalArtifactStore(str(base))))
    else:
      from forecast_ml_framework.feature_store.online import (
        OnlineFeatureStore,
      )
      redis_url = os.environ.get('FORECAST_REDIS_URL') or None
      stores.append(OnlineFeatureStore(redis_url=redis_url))
  return stores


def _requested_kinds(params):
  '''Resolve the store type names requested by params.

  Args:
    params: Params mapping passed to :func:`build_stores`.

  Returns:
    List of ``'offline'`` and/or ``'online'`` names.

  Raises:
    ValueError: When a named store type is unknown.
  '''
  if params.get('stores') is not None:
    specs = params['stores']
    if not isinstance(specs, (list, tuple)):
      raise ValueError('unknown store type')
    kinds = []
    for spec in specs:
      kind = spec.get('type') if isinstance(spec, dict) else spec
      if kind not in ('offline', 'online'):
        raise ValueError('unknown store type')
      kinds.append(kind)
    return kinds
  kind = params.get('store_type', params.get('store'))
  if kind is None:
    return ['offline', 'online']
  if kind == 'both':
    return ['offline', 'online']
  if kind not in ('offline', 'online'):
    raise ValueError('unknown store type')
  return [kind]


def _records_for_store(store):
  '''Enumerate retention records for one store.

  Args:
    store: Offline or online feature store.

  Returns:
    List of record dicts from the frozen enumerators.

  Raises:
    ValueError: When the store type is unknown.
  '''
  if callable(getattr(store, '_records_for', None)):
    return list_online_entities(store)
  return list_snapshots(store)


def _check_actor(actor):
  '''Raise unless an actor identity was supplied.

  Args:
    actor: Operator identity (hashed by the enforcer).

  Raises:
    ValueError: When the actor is blank.
  '''
  if actor is None or (isinstance(actor, str)
                       and not actor.strip()):
    raise ValueError('a non-empty actor is required')


def _check_params(params):
  '''Raise unless params carry an int retention_days.

  Args:
    params: Retention params mapping.

  Raises:
    ValueError: When retention_days is missing or bad.
  '''
  if not isinstance(params, dict):
    raise ValueError('params must be a mapping')
  days = params.get('retention_days')
  if isinstance(days, bool) or not isinstance(days, int):
    raise ValueError('retention_days must be an int')


def run_retention_sweep(*, stores: list, params: dict, actor: str,
                        dry_run: bool = False, now=None, clock=None,
                        log=None) -> dict:
  '''Sweep every store through the frozen retention enforcer.

  Args:
    stores: Offline and/or online feature stores.
    params: Retention params with ``retention_days``.
    actor: Operator identity. Only its scrub hash is
      audited, never the clear value.
    dry_run: When True, plan with the pure checker only
      and delete nothing (``swept`` stays 0).
    now: Optional reference time for the planner.
    clock: Optional injectable clock for the planner.
    log: Optional list receiving the audit entries. The
      same list is returned as ``audit_entries``.

  Returns:
    Dict ``{'swept': int, 'kept': int, 'audit_entries':
    [...]}`` with counts only, never clear identities.

  Raises:
    ValueError: When the actor or params are invalid or
      a store type is unknown.
    Exception: Store errors abort the sweep after the
      frozen enforcer appends a partial audit entry.
  '''
  _check_actor(actor)
  _check_params(params)
  audit_entries = log if log is not None else []
  swept = 0
  kept = 0
  for store in stores or []:
    if dry_run:
      records = _records_for_store(store)
      plan = check_retention(records, params, now=now, clock=clock)
      kept += len(plan['keep'])
      continue
    result = enforce_retention(store, params, actor, now=now,
                               clock=clock, log=audit_entries)
    swept += len(result['delete'])
    kept += len(result['keep'])
  return {'swept': swept, 'kept': kept,
          'audit_entries': audit_entries}


def run_erasure(*, subject_id: str, stores: list, artifacts: dict,
                actor: str, log=None, at=None, now=None,
                clock=None) -> dict:
  '''Erase one subject via the frozen erasure surface.

  This is the library path gated on ADR D-ERASURE-CLI-1.
  The clear ``subject_id`` is taken only from the secure
  env channel by the CLI caller; it is never logged here.

  Args:
    subject_id: Clear subject id (never logged).
    stores: Feature stores to delete rows from.
    artifacts: Artifact dicts receiving the markers.
    actor: Operator identity (hashed into the audit).
    log: Optional list receiving the audit entry.
    at: Explicit requested-at stamp for the markers.
    now: Fallback requested-at stamp.
    clock: Fallback clock for the requested-at stamp.

  Returns:
    The frozen erasure report dict with hashes only.

  Raises:
    ValueError: When the subject or actor is blank or a
      store/artifact shape is unknown.
    Exception: Store errors propagate after a partial
      audit entry, per the frozen erasure surface.
  '''
  if subject_id is None or (isinstance(subject_id, str)
                            and not subject_id.strip()):
    raise ValueError('a non-empty subject_id is required')
  _check_actor(actor)
  return erase_subject(subject_id, stores=stores,
                       artifacts=artifacts, actor=actor, log=log,
                       at=at, now=now, clock=clock)


def _actor_from_env():
  '''Return the operator actor from the environment.

  Returns:
    The ``FORECAST_ACTOR`` or ``FORECAST_OPERATOR`` env
    value, else the retention job service name.
  '''
  for name in _ACTOR_ENVS:
    value = os.environ.get(name)
    if value and str(value).strip():
      return str(value)
  return _DEFAULT_ACTOR


def main(argv: list[str] | None = None) -> int:
  '''Run the retention or secure-channel erasure CLI.

  Args:
    argv: Argument list, or None for ``sys.argv``.

  Returns:
    0 on success, 2 on validation error. Output carries
    counts and hashes only, never clear identities or
    secret values.
  '''
  parser = argparse.ArgumentParser(
      prog='retention_job',
      description='Retention sweep and erasure CLI.')
  parser.add_argument('--retention', action='store_true',
                      help='run the retention sweep (default)')
  parser.add_argument('--erase', '--erase-subject',
                      action='store_true', dest='erase',
                      help='erase the env subject id')
  parser.add_argument('--dry-run', action='store_true',
                      help='plan only, delete nothing')
  parser.add_argument('--governance-params',
                      default=_DEFAULT_PARAMS,
                      help='platform params file path')
  args = parser.parse_args(argv)
  subject_id = None
  if args.erase:
    if args.dry_run:
      print('error: --dry-run cannot be combined with --erase')
      return 2
    subject_id = os.environ.get(_ERASE_ENV)
    if subject_id is None or not str(subject_id).strip():
      print('error: env subject id is required for --erase')
      return 2
  try:
    params = load_governance_params(args.governance_params)
  except ValueError:
    print('error: invalid governance params')
    return 2
  actor = _actor_from_env()
  try:
    stores = build_stores(params)
    if args.erase:
      report = run_erasure(subject_id=subject_id, stores=stores,
                           artifacts={}, actor=actor)
      digest = report['subject_hash']
      removed = len(report['removed'])
      actor_hash = subject_hash(actor)
      print(f'erased subject_hash={digest} removed={removed} '
            f'actor_hash={actor_hash}')
    else:
      result = run_retention_sweep(
          stores=stores, params=params, actor=actor,
          dry_run=args.dry_run)
      swept = result['swept']
      kept = result['kept']
      actor_hash = subject_hash(actor)
      print(f'retention swept={swept} kept={kept} '
            f'actor_hash={actor_hash}')
  except Exception:
    print('error: job failed')
    return 2
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
