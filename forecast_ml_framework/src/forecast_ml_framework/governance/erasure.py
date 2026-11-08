#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''GDPR erasure: delete rows, flag artifacts for retrain.

``erase_subject`` removes feature rows through the frozen
store surfaces (offline ``delete_entity``, online ``delete``
with its conditional Redis rule), then writes the exact
single-slot erasure markers documented per artifact: the
fitted registry carries ``__meta__.erasure`` and the
best-model envelope carries the top-level ``erasure`` key,
both ``{subject_hash, requested_at, action}`` with
``action`` ``'retrain_required'``. Fitted statistics are
never silently edited; the marker flags the artifact for
the retraining pipeline.

Persistence contract: this module mutates the caller's
in-memory artifact dicts in place. The caller re-saves them
through the existing writers (the fitted-registry JSON
dataset and the best-model restricted-pickle dataset). No
save callables and no dataset I/O live here.

Single-slot evidence note: a second subject erasure
overwrites the single-slot marker, so the prior hash is
lost from the artifact itself. Both hashes survive in the
appended audit entries; callers needing both must read the
audit trail, not the marker.

Predictions and monitoring rows are scoped out of P1: no
delete surface exists for them in this module. They are
reported as retained under legal hold, with the deferral
reason recorded below (D-PRED-DELETE-1 governs any future
surface plus the P11 gate).
'''
# pylint: disable=broad-exception-caught

from forecast_ml_framework.governance.audit import audit_log
from forecast_ml_framework.governance.retention import (
  _iso_now as _now_iso)
from forecast_ml_framework.governance.scrub import subject_hash

_RETAINED_LEGAL_HOLD = (
  'predictions/monitoring delete surface deferred '
  'per D-PRED-DELETE-1')


def _require_actor(actor):
  '''Raise unless an actor identity was supplied.'''
  if actor is None or (isinstance(actor, str)
                       and not actor.strip()):
    raise ValueError('erase_subject requires a non-empty actor')


def _is_present(store, subject_id):
  '''Return True when ``get_features`` finds the subject.

  Args:
    store: Feature store with a ``get_features`` method.
    subject_id: Clear subject id (never logged).

  Returns:
    True when a re-read at a far-future cutoff finds the
    subject; False when the store has no readable surface
    or the subject is absent.
  '''
  get_fn = getattr(store, 'get_features', None)
  if not callable(get_fn):
    return False
  result = get_fn([subject_id], as_of='9999-01-01')
  if isinstance(result, dict):
    return subject_id in result
  empty = getattr(result, 'empty', None)
  if empty is not None:
    return not bool(empty)
  try:
    return len(result) > 0
  except TypeError:
    return bool(result)


def _erase_from_store(store, subject_id, digest):
  '''Delete the subject from one store and verify the re-read.

  Args:
    store: Offline store (``delete_entity``) or online
      store (``delete``).
    subject_id: Clear subject id (never logged).
    digest: Scrub hash of the subject, for report labels.

  Returns:
    A hashed removal label when the subject was present,
    else None (a not-found subject is not an error).

  Raises:
    ValueError: When the store type is unknown.
    RuntimeError: When the subject survives deletion.
    Exception: Store errors (for example Redis) propagate.
  '''
  delete_entity = getattr(store, 'delete_entity', None)
  if callable(delete_entity):
    was_present = _is_present(store, subject_id)
    delete_entity(subject_id)
    if _is_present(store, subject_id):
      raise RuntimeError(
          'erasure verification failed for subject hash '
          + digest)
    return f'offline:{digest}' if was_present else None
  delete_fn = getattr(store, 'delete', None)
  if callable(delete_fn):
    was_present = _is_present(store, subject_id)
    delete_fn(subject_id)
    if _is_present(store, subject_id):
      raise RuntimeError(
          'erasure verification failed for subject hash '
          + digest)
    return f'online:{digest}' if was_present else None
  raise ValueError('unknown store/record type')


def _apply_markers(artifacts, digest, at=None, now=None, clock=None):
  '''Write exact erasure markers into artifact dicts in place.

  Args:
    artifacts: Mapping with optional ``fitted_registry``
      and ``best_model`` dicts (or None values).
    digest: Scrub hash of the erased subject.
    at: Explicit requested-at stamp.
    now: Fallback requested-at stamp.
    clock: Fallback clock for the requested-at stamp.

  Raises:
    ValueError: When artifacts or a present artifact has
      the wrong shape.
  '''
  if artifacts is None:
    return
  if not isinstance(artifacts, dict):
    raise ValueError('artifacts must be a dict')
  marker = {'subject_hash': digest,
            'requested_at': _now_iso(at=at, now=now, clock=clock),
            'action': 'retrain_required'}
  registry = artifacts.get('fitted_registry')
  if registry is not None:
    if not isinstance(registry, dict):
      raise ValueError('fitted_registry artifact must be a dict')
    meta = registry.get('__meta__')
    if meta is None:
      registry['__meta__'] = {'erasure': dict(marker)}
    elif not isinstance(meta, dict):
      raise ValueError('fitted_registry __meta__ must be a dict')
    else:
      meta['erasure'] = dict(marker)
  envelope = artifacts.get('best_model')
  if envelope is not None:
    if not isinstance(envelope, dict):
      raise ValueError('best_model artifact must be a dict')
    envelope['erasure'] = dict(marker)


def erase_subject(subject_id, *, stores, artifacts, actor,
                  log=None, at=None, now=None, clock=None):
  '''Erase one subject and flag artifacts for retrain.

  Args:
    subject_id: Clear subject id. Only its scrub hash is
      ever returned, audited, or logged.
    stores: Feature stores (offline and/or online). Rows
      are removed via ``delete_entity`` / ``delete`` and
      verified by re-read; store errors propagate.
    artifacts: Mapping with optional ``fitted_registry``
      and ``best_model`` dicts. Markers are written in
      place; the caller persists them via the existing
      JSON / restricted-pickle writers.
    actor: Operator identity (hashed into the audit entry).
    log: Optional list to append the audit entry to.
    at: Explicit requested-at stamp for the markers.
    now: Fallback requested-at stamp.
    clock: Fallback clock for the requested-at stamp.

  Returns:
    Dict with ``subject_hash`` (hash only, never clear),
    ``removed`` (hashed labels), ``retained_legal_hold``
    (the scoped-out predictions/monitoring deferral),
    ``audit_id`` (the audit entry timestamp), and
    ``audit_entry``. A not-found subject returns empty
    ``removed`` plus an audit entry, not an error.

  Raises:
    ValueError: When the subject/actor is blank, the
      artifacts shape is wrong, or a store type is unknown.
    RuntimeError: When a subject survives deletion.
    Exception: Store errors propagate only after a
      partial audit entry (hash only, ``partial: True``)
      is appended, mirroring the retention partial
      pattern: a partial erasure is never claimed done.
  '''
  if subject_id is None or (isinstance(subject_id, str)
                            and not subject_id.strip()):
    raise ValueError('erase_subject requires a subject_id')
  _require_actor(actor)
  digest = subject_hash(subject_id)
  actor_hash = subject_hash(actor)
  removed = []
  try:
    for store in stores or []:
      label = _erase_from_store(store, subject_id, digest)
      if label is not None:
        removed.append(label)
  except Exception:
    audit_log(
        'erasure', actor_hash,
        {'subject_hash': digest, 'removed': list(removed),
         'partial': True}, log=log)
    raise
  _apply_markers(artifacts, digest, at=at, now=now, clock=clock)
  retained = [_RETAINED_LEGAL_HOLD]
  entry = audit_log(
      'erasure', actor_hash,
      {'subject_hash': digest, 'removed': list(removed),
       'retained_legal_hold': list(retained)}, log=log)
  return {'subject_hash': digest, 'removed': removed,
          'retained_legal_hold': retained,
          'audit_id': entry.get('timestamp'),
          'audit_entry': entry}
