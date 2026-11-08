#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Hash-only consent ledger (pure, caller-persisted).

The ledger is a caller-supplied dict: this module invents
no consent database, API, or broker. Entries are keyed by
the scrub-leaf subject hash and store the actor hash only,
so a clear subject or actor id never appears as a ledger
key or value. The caller persists the ledger (for example
through the catalog); this module only reads and mutates
the supplied dict in place.

The D1 feature-store write guard consumes
:func:`check_consent` when a ledger is supplied and fails
closed without a granted record. The guard itself lives in
the stores, not here; this module owns the ledger
semantics only.
'''

from datetime import datetime, timezone

from forecast_ml_framework.governance.audit import audit_log
from forecast_ml_framework.governance.scrub import subject_hash


def _now_iso(at=None):
  '''Resolve an ``at`` stamp to ISO text (UTC now if None).'''
  stamp = at if at is not None else datetime.now(timezone.utc)
  iso = getattr(stamp, 'isoformat', None)
  if callable(iso):
    return iso()
  return str(stamp)


def _require_ledger(ledger):
  '''Raise unless the ledger is a dict.'''
  if not isinstance(ledger, dict):
    raise ValueError('consent ledger must be a dict')


def _require_subject(subject_id):
  '''Raise unless a subject id was supplied.'''
  if subject_id is None or (isinstance(subject_id, str)
                            and not subject_id.strip()):
    raise ValueError('consent requires a subject_id')


def _require_actor(actor):
  '''Raise unless an actor identity was supplied.'''
  if actor is None or (isinstance(actor, str)
                       and not actor.strip()):
    raise ValueError('consent requires a non-empty actor')


def record_consent(ledger, subject_id, *, granted, actor,
                   at=None, log=None):
  '''Record a consent decision under the subject hash only.

  Args:
    ledger: Caller-supplied dict, mutated in place. The
      entry is stored under ``subject_hash(subject_id)``
      and itself carries ``subject_hash``,
      ``actor_hash``, ``granted``, and ``at``; no clear id
      is stored as a key or value.
    subject_id: Clear subject id (never stored clear).
    granted: Decision (True for grant, False otherwise).
    actor: Operator identity (stored as its hash only).
    at: Decision stamp (datetime or text); the current
      UTC time when omitted.
    log: Optional list to append the audit entry to.

  Returns:
    A copy of the stored entry dict.

  Raises:
    ValueError: When the ledger, subject, or actor is
      invalid.
  '''
  _require_ledger(ledger)
  _require_subject(subject_id)
  _require_actor(actor)
  digest = subject_hash(subject_id)
  actor_hash = subject_hash(actor)
  entry = {'subject_hash': digest, 'granted': bool(granted),
           'actor_hash': actor_hash, 'at': _now_iso(at)}
  ledger[digest] = entry
  audit_log('consent_record', actor_hash,
            {'subject_hash': digest, 'granted': entry['granted']},
            log=log)
  return dict(entry)


def check_consent(ledger, subject_id):
  '''Return True only for a latest granted, unrevoked record.

  Args:
    ledger: Caller-supplied consent ledger dict.
    subject_id: Clear subject id (hashed for the lookup).

  Returns:
    True when the latest entry for the subject hash has
    ``granted`` exactly True. Missing, malformed, refused,
    and revoked (tombstoned ``granted`` False) records all
    return False; this never raises on lookup data.
  '''
  if not isinstance(ledger, dict) or subject_id is None:
    return False
  entry = ledger.get(subject_hash(subject_id))
  if not isinstance(entry, dict):
    return False
  return entry.get('granted') is True


def revoke_consent(ledger, subject_id, *, actor, at=None,
                   log=None):
  '''Revoke consent, leaving a tombstone hash record.

  The hash record is never deleted (audit trail): it is
  rewritten with ``granted`` False under the same hash key,
  so :func:`check_consent` returns False from then on.

  Workflow: revocation alone does not delete rows. The
  caller MUST then call ``erase_subject`` for the subject
  (revoke to erase) so feature rows and artifact markers
  follow the revocation.

  Args:
    ledger: Caller-supplied dict, mutated in place.
    subject_id: Clear subject id (never stored clear).
    actor: Operator identity (stored as its hash only).
    at: Revocation stamp (datetime or text); the current
      UTC time when omitted.
    log: Optional list to append the audit entry to.

  Returns:
    Dict ``{'subject_hash': hash}`` (hash only).

  Raises:
    ValueError: When the ledger, subject, or actor is
      invalid.
  '''
  _require_ledger(ledger)
  _require_subject(subject_id)
  _require_actor(actor)
  digest = subject_hash(subject_id)
  actor_hash = subject_hash(actor)
  ledger[digest] = {'subject_hash': digest, 'granted': False,
                    'actor_hash': actor_hash, 'at': _now_iso(at)}
  audit_log('consent_revoke', actor_hash,
            {'subject_hash': digest, 'granted': False}, log=log)
  return {'subject_hash': digest}
