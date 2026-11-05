#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Audit log entries: actor-required, secret-scrubbed, timestamped.

Pure functions only: no persistence. The caller appends the
returned dict to its own log store. Details are deep-scrubbed
of secret-like keys (password/secret/token/api_key and similar,
matched on the normalized key name -- lowercased with all
non-alphanumerics stripped, so ``accessKey``, ``access_key``
and ``AWSAccessKeyId`` all match -- recursively) so secrets
never reach the audit trail. Free-text ``event``/``actor``
strings and string detail values are scrubbed for
``key=value`` secret patterns as well.
'''

import copy
import re
from datetime import datetime, timezone

REDACTED = '***REDACTED***'

_SECRET_MARKERS = ('password', 'passwd', 'pwd', 'pass', 'secret',
                   'token',
                   'api_key', 'apikey', 'api-key', 'access_key',
                   'accesskey', 'private_key', 'privatekey',
                   'authorization', 'credential', 'passphrase',
                   'bearer', 'auth', 'cookie', 'awsaccesskeyid',
                   'awssecretaccesskey')

_SECRET_TEXT_RE = re.compile(
    r'(["\']?[A-Za-z0-9_.\-]*?(?:password|passwd|pwd|passphrase|'
    r'pass|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|'
    r'bearer|auth|cookie|credential|authorization)'
    r'[A-Za-z0-9_.\-]*?["\']?\s*[:=]\s*["\']?)([^\s,;"\']+)',
    re.IGNORECASE)
_BEARER_TEXT_RE = re.compile(
    r'\b(Bearer\s+)([A-Za-z0-9_.\-]+)', re.IGNORECASE)
_URL_CRED_RE = re.compile(
    r'(://[^/\s:@]+:)([^\s/]+)(@[^/\s@]+)', re.IGNORECASE)


def _normalize_key(key):
  '''Return ``key`` lowercased with non-alphanumerics stripped.'''
  return re.sub(r'[^a-z0-9]', '', str(key).lower())


_NORMALIZED_MARKERS = tuple(
    _normalize_key(marker) for marker in _SECRET_MARKERS)


def _is_secret_key(key):
  '''Return True when ``key`` looks secret-bearing.'''
  normalized = _normalize_key(key)
  return any(marker in normalized for marker in _NORMALIZED_MARKERS)


def _scrub_text(text):
  '''Redact secret values in a free-text string.

  Args:
    text: String that may embed ``key=value`` secret patterns
      (or a bare ``Bearer <token>``).

  Returns:
    The string with secret values replaced by ``REDACTED``.
  '''
  if not isinstance(text, str):
    return text
  text = _URL_CRED_RE.sub(r'\1' + REDACTED + r'\3', text)
  text = _BEARER_TEXT_RE.sub(r'\1' + REDACTED, text)
  return _SECRET_TEXT_RE.sub(r'\1' + REDACTED, text)


def scrub_details(details):
  '''Return a deep copy of ``details`` with secrets redacted.

  Args:
    details: Mapping to scrub; nested mappings and lists are
      scrubbed recursively. String values (and a bare string
      ``details``) are scrubbed for ``key=value`` secret
      patterns; other non-mapping values are returned as-is.

  Returns:
    A new object; the input is never mutated.
  '''
  if isinstance(details, dict):
    return {
      key: (REDACTED if _is_secret_key(key)
            else scrub_details(value))
      for key, value in details.items()
    }
  if isinstance(details, list):
    return [scrub_details(item) for item in details]
  if isinstance(details, tuple):
    return tuple(scrub_details(item) for item in details)
  if isinstance(details, frozenset):
    return frozenset(scrub_details(item) for item in details)
  if isinstance(details, set):
    return {scrub_details(item) for item in details}
  if isinstance(details, (bytes, bytearray)):
    return _scrub_text(details.decode('utf-8', errors='replace'))
  if isinstance(details, str):
    return _scrub_text(details)
  return copy.deepcopy(details)


def _resolve_timestamp(clock, now):
  '''Resolve the entry timestamp via injected clock/now.'''
  stamp = now
  if stamp is None:
    stamp = clock() if callable(clock) else clock
  if stamp is None:
    stamp = datetime.now(timezone.utc)
  if isinstance(stamp, datetime):
    return stamp.isoformat()
  return str(stamp)


def audit_log(event, actor, details=None, clock=None, now=None,
              log=None):
  '''Build (and optionally append) one audit log entry.

  Args:
    event: The event name, e.g. ``'train'`` or ``'promote'``.
      Secret ``key=value`` patterns are redacted.
    actor: Who performed the event. Required, non-blank.
      Secret ``key=value`` patterns are redacted.
    details: Optional mapping; secret-like values are redacted.
      A bare string is scrubbed for secret patterns.
    clock: Optional injectable clock: a callable returning a
      datetime (or a datetime itself) used for the timestamp.
    now: Optional explicit timestamp (datetime or string),
      taking precedence over ``clock``.
    log: Optional list to append the entry to. The entry is
      returned either way; the caller owns persistence.

  Returns:
    Dict with ``event``, ``actor``, scrubbed ``details`` and
    an ISO-8601 ``timestamp``.

  Raises:
    ValueError: When ``actor`` is empty or blank.
  '''
  if actor is None or not str(actor).strip():
    raise ValueError('audit_log requires a non-empty actor')
  entry = {
    'event': _scrub_text(event),
    'actor': _scrub_text(actor),
    'details': scrub_details(details if details is not None else {}),
    'timestamp': _resolve_timestamp(clock, now),
  }
  if log is not None:
    log.append(entry)
  return entry
