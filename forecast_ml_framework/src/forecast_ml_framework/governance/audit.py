#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Audit log entries: actor-required, secret-scrubbed, timestamped.

Secret redaction rules live in ``governance.scrub``; see that
module for the full marker, filler and evasion policy.
'''

from datetime import datetime, timezone

from forecast_ml_framework.governance.scrub import (
    REDACTED,
    decode_bytes_scrubbed as _decode_bytes_scrubbed,
    safe_str as _safe_str,
    scrub_text as _scrub_text,
    scrub_details,
)

__all__ = ['REDACTED', 'audit_log', 'scrub_details']


def _clean_text(value):
  '''Return ``value`` as scrubbed, JSON-safe text.

  Bytes/bytearray are decoded with a guard: a hostile
  ``decode`` that raises or returns non-str becomes
  ``REDACTED`` (mirroring ``scrub_details``), never a
  crash or a leak. Other non-str values go through
  ``_safe_str`` so a hostile ``__str__`` becomes
  ``'<unrepresentable>'`` instead of raising; the
  resulting text is then scrubbed like a plain string.
  '''
  if isinstance(value, (bytes, bytearray)):
    return _decode_bytes_scrubbed(value)
  if isinstance(value, str):
    return _scrub_text(value)
  text = _safe_str(value)
  if text == '<unrepresentable>':
    return text
  return _scrub_text(text)


def _resolve_timestamp(clock, now):
  '''Resolve the entry timestamp via injected clock/now.

  The stamp is always a scrubbed, JSON-safe string:
  datetimes go through a guarded ``isoformat`` (a
  hostile override that raises or returns a secret
  falls back to ``_safe_str``, then scrubbed), bytes
  through the ``_clean_text`` decode guard, and any
  other value through ``_safe_str`` then scrubbed.
  A hostile clock callable that raises falls back to
  the current UTC time instead of crashing.
  '''
  stamp = now
  if stamp is None:
    if callable(clock):
      try:
        stamp = clock()
      except Exception:
        stamp = None
    else:
      stamp = clock
  if stamp is None:
    stamp = datetime.now(timezone.utc)
  if isinstance(stamp, datetime):
    try:
      text = stamp.isoformat()
    except Exception:
      text = _safe_str(stamp)
    if not isinstance(text, str):
      return '<unrepresentable>'
    return _scrub_text(text)
  return _clean_text(stamp)


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
  if actor is None:
    raise ValueError('audit_log requires a non-empty actor')
  actor_text = _clean_text(actor)
  if not actor_text.strip():
    raise ValueError('audit_log requires a non-empty actor')
  entry = {
    'event': _clean_text(event),
    'actor': actor_text,
    'details': scrub_details(details if details is not None else {}),
    'timestamp': _resolve_timestamp(clock, now),
  }
  if log is not None:
    log.append(entry)
  return entry
