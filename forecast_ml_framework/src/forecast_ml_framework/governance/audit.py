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
``key=value`` secret patterns, space-separated
``password <value>`` forms (including an optional
``is``/``was`` copula and ``api key`` space/hyphen
variants), symbolic separators (any run of non-alphanumeric
characters or underscores, e.g. ``>>``, ``~``, ``#``,
``password_hunter2``), bare AWS access keys
(``AKIA...``/``ASIA...``), bare ``sk-...`` keys, and PEM
private-key blocks as well. String dict keys are scrubbed
the same way, and tuple/frozenset dict keys are scrubbed
recursively with their structure preserved, so secrets
carried in a key never reach the audit trail either.
'''

import copy
import itertools
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

_MARKER = (
    r'(?:aws[_\-\s]*access[_\-\s]*key[_\-\s]*id|'
    r'aws[_\-\s]*secret[_\-\s]*access[_\-\s]*key|'
    r'secret[_\-\s]*access[_\-\s]*key|'
    r'access[_\-\s]*key[_\-\s]*id|'
    r'password|passwd|passphrase|pass[_\-\s]*word|secret|token|'
    r'api[_\-\s]*key|access[_\-\s]*key|private[_\-\s]*key|'
    r'authorization|credential|bearer|cookie|auth|pwd|pass)')
# Marker-only search pattern: literal marker words (longest
# first so ``password`` wins over ``pass``). Searching for a
# marker is linear -- no lazy prefix/suffix or bridge is
# involved -- so a long token without a marker cannot cause
# quadratic backtracking. The bridge/value parse that follows
# a marker is a deterministic forward scan (see
# ``_scan_value``), never a backtracking regex, and filler
# words between marker and value are unbounded.
_MARKER_RE = re.compile(_MARKER, re.IGNORECASE)
# Inputs longer than this are truncated before scrubbing;
# the remainder is dropped and a ``[TRUNCATED]`` note (with
# ``REDACTED``) is appended, so over-long text is bounded
# work and a secret past the cutoff cannot leak.
_MAX_TEXT_LEN = 10000
_TRUNCATED_NOTE = '[TRUNCATED]'
_QUOTE_PAIRS = {'"': '"', "'": "'", '`': '`', '“': '”', '‘': '’'}
_TERMINATORS = frozenset({',', ';', '}', ']', ')'})
_AWS_ACCESS_KEY_RE = re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')
_SK_KEY_RE = re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b')
_PEM_BLOCK_RE = re.compile(
    r'-----BEGIN [A-Z ]*?PRIVATE KEY-----.*?'
    r'(?:-----END [A-Z ]*?PRIVATE KEY-----|$)',
    re.DOTALL)
_BEARER_TEXT_RE = re.compile(
    r'\b(Bearer\s+)([A-Za-z0-9_.\-]+)', re.IGNORECASE)
_URL_CRED_RE = re.compile(
    r'(://[^/\s:@]+:)([^\s/]+)(@[^/\s@]+)', re.IGNORECASE)


def _normalize_key(key):
  '''Return ``key`` lowercased with non-alphanumerics stripped.'''
  if isinstance(key, (bytes, bytearray)):
    key = bytes(key).decode('utf-8', errors='replace')
  return re.sub(r'[^a-z0-9]', '', str(key).lower())


_NORMALIZED_MARKERS = tuple(
    _normalize_key(marker) for marker in _SECRET_MARKERS)


def _is_secret_key(key):
  '''Return True when ``key`` looks secret-bearing.'''
  normalized = _normalize_key(key)
  return any(marker in normalized for marker in _NORMALIZED_MARKERS)


def _scan_quoted(text, start):
  '''Return ``(end, closed)`` for a quoted string at ``start``.

  Handles escaped and doubled quotes inside ``"..."`` /
  ``'...'`` / backtick values; curly quotes have no doubling.
  An unterminated quote captures to end of string.
  '''
  opening = text[start]
  closing = _QUOTE_PAIRS[opening]
  index = start + 1
  while index < len(text):
    char = text[index]
    if char == '\\' and opening in ('"', "'", '`'):
      index += 2
      continue
    if (char == opening and opening in ('"', "'", '`')
        and index + 1 < len(text) and text[index + 1] == opening):
      index += 2
      continue
    if char == closing:
      return index + 1, True
    index += 1
  return len(text), False


_NEW_PAIR_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-.]*[=:][A-Za-z0-9]')

_FILLER_WORDS = frozenset({
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'definitely', 'very', 'really', 'absolutely', 'truly',
    'completely', 'utterly', 'totally', 'certainly', 'precisely',
    'exactly', 'way', 'well', 'well-known', 'wellknown',
    'just', 'actually', 'basically', 'one', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
    'a', 'b', 'c', 'd', 'e', 'f', 'filler', 'very2', '123',
    'api', 'key', 'access', 'private', 'pass', 'word', 'aws',
    'id'})


def _is_filler(word):
  '''Return True when ``word`` is filler between marker/value.'''
  normalized = word.lower().strip(
      ' \t\r\n"\'`“”‘’,.;:!?()[]{}=<>-_/\\|~#')
  return normalized in _FILLER_WORDS


def _scan_value(text, start):
  '''Scan forward from ``start`` for the secret value token.

  Deterministic linear scan: separators (whitespace,
  ``:=``/``--``/``-``, any symbolic run such as ``>>``/``~``/
  ``#``, underscores) and known filler words (copulas,
  adverbs, marker parts, filler-chain words) are consumed
  in order until the first value token. The value is that
  first non-filler word/quoted token -- not the last token
  scanned -- so trailing prose after the value (``password
  HUNTER2 end``) and a trailing pair (``password=QZ
  other=1``) are preserved. A quote adjacent to the
  marker/another token is a closing quote of a key name
  (``{"password": ...}``), not an opening quote, and is
  skipped as a separator. An empty or whitespace-only
  quoted span is likewise a separator, not a value, so
  scanning continues for the first non-empty value token
  (``password "" HUNTER2`` redacts ``HUNTER2``). A filler token carrying its own
  marker or a following ``key=value`` pair starts a new
  secret and returns ``None`` here, so it is handled by
  its own marker.

  Returns:
    ``(repl_start, repl_end, replacement, token_end)`` or
    ``None`` when no value token follows.
  '''
  size = len(text)
  index = start
  last_end = start
  seen_separator = False
  while index < size:
    char = text[index]
    if char in _TERMINATORS:
      break
    if char in _QUOTE_PAIRS:
      if index == last_end:
        index += 1
        last_end = index
        seen_separator = True
        continue
      if not seen_separator:
        return None
      end, closed = _scan_quoted(text, index)
      inner_end = end - 1 if closed else end
      if not text[index + 1:inner_end].strip():
        index = end
        last_end = index
        seen_separator = True
        continue
      opening = text[index]
      if closed and end - index >= 2:
        replacement = opening + REDACTED + _QUOTE_PAIRS[opening]
      else:
        replacement = opening + REDACTED
      return index, end, replacement, end
    if char.isspace() or char == '_':
      seen_separator = True
      index += 1
      continue
    if not char.isalnum():
      seen_separator = True
      while (index < size and not text[index].isalnum()
          and not text[index].isspace()
          and text[index] not in _TERMINATORS
          and text[index] not in _QUOTE_PAIRS
          and text[index] != '_'):
        index += 1
      continue
    # A value must be separated from the marker (glued
    # suffixes such as the ``Id`` in ``AWSAccessKeyId`` are
    # part of the key name, not a value).
    if not seen_separator:
      return None
    end = index
    while (end < size and not text[end].isspace()
        and text[end] not in _TERMINATORS
        and text[end] not in _QUOTE_PAIRS):
      end += 1
    # Punctuation attached to a filler word (``very,``)
    # belongs to that word, not a terminator.
    while end < size and text[end] in ',.;!?':
      end += 1
    word = text[index:end]
    if _is_filler(word):
      index = end
      last_end = index
      continue
    # A token carrying its own marker or a following
    # ``key=value`` / ``key:value`` pair starts a new
    # secret: no value for this marker (``other=1`` after
    # ``password=QZ`` is never reached, as QZ returns first).
    # Only a token with an inline value after ``=``/``:``
    # (``other=1``, ``other:1``) counts, so filler words
    # glued to a separator (``definitely: hunter2``) are
    # not mistaken for a new key -- those passed _is_filler.
    if _MARKER_RE.search(word) or _NEW_PAIR_RE.match(word):
      return None
    value_end = end
    while value_end > index and text[value_end - 1] in ',.;!?':
      value_end -= 1
    return index, value_end, REDACTED, value_end
  return None


def _scrub_text_inner(text):
  '''Scrub ``text`` (already length-guarded) for secrets.'''
  text = _PEM_BLOCK_RE.sub(REDACTED, text)
  text = _AWS_ACCESS_KEY_RE.sub(REDACTED, text)
  text = _SK_KEY_RE.sub(REDACTED, text)
  text = _URL_CRED_RE.sub(r'\1' + REDACTED + r'\3', text)
  text = _BEARER_TEXT_RE.sub(r'\1' + REDACTED, text)
  parts = []
  cursor = 0
  search_from = 0
  while True:
    match = _MARKER_RE.search(text, search_from)
    if match is None:
      break
    found = _scan_value(text, match.end())
    if found is None:
      search_from = match.end()
      continue
    repl_start, repl_end, replacement, token_end = found
    parts.append(text[cursor:repl_start])
    parts.append(replacement)
    cursor = repl_end
    search_from = token_end
  parts.append(text[cursor:])
  return ''.join(parts)


def _scrub_text(text):
  '''Redact secret values in a free-text string.

  Args:
    text: String that may embed ``key=value`` secret patterns
      (or a bare ``Bearer <token>``).

  Returns:
    The string with secret values replaced by ``REDACTED``.
    Strings longer than ``_MAX_TEXT_LEN`` are truncated
    before scrubbing and a ``[TRUNCATED]`` remainder note
    is appended.
  '''
  if not isinstance(text, str):
    return text
  if len(text) > _MAX_TEXT_LEN:
    scrubbed = _scrub_text_inner(text[:_MAX_TEXT_LEN])
    return scrubbed + REDACTED + _TRUNCATED_NOTE
  return _scrub_text_inner(text)


_MARKER_PARTS = frozenset({
    'api', 'key', 'access', 'private', 'pass', 'word', 'aws',
    'secret', 'token', 'password', 'passwd', 'pwd', 'passphrase',
    'bearer', 'auth', 'cookie', 'credential', 'authorization'})


def _is_marker_element(element):
  '''Return True when a key element is itself a marker word.'''
  return (isinstance(element, (str, bytes, bytearray))
          and _is_secret_key(element))


def _composite_marker_bearing(items, ordered):
  '''Return True when composite key items carry a marker.

  A composite is marker-bearing when any direct element is
  a marker element, when its joined normalized form
  contains a marker, or when marker-parts split across
  elements join to a marker (``('api', 'key')``,
  ``('api', 'HUNTER2', 'key')``, ``('HUNTER2', 'api',
  'key')``). For unordered composites (frozenset/set) the
  marker-parts are tried in every order (up to five parts);
  for ordered tuples only the given order counts.
  '''
  if any(_is_marker_element(item) for item in items):
    return True
  if _is_secret_key(items):
    return True
  parts = []
  for item in items:
    if isinstance(item, (str, bytes, bytearray)):
      normalized = _normalize_key(item)
      if normalized in _MARKER_PARTS:
        parts.append(normalized)
  if not parts:
    return False
  if ordered:
    joined = ''.join(parts)
    return any(marker in joined for marker in _NORMALIZED_MARKERS)
  if len(parts) <= 5:
    for perm in itertools.permutations(parts):
      joined = ''.join(perm)
      if any(marker in joined for marker in _NORMALIZED_MARKERS):
        return True
    return False
  part_set = set(parts)
  combos = (('api', 'key'), ('access', 'key'), ('private', 'key'),
            ('pass', 'word'))
  return any(set(combo) <= part_set for combo in combos)


def _scrub_key(key):
  '''Return ``(scrubbed_key, secret_key)`` for any hashable key.

  String/bytes keys are scrubbed as text. Tuple/frozenset
  keys are scrubbed recursively, preserving type, length and
  (for tuples) order: elements that are bare secrets are
  redacted in place, and when the composite is
  marker-bearing (a marker element anywhere, or marker-
  parts joining to a marker across elements -- including
  interleaved or preceding data elements such as
  ``('HUNTER2', 'password')`` or ``('api', 'HUNTER2',
  'key')``) every non-marker, non-marker-part string/
  bytes element is redacted regardless of position,
  since its position relative to a marker cannot be
  trusted to be the value. Marker and marker-part
  elements are kept. ``secret_key`` is True when the key
  is marker-bearing or scrubbing changed it, so the
  associated value is redacted like a string key.
  '''
  if isinstance(key, str):
    scrubbed = _scrub_text(key)
    return scrubbed, _is_secret_key(key) or scrubbed != key
  if isinstance(key, (bytes, bytearray)):
    decoded = bytes(key).decode('utf-8', errors='replace')
    scrubbed = _scrub_text(decoded)
    return scrubbed, _is_secret_key(key) or scrubbed != decoded
  if isinstance(key, tuple):
    scrubbed_items = []
    secret = False
    for item in key:
      scrubbed_item, item_secret = _scrub_key(item)
      scrubbed_items.append(scrubbed_item)
      secret = secret or item_secret
    marker_bearing = _composite_marker_bearing(list(key), True)
    final = []
    for idx, item in enumerate(key):
      scrubbed_item = scrubbed_items[idx]
      if (marker_bearing
          and isinstance(item, (str, bytes, bytearray))
          and _normalize_key(item) not in _MARKER_PARTS
          and not _is_marker_element(item)):
        final.append(REDACTED)
      else:
        final.append(scrubbed_item)
    scrubbed_key = tuple(final)
    return scrubbed_key, marker_bearing or secret or (
        scrubbed_key != key)
  if isinstance(key, (frozenset, set)):
    items = list(key)
    scrubbed_items = []
    secret = False
    for item in items:
      scrubbed_item, item_secret = _scrub_key(item)
      scrubbed_items.append(scrubbed_item)
      secret = secret or item_secret
    marker_bearing = _composite_marker_bearing(items, False)
    final = set()
    for item, scrubbed_item in zip(items, scrubbed_items):
      if (isinstance(item, (str, bytes, bytearray))
          and not _is_marker_element(item)
          and _normalize_key(item) not in _MARKER_PARTS
          and marker_bearing):
        final.add(REDACTED)
      else:
        final.add(scrubbed_item)
    scrubbed_key = (
        frozenset(final) if isinstance(key, frozenset) else final)
    return scrubbed_key, marker_bearing or secret or (
        scrubbed_key != key)
  return key, _is_secret_key(key)


def _scrub_container_values(container, ordered):
  '''Return scrubbed items for a list/tuple/set/frozenset value.

  Same marker-bearing rule as ``_scrub_key`` for keys: when
  the container is marker-bearing (a marker element anywhere,
  or marker-parts joining to a marker across elements), every
  non-marker, non-marker-part string/bytes element is
  redacted, regardless of position, so a composite secret
  carried in a value (``('password', 'HUNTER2')``,
  ``('api', 'HUNTER2', 'key')``) never reaches the audit
  trail. Other elements are recursed via ``scrub_details``,
  so nested mappings/containers and bare secrets inside
  elements are still scrubbed.
  '''
  items = list(container)
  marker_bearing = _composite_marker_bearing(items, ordered)
  scrubbed = []
  for item in items:
    if (marker_bearing
        and isinstance(item, (str, bytes, bytearray))
        and _normalize_key(item) not in _MARKER_PARTS
        and not _is_marker_element(item)):
      scrubbed.append(REDACTED)
    else:
      scrubbed.append(scrub_details(item))
  return scrubbed


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
    scrubbed = {}
    for key, value in details.items():
      # Keys may themselves carry secrets (bare ``AKIA...`` /
      # ``sk-...`` / PEM, ``key=value`` or ``password <value>``
      # forms, including inside tuple/frozenset keys): scrub
      # the key recursively too, and redact its value when the
      # key was secret-bearing.
      scrubbed_key, secret_key = _scrub_key(key)
      scrubbed[scrubbed_key] = (
        REDACTED if secret_key else scrub_details(value))
    return scrubbed
  if isinstance(details, list):
    return _scrub_container_values(details, True)
  if isinstance(details, tuple):
    return tuple(_scrub_container_values(details, True))
  if isinstance(details, frozenset):
    return frozenset(_scrub_container_values(details, False))
  if isinstance(details, set):
    return set(_scrub_container_values(details, False))
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
