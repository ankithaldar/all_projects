#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Secret scrubbing facade (canonical public leaf).

Markers/confusables/key normalization live in
``scrub_keys``; text/composite scrubbing lives in
``scrub_text``. This module re-exports their
names verbatim (same objects, same public API)
plus ``subject_hash`` and ``scrub_details`` with
key/container helpers, so callers and tests keep
their ``governance.scrub`` imports unchanged.

Leaf invariant: this facade loads its same-leaf
submodules dynamically (``importlib``) so the
static import surface stays stdlib-only
(``hashlib``/``itertools``/``importlib``) and
telemetry/governance-audit can never be pulled
in transitively. Both submodules are
stdlib-only leaves, so the dynamic load cannot
cycle. Static placeholders below keep every
re-exported name visible to static analysis;
they are rebound to the leaf objects at import.
Owns: ``subject_hash``, key/container
scrubbing, ``scrub_details`` dispatch.
'''

import hashlib
import importlib
import itertools

# Static placeholders below keep every re-exported
# name visible to static analysis as a callable;
# importlib rebinds them to the leaf objects.

REDACTED = None
_ALIGN_CANDIDATES = None
_AWS_ACCESS_KEY_RE = None
_BARE_SCHEME_TEXT_RE = None
_BEARER_TEXT_RE = None
_COMPOUND_PATTERNS = None
_CONFUSABLES = None
_CONFUSABLES_ALT = None
_CONFUSABLE_TABLE = None
_CONFUSABLE_TABLE_ALT = None
_FILLER_WORDS = None
_GLUED_PAREN_DIGIT_RE = None
_GLUE_FILLER_WORDS = None
_GLUE_FLANK_PARTS = None
_GLUE_RUN_RE = None
_GLUE_TOKEN_RE = None
_GLUE_TOKEN_WORDS = None
_HYBRID_CHAR_CACHE = None
_INTRA_FILLER = None
_INTRA_FILLER_SET = None
_INTRA_FILLER_WORDS = None
_MARKER = None
_MARKER_PARTS = None
_MARKER_PHRASE_PARTS = None
_MARKER_RE = None
_MAX_COMPOSITE_DEPTH = None
_MAX_ITEMS = None
_MAX_KEY_LEN = None
_MAX_TEXT_LEN = None
_NEW_PAIR_RE = None
_NON_ALNUM_RE = None
_NORMALIZED_MARKERS = None
_PEM_BLOCK_RE = None
_QUOTE_PAIRS = None
_SECRET_MARKERS = None
_SEP = None
_SK_KEY_RE = None
_SPACE_MARK_CPS = None
_SPACE_MARK_TABLE = None
_SPLIT_LETTERS = None
_SPLIT_LETTERS_ALT = None
_SPLIT_LETTER_TABLE = None
_SPLIT_LETTER_TABLE_ALT = None
_TERMINATORS = None
_TOKEN_RE = None
_TRUNCATED_NOTE = None
_URL_CRED_RE = None
_ZERO_WIDTH = None
def _align_opcodes(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _apply_spans(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _build_hybrid_text(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _build_hybrid_text_via_facade(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _char_hybrid_segments(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _choose_align_step(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _combine_skeleton_results(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _composite_has_over_cap_leaf(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _composite_marker_bearing(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _compound_tokens(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _decode_bytes_scrubbed(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _drop_format_and_modifier_chars(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _drop_nonspacing_marks(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _equal_run(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _extend_past_terminators(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _finalize_mapped_span(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _find_compound_marker(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _flatten_composite(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hashable(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hybrid_contains(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hybrid_exact(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hybrid_key_reading(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hybrid_try_targets(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _hybrid_units_from_original(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_base64ish_token(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_filler(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_ly_adverb(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_marker_element(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_marker_phrase_word(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _is_secret_key(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _key_as_text(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _key_middle_has_marker(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _map_alt_span(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _map_alt_span_block(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _marker_part(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _match_compound_at(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _normalize_key(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _normalize_unicode(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _ordered_parts(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _ordered_parts_dict(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _safe_list(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _safe_repr(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _safe_str(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_full_token_end(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_glued_branch(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_quoted(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_quoted_branch(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_token_end(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_value(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scan_word_branch(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_text(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_text_hybrid(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_text_inner_spans(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_text_inputs(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_text_skeletons(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _scrub_type_name(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _short_glued_continue_end(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _skip_symbol_run(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _spans_from_set(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _strip_glued_filler(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _strip_glued_filler_run(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _sub_spans(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _trim_value_end(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _union_spans(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError
def _unknown_key(*_a, **_k):
  '''Stub; rebound at import.'''
  raise NotImplementedError

_keys = importlib.import_module(
    'forecast_ml_framework.governance.scrub_keys')
_text = importlib.import_module(
    'forecast_ml_framework.governance.scrub_text')
globals().update({
    _k: _v for _k, _v in vars(_keys).items()
    if not _k.startswith('__')})
globals().update({
    _k: _v for _k, _v in vars(_text).items()
    if not _k.startswith('__')})
del _keys, _text

__all__ = [
    'MAX_COMPOSITE_DEPTH',
    'MAX_ITEMS',
    'REDACTED',
    '_ALIGN_CANDIDATES',
    '_AWS_ACCESS_KEY_RE',
    '_BARE_SCHEME_TEXT_RE',
    '_BEARER_TEXT_RE',
    '_COMPOUND_PATTERNS',
    '_CONFUSABLES',
    '_CONFUSABLES_ALT',
    '_CONFUSABLE_TABLE',
    '_CONFUSABLE_TABLE_ALT',
    '_FILLER_WORDS',
    '_GLUED_PAREN_DIGIT_RE',
    '_GLUE_FILLER_WORDS',
    '_GLUE_FLANK_PARTS',
    '_GLUE_RUN_RE',
    '_GLUE_TOKEN_RE',
    '_GLUE_TOKEN_WORDS',
    '_HYBRID_CHAR_CACHE',
    '_INTRA_FILLER',
    '_INTRA_FILLER_SET',
    '_INTRA_FILLER_WORDS',
    '_MARKER',
    '_MARKER_PARTS',
    '_MARKER_PHRASE_PARTS',
    '_MARKER_RE',
    '_MAX_COMPOSITE_DEPTH',
    '_MAX_ITEMS',
    '_MAX_KEY_LEN',
    '_MAX_TEXT_LEN',
    '_NEW_PAIR_RE',
    '_NON_ALNUM_RE',
    '_NORMALIZED_MARKERS',
    '_PEM_BLOCK_RE',
    '_QUOTE_PAIRS',
    '_SECRET_MARKERS',
    '_SEP',
    '_SK_KEY_RE',
    '_SPACE_MARK_CPS',
    '_SPACE_MARK_TABLE',
    '_SPLIT_LETTERS',
    '_SPLIT_LETTERS_ALT',
    '_SPLIT_LETTER_TABLE',
    '_SPLIT_LETTER_TABLE_ALT',
    '_TERMINATORS',
    '_TOKEN_RE',
    '_TRUNCATED_NOTE',
    '_URL_CRED_RE',
    '_ZERO_WIDTH',
    '_align_opcodes',
    '_apply_spans',
    '_build_hybrid_text',
    '_build_hybrid_text_via_facade',
    '_char_hybrid_segments',
    '_choose_align_step',
    '_combine_skeleton_results',
    '_composite_has_over_cap_leaf',
    '_composite_marker_bearing',
    '_compound_tokens',
    '_decode_bytes_scrubbed',
    '_drop_format_and_modifier_chars',
    '_drop_nonspacing_marks',
    '_equal_run',
    '_extend_past_terminators',
    '_finalize_mapped_span',
    '_find_compound_marker',
    '_flatten_composite',
    '_hashable',
    '_hybrid_contains',
    '_hybrid_exact',
    '_hybrid_key_reading',
    '_hybrid_try_targets',
    '_hybrid_units_from_original',
    '_is_base64ish_token',
    '_is_filler',
    '_is_ly_adverb',
    '_is_marker_element',
    '_is_marker_phrase_word',
    '_is_secret_key',
    '_key_as_text',
    '_key_middle_has_marker',
    '_map_alt_span',
    '_map_alt_span_block',
    '_marker_part',
    '_match_compound_at',
    '_normalize_key',
    '_normalize_unicode',
    '_ordered_parts',
    '_ordered_parts_dict',
    '_safe_list',
    '_safe_repr',
    '_safe_str',
    '_scan_full_token_end',
    '_scan_glued_branch',
    '_scan_quoted',
    '_scan_quoted_branch',
    '_scan_token_end',
    '_scan_value',
    '_scan_word_branch',
    '_scrub_text',
    '_scrub_text_hybrid',
    '_scrub_text_inner_spans',
    '_scrub_text_inputs',
    '_scrub_text_skeletons',
    '_scrub_type_name',
    '_short_glued_continue_end',
    '_skip_symbol_run',
    '_spans_from_set',
    '_strip_glued_filler',
    '_strip_glued_filler_run',
    '_sub_spans',
    '_trim_value_end',
    '_union_spans',
    '_unknown_key',
    'composite_marker_bearing',
    'decode_bytes_scrubbed',
    'flatten_composite',
    'is_marker_element',
    'is_secret_key',
    'marker_part',
    'normalize_key',
    'normalize_unicode',
    'safe_list',
    'safe_repr',
    'safe_str',
    'scrub_details',
    'scrub_text',
    'scrub_type_name',
    'subject_hash',
    'unknown_key',
]

def subject_hash(subject_id: str) -> str:
  '''Return the sole sha256 subject hash (64-hex lowercase).

  This is the ONLY sha256 subject-hash in the codebase: every
  erasure marker, log, audit, and report that must reference a
  subject without leaking its clear ID computes
  :func:`subject_hash` here (stdlib ``hashlib`` leaf, no
  telemetry/audit imports).

  Args:
    subject_id: Clear subject/entity identifier to hash.

  Returns:
    Lowercase 64-character hex digest of the utf-8 encoding.
  '''
  return hashlib.sha256(str(subject_id).encode('utf-8')).hexdigest()

def _scrub_key_items(items, marker_bearing, seen_next, depth):
  '''Scrub each composite item, returning ``(items, secret)``.'''
  scrubbed_items = []
  secret = False
  for item in items:
    scrubbed_item, item_secret = _scrub_key(
        item, marker_bearing, seen_next, depth + 1)
    scrubbed_items.append(scrubbed_item)
    secret = secret or item_secret
  return scrubbed_items, secret


def _scrub_key(key, _inherited_marker=False, _seen=None, _depth=0):
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
  ``_inherited_marker`` carries an outer composite's
  marker-bearing context into nested tuples, so a secret
  nested at any depth under a marker is redacted.
  Cyclic/deep keys yield ``'<cyclic>'``/``'<max-depth>'``
  with ``secret_key`` True, mirroring ``scrub_details``.
  '''
  if isinstance(key, str):
    return _scrub_str_key(key)
  if isinstance(key, (bytes, bytearray)):
    return _scrub_bytes_key(key)
  if isinstance(key, tuple):
    return _scrub_tuple_key(key, _inherited_marker, _seen, _depth)
  if isinstance(key, (frozenset, set)):
    return _scrub_set_key(key, _inherited_marker, _seen, _depth)
  if key is None or isinstance(key, (bool, int, float, complex)):
    return key, False
  # Unknown / hostile key: never keep the original object (its
  # ``__str__``/``__repr__`` would execute and could leak or
  # raise downstream). See ``_unknown_key``: always a
  # type-only placeholder; scrubbing decides the secret flag.
  return _unknown_key(key)


def _scrub_str_key(key):
  '''Scrub a str key (loop14 split).'''
  # Cycle7 Q2 (F-SCRUB-LONGKEY): detect on the full key
  # before truncation, and fail closed on truncation: a
  # key past the cap is unverifiable (a marker could sit
  # anywhere past it), so its value is redacted like a
  # truncated free text (``_scrub_text`` suffix).
  over_cap = len(key) > _MAX_KEY_LEN
  secret = _is_secret_key(key)
  # Loop20 P4: cap key length before scrub.
  if over_cap:
    key = key[:_MAX_KEY_LEN]
  scrubbed = _scrub_text(key)
  return scrubbed, secret or over_cap or scrubbed != key


def _scrub_bytes_key(key):
  '''Scrub a bytes key (loop14 split).'''
  # Cycle7 Q2 (F-SCRUB-LONGKEY): as ``_scrub_str_key``.
  over_cap = len(key) > _MAX_KEY_LEN
  secret = _is_secret_key(key)
  # Loop20 P4: cap decoded bytes key length.
  if over_cap:
    key = bytes(key)[:_MAX_KEY_LEN]
  decoded = bytes(key).decode('utf-8', errors='replace')
  scrubbed = _scrub_text(decoded)
  return scrubbed, secret or over_cap or scrubbed != decoded


def _finalize_tuple_key(items, scrubbed_items, marker_bearing):
  '''Redact non-marker tuple elements (loop14 split).'''
  final = []
  for idx, item in enumerate(items):
    scrubbed_item = scrubbed_items[idx]
    if (marker_bearing
        and isinstance(item, (str, bytes, bytearray))
        and _marker_part(item) is None
        and not _is_marker_element(item)):
      final.append(REDACTED)
    else:
      final.append(scrubbed_item)
  return final


def _scrub_tuple_key(key, _inherited_marker, _seen, _depth):
  '''Scrub a tuple key (loop14 split).'''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>', True
  if id(key) in _seen:
    return '<cyclic>', True
  seen_next = _seen | {id(key)}
  items = _safe_list(key)
  if items is None:
    return _unknown_key(key)
  # Cycle8 Q5 (C8-01): an over-cap leaf is unverifiable, so
  # treat it as marker-bearing for sibling finalization only
  # (``_is_secret_key`` itself stays non-blanket-fail-closed).
  unverifiable = _composite_has_over_cap_leaf(
      items, _seen=_seen, _depth=_depth)
  marker_bearing = (
      _inherited_marker
      or unverifiable
      or _composite_marker_bearing(
          items, True, _seen=_seen, _depth=_depth))
  scrubbed_items, secret = _scrub_key_items(
      items, marker_bearing, seen_next, _depth)
  scrubbed_key = tuple(
    _finalize_tuple_key(items, scrubbed_items, marker_bearing))
  try:
    changed = scrubbed_key != tuple(items)
  except Exception:
    changed = True
  return scrubbed_key, marker_bearing or secret or changed


def _finalize_set_key(items, scrubbed_items, marker_bearing):
  '''Redact non-marker set elements (loop14 split).'''
  final = set()
  for item, scrubbed_item in zip(items, scrubbed_items):
    if (isinstance(item, (str, bytes, bytearray))
        and not _is_marker_element(item)
        and _marker_part(item) is None
        and marker_bearing):
      final.add(REDACTED)
    else:
      final.add(_hashable(scrubbed_item))
  return final


def _scrub_set_key(key, _inherited_marker, _seen, _depth):
  '''Scrub a frozenset/set key (loop14 split).'''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>', True
  if id(key) in _seen:
    return '<cyclic>', True
  seen_next = _seen | {id(key)}
  items = _safe_list(key)
  if items is None:
    return _unknown_key(key)
  # Cycle8 Q5 (C8-01): as ``_scrub_tuple_key``.
  unverifiable = _composite_has_over_cap_leaf(
      items, _seen=_seen, _depth=_depth)
  marker_bearing = (
      _inherited_marker
      or unverifiable
      or _composite_marker_bearing(
          items, False, _seen=_seen, _depth=_depth))
  scrubbed_items, secret = _scrub_key_items(
      items, marker_bearing, seen_next, _depth)
  final = _finalize_set_key(items, scrubbed_items, marker_bearing)
  scrubbed_key = (
      frozenset(final) if isinstance(key, frozenset) else final)
  try:
    original = (
        frozenset(items) if isinstance(key, frozenset)
        else set(items))
    changed = scrubbed_key != original
  except Exception:
    changed = True
  return scrubbed_key, marker_bearing or secret or changed


def _scrub_container_values(
    container, ordered, _inherited_marker=False,
    _seen=None, _depth=0):
  '''Return scrubbed items for a list/tuple/set/frozenset value.

  Same marker-bearing rule as ``_scrub_key`` for keys: when
  the container is marker-bearing (a marker element anywhere,
  or marker-parts joining to a marker across elements), every
  non-marker, non-marker-part string/bytes element is
  redacted, regardless of position, so a composite secret
  carried in a value (``('password', 'HUNTER2')``,
  ``('api', 'HUNTER2', 'key')``) never reaches the audit
  trail. ``_inherited_marker`` carries an outer composite's
  marker-bearing context into nested containers at any
  depth. Other elements are recursed via ``scrub_details``,
  so nested mappings/containers and bare secrets inside
  elements are still scrubbed. ``_seen`` holds ids on the
  current path and ``_depth`` caps recursion at
  ``_MAX_COMPOSITE_DEPTH``: a cyclic element yields
  ``'<cyclic>'`` and a too-deep element ``'<max-depth>'``
  (same placeholders as ``telemetry.exceptions._coerce``)
  instead of raising ``RecursionError``.
  '''
  # Probe one past the cap so a truncated sentinel already
  # present (e.g. from ``telemetry.exceptions._coerce``) at
  # index ``_MAX_ITEMS`` is preserved, while hostile /
  # infinite iteration is still bounded at ``_MAX_ITEMS + 1``.
  if _seen is None:
    _seen = frozenset()
  seen_next = _seen | {id(container)}
  try:
    items = list(
        itertools.islice(container, _MAX_ITEMS + 1))
  except Exception:
    return [{'type': _scrub_type_name(type(container).__name__)}]
  # Cycle8 Q5 (C8-01): as ``_scrub_tuple_key`` -- an over-cap
  # leaf is unverifiable, so its container siblings redact.
  unverifiable = _composite_has_over_cap_leaf(
      items, _seen=_seen, _depth=_depth)
  marker_bearing = (
      _inherited_marker
      or unverifiable
      or _composite_marker_bearing(
          items, ordered, _seen=_seen, _depth=_depth))
  return [
    _scrub_container_item(item, marker_bearing, seen_next, _depth)
    for item in items]


def _scrub_container_item(item, marker_bearing, seen_next, _depth):
  '''Scrub one container item (loop14 split).'''
  if (marker_bearing
      and isinstance(item, (str, bytes, bytearray))
      and _marker_part(item) is None
      and not _is_marker_element(item)):
    return REDACTED
  if isinstance(item, (list, tuple)):
    if id(item) in seen_next:
      return '<cyclic>'
    if _depth + 1 > _MAX_COMPOSITE_DEPTH:
      return '<max-depth>'
    if isinstance(item, tuple):
      return tuple(_scrub_container_values(
        item, True, marker_bearing, seen_next, _depth + 1))
    return _scrub_container_values(
      item, True, marker_bearing, seen_next, _depth + 1)
  if isinstance(item, (frozenset, set)):
    return _scrub_container_set_item(
      item, marker_bearing, seen_next, _depth)
  if isinstance(item, dict):
    return scrub_details(item, marker_bearing, seen_next, _depth + 1)
  return scrub_details(item, _seen=seen_next, _depth=_depth + 1)


def _scrub_container_set_item(item, marker_bearing, seen_next, _depth):
  '''Scrub a set/frozenset item, keeping it hashable.'''
  if id(item) in seen_next:
    return '<cyclic>'
  if _depth + 1 > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>'
  # Set elements must stay hashable: an unknown element
  # scrubs to ``{'type': name}`` (a dict), so wrap every
  # scrubbed element with ``_hashable`` (safe repr).
  scrubbed_set = [
    _hashable(v) for v in _scrub_container_values(
      item, False, marker_bearing, seen_next, _depth + 1)]
  return (frozenset(scrubbed_set)
          if isinstance(item, frozenset) else set(scrubbed_set))




def scrub_details(
    details, _inherited_marker=False, _seen=None, _depth=0):
  '''Return a deep copy of ``details`` with secrets redacted.

  Args:
    details: Mapping to scrub; nested mappings and lists are
      scrubbed recursively. String values (and a bare string
      ``details``) are scrubbed for ``key=value`` secret
      patterns; scalar values are returned as-is.

  Returns:
    A new object; the input is never mutated. Only
    dict/list/tuple/set/frozenset/scalars/str/bytes are
    traversed: unknown objects are never deep-copied (which
    could execute ``__deepcopy__``/``__reduce__`` and fail on
    file handles) and are replaced by ``{'type': name}``.
    Cyclic containers (a dict/list/tuple/set referring to an
    ancestor on the current path) yield the string
    ``'<cyclic>'`` for the repeat, and nesting past
    ``_MAX_COMPOSITE_DEPTH`` yields ``'<max-depth>'`` --
    the same placeholders as ``_flatten_composite`` and
    ``telemetry.exceptions._coerce`` -- instead of raising
    ``RecursionError``. ``_seen``/``_depth`` carry that
    path state through dict/list/tuple/set/frozenset
    recursion.
  '''
  if _seen is None:
    _seen = frozenset()
  if isinstance(details, (dict, list, tuple, frozenset, set)):
    if _depth > _MAX_COMPOSITE_DEPTH:
      return '<max-depth>'
    if id(details) in _seen:
      return '<cyclic>'
    _seen = _seen | {id(details)}
  if isinstance(details, dict):
    return _scrub_dict_details(
      details, _inherited_marker, _seen, _depth)
  if isinstance(details, (list, tuple, frozenset, set)):
    return _scrub_container_details(
      details, _inherited_marker, _seen, _depth)
  return _scrub_leaf_details(details, _inherited_marker)


def _scrub_dict_details(details, _inherited_marker, _seen, _depth):
  '''Scrub a dict ``details`` (loop14 split).'''
  try:
    # ``+ 1`` preserves a truncated sentinel key already
    # present at index ``_MAX_ITEMS`` (see above).
    items_list = list(
        itertools.islice(details.items(), _MAX_ITEMS + 1))
  except Exception:
    return {'type': _scrub_type_name(type(details).__name__)}
  try:
    items_list = [(key, value) for key, value in items_list]
  except (TypeError, ValueError):
    return {'type': _scrub_type_name(type(details).__name__)}
  scrubbed = {}
  for key, value in items_list:
    # Keys may themselves carry secrets (bare ``AKIA...`` /
    # ``sk-...`` / PEM, ``key=value`` or ``password <value>``
    # forms, including inside tuple/frozenset keys): scrub
    # the key recursively too, and redact its value when the
    # key was secret-bearing. Under an inherited marker (a
    # dict nested in a marker-bearing container) the marker
    # is propagated into key scrubbing, so a tuple key's
    # non-marker elements are redacted, and a plain
    # non-marker string/bytes key -- itself secret payload
    # under the marker -- is replaced by ``REDACTED``
    # instead of leaking. Marker/marker-part key names are
    # kept; every value leaf is likewise redacted,
    # recursively.
    scrubbed_key, secret_key = _scrub_key(
        key, _inherited_marker, _seen, _depth)
    if (_inherited_marker
        and isinstance(key, (str, bytes, bytearray))
        and not _is_marker_element(key)
        and _marker_part(key) is None):
      scrubbed[REDACTED] = scrub_details(
          value, True, _seen, _depth + 1)
    elif secret_key:
      scrubbed[_hashable(scrubbed_key)] = REDACTED
    elif _inherited_marker:
      scrubbed[_hashable(scrubbed_key)] = scrub_details(
          value, True, _seen, _depth + 1)
    else:
      scrubbed[_hashable(scrubbed_key)] = scrub_details(
          value, False, _seen, _depth + 1)
  return scrubbed


def _scrub_container_details(
    details, _inherited_marker, _seen, _depth):
  '''Scrub a list/tuple/set/frozenset (loop14 split).'''
  if isinstance(details, list):
    return _scrub_container_values(
        details, True, _inherited_marker, _seen, _depth)
  if isinstance(details, tuple):
    return tuple(_scrub_container_values(
        details, True, _inherited_marker, _seen, _depth))
  if isinstance(details, frozenset):
    return frozenset(
        _hashable(v) for v in _scrub_container_values(
            details, False, _inherited_marker, _seen, _depth))
  return set(
      _hashable(v) for v in _scrub_container_values(
          details, False, _inherited_marker, _seen, _depth))


def _scrub_leaf_details(details, _inherited_marker):
  '''Scrub a scalar/bytes/str leaf (loop14 split).'''
  if isinstance(details, (bytes, bytearray)):
    if _inherited_marker:
      return REDACTED
    return _decode_bytes_scrubbed(details)
  if isinstance(details, str):
    return REDACTED if _inherited_marker else _scrub_text(details)
  if details is None or isinstance(
      details, (bool, int, float, complex)):
    return details
  return {'type': _scrub_type_name(type(details).__name__)}


safe_str = _safe_str
safe_repr = _safe_repr
safe_list = _safe_list
scrub_text = _scrub_text
scrub_type_name = _scrub_type_name
unknown_key = _unknown_key
is_secret_key = _is_secret_key
normalize_key = _normalize_key
normalize_unicode = _normalize_unicode
decode_bytes_scrubbed = _decode_bytes_scrubbed
composite_marker_bearing = _composite_marker_bearing
marker_part = _marker_part
is_marker_element = _is_marker_element
flatten_composite = _flatten_composite
MAX_ITEMS = _MAX_ITEMS
MAX_COMPOSITE_DEPTH = _MAX_COMPOSITE_DEPTH
