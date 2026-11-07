#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Typed telemetry exception hierarchy.

Errors carry a stable ``code``, the ``stage`` that raised
them, a user-facing message, a scrubbed JSON-safe
``context`` payload, and a ``retryable`` flag. Context
is cap+coerce-then-scrubbed on construction (same order
as the logger factory): oversized or non-JSON-safe
values are reduced first, then secrets are redacted via
``governance.scrub.scrub_details``. This module logs
only at debug level in ``_merge_wrap_context``; all
other logging lives in the guard/hooks layers.

Public API contract (A2 adjudication): the factories
(``DataIngestionError.failed``, ``TrainingFailedError.oom``,
etc.) and ``CODE_REGISTRY`` are the exported public API
of this package (covered by ``tests/test_exceptions.py``),
not dead internal code. They must not be deleted on
YAGNI grounds; only ``FrameworkError`` + ``wrap`` alone
would silently break downstream emitters.

A3 adjudication (``_coerce`` vs ``scrub_details``):
``_coerce`` is intentionally retained. ``scrub_details``
is the secret-redaction owner (this module already
imports its leaf helpers and caps so the two cannot
drift), but ``scrub_details`` does not cap size,
encode bytes, summarise DataFrames, detect cycles,
reject OmegaConf containers, or preserve tuple/
frozenset key structure for downstream scrubbing.
Replacing ``_coerce`` with a bare ``scrub_details``
call changes observable ``context`` values (verified by
the 133 exception tests), so unification is deferred:
keep ``_coerce`` as the cap/coerce stage, keep the
cap+coerce-then-scrub order, and change the two
together in one tested commit if unification is ever
attempted.
'''

import base64
import itertools
import logging
import math
import re
import threading
from collections.abc import Mapping
from datetime import date, datetime
from types import MappingProxyType

# Cross-package imports from the scrub leaf use its
# public aliases (loop17-arch medium): telemetry
# shares the leaf's safe/coerce/marker helpers and
# its caps (MAX_ITEMS/MAX_COMPOSITE_DEPTH) so
# scrubbing and coercion cannot drift apart. The
# ``_`` names below are local aliases for the same
# public objects; do not import them from here.
from forecast_ml_framework.governance.scrub import (
    MAX_COMPOSITE_DEPTH as _MAX_COMPOSITE_DEPTH,
    MAX_ITEMS as _MAX_ITEMS,
    REDACTED,
    composite_marker_bearing as _composite_marker_bearing,
    is_marker_element as _is_marker_element,
    marker_part as _marker_part,
    safe_list as _safe_list,
    safe_repr as _safe_repr,
    safe_str as _safe_str,
    scrub_details,
    scrub_type_name as _scrub_type_name,
    unknown_key as _unknown_key,
)

_UNSET = object()

LOG = logging.getLogger(__name__)

_MAX_STRING_LEN = 8192
_MAX_DEPTH = _MAX_COMPOSITE_DEPTH
_STAGE_MAX_LEN = 200
_CODE_RE = re.compile(r'[A-Z][A-Z0-9_]*')

#: Immutable registry of factory codes to their error class.
#: Populated by :func:`_factory`; duplicate registration is
#: rejected with ``ValueError``.
_CODE_REGISTRY_DICT = {}
CODE_REGISTRY = MappingProxyType(_CODE_REGISTRY_DICT)
_REGISTRY_LOCK = threading.Lock()

#: Known error class names for ``to_dict()['error_type']``.
#: ``type(self).__name__`` is attacker-controlled via a
#: dynamic subclass (``type('Zq9SENTINEL7x', ...)``), so
#: ``to_dict`` only echoes a name from this allowlist and
#: falls back to ``'FrameworkError'`` otherwise (a plain
#: scrub of the name cannot tell a sentinel class name
#: from a legitimate one).
_KNOWN_ERROR_TYPES = frozenset({
  'FrameworkError', 'DataIngestionError',
  'DataValidationError', 'FeatureEngineeringError',
  'TrainingFailedError', 'EvaluationError',
  'InterpretationError', 'ResilienceError',
  'CircuitOpenError',
})


def _safe_bool(value):
  '''Return ``bool(value)`` or ``False`` when that raises.'''
  try:
    return bool(value)
  except Exception:
    return False


def _truncate(text, limit=_MAX_STRING_LEN):
  '''Truncate a string to the context cap.'''
  text = _safe_str(text)
  if len(text) > limit:
    return text[:limit] + '...[TRUNCATED]'
  return text


def _clean_scrubbed(text, limit=_MAX_STRING_LEN):
  '''Scrub secrets from ``text`` and cap it at ``limit``.

  Args:
    text: Raw text, or None.
    limit: Maximum chars before truncation.

  Returns:
    A scrubbed string capped at ``limit`` chars.
  '''
  if text is None:
    return ''
  raw = _safe_str(text)
  if raw == '<unrepresentable>':
    return raw
  scrubbed = scrub_details(raw)
  scrubbed_text = (
      scrubbed if isinstance(scrubbed, str) else _safe_str(scrubbed))
  return _truncate(scrubbed_text, limit)


def _clean_stage(text):
  '''Scrub secrets from and cap a stage string at 200 chars.

  Args:
    text: Raw stage, or None.

  Returns:
    A scrubbed string capped at ``_STAGE_MAX_LEN`` chars.
  '''
  return _clean_scrubbed(text, _STAGE_MAX_LEN)


def _is_omegaconf_container(value):
  '''Return True for OmegaConf DictConfig/ListConfig values.

  Detection is by module/class name without importing
  omegaconf (an optional dependency): a ``DictConfig`` /
  ``ListConfig`` class name, or any MRO entry whose module
  is ``omegaconf.*``. Such containers are rejected as
  contexts because ``dict(context)`` / item access would
  resolve ``${oc.env:...}`` interpolations and leak env
  secrets.
  '''
  try:
    classes = type(value).__mro__
  except Exception:
    return False
  for cls in classes:
    module = getattr(cls, '__module__', '') or ''
    if module.split('.')[0] == 'omegaconf':
      return True
    if cls.__name__ in ('DictConfig', 'ListConfig'):
      return True
  return False


def _unknown_key_placeholder(key):
  '''Return ``(placeholder, secret)`` for an unknown dict key.

  Unknown keys always become a type-only placeholder
  ``<TypeName>``, never their scrubbed repr: keeping a
  scrubbed repr leaks when normalization alone changes a
  character without redacting (``áZq``, fullwidth,
  zero-width, combining). Unrepresentable keys stay
  ``<unrepresentable>``. ``secret`` is the flag from
  ``governance.scrub._unknown_key``: the value under a
  secret-bearing unknown key must be redacted in
  ``_coerce`` exactly as ``scrub_details`` redacts it,
  otherwise the placeholder loses the marker and the
  sentinel value leaks through telemetry.
  '''
  return _unknown_key(key)


def _clean_code(text):
  '''Scrub and cap a code string (re-cleaned on output).'''
  return _clean_stage(text)


def _clean_user_message(text):
  '''Scrub secrets from and truncate a user message.

  Args:
    text: Raw user message, or None.

  Returns:
    A scrubbed, truncated string; never leaks secrets via
    ``str(error)`` or ``user_message``. A hostile
    ``__str__`` becomes ``'<unrepresentable>'``.
  '''
  return _clean_scrubbed(text, _MAX_STRING_LEN)


def _should_redact_marker_bytes(marker_bearing, value):
  '''Return True when marker context redacts ``value``.

  Args:
    marker_bearing: Whether the composite carries a marker.
    value: Raw element to test.

  Returns:
    True for a non-marker, non-marker-part bytes element
    under a marker-bearing composite.
  '''
  return bool(
      marker_bearing
      and isinstance(value, (bytes, bytearray))
      and _marker_part(value) is None
      and not _is_marker_element(value))


def _scrub_notes(notes):
  '''Return a scrubbed copy of ``__notes__`` strings.'''
  try:
    notes_list = list(itertools.islice(notes, _MAX_ITEMS))
  except Exception:
    return []
  scrubbed_notes = []
  for note in notes_list:
    scrubbed_notes.append(_clean_user_message(note))
  return scrubbed_notes


def _coerce_key_items(key, seen, depth, make):
  '''Coerce tuple/frozenset key items via ``make``.'''
  if id(key) in seen:
    return '<cyclic>', True
  seen = seen | {id(key)}
  items = _safe_list(key)
  if items is None:
    return _unknown_key_placeholder(key)
  coerced_items = []
  secret = False
  for item in items:
    coerced_item, item_secret = _coerce_key(
        item, seen, depth + 1)
    coerced_items.append(coerced_item)
    secret = secret or item_secret
  return make(coerced_items), secret


def _coerce_key(key, seen, depth):
  '''Coerce a dict key preserving tuple/frozenset/scalar type.

  Args:
    key: Raw dict key.
    seen: Ids of containers on the current path.
    depth: Current nesting depth.

  Returns:
    A ``(coerced_key, secret)`` tuple. The coerced key is
    hashable; tuple/frozenset structure is preserved
    element-wise so ``scrub_details`` can apply its
    composite-key scrubbing, and scalar keys keep their
    type. ``secret`` propagates the unknown-key secret
    flag (see ``_unknown_key_placeholder``) up through
    tuple/frozenset/nested elements, so ``_coerce`` can
    redact the value under a secret-bearing unknown key
    instead of losing the flag at the placeholder.
    String/bytes/scalar keys return ``False``: the
    downstream scrub still detects marker-bearing keys
    itself.
  '''
  if key is None or isinstance(key, (bool, int)):
    return key, False
  if isinstance(key, float):
    return (key if math.isfinite(key) else '<null>'), False
  if isinstance(key, str):
    return _truncate(key), False
  if isinstance(key, (bytes, bytearray)):
    # Preserve bytes unencoded (never base64 here): the leaf
    # scrub's bytes branch / composite-key logic must see the
    # raw bytes so a ``b'password'`` key redacts its value.
    # Loop20 P4: cap bytes keys like str keys (8192).
    # R-C9-2: truncated bytes carry the str ``...[TRUNCATED]``
    # suffix so downstream scrub over-cap detection
    # (``len > 8192``) still fires; exactly 8192 erased it.
    raw = bytes(key)
    if len(raw) > _MAX_STRING_LEN:
      return raw[:_MAX_STRING_LEN] + b'...[TRUNCATED]', False
    return raw, False
  if isinstance(key, (datetime, date)):
    try:
      return key.isoformat(), False
    except Exception:
      # Hostile isoformat: type-only fallback, never
      # ``str(key)`` (which could leak or raise again). The
      # unhashable dict is converted to a ``<Type>``
      # placeholder by the caller's hash check.
      return {'type': _scrub_type_name(type(key).__name__)}, False
  if depth > _MAX_DEPTH:
    return '<max-depth>', True
  if isinstance(key, tuple):
    return _coerce_key_items(key, seen, depth, tuple)
  if isinstance(key, frozenset):
    return _coerce_key_items(key, seen, depth, frozenset)
  coerced = _coerce(key, seen, depth)
  try:
    hash(coerced)
  except Exception:
    # Unhashable coercion (e.g. a hostile/unknown key that
    # coerced to a dict): use a type-only placeholder plus
    # the secret flag, never ``str(key)`` via ``_truncate``
    # (which would execute a hostile ``__str__`` and leak
    # its secret). A raising ``__repr__`` becomes
    # ``'<unrepresentable>'`` with ``secret`` True.
    return _unknown_key_placeholder(key)
  return coerced, False


def _coerce_dataframe(value):
  '''Coerce a DataFrame-like value to metadata, safely.

  Args:
    value: Candidate DataFrame-like value.

  Returns:
    A metadata dict for validated DataFrame-like values, or
    ``{'type': name}`` only for non-DataFrame / unvalidated
    objects (never a ``str(value)`` dump, which could leak
    secrets carried by ``__str__``).
  '''
  try:
    shape = getattr(value, 'shape', None)
    if shape is None or not hasattr(value, 'dtypes'):
      return {'type': _scrub_type_name(type(value).__name__)}
    shape_items = list(itertools.islice(shape, _MAX_ITEMS))
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 0
        for n in shape_items):
      raise ValueError('invalid DataFrame shape')
    dtypes = getattr(value, 'dtypes', {})
    columns_obj = getattr(value, 'columns', None)
    if columns_obj is None:
      columns_obj = []
    columns = [
        _truncate(c)
        for c in itertools.islice(columns_obj, _MAX_ITEMS)
    ]
    return {
        'type': _scrub_type_name(type(value).__name__),
        'shape': shape_items,
        'columns': columns,
        'dtypes': _truncate(dtypes),
    }
  except Exception:
    return {'type': _scrub_type_name(type(value).__name__)}


def _coerce_scalar(value, _inherited_marker=False):
  '''Coerce scalar ``value`` or return ``_UNSET``.

  Split from ``_coerce`` (loop14 scrub/coerce phases)
  so each phase stays within the cyclomatic budget.
  Scalars are handled before the depth cap, exactly as
  before: a deep scalar still coerces, a deep container
  is capped by the dispatch in ``_coerce``.
  '''
  if value is None or isinstance(value, (bool, int)):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else '<null>'
  if isinstance(value, str):
    return _truncate(value)
  if isinstance(value, (bytes, bytearray)):
    if _should_redact_marker_bytes(_inherited_marker, value):
      return REDACTED
    capped = bytes(value)[:_MAX_STRING_LEN]
    scrubbed = scrub_details(capped)
    scrubbed_text = (
        scrubbed if isinstance(scrubbed, str) else _safe_str(scrubbed))
    encoded = base64.b64encode(
        scrubbed_text.encode('utf-8')).decode('ascii')
    return _truncate(encoded)
  if isinstance(value, (datetime, date)):
    try:
      return value.isoformat()
    except Exception:
      # Hostile isoformat: type-only fallback, never
      # ``str(value)`` (which could leak or raise again).
      return {'type': _scrub_type_name(type(value).__name__)}
  return _UNSET


def _coerce_dict(value, seen, depth, _inherited_marker=False):
  '''Coerce a dict ``value`` to a JSON-safe dict.'''
  if id(value) in seen:
    return '<cyclic>'
  seen = seen | {id(value)}
  try:
    items_list = list(
        itertools.islice(value.items(), _MAX_ITEMS + 1))
  except Exception:
    return {'type': _scrub_type_name(type(value).__name__)}
  try:
    items_list = [(key, item) for key, item in items_list]
  except (TypeError, ValueError):
    return {'type': _scrub_type_name(type(value).__name__)}
  out = {}
  for index, (key, item) in enumerate(items_list):
    if index >= _MAX_ITEMS:
      out['...[TRUNCATED]'] = True
      break
    coerced_key, secret_key = _coerce_key(key, seen, depth + 1)
    try:
      hash(coerced_key)
    except Exception:
      coerced_key, secret_key = _unknown_key_placeholder(key)
    if secret_key:
      # Secret-bearing unknown key: redact the value here,
      # exactly as ``scrub_details`` does, because the
      # placeholder key alone no longer carries the marker
      # the downstream scrub would need to redact it.
      out[coerced_key] = REDACTED
    else:
      out[coerced_key] = _coerce(
          item, seen, depth + 1, _inherited_marker)
  return out


def _coerce_sequence(value, seen, depth, _inherited_marker=False):
  '''Coerce a list/tuple ``value`` to a JSON-safe list.'''
  if id(value) in seen:
    return '<cyclic>'
  seen = seen | {id(value)}
  try:
    probed = list(itertools.islice(value, _MAX_ITEMS + 1))
  except Exception:
    return {'type': _scrub_type_name(type(value).__name__)}
  raw_items = probed[:_MAX_ITEMS]
  # Composite detection recurses into nested tuples at
  # any depth (via scrub._composite_marker_bearing), and
  # the marker context is inherited by nested containers
  # so nested bytes are redacted before base64.
  marker_bearing = (
      _inherited_marker
      or _composite_marker_bearing(
          raw_items, True, _seen=seen, _depth=depth))
  coerced = []
  for item in raw_items:
    # Composite-marker detection before base64: a bytes
    # element in a marker-bearing composite (e.g.
    # ``('api', b'HUNTER2', 'key')``) must be redacted,
    # never base64-encoded and left reversible.
    if _should_redact_marker_bytes(marker_bearing, item):
      coerced.append(REDACTED)
    else:
      coerced.append(_coerce(
          item, seen, depth + 1, marker_bearing))
  if len(probed) > _MAX_ITEMS:
    coerced.append('...[TRUNCATED]')
  return coerced


def _coerce_set(value, seen, depth, _inherited_marker=False):
  '''Coerce a set/frozenset ``value`` to a sorted list.'''
  if id(value) in seen:
    return '<cyclic>'
  seen = seen | {id(value)}
  raw_items = _safe_list(value)
  if raw_items is None:
    return {'type': _scrub_type_name(type(value).__name__)}
  marker_bearing = (
      _inherited_marker
      or _composite_marker_bearing(
          raw_items, False, _seen=seen, _depth=depth))
  coerced = []
  for item in sorted(raw_items, key=_safe_repr)[:_MAX_ITEMS]:
    if _should_redact_marker_bytes(marker_bearing, item):
      coerced.append(REDACTED)
    else:
      coerced.append(_coerce(
          item, seen, depth + 1, marker_bearing))
  return coerced


def _coerce(value, seen, depth, _inherited_marker=False):
  '''Reduce ``value`` to a JSON-safe form, capping size.

  Args:
    value: Value to reduce.
    seen: Ids of containers on the current path, for
      cyclic-reference detection.
    depth: Current nesting depth.
    _inherited_marker: Outer composite's marker-bearing
      context, carried into nested containers at any
      depth so nested bytes are redacted before base64.

  Returns:
    A JSON-safe value; cyclic refs become the string
    ``'<cyclic>'`` and unknown objects become
    ``{'type': name}`` only (DataFrame-like validated
    objects become a metadata summary). Dict values under
    secret-bearing unknown keys (see ``_coerce_key``)
    become ``REDACTED`` directly.
  '''
  scalar = _coerce_scalar(value, _inherited_marker)
  if scalar is not _UNSET:
    return scalar
  if depth > _MAX_DEPTH:
    return '<max-depth>'
  if _is_omegaconf_container(value):
    # Never item-access an OmegaConf container: that would
    # resolve interpolations (e.g. ``oc.env``) and leak.
    return {'type': _scrub_type_name(type(value).__name__)}
  if isinstance(value, dict):
    return _coerce_dict(value, seen, depth, _inherited_marker)
  if isinstance(value, (list, tuple)):
    return _coerce_sequence(value, seen, depth, _inherited_marker)
  if isinstance(value, (set, frozenset)):
    return _coerce_set(value, seen, depth, _inherited_marker)
  return _coerce_dataframe(value)


def _clean_context(context):
  '''Cap+coerce then scrub a context mapping.

  Args:
    context: Raw context mapping, or None.

  Returns:
    A new scrubbed, JSON-safe dict; never mutates input.

  Raises:
    TypeError: If ``context`` is not None and not a mapping,
      or is an OmegaConf DictConfig/ListConfig container
      (``dict(context)`` would resolve interpolations).
  '''
  if context is None:
    return {}
  _validate_context_mapping(context)
  try:
    if len(context) == 0:
      return {}
  except Exception:
    return {}
  try:
    plain = dict(
        itertools.islice(context.items(), _MAX_ITEMS + 1))
  except Exception:
    return {}
  if not plain:
    return {}
  coerced = _coerce(plain, frozenset(), 0)
  scrubbed = scrub_details(coerced)
  return scrubbed if isinstance(scrubbed, dict) else {}


def _stringify_composite_keys(value):
  '''Return ``value`` with composite dict keys stringified.

  Tuple/frozenset/set keys are kept in the in-memory
  ``context`` (so callers can use them as keys) but break
  ``json.dumps``; ``to_dict()`` stringifies them (via
  ``str``) recursively so its payload is JSON-safe. Only
  ``to_dict()`` uses this helper.
  '''
  if isinstance(value, dict):
    out = {}
    for key, item in value.items():
      new_key = (
          str(key)
          if isinstance(key, (tuple, frozenset, set)) else key)
      out[new_key] = _stringify_composite_keys(item)
    return out
  if isinstance(value, list):
    return [_stringify_composite_keys(item) for item in value]
  if isinstance(value, tuple):
    return [_stringify_composite_keys(item) for item in value]
  return value


_POS_NAMES = ('code', 'stage', 'user_message', 'context', 'retryable')


def _is_supplied(name, kwarg):
  '''Return True when a keyword counts as supplied.'''
  return kwarg is not _UNSET and not (
    name == 'code' and kwarg is None)


def _validate_positional_arity(args, owner_name='FrameworkError'):
  '''Reject more than five positional arguments.'''
  if len(args) > 5:
    raise TypeError(
      f'{_scrub_type_name(_safe_str(owner_name))} takes at '
      f'most 5 positional arguments ({len(args)} given)')


def _check_positional_collisions(args, kw_by_name):
  '''Reject positional+keyword duplicates (multi case).'''
  for _name in _POS_NAMES[:len(args)]:
    if _is_supplied(_name, kw_by_name[_name]):
      raise TypeError(f'multiple values for argument: {_name}')


def _resolve_multi_positionals(
    args, code, stage, user_message, context, retryable):
  '''Resolve multiple positionals against keywords.'''
  kw_by_name = {
    'code': code, 'stage': stage, 'user_message': user_message,
    'context': context, 'retryable': retryable}
  _check_positional_collisions(args, kw_by_name)
  for _name, _value in zip(_POS_NAMES, args):
    kw_by_name[_name] = _value
  return (
    kw_by_name['code'], kw_by_name['stage'],
    kw_by_name['user_message'], kw_by_name['context'],
    kw_by_name['retryable'])


def _resolve_single_positional(args, user_message, message):
  '''Resolve a single positional as the user message.'''
  if user_message is not _UNSET or message is not _UNSET:
    raise TypeError('multiple values for argument: user_message')
  return args[0]


def _apply_ctor_defaults(stage, user_message, context, retryable,
                         message):
  '''Apply _UNSET defaults and the ``message`` alias.'''
  if stage is _UNSET:
    stage = ''
  if user_message is _UNSET:
    user_message = None
  if context is _UNSET:
    context = None
  if retryable is _UNSET:
    retryable = False
  if message is _UNSET:
    message = None
  if message is not None:
    if user_message is not None:
      raise TypeError('multiple values for argument: user_message')
    user_message = message
  return stage, user_message, context, retryable


def _resolve_ctor_args(
    args, code, stage, user_message, context, retryable, message,
    owner_name='FrameworkError'):
  '''Resolve positional/keyword ctor args to a 5-tuple.

  Returns ``(code, stage, user_message, context,
  retryable)`` with defaults applied. Split from
  ``FrameworkError.__init__`` (loop14) so the ctor
  stays within the cyclomatic budget; no behavior
  change -- positional order, collision errors, and
  the ``message`` alias are preserved exactly.
  '''
  _validate_positional_arity(args, owner_name)
  if args:
    if len(args) == 1:
      user_message = _resolve_single_positional(
        args, user_message, message)
    else:
      code, stage, user_message, context, retryable = (
        _resolve_multi_positionals(
          args, code, stage, user_message, context, retryable))
  stage, user_message, context, retryable = _apply_ctor_defaults(
    stage, user_message, context, retryable, message)
  return code, stage, user_message, context, retryable


def _validate_code(effective_code, prefix, owner_name):
  '''Validate ``effective_code`` against pattern/prefix.

  Returns the code unchanged; raises ``ValueError``
  with the code scrubbed (never echoed raw) on
  failure. Split from the ctor (loop14).
  '''
  if _CODE_RE.fullmatch(effective_code) is None:
    raise ValueError(
      'error code must fully match [A-Z][A-Z0-9_]*, '
      f'got {_clean_code(effective_code)!r}')
  if prefix is not None and not effective_code.startswith(prefix):
    raise ValueError(
      f'{_scrub_type_name(_safe_str(owner_name))} code must '
      f'start with {prefix!r}, got '
      f'{_clean_code(effective_code)!r}')
  return effective_code


class FrameworkError(Exception):
  '''Base typed framework error.

  Args:
    code: Stable machine code, e.g. ``'TRAIN_OOM'``. Must
      fully match ``[A-Z][A-Z0-9_]*``; pass via the
      ``code=`` keyword. Positional order is ``(code,
      stage, user_message, context, retryable)`` when
      multiple positionals are given; a single positional
      string is always the user message, never a code.
    stage: Pipeline stage that raised, e.g.
      ``'modeling'``.
    user_message: Human-readable message; scrubbed for
      secrets and truncated on construction.
    context: Extra payload; must be a mapping or None
      (``TypeError`` otherwise), cap+coerce then scrubbed.
    retryable: Whether a caller may retry the failed
      operation.

  Raises:
    TypeError: If ``context`` is not a mapping or None,
      if more than five positional args are given, or if
      a positional and a keyword supply the same parameter.
    ValueError: If ``code`` does not fully match the
      code pattern or violates the subclass prefix.
  '''

  #: Class-level default code used when none is given.
  default_code = 'FRAMEWORK_ERROR'
  #: Required code prefix for subclasses, or None.
  _code_prefix = None

  def __init__(
      self,
      *args,
      code=None,
      stage=_UNSET,
      user_message=_UNSET,
      context=_UNSET,
      retryable=_UNSET,
      message=_UNSET,
  ):
    '''Initialize a typed error, scrubbing context.'''
    code, stage, user_message, context, retryable = (
      _resolve_ctor_args(
        args, code, stage, user_message, context, retryable,
        message, owner_name=_scrub_type_name(type(self).__name__)))
    effective_code = _safe_str(
        code if code is not None else self.default_code)
    _validate_code(
      effective_code, self._code_prefix, type(self).__name__)
    self.code = effective_code
    self.stage = _clean_stage(stage)
    self.user_message = _clean_user_message(user_message)
    self.context = _clean_context(context)
    self.retryable = _safe_bool(retryable)
    super().__init__(str(self))

  @property
  def message(self):
    '''Alias for ``user_message``.'''
    return self.user_message

  def __str__(self):
    '''Return ``'[<code>] <user_message>'``, re-cleaned.

    Attributes are plain (mutable) attributes, so
    ``__str__`` re-cleans ``code``/``user_message`` at
    output time: a post-construction mutation cannot
    smuggle a secret into ``str(error)``.
    '''
    return (
        f'[{_clean_code(getattr(self, "code", None))}] '
        f'{_clean_user_message(getattr(self, "user_message", None))}')

  def to_dict(self):
    '''Return a JSON-safe envelope of this error.

    ``code``/``stage``/``user_message``/``context`` are
    re-cleaned at output time (attributes are mutable),
    so post-construction mutation cannot leak secrets
    through ``to_dict()`` either.

    ``error_type`` is allowlisted (see
    ``_KNOWN_ERROR_TYPES``): a dynamic subclass name can
    carry a sentinel, so unknown names fall back to
    ``'FrameworkError'``. ``context`` is total: a
    post-construction mutation to a non-mapping or
    OmegaConf container (which ``_clean_context`` rejects
    with ``TypeError`` on construction) yields ``{}``
    here instead of raising.

    Returns:
      Dict with ``code``, ``stage``, ``user_message``,
      scrubbed ``context``, ``retryable``, and
      ``error_type``; safe for ``json.dumps``.
    '''
    try:
      cleaned_context = _clean_context(
        getattr(self, 'context', None))
    except TypeError:
      cleaned_context = {}
    error_type = type(self).__name__
    if error_type not in _KNOWN_ERROR_TYPES:
      error_type = 'FrameworkError'
    return {
        'code': _clean_code(getattr(self, 'code', None)),
        'stage': _clean_stage(getattr(self, 'stage', None)),
        'user_message': _clean_user_message(
            getattr(self, 'user_message', None)),
        'context': _stringify_composite_keys(cleaned_context),
        'retryable': _safe_bool(getattr(self, 'retryable', False)),
        'error_type': error_type,
    }

  @classmethod
  def _build(cls, code, user_message, stage, context, retryable):
    '''Build an instance with an explicit code.'''
    return cls(
        user_message, stage=stage, code=code,
        context=context, retryable=retryable)


def _register_default(code, cls):
  '''Register a class default code if not already present.

  Args:
    code: Default code to register.
    cls: Error class for the code.

  Raises:
    ValueError: If ``code`` is registered for another class.
  '''
  with _REGISTRY_LOCK:
    existing = _CODE_REGISTRY_DICT.get(code)
    if existing is None:
      _CODE_REGISTRY_DICT[code] = cls
    elif existing is not cls:
      raise ValueError(
          'duplicate error code registration: '
          f'{_clean_code(code)!r}')


def _factory(cls, code, default_message, retryable):
  '''Return a factory classmethod for ``cls``.

  Args:
    cls: Subclass to build.
    code: Stable code for the factory.
    default_message: Message when none is supplied.
    retryable: Default retry flag for the factory.

  Returns:
    A classmethod building an instance of ``cls``.

  Raises:
    ValueError: If ``code`` is invalid, violates the
      ``cls`` prefix, or is already registered.
  '''
  code = _safe_str(code)
  if _CODE_RE.fullmatch(code) is None:
    raise ValueError(
        'factory code must fully match [A-Z][A-Z0-9_]*, '
        f'got {_clean_code(code)!r}')
  prefix = getattr(cls, '_code_prefix', None)
  if prefix is not None and not code.startswith(prefix):
    raise ValueError(
        f'{_scrub_type_name(cls.__name__)} code must start '
        f'with {prefix!r}, got {_clean_code(code)!r}')
  with _REGISTRY_LOCK:
    if code in _CODE_REGISTRY_DICT:
      raise ValueError(
          'duplicate error code registration: '
          f'{_clean_code(code)!r}')
    _CODE_REGISTRY_DICT[code] = cls

  @classmethod
  def _make(
      factory_cls, user_message=None, *, stage='', context=None,
      retryable=retryable,
  ):
    # Loop20 P5: ``is None`` not truthiness -- a hostile
    # ``__bool__`` must not crash and a supplied empty
    # string is passed through ctor cleaning, not
    # replaced by the default.
    resolved = (
        default_message if user_message is None else user_message)
    return factory_cls._build(
        code, resolved, stage, context, retryable)

  return _make


class DataIngestionError(FrameworkError):
  '''An ingestion-stage failure (``INGEST_*``).'''

  default_code = 'INGEST_ERROR'
  _code_prefix = 'INGEST_'


class DataValidationError(FrameworkError):
  '''A validation-stage failure (``VALIDATE_*``).'''

  default_code = 'VALIDATE_ERROR'
  _code_prefix = 'VALIDATE_'


class FeatureEngineeringError(FrameworkError):
  '''A feature-engineering failure (``FEATURE_*``).'''

  default_code = 'FEATURE_ERROR'
  _code_prefix = 'FEATURE_'


class TrainingFailedError(FrameworkError):
  '''A training-stage failure (``TRAIN_*``).'''

  default_code = 'TRAIN_ERROR'
  _code_prefix = 'TRAIN_'


class EvaluationError(FrameworkError):
  '''An evaluation-stage failure (``EVAL_*``).'''

  default_code = 'EVAL_ERROR'
  _code_prefix = 'EVAL_'


class InterpretationError(FrameworkError):
  '''An interpretability-stage failure (``INTERP_*``).'''

  default_code = 'INTERP_ERROR'
  _code_prefix = 'INTERP_'


class ResilienceError(FrameworkError):
  '''A resilience failure (``RESILIENCE_*``).'''

  default_code = 'RESILIENCE_ERROR'
  _code_prefix = 'RESILIENCE_'


class CircuitOpenError(ResilienceError):
  '''A circuit breaker is open (``RESILIENCE_*``).'''

  default_code = 'RESILIENCE_CIRCUIT_OPEN'


DataIngestionError.failed = _factory(
    DataIngestionError, 'INGEST_FAILED',
    'data ingestion failed', False)
DataIngestionError.source_unavailable = _factory(
    DataIngestionError, 'INGEST_UNAVAILABLE',
    'ingestion source unavailable', True)
DataIngestionError.file_not_found = _factory(
    DataIngestionError, 'INGEST_NOT_FOUND',
    'ingestion file not found', False)
DataValidationError.failed = _factory(
    DataValidationError, 'VALIDATE_FAILED',
    'data validation failed', False)
DataValidationError.schema_mismatch = _factory(
    DataValidationError, 'VALIDATE_SCHEMA',
    'schema mismatch', False)
DataValidationError.null_check_failed = _factory(
    DataValidationError, 'VALIDATE_NULLS',
    'null check failed', False)
FeatureEngineeringError.failed = _factory(
    FeatureEngineeringError, 'FEATURE_FAILED',
    'feature engineering failed', False)
FeatureEngineeringError.transformation_failed = _factory(
    FeatureEngineeringError, 'FEATURE_TRANSFORM',
    'feature transformation failed', False)
FeatureEngineeringError.missing_column = _factory(
    FeatureEngineeringError, 'FEATURE_MISSING_COLUMN',
    'required column missing', False)
TrainingFailedError.failed = _factory(
    TrainingFailedError, 'TRAIN_FAILED',
    'training failed', False)
TrainingFailedError.oom = _factory(
    TrainingFailedError, 'TRAIN_OOM',
    'training ran out of memory', True)
TrainingFailedError.diverged = _factory(
    TrainingFailedError, 'TRAIN_DIVERGED',
    'training diverged', False)
EvaluationError.failed = _factory(
    EvaluationError, 'EVAL_FAILED',
    'evaluation failed', False)
EvaluationError.metric_failed = _factory(
    EvaluationError, 'EVAL_METRIC',
    'metric computation failed', False)
EvaluationError.missing_ground_truth = _factory(
    EvaluationError, 'EVAL_NO_GROUND_TRUTH',
    'ground truth missing', False)
InterpretationError.failed = _factory(
    InterpretationError, 'INTERP_FAILED',
    'interpretation failed', False)
InterpretationError.shap_failed = _factory(
    InterpretationError, 'INTERP_SHAP_FAILED',
    'SHAP explanation failed', False)
InterpretationError.backend_unavailable = _factory(
    InterpretationError, 'INTERP_BACKEND_UNAVAILABLE',
    'interpretation backend unavailable', True)
ResilienceError.failed = _factory(
    ResilienceError, 'RESILIENCE_FAILED',
    'resilience operation failed', True)
ResilienceError.exhausted = _factory(
    ResilienceError, 'RESILIENCE_EXHAUSTED',
    'retry attempts exhausted', False)
ResilienceError.timeout = _factory(
    ResilienceError, 'RESILIENCE_TIMEOUT',
    'operation timed out', True)
CircuitOpenError.open = _factory(
    CircuitOpenError, 'RESILIENCE_CIRCUIT_OPEN',
    'circuit breaker is open', True)

_register_default('FRAMEWORK_ERROR', FrameworkError)
_register_default('INGEST_ERROR', DataIngestionError)
_register_default('VALIDATE_ERROR', DataValidationError)
_register_default('FEATURE_ERROR', FeatureEngineeringError)
_register_default('TRAIN_ERROR', TrainingFailedError)
_register_default('EVAL_ERROR', EvaluationError)
_register_default('INTERP_ERROR', InterpretationError)
_register_default('RESILIENCE_ERROR', ResilienceError)
_register_default('RESILIENCE_CIRCUIT_OPEN', CircuitOpenError)


def _validate_context_mapping(context):
  '''Reject non-mapping/OmegaConf ctor and wrap contexts.'''
  if context is not None and _is_omegaconf_container(context):
    raise TypeError(
      'context must be a mapping or None, got '
      f'{_scrub_type_name(type(context).__name__)}')
  if context is not None and not isinstance(context, Mapping):
    raise TypeError(
      'context must be a mapping or None, got '
      f'{_scrub_type_name(type(context).__name__)}')


def _merge_wrap_context(base, context):
  '''Merge ``context`` into a copy of ``base``.

  A malformed ``context`` (hostile ``__bool__`` or a mapping
  whose ``items()`` fails) is skipped with a debug log and
  the base context is returned unchanged.
  '''
  merged = dict(base)
  try:
    if context:
      try:
        merged.update(dict(
            itertools.islice(context.items(), _MAX_ITEMS + 1)))
      except (AttributeError, TypeError, ValueError) as exc:
        LOG.debug('wrap context merge skipped: %r', exc)
  except (AttributeError, TypeError, ValueError) as exc:
    LOG.debug('wrap context merge skipped: %r', exc)
  return merged


def _copy_wrap_notes(target, source):
  '''Scrub-copy ``__notes__`` from ``source`` to target.'''
  notes = getattr(source, '__notes__', None)
  if notes is not None:
    target.__notes__ = _scrub_notes(notes)


def _wrap_typed(exc, stage, code, text, context, retryable):
  '''Copy-on-wrap a typed error (see ``wrap``).'''
  _validate_context_mapping(context)
  merged = _merge_wrap_context(exc.context, context)
  copied = type(exc)(
    code=code if code is not None else exc.code,
    stage=stage if stage is not None else exc.stage,
    user_message=text if text is not None else exc.user_message,
    context=merged,
    retryable=(
      _safe_bool(retryable) if retryable is not None
      else exc.retryable),
  )
  copied.__cause__ = exc.__cause__
  copied.__context__ = exc.__context__
  copied.__traceback__ = exc.__traceback__
  copied.__suppress_context__ = exc.__suppress_context__
  _copy_wrap_notes(copied, exc)
  return copied


def _wrap_untyped(exc, stage, code, text, context, retryable):
  '''Wrap an untyped error (see ``wrap``).'''
  wrapped = FrameworkError(
    code=code if code is not None else FrameworkError.default_code,
    stage=stage if stage is not None else '',
    user_message=text if text is not None else _safe_str(exc),
    context=context,
    retryable=(
      _safe_bool(retryable) if retryable is not None else False),
  )
  wrapped.__cause__ = exc
  wrapped.__context__ = exc.__context__
  wrapped.__traceback__ = exc.__traceback__
  wrapped.__suppress_context__ = getattr(
      exc, '__suppress_context__', False)
  _copy_wrap_notes(wrapped, exc)
  return wrapped


def wrap(
    exc, stage=None, code=None, user_message=None, context=None,
    retryable=None, *, message=None,
):
  '''Wrap ``exc`` as a :class:`FrameworkError`.

  Copy-on-wrap: never mutates ``exc``. A typed error
  yields a new instance of the same class with any
  supplied ``stage``/``code``/``user_message``/
  ``context``/``retryable`` overrides applied through
  the constructor (so prefix/code validation, message
  scrubbing, and context cleaning all apply), while
  ``__cause__``, ``__context__``, ``__traceback__``,
  and ``__suppress_context__`` are preserved. Wrapping
  is idempotent: re-wrapping a copy with no overrides
  yields an equal error. An untyped error becomes a
  new ``FrameworkError`` with ``__cause__`` set to the
  original and ``user_message`` scrubbed (including
  the ``str(exc)`` default); callers should
  ``raise wrap(...) from exc`` to keep the chain.

  Cause chain: ``__cause__``/``__context__`` are
  preserved by reference (never copied or scrubbed),
  so the original exception -- which may itself carry
  unscrubbed secrets in its message/notes -- stays
  reachable via the chain. ``to_dict()``/``__str__``
  of the wrapper never include the cause; do not log
  the cause chain directly. ``__notes__`` are the
  exception: they are scrubbed (``_scrub_notes``) on
  the copy, since notes are plain strings carried on
  the wrapper itself.

  Logging contract (envelope-only): log only the
  wrapper's envelope -- ``to_dict()`` or
  ``str(wrapper)`` -- never the cause chain,
  traceback, or ``exc_info`` of the wrapped
  exception. ``wrap`` preserves ``__cause__`` /
  ``__context__`` / ``__traceback__`` by reference
  for chaining, and the original may carry
  unscrubbed secrets, so logging ``exc_info`` or
  formatting the traceback would leak them even
  though the envelope is scrubbed. This contract
  does not scrub or drop the chain (that would
  break chaining); callers must not log it.

  Args:
    exc: Exception to wrap.
    stage: Stage to record.
    code: Code override (validated like the ctor).
    user_message: Message override (scrubbed).
    context: Context to merge (typed) or attach;
      must be a mapping or None.
    retryable: Retry flag override.
    message: Alias for ``user_message``.

  Returns:
    A new ``FrameworkError``; ``exc`` is unchanged.

  Raises:
    TypeError: If ``context`` is not a mapping or None,
      or if both ``user_message`` and ``message`` are given.
    ValueError: If ``code`` fails ctor validation.
  '''
  if user_message is not None and message is not None:
    raise TypeError('multiple values for argument: user_message')
  text = user_message if user_message is not None else message
  if isinstance(exc, FrameworkError):
    return _wrap_typed(exc, stage, code, text, context, retryable)
  return _wrap_untyped(exc, stage, code, text, context, retryable)


__all__ = [
    'CODE_REGISTRY',
    'CircuitOpenError',
    'DataIngestionError',
    'DataValidationError',
    'EvaluationError',
    'FeatureEngineeringError',
    'FrameworkError',
    'InterpretationError',
    'ResilienceError',
    'TrainingFailedError',
    'wrap',
]
