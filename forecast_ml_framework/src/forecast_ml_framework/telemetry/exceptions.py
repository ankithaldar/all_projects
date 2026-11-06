#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Typed telemetry exception hierarchy.

Errors carry a stable ``code``, the ``stage`` that raised
them, a user-facing message, a scrubbed JSON-safe
``context`` payload, and a ``retryable`` flag. Context
is cap+coerce-then-scrubbed on construction (same order
as the logger factory): oversized or non-JSON-safe
values are reduced first, then secrets are redacted via
``governance.audit.scrub_details``. This module never
logs; logging lives in the guard/hooks layers.
'''

import base64
import re
from collections.abc import Mapping
from datetime import date, datetime
from types import MappingProxyType

from forecast_ml_framework.governance.audit import scrub_details

_MAX_STRING_LEN = 8192
_MAX_DEPTH = 10
_MAX_ITEMS = 1000
_CODE_RE = re.compile(r'[A-Z][A-Z0-9_]*')

#: Immutable registry of factory codes to their error class.
#: Populated by :func:`_factory`; duplicate registration is
#: rejected with ``ValueError``.
_CODE_REGISTRY_DICT = {}
CODE_REGISTRY = MappingProxyType(_CODE_REGISTRY_DICT)
_CODE_REGISTRY = CODE_REGISTRY


def _truncate(text):
  '''Truncate a string to the context cap.'''
  text = str(text)
  if len(text) > _MAX_STRING_LEN:
    return text[:_MAX_STRING_LEN] + '...[TRUNCATED]'
  return text


def _clean_user_message(text):
  '''Scrub secrets from and truncate a user message.

  Args:
    text: Raw user message, or None.

  Returns:
    A scrubbed, truncated string; never leaks secrets via
    ``str(error)`` or ``user_message``.
  '''
  if text is None:
    return ''
  scrubbed = scrub_details(str(text))
  return _truncate(scrubbed if isinstance(scrubbed, str) else str(scrubbed))


def _coerce(value, seen, depth):
  '''Reduce ``value`` to a JSON-safe form, capping size.

  Args:
    value: Value to reduce.
    seen: Ids of containers on the current path, for
      cyclic-reference detection.
    depth: Current nesting depth.

  Returns:
    A JSON-safe value; cyclic refs become the string
    ``'<cyclic>'`` and unknown objects become strings
    (DataFrame-like objects become a metadata summary).
  '''
  if value is None or isinstance(value, (bool, int, float)):
    return value
  if isinstance(value, str):
    return _truncate(value)
  if isinstance(value, (bytes, bytearray)):
    scrubbed = scrub_details(bytes(value))
    scrubbed_text = (
        scrubbed if isinstance(scrubbed, str) else str(scrubbed))
    encoded = base64.b64encode(
        scrubbed_text.encode('utf-8')).decode('ascii')
    return _truncate(encoded)
  if isinstance(value, (datetime, date)):
    return value.isoformat()
  if depth > _MAX_DEPTH:
    return '<max-depth>'
  if isinstance(value, dict):
    if id(value) in seen:
      return '<cyclic>'
    seen = seen | {id(value)}
    out = {}
    for index, (key, item) in enumerate(value.items()):
      if index >= _MAX_ITEMS:
        out['...[TRUNCATED]'] = True
        break
      out[_truncate(key)] = _coerce(item, seen, depth + 1)
    return out
  if isinstance(value, (list, tuple)):
    if id(value) in seen:
      return '<cyclic>'
    seen = seen | {id(value)}
    items = list(value)[:_MAX_ITEMS]
    coerced = [_coerce(v, seen, depth + 1) for v in items]
    if len(value) > _MAX_ITEMS:
      coerced.append('...[TRUNCATED]')
    return coerced
  if isinstance(value, (set, frozenset)):
    items = sorted(value, key=lambda v: repr(v))[:_MAX_ITEMS]
    return [_coerce(v, seen, depth + 1) for v in items]
  shape = getattr(value, 'shape', None)
  if shape is not None and hasattr(value, 'dtypes'):
    dtypes = getattr(value, 'dtypes', {})
    try:
      columns = list(getattr(value, 'columns', []) or [])
    except TypeError:
      columns = []
    return {
        'type': type(value).__name__,
        'shape': [int(n) for n in shape],
        'columns': [_truncate(c) for c in columns[:_MAX_ITEMS]],
        'dtypes': _truncate(dtypes),
    }
  return _truncate(value)


def _clean_context(context):
  '''Cap+coerce then scrub a context mapping.

  Args:
    context: Raw context mapping, or None.

  Returns:
    A new scrubbed, JSON-safe dict; never mutates input.

  Raises:
    TypeError: If ``context`` is not None and not a mapping.
  '''
  if context is None:
    return {}
  if not isinstance(context, Mapping):
    raise TypeError(
        'context must be a mapping or None, got '
        f'{type(context).__name__}')
  if not context:
    return {}
  coerced = _coerce(dict(context), frozenset(), 0)
  scrubbed = scrub_details(coerced)
  return scrubbed if isinstance(scrubbed, dict) else {}


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
    TypeError: If ``context`` is not a mapping or None.
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
      stage='',
      user_message=None,
      context=None,
      retryable=False,
      message=None,
  ):
    '''Initialize a typed error, scrubbing context.'''
    if args:
      args = list(args)
      if len(args) == 1:
        if user_message is None:
          user_message = args[0]
      else:
        code, stage, user_message, context, retryable = (
            list(args) + [None] * 5)[:5]
    if message is not None and user_message is None:
      user_message = message
    effective_code = str(code or self.default_code)
    if _CODE_RE.fullmatch(effective_code) is None:
      raise ValueError(
          'error code must fully match [A-Z][A-Z0-9_]*, '
          f'got {effective_code!r}')
    prefix = self._code_prefix
    if prefix is not None and not effective_code.startswith(prefix):
      raise ValueError(
          f'{type(self).__name__} code must start with '
          f'{prefix!r}, got {effective_code!r}')
    self.code = effective_code
    self.stage = str(stage or '')
    self.user_message = _clean_user_message(user_message)
    self.context = _clean_context(context)
    self.retryable = bool(retryable)
    super().__init__(str(self))

  @property
  def message(self):
    '''Alias for ``user_message``.'''
    return self.user_message

  def __str__(self):
    '''Return ``'[<code>] <user_message>'``.'''
    return f'[{self.code}] {self.user_message}'

  def to_dict(self):
    '''Return a JSON-safe envelope of this error.

    Returns:
      Dict with ``code``, ``stage``, ``user_message``,
      scrubbed ``context``, ``retryable``, and
      ``error_type``; safe for ``json.dumps``.
    '''
    return {
        'code': self.code,
        'stage': self.stage,
        'user_message': self.user_message,
        'context': _clean_context(self.context),
        'retryable': bool(self.retryable),
        'error_type': type(self).__name__,
    }

  @classmethod
  def _build(cls, code, user_message, stage, context, retryable):
    '''Build an instance with an explicit code.'''
    return cls(
        user_message, stage=stage, code=code,
        context=context, retryable=retryable)


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
  if _CODE_RE.fullmatch(str(code)) is None:
    raise ValueError(
        'factory code must fully match [A-Z][A-Z0-9_]*, '
        f'got {code!r}')
  prefix = getattr(cls, '_code_prefix', None)
  if prefix is not None and not str(code).startswith(prefix):
    raise ValueError(
        f'{cls.__name__} code must start with '
        f'{prefix!r}, got {code!r}')
  if code in _CODE_REGISTRY_DICT:
    raise ValueError(f'duplicate error code registration: {code!r}')
  _CODE_REGISTRY_DICT[code] = cls

  @classmethod
  def _make(
      factory_cls, user_message=None, *, stage='', context=None,
      retryable=retryable,
  ):
    return factory_cls._build(
        code, user_message or default_message, stage, context,
        retryable)

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
    TypeError: If ``context`` is not a mapping or None.
    ValueError: If ``code`` fails ctor validation.
  '''
  text = user_message if user_message is not None else message
  if isinstance(exc, FrameworkError):
    if context is not None and not isinstance(context, Mapping):
      raise TypeError(
          'context must be a mapping or None, got '
          f'{type(context).__name__}')
    if context:
      merged = dict(exc.context)
      merged.update(dict(context))
    else:
      merged = dict(exc.context)
    copied = type(exc)(
        code=code or exc.code,
        stage=stage or exc.stage,
        user_message=text if text is not None else exc.user_message,
        context=merged,
        retryable=(
            bool(retryable) if retryable is not None
            else exc.retryable),
    )
    copied.__cause__ = exc.__cause__
    copied.__context__ = exc.__context__
    copied.__traceback__ = exc.__traceback__
    copied.__suppress_context__ = exc.__suppress_context__
    notes = getattr(exc, '__notes__', None)
    if notes is not None:
      copied.__notes__ = list(notes)
    return copied
  wrapped = FrameworkError(
      code=code or FrameworkError.default_code,
      stage=stage or '',
      user_message=text if text is not None else str(exc),
      context=context,
      retryable=bool(retryable) if retryable is not None else False,
  )
  wrapped.__cause__ = exc
  wrapped.__suppress_context__ = False
  return wrapped


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
