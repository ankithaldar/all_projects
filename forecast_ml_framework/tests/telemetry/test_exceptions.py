#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for telemetry exception hierarchy.'''

import json

import pytest

from forecast_ml_framework.telemetry.exceptions import (
  CircuitOpenError,
  DataIngestionError,
  DataValidationError,
  EvaluationError,
  FeatureEngineeringError,
  FrameworkError,
  InterpretationError,
  ResilienceError,
  TrainingFailedError,
  wrap,
)


def test_framework_error_str_and_attrs():
  '''Str is bracketed code plus message.'''
  err = FrameworkError(
      code='TRAIN_FAILED',
      stage='modeling',
      user_message='boom',
      context={'epoch': 3},
      retryable=True,
  )
  assert str(err) == '[TRAIN_FAILED] boom'
  assert err.code == 'TRAIN_FAILED'
  assert err.stage == 'modeling'
  assert err.user_message == 'boom'
  assert err.context == {'epoch': 3}
  assert err.retryable is True


def test_framework_error_positional_order():
  '''Positional order is code stage message context retryable.'''
  err = FrameworkError(
      'EVAL_FAILED', 'modeling', 'bad metric', {'m': 1}, False)
  assert err.code == 'EVAL_FAILED'
  assert err.stage == 'modeling'
  assert err.user_message == 'bad metric'
  assert err.context == {'m': 1}
  assert err.retryable is False


@pytest.mark.parametrize(
    'cls,code',
    [
        (DataIngestionError, 'INGEST_ERROR'),
        (DataValidationError, 'VALIDATE_ERROR'),
        (FeatureEngineeringError, 'FEATURE_ERROR'),
        (TrainingFailedError, 'TRAIN_ERROR'),
        (EvaluationError, 'EVAL_ERROR'),
        (InterpretationError, 'INTERP_ERROR'),
        (ResilienceError, 'RESILIENCE_ERROR'),
        (CircuitOpenError, 'RESILIENCE_CIRCUIT_OPEN'),
    ],
)
def test_subclass_caught_as_framework_error(cls, code):
  '''Every subclass is a FrameworkError with stable code.'''
  err = cls('failed', stage='eda')
  assert isinstance(err, FrameworkError)
  assert err.code == code
  assert err.code.startswith(code.split('_')[0] + '_')
  with pytest.raises(FrameworkError):
    raise err


def test_subclass_code_override_within_prefix():
  '''Subclass accepts code override inside its prefix.'''
  err = DataIngestionError(
      'timeout', stage='eda', code='INGEST_TIMEOUT')
  assert err.code == 'INGEST_TIMEOUT'


def test_context_secret_redacted():
  '''Secret context values are scrubbed on construction.'''
  err = FrameworkError(
      code='X_ERROR',
      stage='eda',
      user_message='failed',
      context={'api_key': 'hunter2', 'nested': {'token': 'abc'}},
  )
  assert err.context['api_key'] == '***REDACTED***'
  assert err.context['nested']['token'] == '***REDACTED***'
  assert 'hunter2' not in json.dumps(err.context)


def test_context_coerced_json_safe():
  '''Sets and bytes in context become JSON-safe.'''
  err = FrameworkError(
      code='X_ERROR',
      stage='eda',
      user_message='failed',
      context={'tags': {'b', 'a'}, 'raw': b'ab'},
  )
  json.dumps(err.context)
  assert err.context['tags'] == ['a', 'b']
  assert isinstance(err.context['raw'], str)
  cyclic = {}
  cyclic['self'] = cyclic
  err2 = FrameworkError(
      code='X_ERROR', user_message='x', context=cyclic)
  json.dumps(err2.to_dict())
  assert err2.context['self']['self'] == '<cyclic>'


def test_to_dict_roundtrip_json():
  '''to_dict round-trips through JSON unchanged.'''
  err = TrainingFailedError(
      'no convergence', stage='modeling',
      context={'epoch': 1}, retryable=True)
  payload = err.to_dict()
  assert json.loads(json.dumps(payload)) == payload
  assert payload['code'] == 'TRAIN_ERROR'
  assert payload['stage'] == 'modeling'
  assert payload['user_message'] == 'no convergence'
  assert payload['context'] == {'epoch': 1}
  assert payload['retryable'] is True


def test_raise_from_preserves_cause():
  '''Raising from keeps __cause__ and traceback.'''
  try:
    try:
      raise ValueError('root cause')
    except ValueError as old:
      raise TrainingFailedError('train failed') from old
  except TrainingFailedError as err:
    assert isinstance(err.__cause__, ValueError)
    assert str(err.__cause__) == 'root cause'
    assert err.__traceback__ is not None


def test_wrap_typed_updates_stage_preserves():
  '''wrap on typed error updates stage, keeps code/cause.'''
  original = DataValidationError('bad data', stage='data_prep')
  original.__cause__ = ValueError('root')
  wrapped = wrap(original, stage='modeling')
  assert wrapped is not original
  assert original.stage == 'data_prep'
  assert wrapped.stage == 'modeling'
  assert wrapped.code == 'VALIDATE_ERROR'
  assert wrapped.__cause__ is not None


def test_wrap_typed_no_stage_keeps_stage():
  '''wrap without a new stage keeps the original stage.'''
  original = DataValidationError('bad', stage='data_prep')
  assert wrap(original).stage == 'data_prep'


def test_wrap_untyped_becomes_framework_error():
  '''wrap converts unknown exceptions preserving cause.'''
  old = RuntimeError('kaboom')
  wrapped = wrap(
      old, stage='modeling', code='TRAIN_FAILED',
      user_message='train broke', context={'epoch': 2})
  assert isinstance(wrapped, FrameworkError)
  assert wrapped.code == 'TRAIN_FAILED'
  assert wrapped.stage == 'modeling'
  assert wrapped.user_message == 'train broke'
  assert wrapped.context == {'epoch': 2}
  assert wrapped.__cause__ is old


def test_wrap_untyped_defaults():
  '''wrap defaults message from the original exception.'''
  old = ValueError('plain failure')
  wrapped = wrap(old, stage='eda')
  assert wrapped.user_message == 'plain failure'
  assert wrapped.code == 'FRAMEWORK_ERROR'


def test_training_oom_factory():
  '''TrainingFailedError.oom builds a retryable OOM error.'''
  err = TrainingFailedError.oom(stage='modeling')
  assert isinstance(err, TrainingFailedError)
  assert err.code == 'TRAIN_OOM'
  assert err.retryable is True
  assert err.stage == 'modeling'


def test_factories_on_other_subclasses():
  '''Factory classmethods exist and set prefixed codes.'''
  cases = [
      (DataIngestionError.failed(), 'INGEST_FAILED'),
      (DataValidationError.schema_mismatch(), 'VALIDATE_SCHEMA'),
      (FeatureEngineeringError.failed(), 'FEATURE_FAILED'),
      (EvaluationError.failed(), 'EVAL_FAILED'),
      (InterpretationError.failed(), 'INTERP_FAILED'),
      (ResilienceError.exhausted(), 'RESILIENCE_EXHAUSTED'),
      (CircuitOpenError.open(), 'RESILIENCE_CIRCUIT_OPEN'),
  ]
  for err, code in cases:
    assert isinstance(err, FrameworkError)
    assert err.code == code


def test_circuit_open_is_resilience_error():
  '''CircuitOpenError is a ResilienceError, retryable.'''
  err = CircuitOpenError.open(stage='modeling')
  assert isinstance(err, ResilienceError)
  assert err.retryable is True


def test_long_string_truncated():
  '''Oversized context strings are capped.'''
  err = FrameworkError(
      user_message='x', context={'blob': 'a' * 9000})
  assert err.context['blob'].endswith('...[TRUNCATED]')
  assert len(err.context['blob']) <= 8300


def test_tuple_datetime_and_unknown_coerced():
  '''Tuples, dates, and objects become JSON-safe.'''
  import datetime

  class Thing:
    '''Plain object coerced via str.'''

    def __str__(self):
      '''Return the string representation.'''
      return 'thing!'

  err = FrameworkError(
      user_message='x',
      context={
          'pair': (1, 'two'),
          'when': datetime.datetime(
              2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
          'thing': Thing(),
      })
  assert err.context['pair'] == [1, 'two']
  assert err.context['when'] == '2026-01-02T03:04:05+00:00'
  assert err.context['thing'] == {'type': 'Thing'}
  json.dumps(err.to_dict())


def test_dataframe_like_becomes_metadata():
  '''DataFrame-like values become shape/column metadata.'''

  class Frame:
    '''Minimal DataFrame stand-in.'''

    shape = (2, 2)
    columns = ('a', 'b')
    dtypes = 'int64'

  err = FrameworkError(
      user_message='x', context={'frame': Frame()})
  assert err.context['frame']['shape'] == [2, 2]
  assert err.context['frame']['columns'] == ['a', 'b']
  assert 'type' in err.context['frame']


def test_deep_nesting_capped():
  '''Deeply nested context hits the depth cap.'''
  payload = current = {}
  for _ in range(15):
    nxt = {}
    current['next'] = nxt
    current = nxt
  err = FrameworkError(user_message='x', context=payload)
  assert '<max-depth>' in json.dumps(err.context)


def test_many_items_truncated():
  '''Oversized lists and dicts are item-capped.'''
  err = FrameworkError(
      user_message='x',
      context={'items': list(range(1005))})
  assert err.context['items'][-1] == '...[TRUNCATED]'
  big = {f'k{i}': i for i in range(1005)}
  err2 = FrameworkError(user_message='x', context=big)
  assert err2.context['...[TRUNCATED]'] is True


def test_cyclic_list_coerced():
  '''A cyclic list becomes a cyclic marker.'''
  items = [1]
  items.append(items)
  err = FrameworkError(user_message='x', context={'i': items})
  assert err.context['i'][1] == '<cyclic>'


def test_single_positional_code_lookalike():
  '''A lone string is the user message, never a code.'''
  err = FrameworkError('TRAIN_OOM')
  assert err.code == 'FRAMEWORK_ERROR'
  assert err.user_message == 'TRAIN_OOM'


def test_message_alias_and_property():
  '''message kwarg and property mirror user_message.'''
  err = FrameworkError(message='via alias')
  assert err.user_message == 'via alias'
  assert err.message == 'via alias'


def test_bad_subclass_code_rejected():
  '''A subclass code outside its prefix raises.'''
  with pytest.raises(ValueError):
    DataIngestionError('x', code='TRAIN_OOM')


def test_wrap_typed_overrides():
  '''wrap overrides merge on typed errors.'''
  err = DataValidationError('bad', stage='eda')
  out = wrap(
      err, stage='modeling', user_message='new',
      context={'epoch': 1}, retryable=True)
  assert out.stage == 'modeling'
  assert out.user_message == 'new'
  assert out.context == {'epoch': 1}
  assert out.retryable is True


def test_defaults_without_args():
  '''FrameworkError can be raised with no arguments.'''
  err = FrameworkError()
  assert err.code == 'FRAMEWORK_ERROR'
  assert err.stage == ''
  assert err.user_message == ''
  assert err.context == {}
  assert err.retryable is False
  assert str(err) == '[FRAMEWORK_ERROR] '


def test_user_message_scrubbed_on_construction():
  '''Secrets in user_message are scrubbed in ctor and str.'''
  err = FrameworkError(
      code='X_ERROR', user_message='failed password=hunter2')
  assert 'hunter2' not in err.user_message
  assert 'hunter2' not in str(err)
  assert '***REDACTED***' in err.user_message


def test_user_message_truncated():
  '''Oversized user_message is capped like context strings.'''
  err = FrameworkError(code='X_ERROR', user_message='a' * 9000)
  assert err.user_message.endswith('...[TRUNCATED]')
  assert len(err.user_message) <= 8300


def test_wrap_typed_scrubs_user_message():
  '''wrap scrubs a typed user_message override.'''
  err = DataValidationError('bad', stage='eda')
  out = wrap(err, user_message='leak api_key=hunter2')
  assert 'hunter2' not in out.user_message
  assert 'hunter2' not in str(out)


def test_wrap_untyped_scrubs_str_exc_default():
  '''wrap scrubs str(exc) when no message is supplied.'''
  old = RuntimeError('failed password=hunter2')
  out = wrap(old, stage='eda')
  assert 'hunter2' not in out.user_message
  assert 'hunter2' not in str(out)


def test_bytes_secret_not_reversibly_leaked():
  '''Bytes are scrubbed before base64, so decode leaks nothing.'''
  import base64

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'raw': b'password=hunter2'})
  decoded = base64.b64decode(err.context['raw']).decode(
      'utf-8', errors='replace')
  assert 'hunter2' not in decoded
  assert 'hunter2' not in json.dumps(err.context)


def test_bytes_truncated():
  '''Oversized bytes base64 output is capped.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'raw': b'a' * 9000})
  assert err.context['raw'].endswith('...[TRUNCATED]')
  assert len(err.context['raw']) <= 8300


def test_wrap_typed_code_prefix_mismatch_rejected():
  '''wrap typed code override must satisfy the prefix guard.'''
  err = DataValidationError('bad', stage='eda')
  with pytest.raises(ValueError):
    wrap(err, code='TRAIN_OOM')


def test_wrap_copy_on_wrap_preserves_cause_and_traceback():
  '''wrap returns a copy and leaves the original untouched.'''
  try:
    try:
      raise ValueError('root')
    except ValueError as root:
      raise DataValidationError(
          'bad', stage='data_prep', context={'a': 1}) from root
  except DataValidationError as original:
    wrapped = wrap(original, stage='modeling',
                   context={'b': 2}, retryable=True)
    assert wrapped is not original
    assert original.stage == 'data_prep'
    assert original.context == {'a': 1}
    assert original.retryable is False
    assert wrapped.stage == 'modeling'
    assert wrapped.context == {'a': 1, 'b': 2}
    assert wrapped.retryable is True
    assert wrapped.__cause__ is original.__cause__
    assert wrapped.__traceback__ is original.__traceback__
    again = wrap(wrapped)
    assert again is not wrapped
    assert again.to_dict() == wrapped.to_dict()
    assert again.__cause__ is wrapped.__cause__


@pytest.mark.parametrize(
    'bad_code',
    ['train_oom', 'TRAIN-OOM', 'TRAIN OOM', '1ABC', 'TRAIN_OOM\nEVIL',
     'TRAIN_OOM '],
)
def test_code_fullmatch_validation(bad_code):
  '''Codes must fully match the code pattern.'''
  with pytest.raises(ValueError):
    FrameworkError(code=bad_code, user_message='x')


def test_code_registry_immutable_populated_duplicate_rejected():
  '''Factory codes live in an immutable registry.'''
  from forecast_ml_framework.telemetry.exceptions import (
      CODE_REGISTRY,
      _factory,
  )

  assert 'TRAIN_OOM' in CODE_REGISTRY
  assert CODE_REGISTRY['TRAIN_OOM'] is TrainingFailedError
  assert len(CODE_REGISTRY) >= 20
  with pytest.raises(TypeError):
    CODE_REGISTRY['NEW_CODE'] = FrameworkError
  with pytest.raises(ValueError):
    _factory(DataIngestionError, 'INGEST_FAILED', 'dup', False)


@pytest.mark.parametrize('bad_context', [['a'], 'string', 123, {'a'}])
def test_non_mapping_context_raises_typeerror(bad_context):
  '''Non-mapping context raises a documented TypeError.'''
  with pytest.raises(TypeError):
    FrameworkError(code='X_ERROR', user_message='x',
                   context=bad_context)


def test_wrap_non_mapping_context_raises_typeerror():
  '''wrap with non-mapping context raises TypeError.'''
  with pytest.raises(TypeError):
    wrap(ValueError('x'), context=['not-a-mapping'])


def test_stage_scrubbed_and_capped():
  '''Stage is scrubbed and capped at 200 chars.'''
  err = FrameworkError(
      code='X_ERROR', stage='leak password=hunter2', user_message='x')
  assert 'hunter2' not in err.stage
  assert '***REDACTED***' in err.stage
  long_err = FrameworkError(
      code='X_ERROR', stage='s' * 1000, user_message='x')
  assert long_err.stage.endswith('...[TRUNCATED]')
  assert len(long_err.stage) <= 200 + len('...[TRUNCATED]')


def test_composite_and_scalar_dict_keys_preserved():
  '''Tuple/frozenset/int keys keep their type and are scrubbed.'''
  err = FrameworkError(
      code='X_ERROR',
      user_message='x',
      context={
          ('a', 'b'): 1,
          frozenset({'x', 'y'}): 2,
          3: 'intkey',
          ('password', 'hunter2'): 'secretval',
      },
  )
  keys = list(err.context.keys())
  assert ('a', 'b') in keys
  assert frozenset({'x', 'y'}) in keys
  assert 3 in keys
  assert 'hunter2' not in str(keys)
  assert err.context[('password', '***REDACTED***')] == (
      '***REDACTED***')


def test_dataframe_like_introspection_failure_fallback():
  '''Broken DataFrame-like introspection falls back to str.'''

  class BadFrame:
    '''DataFrame stand-in whose introspection raises.'''

    @property
    def shape(self):
      '''Shape.'''
      raise RuntimeError('shape boom')

    @property
    def dtypes(self):
      '''Dtypes.'''
      raise RuntimeError('dtypes boom')

    @property
    def columns(self):
      '''Columns.'''
      raise RuntimeError('columns boom')

    def __str__(self):
      '''Return the string representation.'''
      return 'bad-frame-str'

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'frame': BadFrame()})
  assert err.context['frame'] == {'type': 'BadFrame'}


def test_dataframe_like_bad_shape_fallback():
  '''Non-int shape falls back to str.'''

  class WeirdFrame:
    '''DataFrame stand-in with a bad shape.'''

    shape = ('a', 2.5)
    columns = ('a', 'b')
    dtypes = 'int64'

    def __str__(self):
      '''Return the string representation.'''
      return 'weird-frame'

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'frame': WeirdFrame()})
  assert err.context['frame'] == {'type': 'WeirdFrame'}


def test_dataframe_like_columns_capped():
  '''Oversized column iteration is capped via islice.'''

  class WideFrame:
    '''DataFrame stand-in with many columns.'''

    shape = (1, 5000)
    columns = tuple(f'c{i}' for i in range(5000))
    dtypes = 'int64'

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'frame': WideFrame()})
  assert len(err.context['frame']['columns']) == 1000


def test_registry_default_codes_registered():
  '''Default codes for FrameworkError and subclasses registered.'''
  from forecast_ml_framework.telemetry.exceptions import CODE_REGISTRY

  assert CODE_REGISTRY['FRAMEWORK_ERROR'] is FrameworkError
  assert CODE_REGISTRY['INGEST_ERROR'] is DataIngestionError
  assert CODE_REGISTRY['VALIDATE_ERROR'] is DataValidationError
  assert CODE_REGISTRY['FEATURE_ERROR'] is FeatureEngineeringError
  assert CODE_REGISTRY['TRAIN_ERROR'] is TrainingFailedError
  assert CODE_REGISTRY['EVAL_ERROR'] is EvaluationError
  assert CODE_REGISTRY['INTERP_ERROR'] is InterpretationError
  assert CODE_REGISTRY['RESILIENCE_ERROR'] is ResilienceError
  assert (
      CODE_REGISTRY['RESILIENCE_CIRCUIT_OPEN'] is CircuitOpenError)


def test_code_registry_dead_alias_removed():
  '''Dead _CODE_REGISTRY alias no longer exists.'''
  import forecast_ml_framework.telemetry.exceptions as exc_mod

  assert not hasattr(exc_mod, '_CODE_REGISTRY')


def test_wrap_explicit_empty_stage_override():
  '''wrap sentinel allows explicit empty stage override.'''
  original = DataValidationError('bad', stage='data_prep')
  wrapped = wrap(original, stage='')
  assert wrapped.stage == ''


def test_wrap_untyped_preserves_context_traceback_notes():
  '''wrap on untyped preserves context/traceback/notes.'''
  old = RuntimeError('boom')
  old.__notes__ = ['a note']
  wrapped = wrap(old, stage='eda')
  assert wrapped.__context__ is old.__context__
  assert wrapped.__traceback__ is old.__traceback__
  assert wrapped.__notes__ == ['a note']
  assert wrapped.__notes__ is not old.__notes__


def test_nan_inf_become_null_marker():
  '''NaN/Inf floats become a null marker string, JSON strict.'''
  err = FrameworkError(
      code='X_ERROR',
      user_message='x',
      context={
          'nan': float('nan'),
          'inf': float('inf'),
          'ninf': float('-inf'),
          'nested': [float('nan')],
      },
  )
  assert err.context['nan'] == '<null>'
  assert err.context['inf'] == '<null>'
  assert err.context['ninf'] == '<null>'
  assert err.context['nested'] == ['<null>']
  json.dumps(err.to_dict(), allow_nan=False)


def test_exceptions_imports_scrub_leaf_not_governance_audit():
  '''telemetry.exceptions imports scrub_details from the scrub
  leaf (governance.scrub), never from governance.audit.

  Import direction is checked at source level (AST) so the
  governance package __init__ side effects cannot mask the
  direct dependency: the leaf import is what keeps
  telemetry from depending on governance.audit (whose
  callers may raise FrameworkError).
  '''
  import ast
  from pathlib import Path

  src = (
      Path(__file__).resolve().parents[2]
      / 'src' / 'forecast_ml_framework' / 'telemetry'
      / 'exceptions.py')
  tree = ast.parse(src.read_text())
  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
      imported.add(node.module or '')
    elif isinstance(node, ast.Import):
      imported.update(alias.name for alias in node.names)
  assert 'forecast_ml_framework.governance.scrub' in imported
  assert not any(name.endswith('governance.audit')
                 for name in imported)
  assert not any('governance.audit' in name for name in imported)
  # The imported function object really is the leaf's.
  import forecast_ml_framework.telemetry.exceptions as exc_mod
  from forecast_ml_framework.governance.scrub import scrub_details

  assert exc_mod.scrub_details is scrub_details


# --- telemetry-final-fixes (RED) ---

def test_bytes_dict_key_secret_redacted():
  '''Bytes password key redacts its value, not base64-leaked.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={b'password': 'hunter2', b'ok': 'keep'})
  blob = json.dumps(err.context, default=str)
  assert 'hunter2' not in blob
  assert 'aHVudGVyMg' not in blob  # base64(hunter2)
  assert err.context['password'] == '***REDACTED***'
  assert err.context['ok'] == 'keep'


def test_tuple_bytes_key_composite_redacted():
  '''Tuple bytes elements composite-marker detected pre-base64.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={(b'api', b'HUNTER2', b'key'): 'v1',
               ('ok', 'keep'): 'keep-value'})
  blob = str(err.context)
  assert 'HUNTER2' not in blob
  assert 'SFVOVEVSUg' not in blob
  assert ('ok', 'keep') in list(err.context.keys())


def test_tuple_value_bytes_composite_before_base64():
  '''Composite bytes in tuple/set values redacted before base64.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'pair': ('api', b'HUNTER2', 'key'),
               'setval': {b'password', b'HUNTER2'}})
  blob = json.dumps(err.context, default=str)
  assert 'HUNTER2' not in blob
  assert 'SFVOVEVSUg' not in blob
  assert '***REDACTED***' in blob


def test_unknown_object_never_str_dumped():
  '''Unknown objects become type-only dicts, never str dump.'''

  class SecretThing:
    '''Test double SecretThing.'''
    def __str__(self):
      '''Return the string representation.'''
      return 'password=hunter2 leak'

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'thing': SecretThing()})
  assert err.context['thing'] == {'type': 'SecretThing'}
  assert 'hunter2' not in json.dumps(err.context)


def test_to_dict_and_str_reclean_mutated_attrs():
  '''Mutated stage/message/code are re-cleaned on output.'''
  err = FrameworkError(
      code='X_ERROR', stage='eda', user_message='ok')
  err.stage = 'leak password=hunter2'
  err.user_message = 'leak password=hunter2'
  payload = err.to_dict()
  assert 'hunter2' not in json.dumps(payload)
  assert 'hunter2' not in str(err)
  assert '***REDACTED***' in payload['stage']
  assert '***REDACTED***' in payload['user_message']


def test_hostile_str_user_message_unrepresentable():
  '''Hostile __str__ becomes <unrepresentable>, no crash.'''

  class BadStr:
    '''Test double BadStr.'''
    def __str__(self):
      '''Return the string representation.'''
      raise RuntimeError('boom')

  err = FrameworkError(code='X_ERROR', user_message=BadStr())
  assert err.user_message == '<unrepresentable>'
  assert str(err) == '[X_ERROR] <unrepresentable>'


def test_wrap_hostile_str_exc_unrepresentable():
  '''wrap() on hostile-__str__ exception uses placeholder.'''

  class BadExc(Exception):
    '''Test double BadExc.'''
    def __str__(self):
      '''Return the string representation.'''
      raise RuntimeError('boom')

  out = wrap(BadExc(), stage='eda')
  assert out.user_message == '<unrepresentable>'


def test_positional_more_than_five_raises_typeerror():
  '''More than five positional args raise TypeError.'''
  with pytest.raises(TypeError):
    FrameworkError('X_ERROR', 'eda', 'msg', {}, False, 'extra')


def test_positional_kwargs_collision_raises_typeerror():
  '''Positional + duplicate kwarg raise TypeError.'''
  with pytest.raises(TypeError):
    FrameworkError(
        'X_ERROR', 'eda', 'msg', code='Y_ERROR')
  with pytest.raises(TypeError):
    FrameworkError('the message', user_message='other')


def test_set_hostile_repr_no_crash():
  '''Set element with hostile __repr__ does not crash coercion.

  Fails if hostile-repr coercion returns {} (empty dict
  cannot hold the set element's type marker) or crashes.
  '''
  class BadRepr:
    '''Test double BadRepr.'''
    def __repr__(self):
      '''Return the representation.'''
      raise RuntimeError('repr boom')

    def __str__(self):
      '''Return the string representation.'''
      return 'badrepr'

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'s': {BadRepr()}})
  dumped = json.dumps(err.context, default=str)
  assert isinstance(dumped, str)
  assert 'BadRepr' in dumped


def test_wrap_scrubs_notes():
  '''__notes__ are scrubbed on wrap (typed and untyped).'''
  old = RuntimeError('boom')
  old.__notes__ = ['password=hunter2', 'plain note']
  out = wrap(old, stage='eda')
  assert 'hunter2' not in str(out.__notes__)
  assert 'plain note' in out.__notes__
  assert '***REDACTED***' in out.__notes__[0]
  typed = DataValidationError('bad', stage='eda')
  typed.__notes__ = ['token=tok-12345']
  out2 = wrap(typed)
  assert 'tok-12345' not in str(out2.__notes__)


# --- loop2-audit-fixes: nested bytes before base64 (RED) ---

def test_nested_tuple_bytes_outer_marker_before_base64():
  '''Nested tuple bytes under outer marker redacted pre-base64.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': ('password', (b'HUNTER2',))})
  blob = json.dumps(err.context, default=str)
  assert 'HUNTER2' not in blob
  assert 'SFVOVEVSUg' not in blob
  assert '***REDACTED***' in blob


def test_nested_tuple_bytes_secret_before_marker_before_base64():
  '''Secret in nested tuple before outer marker redacted.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': ((b'HUNTER2',), 'password')})
  blob = json.dumps(err.context, default=str)
  assert 'HUNTER2' not in blob
  assert 'SFVOVEVSUg' not in blob


# --- loop2-cyclic-flake: depth cap before recursion (RED) ---

def test_deeply_nested_list_coerced_without_recursion_error():
  '''A list nested far past the depth cap is capped, not blown up.

  ``_coerce`` caps depth at ``_MAX_DEPTH``, but composite
  marker detection flattened the raw subtree without a depth
  cap first, so a deep raw list hit ``RecursionError`` in
  ``scrub._flatten_composite`` before the cap could apply
  (depth-dependent: isolation passed, full suite flaked).
  '''
  deep = []
  current = deep
  for _ in range(3000):
    nxt = []
    current.append(nxt)
    current = nxt
  err = FrameworkError(user_message='x', context={'i': deep})
  assert '<max-depth>' in json.dumps(err.context)


def test_deeply_nested_tuple_flattens_without_recursion_error():
  '''Deep tuple nesting flattens under a depth cap.'''
  from forecast_ml_framework.governance.scrub import (
      _flatten_composite,
  )
  deep = ('leaf',)
  for _ in range(3000):
    deep = (deep,)
  flat = list(_flatten_composite(deep))
  assert flat == ['<max-depth>']


def test_flatten_composite_cyclic_yields_marker():
  '''A cyclic composite yields a cyclic marker, not a hang.'''
  from forecast_ml_framework.governance.scrub import (
      _flatten_composite,
  )
  items = [1]
  items.append(items)
  flat = list(_flatten_composite(items))
  assert '<cyclic>' in flat


def test_cyclic_list_coerced_at_deep_stack():
  '''Cyclic list coerces even from a deep call stack.'''
  items = [1]
  items.append(items)

  def _recurse(depth):
    '''Recurse.'''
    if depth == 0:
      return FrameworkError(
          user_message='x', context={'i': items}).context['i'][1]
    return _recurse(depth - 1)

  # Remaining budget is small: the coercion must not spend
  # unbounded frames re-traversing the cycle / subtree.
  assert _recurse(200) == '<cyclic>'


# --- loop4-scrub-fixes: hostile keys / sets / ctor collisions (RED) ---

def test_loop4_hostile_dict_key_no_crash_no_leak():
  '''Test loop4 hostile dict key no crash no leak.'''
  class BadKey:
    '''Test double BadKey.'''
    def __repr__(self):
      '''Return the representation.'''
      raise RuntimeError('repr boom')

    def __str__(self):
      '''Return the string representation.'''
      return 'password=Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 5151

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={BadKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(err.context)
  assert '<unrepresentable>' in str(err.context)
  json.dumps(err.to_dict())


def test_loop4_hostile_tuple_key_no_crash():
  '''Test loop4 hostile tuple key no crash.'''
  class BadKey:
    '''Test double BadKey.'''
    def __repr__(self):
      '''Return the representation.'''
      raise RuntimeError('repr boom')

    def __str__(self):
      '''Return the string representation.'''
      return 'password=Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 5152

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={(BadKey(), 'x'): 'v'})
  assert 'Zq9SENTINEL' not in str(err.context)
  assert '<unrepresentable>' in str(err.context)


def test_loop4_hostile_key_repr_secret_scrubbed():
  '''Test loop4 hostile key repr secret scrubbed.'''
  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return 'password=Zq9SENTINEL'

    def __str__(self):
      '''Return the string representation.'''
      return 'password=Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 5153

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={EvilKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(err.context)


def test_loop4_unknown_object_in_set_context_no_crash():
  '''Test loop4 unknown object in set context no crash.'''
  class Obj:
    '''Test double Obj.'''
    pass

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'s': {Obj()}})
  json.dumps(err.to_dict())
  assert 'Obj' in str(err.context)


def test_loop4_dict_nested_in_marker_tuple_no_key_leak():
  '''Test loop4 dict nested in marker tuple no key leak.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': ('password', {'Zq': 'x'})})
  assert 'Zq' not in str(err.context)
  assert '***REDACTED***' in str(err.context)


def test_loop4_positional_and_collision_typeerrors():
  '''Test loop4 positional and collision typeerrors.'''
  with pytest.raises(TypeError):
    FrameworkError('X_ERROR', 'eda', 'msg', {}, False, 'extra', 'more')
  with pytest.raises(TypeError):
    FrameworkError('X_ERROR', 'eda', 'msg', stage='other')
  with pytest.raises(TypeError):
    FrameworkError('X_ERROR', 'eda', 'msg', {}, False, retryable=True)
  with pytest.raises(TypeError):
    FrameworkError('X_ERROR', 'eda', 'msg', message='alias')
  with pytest.raises(TypeError):
    FrameworkError('only message', message='alias')
  # Single positional is the message; code/stage kwargs are fine.
  err = FrameworkError('hello', code='X_ERROR', stage='eda')
  assert err.user_message == 'hello'
  assert err.code == 'X_ERROR'
  assert err.stage == 'eda'


def test_loop4_cyrillic_homoglyph_still_blocked():
  '''Test loop4 cyrillic homoglyph still blocked.'''
  err = FrameworkError(
      code='X_ERROR', user_message='p\u0430ssword=hunter2',
      context={'note': 'p\u0430ssword=hunter2'})
  assert 'hunter2' not in str(err)
  assert 'hunter2' not in json.dumps(err.to_dict())


# --- loop5-scrub-fixes: homoglyph / unknown keys / hostile (RED) ---

@pytest.mark.parametrize('text', [
    'p\u00e1ssword=hunter2',
    'p\u0251ssword=hunter2',
    'pa\u0301ssword=hunter2',
    'passwo\u044fd=hunter2',
    'p\u044fssword=hunter2',
])
def test_loop5_homoglyph_evasion_scrubbed(text):
  '''Test loop5 homoglyph evasion scrubbed.'''
  err = FrameworkError(
      code='X_ERROR', user_message=text, context={'note': text})
  assert 'hunter2' not in str(err), repr(text)
  assert 'hunter2' not in json.dumps(err.to_dict()), repr(text)
  assert '***REDACTED***' in str(err), repr(text)


def test_loop5_unknown_key_bare_secret_repr_not_preserved():
  '''Test loop5 unknown key bare secret repr not preserved.'''
  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return 'Zq9SENTINEL'

    def __str__(self):
      '''Return the string representation.'''
      return 'Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 5154

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={EvilKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(err.context)
  assert 'EvilKey' in str(err.context)
  json.dumps(err.to_dict())


def test_loop5_unknown_key_bare_secret_in_tuple_and_frozenset():
  '''Test loop5 unknown key bare secret in tuple and frozenset.'''
  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return 'Zq9SENTINEL'

    def __str__(self):
      '''Return the string representation.'''
      return 'Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 5155

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={(EvilKey(), 'x'): 'v', ('ok', 'keep'): 1})
  assert 'Zq9SENTINEL' not in str(err.context)
  err2 = FrameworkError(
      code='X_ERROR', user_message='x',
      context={frozenset({EvilKey()}): 'v'})
  assert 'Zq9SENTINEL' not in str(err2.context)
  str(err.to_dict())
  str(err2.to_dict())


def test_loop5_hostile_tuple_container_no_crash_construction():
  '''Test loop5 hostile tuple container no crash construction.'''
  class EvilTuple(tuple):
    '''Test double EvilTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': EvilTuple((1, 2)), 'ok': 'keep'})
  assert 'boom' not in str(err.context)
  assert err.context['ok'] == 'keep'
  json.dumps(err.to_dict())


def test_loop5_hostile_mapping_no_crash_construction():
  '''Test loop5 hostile mapping no crash construction.'''
  from collections.abc import Mapping

  class EvilMapping(Mapping):
    '''Test double EvilMapping.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

    def __len__(self):
      '''Return the length.'''
      return 1

    def __getitem__(self, key):
      '''Return an item by key.'''
      raise RuntimeError('getitem boom')

  err = FrameworkError(
      code='X_ERROR', user_message='x', context=EvilMapping())
  assert 'boom' not in str(err.context)
  json.dumps(err.to_dict())


def test_loop5_hostile_tuple_key_no_crash_construction():
  '''Test loop5 hostile tuple key no crash construction.'''
  class EvilTuple(tuple):
    '''Test double EvilTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={EvilTuple(('a',)): 'v', 'ok': 'keep'})
  assert 'boom' not in str(err.context)
  assert err.context['ok'] == 'keep'
  json.dumps(err.to_dict())


def test_loop5_hostile_dict_container_no_crash_construction():
  '''Test loop5 hostile dict container no crash construction.'''
  class EvilDict(dict):
    '''Test double EvilDict.'''
    def items(self):
      '''Items.'''
      raise RuntimeError('items boom')

    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': EvilDict({'a': 1}), 'ok': 'keep'})
  assert 'boom' not in str(err.context)
  assert err.context['ok'] == 'keep'
  json.dumps(err.to_dict())


# --- loop6-scrub-fixes: code leak / hostile code / retryable (RED) ---

def test_loop6_code_value_error_does_not_echo_raw_code():
  '''Test loop6 code value error does not echo raw code.'''
  with pytest.raises(ValueError) as excinfo:
    FrameworkError(code='password=hunter2', user_message='x')
  assert 'hunter2' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)


def test_loop6_factory_code_value_error_does_not_echo_raw():
  '''Test loop6 factory code value error does not echo raw.'''
  from forecast_ml_framework.telemetry.exceptions import _factory
  with pytest.raises(ValueError) as excinfo:
    _factory(FrameworkError, 'password=hunter2', 'dup', False)
  assert 'hunter2' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)


def test_loop6_hostile_code_str_no_crash_value_error():
  '''Test loop6 hostile code str no crash value error.'''
  class BadCode:
    '''Test double BadCode.'''
    def __str__(self):
      '''Return the string representation.'''
      raise RuntimeError('str boom')

    def __repr__(self):
      '''Return the representation.'''
      return 'BadCode()'

  with pytest.raises(ValueError) as excinfo:
    FrameworkError(code=BadCode(), user_message='x')
  assert 'boom' not in str(excinfo.value)


def test_loop6_hostile_factory_code_str_no_crash():
  '''Test loop6 hostile factory code str no crash.'''
  from forecast_ml_framework.telemetry.exceptions import _factory

  class BadCode:
    '''Test double BadCode.'''
    def __str__(self):
      '''Return the string representation.'''
      raise RuntimeError('str boom')

    def __repr__(self):
      '''Return the representation.'''
      return 'BadCode()'

  with pytest.raises(ValueError) as excinfo:
    _factory(FrameworkError, BadCode(), 'dup', False)
  assert 'boom' not in str(excinfo.value)


def test_loop6_hostile_retryable_bool_defaults_false():
  '''Test loop6 hostile retryable bool defaults false.'''
  class BadBool:
    '''Test double BadBool.'''
    def __bool__(self):
      '''Return the truth value.'''
      raise RuntimeError('bool boom')

  err = FrameworkError(
      code='X_ERROR', user_message='x', retryable=BadBool())
  assert err.retryable is False


def test_loop6_mutated_retryable_to_dict_safe_bool():
  '''Test loop6 mutated retryable to dict safe bool.'''
  class BadBool:
    '''Test double BadBool.'''
    def __bool__(self):
      '''Return the truth value.'''
      raise RuntimeError('bool boom')

  err = FrameworkError(code='X_ERROR', user_message='x')
  err.retryable = BadBool()
  payload = err.to_dict()
  assert payload['retryable'] is False
  json.dumps(payload)


def test_loop6_hostile_retryable_wrap_defaults_false():
  '''Test loop6 hostile retryable wrap defaults false.'''
  class BadBool:
    '''Test double BadBool.'''
    def __bool__(self):
      '''Return the truth value.'''
      raise RuntimeError('bool boom')

  old = ValueError('plain failure')
  out = wrap(old, stage='eda', retryable=BadBool())
  assert out.retryable is False


def test_loop6_should_redact_marker_bytes_predicate():
  '''Test loop6 should redact marker bytes predicate.'''
  from forecast_ml_framework.telemetry.exceptions import (
    _should_redact_marker_bytes,
  )
  assert _should_redact_marker_bytes(True, b'HUNTER2') is True
  assert _should_redact_marker_bytes(True, b'password') is False
  assert _should_redact_marker_bytes(True, b'api') is False
  assert _should_redact_marker_bytes(False, b'HUNTER2') is False
  assert _should_redact_marker_bytes(True, 'HUNTER2') is False


def test_loop6_clean_scrubbed_shared_helper():
  '''Test loop6 clean scrubbed shared helper.'''
  from forecast_ml_framework.telemetry.exceptions import (
    _clean_code,
    _clean_scrubbed,
    _clean_stage,
    _clean_user_message,
  )
  assert _clean_scrubbed(None, 200) == ''
  assert _clean_scrubbed('password=hunter2', 200) == (
      'password=***REDACTED***')
  assert _clean_stage('password=hunter2') == (
      'password=***REDACTED***')
  assert _clean_code('password=hunter2') == (
      'password=***REDACTED***')
  assert _clean_user_message('password=hunter2') == (
      'password=***REDACTED***')
  assert _clean_scrubbed('x' * 300, 200).endswith(
      '...[TRUNCATED]')


# --- telemetry-final-fixes loop6 gates (RED) ---

def _fake_dict_config():
  '''Fake dict config.'''
  from collections.abc import Mapping

  class DictConfig(Mapping):
    '''Test double DictConfig.'''
    def __getitem__(self, key):
      '''Return an item by key.'''
      return '${oc.env:MY_SECRET_ENV}'

    def __iter__(self):
      '''Return an iterator.'''
      return iter(['k'])

    def __len__(self):
      '''Return the length.'''
      return 1

  return DictConfig()


def test_final_dict_config_context_rejected():
  '''Test final dict config context rejected.'''
  with pytest.raises((TypeError, ValueError)):
    FrameworkError(
        code='X_ERROR', user_message='x', context=_fake_dict_config())


def test_final_dict_config_wrap_merge_rejected():
  '''Test final dict config wrap merge rejected.'''
  original = DataValidationError('bad', stage='eda')
  with pytest.raises((TypeError, ValueError)):
    wrap(original, context=_fake_dict_config())
  with pytest.raises((TypeError, ValueError)):
    wrap(ValueError('x'), context=_fake_dict_config())


def test_final_real_omegaconf_dict_config_rejected():
  '''Test final real omegaconf dict config rejected.'''
  pytest.importorskip('omegaconf')
  from omegaconf import OmegaConf
  cfg = OmegaConf.create({'k': '${oc.env:MY_SECRET_ENV}'})
  with pytest.raises((TypeError, ValueError)):
    FrameworkError(code='X_ERROR', user_message='x', context=cfg)


@pytest.mark.parametrize('repr_text', [
    '\u00e1Zq9SENTINEL',
    '\uff41Zq9SENTINEL',
    'a\u0301Zq9SENTINEL',
    'a\u200bZq9SENTINEL',
])
def test_final_unknown_key_normalization_bare_repr_not_preserved(
        repr_text):
  '''Test final unknown key normalization bare repr not preserved.'''
  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return repr_text

    __str__ = __repr__

    def __hash__(self):
      '''Return the hash value.'''
      return 5160

  err = FrameworkError(
      code='X_ERROR', user_message='x', context={EvilKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(err.context), repr(repr_text)
  assert 'aZq9SENTINEL' not in str(err.context), repr(repr_text)
  assert 'EvilKey' in str(err.context), repr(repr_text)
  json.dumps(err.to_dict())


@pytest.mark.parametrize('text', [
    'secret_key=hunter2',
    'password_hash=hunter2',
    'token_value=hunter2',
    'aws_secret_key=hunter2',
    'api.key=hunter2',
    'access.key=hunter2',
    'pass.word=hunter2',
])
def test_final_compound_and_dotted_markers_scrubbed(text):
  '''Test final compound and dotted markers scrubbed.'''
  err = FrameworkError(
      code='X_ERROR', user_message=text, context={'note': text})
  assert 'hunter2' not in str(err), repr(text)
  assert 'hunter2' not in json.dumps(err.to_dict()), repr(text)
  assert '***REDACTED***' in str(err), repr(text)


@pytest.mark.parametrize('text', [
    'Bearer is tok12345',
    'Bearer was tok12345',
    'Bearer very tok12345',
    'Bearer is very tok12345 end',
])
def test_final_bearer_filler_scrubbed(text):
  '''Test final bearer filler scrubbed.'''
  err = FrameworkError(
      code='X_ERROR', user_message=text, context={'note': text})
  assert 'tok12345' not in str(err), repr(text)
  assert 'tok12345' not in json.dumps(err.to_dict()), repr(text)
  assert '***REDACTED***' in str(err), repr(text)


def test_final_tuple_frozenset_keys_to_dict_json_safe():
  '''Test final tuple frozenset keys to dict json safe.'''
  err = FrameworkError(
      code='X_ERROR',
      user_message='x',
      context={
          ('a', 'b'): 1,
          frozenset({'x', 'y'}): 2,
          ('password', 'hunter2'): 'secretval',
          'nested': {('n1', 'n2'): 3},
      },
  )
  # In-memory form keeps composite keys.
  keys = list(err.context.keys())
  assert ('a', 'b') in keys
  assert frozenset({'x', 'y'}) in keys
  # to_dict form is JSON-safe: composite keys are strings.
  payload = err.to_dict()
  assert json.loads(json.dumps(payload)) == payload
  assert 'hunter2' not in json.dumps(payload)
  assert all(
      isinstance(key, (str, int, float, bool)) or key is None
      for key in payload['context'])
  assert all(
      isinstance(key, (str, int, float, bool)) or key is None
      for key in payload['context']['nested'])


def test_final_infinite_tuple_container_capped():
  '''Test final infinite tuple container capped.'''
  class InfiniteTuple(tuple):
    '''Test double InfiniteTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      count = 0
      while True:
        yield count
        count += 1

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': InfiniteTuple((1, 2))})
  assert len(err.context['k']) <= 1001
  assert err.context['k'][:3] == [0, 1, 2]
  json.dumps(err.to_dict())


def test_final_infinite_list_container_capped():
  '''Test final infinite list container capped.'''
  class InfiniteList(list):
    '''Test double InfiniteList.'''
    def __iter__(self):
      '''Return an iterator.'''
      count = 0
      while True:
        yield count
        count += 1

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': InfiniteList([1])})
  assert len(err.context['k']) <= 1001
  json.dumps(err.to_dict())


def test_final_infinite_dict_items_capped():
  '''Test final infinite dict items capped.'''
  class InfiniteDict(dict):
    '''Test double InfiniteDict.'''
    def items(self):
      '''Items.'''
      count = 0
      while True:
        yield f'k{count}', count
        count += 1

  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': InfiniteDict({'a': 1})})
  assert len(err.context['k']) <= 1001
  json.dumps(err.to_dict())


# --- loop7-nfkc-split-fix: NFKC separator splits marker (RED) ---

@pytest.mark.parametrize('text', [
    'toke\u0149=Zq9SENTINEL7x',
    'crede\u0149tial=Zq9SENTINEL7x',
    'p' + '\u1e9a' + 'ssword=Zq9SENTINEL7x',
    'credenti' + '\u1e9a' + 'l=Zq9SENTINEL7x',
    'cre\u249fential=Zq9SENTINEL7x',
    'toke\u02bcn=Zq9SENTINEL7x',
    'toke\u02c2n=Zq9SENTINEL7x',
])
def test_loop7_nfkc_split_marker_scrubbed_in_framework_error(text):
  '''Test loop7 nfkc split marker scrubbed in framework error.'''
  err = FrameworkError(
      code='X_ERROR', user_message=text, context={'note': text})
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict()), repr(text)
  assert '***REDACTED***' in str(err), repr(text)
  assert '***REDACTED***' in json.dumps(err.to_dict()), repr(text)


# --- loop8-proven-fixes: unknown-key secret flag (RED) ---

class _Loop8EvilKeyPw:
  '''Key whose repr carries a password marker.'''

  def __repr__(self):
    '''Return the representation.'''
    return 'password=Zq9SENTINEL7x'

  __str__ = __repr__

  def __hash__(self):
    '''Return the hash value.'''
    return 4244


class _Loop8BadRepr:
  '''Key whose repr raises, carrying sentinel in error.'''

  def __repr__(self):
    '''Return the representation.'''
    raise RuntimeError('Zq9SENTINEL7x')

  def __str__(self):
    '''Return the string representation.'''
    return 'password=Zq9SENTINEL7x'

  def __hash__(self):
    '''Return the hash value.'''
    return 4245


def test_loop8_unknown_key_secret_value_redacted():
  '''Secret-bearing unknown key redacts value in telemetry.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={_Loop8EvilKeyPw(): 'Zq9SENTINEL7x'})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob
  assert 'Zq9SENTINEL7x' not in str(err.context)
  assert 'password=Zq9SENTINEL7x' not in blob
  assert '***REDACTED***' in blob
  assert '<_Loop8EvilKeyPw>' in blob


def test_loop8_unknown_key_tuple_frozenset_secret_redacted():
  '''Tuple/frozenset unknown secret keys redact value.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={(_Loop8EvilKeyPw(), 'x'): 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict())
  assert '***REDACTED***' in json.dumps(err.to_dict())
  err2 = FrameworkError(
      code='X_ERROR', user_message='x',
      context={frozenset({_Loop8EvilKeyPw()}): 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err2.to_dict())
  assert '***REDACTED***' in json.dumps(err2.to_dict())


def test_loop8_unknown_key_nested_list_secret_redacted():
  '''Nested/list unknown secret keys redact value.'''
  contexts = [
    {'outer': {_Loop8EvilKeyPw(): 'Zq9SENTINEL7x'}},
    {'k': [{_Loop8EvilKeyPw(): 'Zq9SENTINEL7x'}]},
    {'k': [[{_Loop8EvilKeyPw(): 'Zq9SENTINEL7x'}]]},
  ]
  for ctx in contexts:
    err = FrameworkError(
        code='X_ERROR', user_message='x', context=ctx)
    blob = json.dumps(err.to_dict())
    assert 'Zq9SENTINEL7x' not in blob, repr(ctx)
    assert '***REDACTED***' in blob, repr(ctx)


def test_loop8_unknown_key_unrepresentable_secret_redacted():
  '''Unrepresentable unknown key redacts value, repr hidden.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={_Loop8BadRepr(): 'Zq9SENTINEL7x'})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob
  assert '***REDACTED***' in blob
  assert '<unrepresentable>' in blob


# --- loop9-proven-fixes: digraphs, hostile datetime, dedupe (RED) ---

@pytest.mark.parametrize('text', [
    'pa\u00dfword=Zq9SENTINEL7x',
    'pa\u1e9eword=Zq9SENTINEL7x',
    'pa\u00dfwd=Zq9SENTINEL7x',
    'acce\u00df_key=Zq9SENTINEL7x',
    'au\u00fe=Zq9SENTINEL7x',
    'au\u00feorization=Zq9SENTINEL7x',
    'au\u00deorization=Zq9SENTINEL7x',
])
def test_loop9_digraph_markers_scrubbed_in_framework_error(text):
  '''Digraph ss/th markers redact sentinel in FrameworkError.'''
  err = FrameworkError(
      code='X_ERROR', user_message=text, context={'note': text})
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict()), repr(text)
  assert '***REDACTED***' in str(err), repr(text)


@pytest.mark.parametrize('key', [
    'pa\u00dfword',
    'pa\u1e9eword',
    'au\u00feorization',
    'au\u00deorization',
])
def test_loop9_digraph_markers_scrubbed_in_context_key(key):
  '''Digraph ss/th markers redact value under context key.'''
  err = FrameworkError(
      code='X_ERROR', user_message='x',
      context={key: 'Zq9SENTINEL7x', 'ok': 'keep'})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob, repr(key)
  assert '***REDACTED***' in blob, repr(key)
  assert err.context['ok'] == 'keep'


def _loop9_evil_datetime():
  '''Loop9 evil datetime.'''
  from datetime import datetime

  class EvilDate(datetime):
    '''Test double EvilDate.'''
    def isoformat(self, sep='T', timespec='auto'):
      '''Isoformat.'''
      raise RuntimeError('Zq9SENTINEL7x')

  return EvilDate(2024, 1, 1)


def _loop9_evil_date():
  '''Loop9 evil date.'''
  from datetime import date

  class EvilDateOnly(date):
    '''Test double EvilDateOnly.'''
    def isoformat(self, sep='T', timespec='auto'):
      '''Isoformat.'''
      raise RuntimeError('Zq9SENTINEL7x')

  return EvilDateOnly(2024, 1, 1)


def _loop9_evil_time():
  '''Loop9 evil time.'''
  from datetime import time

  class EvilTime(time):
    '''Test double EvilTime.'''
    def isoformat(self, sep='T', timespec='auto'):
      '''Isoformat.'''
      raise RuntimeError('Zq9SENTINEL7x')

  return EvilTime(12, 0)


def test_loop9_hostile_datetime_value_no_crash_type_only():
  '''Hostile datetime/date values fall back to type-only.'''
  from forecast_ml_framework.telemetry.exceptions import _coerce
  evil = _loop9_evil_datetime()
  assert _coerce(evil, frozenset(), 0) == {'type': 'EvilDate'}
  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'k': evil})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob
  assert err.context['k'] == {'type': 'EvilDate'}
  err_date = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': _loop9_evil_date()})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_date.to_dict())
  assert err_date.context['k'] == {'type': 'EvilDateOnly'}
  err_list = FrameworkError(
      code='X_ERROR', user_message='x',
      context={'k': [evil]})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_list.to_dict())
  assert err_list.context['k'] == [{'type': 'EvilDate'}]


def test_loop9_hostile_datetime_key_no_crash_no_leak():
  '''Hostile datetime/date keys never crash, sentinel hidden.'''
  evil = _loop9_evil_datetime()
  err = FrameworkError(
      code='X_ERROR', user_message='x', context={evil: 'v'})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob
  assert 'EvilDate' in blob
  json.dumps(err.to_dict())
  err_date = FrameworkError(
      code='X_ERROR', user_message='x',
      context={_loop9_evil_date(): 'v'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_date.to_dict())
  assert 'EvilDateOnly' in json.dumps(err_date.to_dict())
  err_tuple = FrameworkError(
      code='X_ERROR', user_message='x',
      context={(evil, 'x'): 'v', ('ok', 'keep'): 1})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_tuple.to_dict())
  assert err_tuple.context[('ok', 'keep')] == 1


def test_loop9_hostile_time_probe_no_crash_type_only():
  '''Hostile time probes fall back to type-only, no crash.'''
  evil_time = _loop9_evil_time()
  err = FrameworkError(
      code='X_ERROR', user_message='x', context={'k': evil_time})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob
  assert err.context['k'] == {'type': 'EvilTime'}
  err_key = FrameworkError(
      code='X_ERROR', user_message='x',
      context={evil_time: 'v'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_key.to_dict())
  assert 'EvilTime' in json.dumps(err_key.to_dict())


def test_loop9_safe_helpers_single_home_in_scrub():
  '''Telemetry safe helpers are the scrub leaf's, not copies.'''
  import forecast_ml_framework.telemetry.exceptions as exc_mod
  from forecast_ml_framework.governance.scrub import (
    _safe_list,
    _safe_repr,
    _safe_str,
  )
  assert exc_mod._safe_str is _safe_str
  assert exc_mod._safe_repr is _safe_repr
  assert exc_mod._safe_list is _safe_list

  class BadStr:
    '''Test double BadStr.'''
    def __str__(self):
      '''Return the string representation.'''
      raise RuntimeError('Zq9SENTINEL7x')

  class BadRepr:
    '''Test double BadRepr.'''
    def __repr__(self):
      '''Return the representation.'''
      raise RuntimeError('Zq9SENTINEL7x')

  assert _safe_str(BadStr()) == '<unrepresentable>'
  assert exc_mod._safe_str(BadStr()) == '<unrepresentable>'
  assert _safe_repr(BadRepr()) == '<unrepresentable>'
  assert exc_mod._safe_repr(BadRepr()) == '<unrepresentable>'

  class BadIter:
    '''Test double BadIter.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('Zq9SENTINEL7x')

  assert _safe_list(BadIter()) is None
  assert exc_mod._safe_list(BadIter()) is None


# --- loop12 arch/sec: to_dict total, error_type, hostile dict (RED) ---

@pytest.mark.parametrize('mutated', [
  ['not', 'a', 'mapping'],
  'password=hunter2',
  123,
  {'a'},
])
def test_loop12_to_dict_mutated_context_returns_empty(mutated):
  '''Mutated non-mapping context does not crash to_dict.'''
  err = FrameworkError(code='X_ERROR', user_message='x')
  err.context = mutated
  payload = err.to_dict()
  assert payload['context'] == {}
  json.dumps(payload)


def test_loop12_to_dict_mutated_omegaconf_context_empty():
  '''Mutated OmegaConf context does not crash to_dict.'''
  pytest.importorskip('omegaconf')
  from omegaconf import OmegaConf
  err = FrameworkError(code='X_ERROR', user_message='x')
  err.context = OmegaConf.create({'k': 'v'})
  payload = err.to_dict()
  assert payload['context'] == {}
  json.dumps(payload)


def test_loop12_to_dict_error_type_allowlist():
  '''Dynamic type name carrying a sentinel does not leak.'''
  dyn = type('Zq9SENTINEL7x', (FrameworkError,), {})
  err = dyn(code='X_ERROR', user_message='x')
  payload = err.to_dict()
  assert 'Zq9SENTINEL7x' not in json.dumps(payload)
  assert payload['error_type'] == 'FrameworkError'
  dyn2 = type('password=hunter2', (FrameworkError,), {})
  err2 = dyn2(code='X_ERROR', user_message='x')
  assert 'hunter2' not in json.dumps(err2.to_dict())
  assert err2.to_dict()['error_type'] == 'FrameworkError'
  known = TrainingFailedError(
    code='TRAIN_FAILED', user_message='x')
  assert known.to_dict()['error_type'] == 'TrainingFailedError'


def test_loop12_hostile_dict_items_non_pair_no_crash():
  '''Hostile dict items() with non-pairs falls back to type.'''
  class HostileDict(dict):
    '''Test double HostileDict.'''
    def items(self):
      '''Items.'''
      return [('k', 'v', 'extra'), ('a', 'b')]

  err = FrameworkError(
    code='X_ERROR', user_message='x',
    context={'outer': HostileDict()})
  assert err.context == {'outer': {'type': 'HostileDict'}}
  json.dumps(err.to_dict())
  err_list = FrameworkError(
    code='X_ERROR', user_message='x',
    context={'k': [HostileDict()]})
  assert err_list.context == {'k': [{'type': 'HostileDict'}]}
  json.dumps(err_list.to_dict())


# --- loop14: arch fixes (RED) ---

def test_loop14_telemetry_import_does_not_load_audit():
  '''import telemetry.exceptions must not load audit.'''
  import os
  import subprocess
  import sys
  from pathlib import Path

  root = Path(__file__).resolve().parents[2]
  env = dict(os.environ)
  env['PYTHONPATH'] = (
    str(root / 'src') + os.pathsep + env.get('PYTHONPATH', ''))
  code = '\n'.join([
    'import sys',
    'import forecast_ml_framework.telemetry.exceptions as e',
    'assert \'forecast_ml_framework.governance.audit\' not in sys.modules',
    'assert \'forecast_ml_framework.governance.lineage\' not in sys.modules',
    'assert \'forecast_ml_framework.governance.retention\' not in sys.modules',
    'err = e.FrameworkError(code=\'X_ERROR\', user_message=\'x\')',
    'assert err.to_dict()[\'code\'] == \'X_ERROR\'',
  ])
  proc = subprocess.run(
    [sys.executable, '-c', code], capture_output=True, text=True,
    env=env, timeout=60, check=False)
  assert proc.returncode == 0, proc.stderr


def test_loop14_wrap_envelope_only_logging_excludes_cause():
  '''wrap: envelope-only logging; cause never in to_dict/str.'''
  assert 'envelope-only' in (wrap.__doc__ or '').lower()
  cause = ValueError('password=hunter2 Zq9SENTINEL7x')
  wrapped = wrap(cause, stage='eda', user_message='clean message')
  # Cause is preserved internally by design.
  assert wrapped.__cause__ is cause
  assert 'hunter2' not in json.dumps(wrapped.to_dict())
  assert 'Zq9SENTINEL7x' not in json.dumps(wrapped.to_dict())
  assert 'hunter2' not in str(wrapped)
  assert 'Zq9SENTINEL7x' not in str(wrapped)
  typed_cause = RuntimeError('token=tok-12345')
  typed = DataValidationError('bad', stage='eda')
  typed.__cause__ = typed_cause
  wrapped_typed = wrap(typed, stage='modeling')
  assert wrapped_typed.__cause__ is typed_cause
  assert 'tok-12345' not in json.dumps(wrapped_typed.to_dict())
  assert 'tok-12345' not in str(wrapped_typed)


def test_loop14_mutated_context_and_code_reclean_on_output():
  '''Mutated context/code cannot leak via to_dict/str.'''
  err = FrameworkError(code='X_ERROR', stage='eda', user_message='ok')
  err.context = {'password': 'hunter2', 'nested': {'token': 'tok-1'}}
  err.code = 'password=hunter2'
  payload = err.to_dict()
  assert 'hunter2' not in json.dumps(payload)
  assert 'tok-1' not in json.dumps(payload)
  assert 'hunter2' not in str(err)
  assert payload['context']['password'] == '***REDACTED***'
  err2 = FrameworkError(
    code='X_ERROR', user_message='x', context={'a': 1})
  err2.context['password'] = 'hunter2'
  assert 'hunter2' not in json.dumps(err2.to_dict())


def test_loop14_resolve_ctor_args_and_validate_code_helpers():
  '''Ctor helpers exist and preserve ctor semantics.'''
  import forecast_ml_framework.telemetry.exceptions as exc_mod

  assert callable(exc_mod._resolve_ctor_args)
  assert callable(exc_mod._validate_code)
  assert exc_mod._validate_code('X_ERROR', None, 'FrameworkError') == (
    'X_ERROR')
  with pytest.raises(ValueError):
    exc_mod._validate_code('bad-code', None, 'FrameworkError')
  with pytest.raises(ValueError):
    exc_mod._validate_code('TRAIN_OOM', 'INGEST_', 'DataIngestionError')
  resolved = exc_mod._resolve_ctor_args(
    ('only message',), code='X_ERROR', stage='eda',
    user_message=exc_mod._UNSET, context=exc_mod._UNSET,
    retryable=exc_mod._UNSET, message=exc_mod._UNSET)
  assert resolved == ('X_ERROR', 'eda', 'only message', None, False)


def _loop14_complexity_offenders(path):
  '''Return [(name, lineno, count)] with count > 15.'''
  import ast

  tree = ast.parse(path.read_text())
  offenders = []
  for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    count = 1
    stack = list(node.body)
    while stack:
      child = stack.pop()
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
      if isinstance(child, ast.If):
        count += 1
      elif isinstance(child, ast.For):
        count += 1
      elif isinstance(child, ast.While):
        count += 1
      elif isinstance(child, ast.ExceptHandler):
        count += 1
      elif isinstance(child, ast.BoolOp):
        count += len(child.values) - 1
      elif isinstance(child, ast.IfExp):
        count += 1
      elif isinstance(child, ast.comprehension):
        count += 1 + len(child.ifs)
      stack.extend(ast.iter_child_nodes(child))
    if count > 15:
      offenders.append((node.name, node.lineno, count))
  return offenders


def test_loop14_complexity_sanity_exceptions_and_scrub():
  '''Every function in exceptions/scrub is <= 15 branches.'''
  from pathlib import Path

  base = (
    Path(__file__).resolve().parents[2]
    / 'src' / 'forecast_ml_framework')
  offenders = []
  for rel in ('telemetry/exceptions.py', 'governance/scrub.py'):
    for name, lineno, count in _loop14_complexity_offenders(base / rel):
      offenders.append(f'{rel}:{name} L{lineno}={count}')
  assert offenders == [], offenders


def test_loop14_telemetry_spool_tmp_fixture(telemetry_spool_tmp):
  '''conftest spool tmp fixture provides a usable dir.'''
  assert telemetry_spool_tmp.name == 'spool'
  assert telemetry_spool_tmp.is_dir()


# --- loop17: proven fixes P1-P4 telemetry (RED) ---

@pytest.mark.parametrize('type_name', [
    'password=Zq9SENTINEL7x',
    'passwordZq9SENTINEL7x',
    'token=Zq9SENTINEL7x',
    'Zq9SENTINEL7x',
])
def test_loop17_telemetry_type_name_scrubbed(type_name):
  '''P1 telemetry: dynamic type name does not leak via context.'''
  obj = type(type_name, (), {})()
  err = FrameworkError(
        code='X_ERROR', user_message='oops',
        context={'note': obj})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob, repr(type_name)
  assert 'Zq9SENTINEL7x' not in str(err), repr(type_name)
  err_key = FrameworkError(
        code='X_ERROR', user_message='oops',
        context={obj: 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_key.to_dict())


@pytest.mark.parametrize('text', [
    'api very key=Zq9SENTINEL7x',
    'access very key=Zq9SENTINEL7x',
    'private very key=Zq9SENTINEL7x',
    'api is key=Zq9SENTINEL7x',
    'access very key very id=Zq9SENTINEL7x',
    'password pretty Zq9SENTINEL7x',
    'password is quite Zq9SENTINEL7x',
    'password is pretty much Zq9SENTINEL7x',
    'token is so Zq9SENTINEL7x',
    'secret was too Zq9SENTINEL7x',
    'password somewhat Zq9SENTINEL7x',
    'password extremely Zq9SENTINEL7x',
    'Bearer very)really Zq9SENTINEL7x',
    'Bearer very) Zq9SENTINEL7x',
    'Bearer is] Zq9SENTINEL7x',
    'Bearer very}really Zq9SENTINEL7x',
])
def test_loop17_telemetry_filler_and_bearer_scrubbed(text):
  '''P2-P4 telemetry: filler/bracket patterns do not leak.'''
  err = FrameworkError(
        code='X_ERROR', user_message='oops ' + text,
        context={'note': text})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)


# --- loop17-arch/sec: H1/H2/H3/F1/F2 + suppress (RED) ---

@pytest.mark.parametrize('type_name', [
    'zqsentinelvalue',
    'zq9sentinelvalue',
    '\u00e1Zq9SENTINEL7x',
    '\uff41Zq9SENTINEL7x',
    'a\u0301Zq9SENTINEL7x',
])
def test_loop17_arch_telemetry_type_name_scrubbed(type_name):
  '''H3/F1 telemetry: sentinel type names do not leak.'''
  obj = type(type_name, (), {})()
  err = FrameworkError(
        code='X_ERROR', user_message='oops',
        context={'note': obj})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob, repr(type_name)
  assert 'zqsentinelvalue' not in blob, repr(type_name)
  assert 'Zq9SENTINEL7x' not in str(err), repr(type_name)
  assert '***REDACTED***' in blob, repr(type_name)
  err_key = FrameworkError(
        code='X_ERROR', user_message='oops',
        context={obj: 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err_key.to_dict())
  assert 'zqsentinelvalue' not in json.dumps(err_key.to_dict())


@pytest.mark.parametrize('text', [
    'password tremendously Zq9SENTINEL7x',
    'password literally Zq9SENTINEL7x',
    'password honestly Zq9SENTINEL7x',
    'api tremendously key=Zq9SENTINEL7x',
    'apiverykey=Zq9SENTINEL7x',
    'api well-known key=Zq9SENTINEL7x',
    'api wellknown key=Zq9SENTINEL7x',
])
def test_loop17_arch_telemetry_filler_glued_scrubbed(text):
  '''H1/H2/F2 telemetry: filler patterns do not leak.'''
  err = FrameworkError(
        code='X_ERROR', user_message='oops ' + text,
        context={'note': text})
  blob = json.dumps(err.to_dict())
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)


# --- loop18 P1: type name in exception messages (RED) ---

def test_loop18_clean_context_type_name_scrubbed():
  '''P1: _clean_context TypeError scrubs type name.'''
  obj = type('Zq9SENTINEL7x', (), {})()
  with pytest.raises(TypeError) as excinfo:
    FrameworkError(code='X_ERROR', user_message='x', context=obj)
  assert 'Zq9SENTINEL7x' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)


def test_loop18_validate_context_mapping_type_name_scrubbed():
  '''P1: _validate_context_mapping TypeError scrubs name.'''
  obj = type('Zq9SENTINEL7x', (), {})()
  with pytest.raises(TypeError) as excinfo:
    wrap(ValueError('x'), context=obj)
  assert 'Zq9SENTINEL7x' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)
  original = DataValidationError('bad', stage='eda')
  with pytest.raises(TypeError) as excinfo2:
    wrap(original, context=obj)
  assert 'Zq9SENTINEL7x' not in str(excinfo2.value)


def test_loop18_validate_code_owner_name_scrubbed():
  '''P1: _validate_code prefix error scrubs owner name.'''
  import forecast_ml_framework.telemetry.exceptions as exc_mod

  with pytest.raises(ValueError) as excinfo:
    exc_mod._validate_code(
            'TRAIN_OOM', 'INGEST_', 'Zq9SENTINEL7x')
  assert 'Zq9SENTINEL7x' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)
  dyn = type(
        'Zq9SENTINEL7x', (FrameworkError,),
        {'_code_prefix': 'INGEST_'})
  with pytest.raises(ValueError) as excinfo2:
    dyn(code='TRAIN_OOM', user_message='x')
  assert 'Zq9SENTINEL7x' not in str(excinfo2.value)


def test_loop18_factory_cls_name_scrubbed():
  '''P1: _factory prefix error scrubs cls name.'''
  import forecast_ml_framework.telemetry.exceptions as exc_mod

  dyn = type(
        'Zq9SENTINEL7x', (FrameworkError,),
        {'_code_prefix': 'INGEST_'})
  with pytest.raises(ValueError) as excinfo:
    exc_mod._factory(dyn, 'TRAIN_OOM', 'dup', False)
  assert 'Zq9SENTINEL7x' not in str(excinfo.value)
  assert '***REDACTED***' in str(excinfo.value)


def test_loop17_arch_wrap_untyped_preserves_suppress_context():
  '''Untyped wrap preserves __suppress_context__ flag.'''
  old = RuntimeError('boom')
  old.__suppress_context__ = True
  wrapped = wrap(old, stage='eda')
  assert wrapped.__suppress_context__ is True
  old2 = RuntimeError('boom')
  old2.__suppress_context__ = False
  wrapped2 = wrap(old2, stage='eda')
  assert wrapped2.__suppress_context__ is False


# --- loop19 reviewer fixes (RED) ---

def test_loop19_wrap_both_messages_raises_typeerror():
  '''wrap user_message+message raises TypeError, like ctor.'''
  with pytest.raises(TypeError):
    wrap(ValueError('x'), user_message='a', message='b')
  err = DataValidationError('bad', stage='eda')
  with pytest.raises(TypeError):
    wrap(err, user_message='a', message='b')


def test_loop19_positional_arity_uses_scrubbed_subclass_name():
  '''Arity error names the subclass, scrubbed, not hardcoded.'''
  with pytest.raises(TypeError) as excinfo:
    DataIngestionError('INGEST_X', 'eda', 'msg', {}, False, 'x')
  assert 'DataIngestionError' in str(excinfo.value)
  dyn = type('Zq9SENTINEL7x', (FrameworkError,), {})
  with pytest.raises(TypeError) as excinfo2:
    dyn('X_ERROR', 'eda', 'msg', {}, False, 'x')
  assert 'Zq9SENTINEL7x' not in str(excinfo2.value)
  assert '***REDACTED***' in str(excinfo2.value)


def test_loop19_context_mapping_validation_single_helper():
  '''OmegaConf/Mapping TypeError lives in one helper only.'''
  from pathlib import Path

  src = (
      Path(__file__).resolve().parents[2]
      / 'src' / 'forecast_ml_framework' / 'telemetry'
      / 'exceptions.py').read_text()
  assert src.count('context must be a mapping or None') == 2
  assert '_validate_context_mapping(context)' in src


def test_loop19_telemetry_init_docstring_shipped_api():
  '''telemetry __init__ docstring describes shipped API.'''
  import forecast_ml_framework.telemetry as tel

  doc = tel.__doc__ or ''
  assert 'full API lands' not in doc
  assert 'FrameworkError' in doc
  assert 'wrap' in doc


# --- R-C9-2: bytes truncation suffix (RED) ---

def test_r_c9_2_bytes_tuple_key_over_cap_sentinel_gone_via_to_dict():
  '''Over-cap bytes tuple key redacts sentinel sibling.'''
  sentinel = 'HUNTER2ZQ9SENTINEL'
  big = b'a' * 10000 + b'api very key' + b'a' * 10000
  err = FrameworkError(
      code='X_ERROR', user_message='m',
      context={(big, sentinel): 'hello'})
  assert sentinel not in str(err.to_dict())
  assert sentinel not in str(err.context)


def test_r_c9_2_coerce_key_bytes_truncation_suffix_and_boundary():
  '''Over-cap bytes key carries suffix; exact-cap key unchanged.'''
  from forecast_ml_framework.telemetry.exceptions import _coerce_key

  coerced, _secret = _coerce_key(b'a' * 9000, frozenset(), 0)
  assert coerced.endswith(b'...[TRUNCATED]')
  assert len(coerced) == 8192 + len(b'...[TRUNCATED]')
  exact, _secret_exact = _coerce_key(b'a' * 8192, frozenset(), 0)
  assert exact == b'a' * 8192
