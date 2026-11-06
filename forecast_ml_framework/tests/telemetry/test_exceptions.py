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
  assert err.context['thing'] == 'thing!'
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
