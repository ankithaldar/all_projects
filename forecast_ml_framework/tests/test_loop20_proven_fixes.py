#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Loop20 proven fixes P1-P5.'''
import time

SENTINEL = 'Zq9SENTINEL7x'


def _scrub_text(text):
  '''Scrub text.'''
  from forecast_ml_framework.governance.scrub import scrub_text
  return scrub_text(text)


def test_p1_intra_marker_filler_over_128():
  '''P1 intra marker filler over 128.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import FrameworkError
  cases = [
      f'api {'very ' * 129}key {SENTINEL}',
      f'api {'very ' * 200}key {SENTINEL}',
      f'aws {'very ' * 129}access {'very ' * 129}key id {SENTINEL}',
      f'access {'very ' * 150}key {SENTINEL}',
      f'private {'very ' * 150}key {SENTINEL}',
      f'pass {'very ' * 150}word {SENTINEL}',
      f'secret {'very ' * 150}access key {SENTINEL}',
  ]
  for text in cases:
    assert SENTINEL not in _scrub_text(text), text[:60]
    assert '***REDACTED***' in _scrub_text(text), text[:60]
    assert SENTINEL not in repr(scrub_details({'k': text}))
    assert SENTINEL not in repr(audit_log('e', text))
    err = FrameworkError(text, context={'k': text}, stage=text)
    assert SENTINEL not in repr(err.to_dict())
    assert SENTINEL not in str(err)


def test_p2_atomic_single_letter_ly_filler():
  '''P2 atomic single letter ly filler.'''
  for letter in 'abcdef':
    text = f'api {letter * 10}ly key {SENTINEL}'
    out = _scrub_text(text)
    assert SENTINEL not in out, text
    assert '***REDACTED***' in out, text
  # controls starting g-z already worked
  for letter in 'gpqxyz':
    text = f'api {letter * 10}ly key {SENTINEL}'
    assert SENTINEL not in _scrub_text(text), text


def test_p3_ly_stem_over_24():
  '''P3 ly stem over 24.'''
  for stem in (25, 40, 100):
    text = f'api {'q' * stem}ly key {SENTINEL}'
    out = _scrub_text(text)
    assert SENTINEL not in out, (stem, out[:80])
    assert '***REDACTED***' in out, stem
  for stem in (25, 40):
    text = f'api {'x' * stem}ly key {SENTINEL}'
    assert SENTINEL not in _scrub_text(text), stem


def test_p4_dict_key_dos_bounded():
  '''P4 dict key dos bounded.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  # Long ambiguous key must be bounded (was >3s for 50k, 15s for 20x10k)
  start = time.perf_counter()
  out = scrub_details({'\u044f' * 50000: 'v'})
  elapsed = time.perf_counter() - start
  assert elapsed < 1.0, elapsed
  assert out is not None
  start = time.perf_counter()
  out2 = scrub_details({'\u044f' * 20000: 'v'})
  elapsed2 = time.perf_counter() - start
  assert elapsed2 < 1.0, elapsed2
  assert out2 is not None


def test_p4_long_key_secret_still_redacted():
  '''P4 long key secret still redacted.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  # Secret key detection must not be bypassed by long key
  out = scrub_details({'password': SENTINEL})
  assert SENTINEL not in repr(out)
  # Long key that is a secret marker within cap still redacts value
  out2 = scrub_details({f'api {'very ' * 129}key {SENTINEL}': 'v'})
  assert SENTINEL not in repr(out2)


def test_p5_factory_hostile_bool():
  '''P5 factory hostile bool.'''
  from forecast_ml_framework.telemetry.exceptions import (
      DataIngestionError,
  )

  class EvilBool:
    '''Evil Bool test double.'''
    def __bool__(self):
      '''Return the boolean value.'''
      raise RuntimeError('bool boom')

    def __str__(self):
      '''Return the string value.'''
      return 'evilbool'

  err = DataIngestionError.failed(user_message=EvilBool())
  assert err.to_dict()['user_message'] == 'evilbool'
  assert SENTINEL not in repr(err.to_dict())
  # Empty string is supplied (not None) and must pass through ctor
  err2 = DataIngestionError.failed(user_message='')
  assert err2.to_dict()['user_message'] == ''
  # None still uses default
  err3 = DataIngestionError.failed()
  assert err3.to_dict()['user_message'] == 'data ingestion failed'


def test_redos_long_token_still_fast():
  '''Redos long token still fast.'''
  for text in (
      'password ' + 'a' * 5000,
      'api ' + 'a' * 5000,
      'api' + 'a' * 5000,
      'password ' + 'q' * 5000,
  ):
    start = time.perf_counter()
    out = _scrub_text(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (text[:20], elapsed)
    # api prefix alone is not a marker: preserved
    if text.startswith('api'):
      assert out == text
