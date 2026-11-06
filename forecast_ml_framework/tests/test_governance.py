#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.governance (phase 3).

Lineage (model + stable config hash), audit (actor required,
secret redaction, injectable clock), and retention (keep/delete
split by age with injected now). Pure stdlib; runs in every env.
'''

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone

import pytest


def _run_info():
  '''Sample run_info carrying variables, parameters and outputs.'''
  return {
    'run_id': 'run-1',
    'inputs': ['raw'],
    'features': ['f1', 'f2'],
    'model': 'my-model',
    'outputs': ['predictions'],
    'variables': {'target': 'y', 'horizon': 7},
    'parameters': {'lr': 0.1, 'epochs': 10},
  }


def _catalog():
  '''Sample catalog entries keyed by dataset name.'''
  return {
    'raw': {'type': 'kedro_datasets.pandas.ParquetDataset',
            'filepath': 'data/raw.parquet'},
    'predictions': {'type': 'kedro_datasets.pandas.ParquetDataset',
                    'filepath': 'data/predictions.parquet'},
  }


def _expected_hash(variables, parameters):
  '''Canonical sha256 of variables+parameters, per spec.'''
  canonical = json.dumps(
    {'variables': variables, 'parameters': parameters},
    sort_keys=True, separators=(',', ':')).encode('utf-8')
  return hashlib.sha256(canonical).hexdigest()


# --- lineage ---------------------------------------------------------------

def test_lineage_contains_model_and_config_hash():
  from forecast_ml_framework.governance.lineage import build_lineage
  lineage = build_lineage(_run_info(), _catalog())
  assert lineage['model'] == 'my-model'
  assert lineage['inputs'] == ['raw']
  assert lineage['features'] == ['f1', 'f2']
  assert lineage['outputs'] == ['predictions']
  assert lineage['config_hash'] == _expected_hash(
    {'target': 'y', 'horizon': 7}, {'lr': 0.1, 'epochs': 10})


def test_lineage_config_hash_is_stable_and_order_independent():
  from forecast_ml_framework.governance.lineage import build_lineage
  first = build_lineage(_run_info(), _catalog())
  reordered = _run_info()
  reordered['variables'] = {'horizon': 7, 'target': 'y'}
  reordered['parameters'] = {'epochs': 10, 'lr': 0.1}
  second = build_lineage(reordered, _catalog())
  assert first['config_hash'] == second['config_hash']


def test_lineage_config_hash_changes_with_parameters():
  from forecast_ml_framework.governance.lineage import build_lineage
  changed = _run_info()
  changed['parameters'] = {'lr': 0.2, 'epochs': 10}
  assert build_lineage(changed, _catalog())['config_hash'] != (
    build_lineage(_run_info(), _catalog())['config_hash'])


def test_lineage_defaults_without_optional_fields():
  from forecast_ml_framework.governance.lineage import build_lineage
  lineage = build_lineage({}, {})
  assert lineage['model'] is None
  assert lineage['inputs'] == []
  assert lineage['features'] == []
  assert lineage['outputs'] == []
  assert lineage['config_hash'] == _expected_hash({}, {})


# --- audit -----------------------------------------------------------------

def test_audit_log_entry_has_timestamp_actor_and_event():
  from forecast_ml_framework.governance.audit import audit_log
  fixed = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
  entry = audit_log('train', 'alice', {'lr': 0.1},
                    clock=lambda: fixed)
  assert entry['event'] == 'train'
  assert entry['actor'] == 'alice'
  assert entry['details'] == {'lr': 0.1}
  assert entry['timestamp'] == fixed.isoformat()


@pytest.mark.parametrize('actor', ['', '   ', None])
def test_audit_log_requires_actor(actor):
  from forecast_ml_framework.governance.audit import audit_log
  with pytest.raises(ValueError):
    audit_log('train', actor, {})


def test_audit_log_redacts_secret_like_keys():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
    'password': 'hunter2',
    'client_secret': 'shh',
    'access_token': 'tok-123',
    'api_key': 'key-123',
    'nested': {'db_password': 'hunter2', 'ok': 'keep'},
    'items': [{'token': 'tok-999'}, {'safe': 1}],
    'lr': 0.1,
  }
  entry = audit_log('train', 'alice', details)
  scrubbed = entry['details']
  assert scrubbed['password'] == '***REDACTED***'
  assert scrubbed['client_secret'] == '***REDACTED***'
  assert scrubbed['access_token'] == '***REDACTED***'
  assert scrubbed['api_key'] == '***REDACTED***'
  assert scrubbed['nested']['db_password'] == '***REDACTED***'
  assert scrubbed['nested']['ok'] == 'keep'
  assert scrubbed['items'][0]['token'] == '***REDACTED***'
  assert scrubbed['items'][1]['safe'] == 1
  assert scrubbed['lr'] == 0.1
  blob = json.dumps(entry)
  for raw in ('hunter2', 'shh', 'tok-123', 'key-123', 'tok-999'):
    assert raw not in blob


def test_audit_log_does_not_mutate_details():
  from forecast_ml_framework.governance.audit import audit_log
  details = {'password': 'hunter2', 'nested': {'token': 'abc'}}
  audit_log('train', 'alice', details)
  assert details == {'password': 'hunter2', 'nested': {'token': 'abc'}}


def test_audit_log_redacts_camelcase_and_expanded_markers():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      'accessKey': 'AKIA-secret-1',
      'privateKey': 'priv-secret-2',
      'AWSAccessKeyId': 'AKIA-secret-3',
      'pwd': 'pwd-secret',
      'passphrase': 'phrase-secret',
      'bearer': 'bearer-secret',
      'auth': 'auth-secret',
      'cookie': 'cookie-secret',
      'ok': 'keep',
  }
  entry = audit_log('train', 'alice', details)
  scrubbed = entry['details']
  for key in details:
    if key != 'ok':
      assert scrubbed[key] == '***REDACTED***', key
  assert scrubbed['ok'] == 'keep'
  blob = json.dumps(entry)
  for raw in ('AKIA-secret-1', 'priv-secret-2', 'AKIA-secret-3',
              'pwd-secret', 'phrase-secret', 'bearer-secret',
              'auth-secret', 'cookie-secret'):
    assert raw not in blob


def test_audit_log_scrubs_event_and_actor_secret_patterns():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train password=hunter2', 'alice token=tok-123', {})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert 'tok-123' not in blob
  assert '***REDACTED***' in entry['event']
  assert '***REDACTED***' in entry['actor']


def test_audit_log_scrubs_string_details():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', 'password=hunter2')
  assert 'hunter2' not in json.dumps(entry)
  assert '***REDACTED***' in json.dumps(entry)
  nested = audit_log(
      'train', 'alice', {'note': 'api_key=key-123', 'ok': 'hello'})
  assert 'key-123' not in json.dumps(nested)
  assert nested['details']['ok'] == 'hello'
  quoted = audit_log('train', 'alice password="hunter2"', {})
  assert 'hunter2' not in json.dumps(quoted)


# --- retention -------------------------------------------------------------

def _artifacts(now):
  '''Artifacts aged 1, exactly 30 (boundary) and 31 days.'''
  return [
    {'name': 'fresh', 'created_at': now - timedelta(days=1)},
    {'name': 'boundary', 'created_at': now - timedelta(days=30)},
    {'name': 'old', 'created_at': now - timedelta(days=31)},
  ]


def test_retention_splits_keep_and_delete():
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2026, 11, 5, tzinfo=timezone.utc)
  result = check_retention(
    _artifacts(now), {'retention_days': 30}, now=now)
  assert [a['name'] for a in result['keep']] == ['fresh', 'boundary']
  assert [a['name'] for a in result['delete']] == ['old']


def test_retention_accepts_retention_days_keyword():
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2026, 11, 5, tzinfo=timezone.utc)
  result = check_retention(_artifacts(now), retention_days=30, now=now)
  assert len(result['keep']) == 2
  assert len(result['delete']) == 1


def test_retention_age_days_field():
  from forecast_ml_framework.governance.retention import check_retention
  result = check_retention(
    [{'name': 'a', 'age_days': 5}, {'name': 'b', 'age_days': 90}],
    {'retention_days': 30})
  assert [a['name'] for a in result['keep']] == ['a']
  assert [a['name'] for a in result['delete']] == ['b']


def test_retention_without_age_is_kept():
  from forecast_ml_framework.governance.retention import check_retention
  result = check_retention(
    [{'name': 'unknown'}], {'retention_days': 30})
  assert [a['name'] for a in result['keep']] == ['unknown']
  assert result['delete'] == []


def test_audit_scrubs_quoted_json_secrets():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', '{"password": "hunter2", "ok": "keep"}')
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_url_credentials():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'url': 'https://user:secretpass@example.com/db'})
  blob = json.dumps(entry)
  assert 'secretpass' not in blob
  assert 'user' in blob


def test_audit_scrubs_uppercase_pass_assignment():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train PASS=hunter2', 'alice', {})
  assert 'hunter2' not in json.dumps(entry)


def test_audit_redacts_pass_key():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'pass': 'hunter2', 'ok': 1})
  assert entry['details']['pass'] == '***REDACTED***'
  assert entry['details']['ok'] == 1


def test_audit_scrubs_sets_and_bytes():
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details(
      {'tags': {'password=hunter2'}, 'raw': b'password=hunter2'})
  blob = json.dumps(out, default=str)
  assert 'hunter2' not in blob
  assert '***REDACTED***' in blob
  frozen = scrub_details(frozenset({'token=tok-123'}))
  assert 'tok-123' not in json.dumps(list(frozen))


def test_audit_scrubs_url_password_containing_at():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'url': 'postgres://user:p@ssw0rd@host/db'})
  blob = json.dumps(entry)
  assert 'ssw0rd' not in blob
  assert '***REDACTED***' in blob


def test_pyproject_dependencies_cover_used_and_trim_unused():
  import tomllib
  from pathlib import Path
  raw = Path(__file__).resolve().parents[1] / 'pyproject.toml'
  data = tomllib.loads(raw.read_text())
  deps = data['project']['dependencies']
  for needed in ('kedro-datasets', 'scipy', 'pyarrow'):
    assert needed in deps
  for unused in ('polars', 'seaborn', 'dvc', 'pyyaml'):
    assert unused not in deps
  dev = str(data.get('dependency-groups', {}).get('dev', []))
  assert 'pyyaml' in dev
  extras = str(data['project']['optional-dependencies'])
  for unused in ('polars', 'seaborn', 'dvc'):
    assert unused in extras


# --- Coverage: lineage extraction branches ---


def test_lineage_get_from_object_run_info():
  from types import SimpleNamespace
  from forecast_ml_framework.governance.lineage import build_lineage
  run_info = SimpleNamespace(
      inputs=['raw'], outputs=['pred'], features=['f1'],
      model_name='m1', variables={'a': 1}, parameters={'b': 2})
  out = build_lineage(run_info)
  assert out['inputs'] == ['raw']
  assert out['outputs'] == ['pred']
  assert out['features'] == ['f1']
  assert out['model'] == 'm1'
  assert out['config_hash']


def test_lineage_get_alias_fields_and_config_nested():
  from forecast_ml_framework.governance.lineage import build_lineage
  out = build_lineage({
      'input_datasets': ['i1'], 'output_datasets': ['o1'],
      'feature_cols': ['f9'], 'model_id': 'mid',
      'config': {'variables': {'v': 1}, 'parameters': {'p': 2}}})
  assert out['inputs'] == ['i1']
  assert out['outputs'] == ['o1']
  assert out['features'] == ['f9']
  assert out['model'] == 'mid'
  # Explicit kwargs win over run_info sections.
  out2 = build_lineage(
      {'config': 'not-a-dict'}, variables={'x': 1}, parameters={'y': 2})
  assert out2['config_hash']
  assert build_lineage(None)['model'] is None
  assert build_lineage({})['inputs'] == []
  assert build_lineage({})['outputs'] == []
  assert build_lineage({})['features'] == []


def test_lineage_catalog_names_tagged_entries():
  from forecast_ml_framework.governance.lineage import (
    _catalog_names, build_lineage)
  entries = [
      {'role': 'input', 'name': 'raw'},
      {'kind': 'output', 'dataset': 'pred'},
      {'direction': 'INPUT', 'name': 'raw2'},
      {'name': 'untagged'},
      'not-a-dict',
      {'role': 'output', 'name': 'pred2'},
  ]
  assert _catalog_names(entries, 'input') == ['raw', 'raw2']
  assert _catalog_names(entries, 'output') == ['pred', 'pred2']
  assert _catalog_names([], 'input') == []
  assert _catalog_names(None, 'input') == []
  as_dict = {'a': {'role': 'input', 'name': 'd1'}}
  assert _catalog_names(as_dict, 'input') == ['d1']
  out = build_lineage({}, catalog_entries=entries)
  assert out['inputs'] == ['raw', 'raw2']
  assert out['outputs'] == ['pred', 'pred2']


# --- Coverage: retention clock injection / parsing branches ---


def test_retention_resolve_days_variants():
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2024, 1, 31, tzinfo=timezone.utc)
  arts = [{'age_days': 100}, {'age_days': 1}]
  assert check_retention(arts, params=10, now=now)['delete'] == [arts[0]]
  assert check_retention(arts, params=10.0, now=now)['keep'] == [arts[1]]
  for key in ('retention', 'ttl_days', 'max_age_days'):
    out = check_retention(arts, params={key: 10}, now=now)
    assert len(out['delete']) == 1
  out = check_retention(arts, retention_days=10, now=now)
  assert len(out['delete']) == 1
  for bad_params in (None, {}, {'other': 1}, 'ten', object()):
    with pytest.raises(ValueError, match='retention_days'):
      check_retention(arts, params=bad_params)
  with pytest.raises(ValueError, match='non-negative'):
    check_retention(arts, params=-1)
  with pytest.raises(ValueError, match='non-negative'):
    check_retention(arts, retention_days=-0.5)


def test_retention_clock_and_now_injection():
  from forecast_ml_framework.governance.retention import check_retention
  arts = [{'name': 'old', 'created_at': '2024-01-01T00:00:00Z'},
          {'name': 'new', 'created_at': '2024-01-30T00:00:00+00:00'}]
  # Callable clock returning a datetime.
  out = check_retention(
      arts, params=10,
      clock=lambda: datetime(2024, 1, 31, tzinfo=timezone.utc))
  assert out['delete'] == [arts[0]]
  # Clock returning an epoch float.
  epoch = datetime(2024, 1, 31, tzinfo=timezone.utc).timestamp()
  out = check_retention(arts, params=10, clock=lambda: epoch)
  assert out['delete'] == [arts[0]]
  # now as epoch and as ISO string.
  assert check_retention(arts, params=10, now=epoch)['delete'] == [arts[0]]
  out = check_retention(arts, params=10, now='2024-01-31T00:00:00Z')
  assert out['delete'] == [arts[0]]
  # Naive reference vs naive stamp still aligns.
  out = check_retention(
      [{'age_days': 1}], params=10, now=datetime(2024, 1, 31))
  assert len(out['keep']) == 1


def test_retention_age_from_datetime_and_mixed_tz():
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2024, 1, 31, 12, tzinfo=timezone.utc)
  arts = [
      {'created_at': datetime(2024, 1, 1)},  # naive, aligns to UTC
      {'timestamp': 0},  # epoch 1970 -> very old
      {'date': '2024-01-30T00:00:00'},  # naive ISO
      {'name': 'no-date'},  # conservatively kept
      'not-a-dict',  # conservatively kept
  ]
  out = check_retention(arts, params=10, now=now)
  assert arts[0] in out['delete']
  assert arts[1] in out['delete']
  assert arts[2] in out['keep']
  assert arts[3] in out['keep']
  assert arts[4] in out['keep']


def test_retention_dict_input_returns_names():
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2024, 1, 31, tzinfo=timezone.utc)
  artifacts = {
      'old': datetime(2024, 1, 1, tzinfo=timezone.utc),
      'new': '2024-01-30T00:00:00+00:00',
      'epoch': 0,
  }
  out = check_retention(artifacts, params=10, now=now)
  assert out == {'keep': ['new'], 'delete': ['old', 'epoch']}


# --- cycle3 bare-secret audit (RED) ---

def test_audit_scrubs_bare_aws_access_key():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train AKIAIOSFODNN7EXAMPLE', 'alice',
      {'note': 'key is AKIAIOSFODNN7EXAMPLE here'})
  blob = json.dumps(entry)
  assert 'AKIAIOSFODNN7EXAMPLE' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_bare_openai_sk_key():
  from forecast_ml_framework.governance.audit import audit_log
  for raw in ('sk-1234567890abcdefghij', 'sk-proj-1234567890abcdefghij'):
    entry = audit_log('train', 'alice', {'note': raw})
    assert raw not in json.dumps(entry)
    assert '***REDACTED***' in json.dumps(entry)


def test_audit_scrubs_pem_private_key_block():
  from forecast_ml_framework.governance.audit import audit_log
  pem = ('-----BEGIN PRIVATE KEY-----\n'
         'MIIEvQIBADANBgkqhkiG9w0BAQEXXYZC\n'
         '-----END PRIVATE KEY-----')
  entry = audit_log('train', 'alice', {'cert': pem})
  blob = json.dumps(entry)
  assert 'MIIEvQIBADANBgkqhkiG9w0BAQEXXYZC' not in blob
  assert 'PRIVATE KEY' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_space_separated_secret():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password hunter2', 'secret hunter2', 'api_key hunter2'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_bare_secrets_in_sets():
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details(
      {'tags': {'AKIAIOSFODNN7EXAMPLE', 'sk-1234567890abcdefghij'}})
  blob = json.dumps(out, default=str)
  assert 'AKIAIOSFODNN7EXAMPLE' not in blob
  assert 'sk-1234567890abcdefghij' not in blob


def test_audit_scrubs_bare_aws_asia_session_key():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train ASIAIOSFODNN7EXAMPLE', 'alice',
      {'note': 'key is ASIAIOSFODNN7EXAMPLE here'})
  blob = json.dumps(entry)
  assert 'ASIAIOSFODNN7EXAMPLE' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_space_secret_with_copula():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password is hunter2', 'password was hunter2',
               'secret is hunter2', 'token was tok-123'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert 'tok-123' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_api_key_space_and_hyphen_variants():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('api key hunter2', 'api-key hunter2',
               'api key is hunter2', 'api-key was hunter2',
               'access key hunter2', 'private key hunter2'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_space_secret_quoted_and_separators():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password "hunter2"', "password 'hunter2'",
               'secret is: hunter2', 'password is "hunter2"',
               'password - hunter2', 'secret was: hunter2',
               'secret is: "hunter2"', 'password - "hunter2"',
               'password is= hunter2', 'token was: tok-123'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert 'tok-123' not in blob, text
    assert '***REDACTED***' in blob, text


# --- loop4: quoted multi-word / comma / api-separator audit (RED) ---

def test_audit_scrubs_quoted_multiword_secret():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password "hunter2 secret phrase"',
               "password 'hunter2 secret phrase'",
               'password="hunter2 secret phrase"',
               'password is "hunter2 secret phrase"',
               '{"password": "hunter2 secret phrase"}'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    for raw in ('hunter2', 'secret', 'phrase'):
      assert raw not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_quoted_comma_semicolon_secret():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('token: "abc,def"', "token: 'abc,def'",
               'token: "abc;def"', 'token="abc,def;ghi"'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    for raw in ('abc', 'def', 'ghi'):
      assert raw not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_api_key_multi_space_and_tab():
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('api  key   abc123', 'api\tkey abc123',
               'api \t key abc123', 'api   key is abc123',
               'access  key abc123', 'private\tkey abc123'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'abc123' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)


# --- loop5: quote/adverb/spaced-marker audit (RED) ---

@pytest.mark.parametrize('text', [
    'password=`hunter2`',
    'password=\u201chunter2\u201d',
    'password="hunter2',
    "password='hunter2",
    'password="hunter2""',
    "password='hunter2''s'",
    'password="hunter2\\"x"',
    'password definitely hunter2',
    'pass word hunter2',
])
def test_audit_scrubs_loop5_quote_adverb_spaced_pocs(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


@pytest.mark.parametrize('text', [
    'password is definitely hunter2',
    'password: definitely hunter2',
    'password= definitely hunter2',
    'password=`hunter2',
    'password=\u201chunter2',
    'pass word=hunter2',
    'pass word: hunter2',
    'token really is hunter2',
])
def test_audit_scrubs_loop5_filler_and_unterminated_variants(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_leaves_normal_text_unchanged():
  from forecast_ml_framework.governance.audit import audit_log
  text = 'train completed successfully learning_rate 0.1'
  entry = audit_log('train', 'alice', {'note': text})
  assert entry['details']['note'] == text


# --- loop6: filler-before-separator / broad filler audit (RED) ---

@pytest.mark.parametrize('text', [
    'password definitely: hunter2',
    'password definitely= hunter2',
    'password definitely := hunter2',
    'password very really definitely absolutely truly hunter2',
    'password very2 hunter2',
    'password very, hunter2',
    'password well-known hunter2',
    'password 123 hunter2',
    'password a b c d hunter2',
    'password really: hunter2',
    'password -- hunter2',
    'password definitely -- hunter2',
    'password : = hunter2',
    'password := hunter2',
    'password way hunter2',
    'password definitely way hunter2',
    'password very really definitely: hunter2',
    'token well-known = tok12345',
    'secret a b c d e f hunter2',
])
def test_audit_scrubs_loop6_filler_and_separator_pocs(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert 'tok12345' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


# --- loop7: symbolic separators / dict-key secrets audit (RED) ---

@pytest.mark.parametrize('text', [
    'password >> hunter2',
    'password ~ hunter2',
    'password > hunter2',
    'password / hunter2',
    'password | hunter2',
    'password * hunter2',
    'password ! hunter2',
    'password ? hunter2',
    'password @ hunter2',
    'password # hunter2',
    'password $ hunter2',
    'password >>/ hunter2',
    'password >>/~ hunter2',
    'password /=>//\\/|/*/!!!/???/@/#/$ hunter2',
    'password >>/~ /=>//\\/|/*/!!!/???/@/#/$ hunter2',
    'password_hunter2',
    'password>>hunter2',
    'password~hunter2',
    'password>>/~hunter2',
    'token >> tok12345',
    'secret # hunter2',
])
def test_audit_scrubs_loop7_symbolic_separator_pocs(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert 'tok12345' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_symbolic_separator_redaction_is_idempotent():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', {'note': 'password >> hunter2'})
  assert entry['details']['note'] == 'password >> ***REDACTED***'


def test_audit_scrubs_secret_bearing_dict_keys():
  from forecast_ml_framework.governance.audit import audit_log
  pem = ('-----BEGIN PRIVATE KEY-----\n'
         'MIIEvQIBADANBgkqhkiG9w0BAQEXXYZC\n'
         '-----END PRIVATE KEY-----')
  details = {
      'AKIAIOSFODNN7EXAMPLE': 'x',
      'password hunter2': 'y',
      'sk-1234567890abcdefghij': 'z',
      'password=hunter2': 'w',
      pem: 'p',
      'nested': {'AKIAIOSFODNN7EXAMPLE': 'q', 'ok': 'keep'},
      'ok': 'keep',
      'password': 'hunter2',
  }
  entry = audit_log('train', 'alice', details)
  blob = json.dumps(entry)
  for raw in ('AKIAIOSFODNN7EXAMPLE', 'hunter2',
              'sk-1234567890abcdefghij', 'MIIEvQIBADANBgkqhkiG9w0BAQEXXYZC',
              'PRIVATE KEY'):
    assert raw not in blob, raw
  assert '***REDACTED***' in blob
  assert entry['details']['ok'] == 'keep'
  assert entry['details']['nested']['ok'] == 'keep'
  # Plain marker key names (no secret value) are preserved.
  assert 'password' in entry['details']


# --- loop8: non-string dict keys / long filler chains (RED) ---

def test_audit_scrubs_tuple_dict_keys():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      ('AKIAIOSFODNN7EXAMPLE',): 'v1',
      ('password', 'HUNTER2'): 'v2',
      ('token', 'tok-12345', 'extra'): 'v3',
      (('password', 'hunter2'), 'ok-label'): 'v4',
      ('ok', 'keep'): 'keep-value',
      ('nested', 'value'): 42,
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  for raw in ('AKIAIOSFODNN7EXAMPLE', 'HUNTER2', 'hunter2', 'tok-12345'):
    assert raw not in blob, raw
  assert '***REDACTED***' in blob
  # Key structure preserved.
  keys = list(entry['details'].keys())
  assert ('ok', 'keep') in keys
  assert ('nested', 'value') in keys
  assert entry['details'][('ok', 'keep')] == 'keep-value'
  assert entry['details'][('nested', 'value')] == 42
  tuple_keys = [k for k in keys if isinstance(k, tuple)]
  assert len(tuple_keys) == 6
  assert all(isinstance(k, tuple) for k in tuple_keys)


def test_audit_scrubs_frozenset_dict_keys():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      frozenset({'AKIAIOSFODNN7EXAMPLE'}): 'v1',
      frozenset({'password', 'HUNTER2'}): 'v2',
      frozenset({'token=tok-12345'}): 'v3',
      frozenset({'ok', 'keep'}): 'keep-value',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  for raw in ('AKIAIOSFODNN7EXAMPLE', 'HUNTER2', 'tok-12345'):
    assert raw not in blob, raw
  assert '***REDACTED***' in blob
  keys = list(entry['details'].keys())
  assert all(isinstance(k, frozenset) for k in keys)
  assert frozenset({'ok', 'keep'}) in keys
  assert entry['details'][frozenset({'ok', 'keep'})] == 'keep-value'


def test_audit_scrubs_nested_tuple_frozenset_dict_key():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      (frozenset({'AKIAIOSFODNN7EXAMPLE'}), 'label'): 'v1',
      (frozenset({'password', 'hunter2'}),): 'v2',
      'ok': 'keep',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  for raw in ('AKIAIOSFODNN7EXAMPLE', 'hunter2'):
    assert raw not in blob, raw
  assert entry['details']['ok'] == 'keep'


def test_audit_scrubs_long_filler_chain():
  from forecast_ml_framework.governance.audit import audit_log
  texts = [
      'password one two three four five six seven HUNTER2',
      'password one two three four five six seven eight nine ten '
      'eleven twelve thirteen HUNTER2',
      'password very really definitely absolutely truly completely '
      'utterly totally certainly precisely exactly hunter2',
      'token one two three four five six seven eight tok12345',
  ]
  for text in texts:
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'HUNTER2' not in blob, repr(text)
    assert 'hunter2' not in blob, repr(text)
    assert 'tok12345' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)


# --- loop9: composite-key order, filler cap, scrub DoS (RED) ---

def test_audit_scrubs_secret_before_marker_in_tuple_key():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      ('HUNTER2', 'password'): 'v1',
      ('HUNTER2', 'api', 'key'): 'v2',
      ('password', 'HUNTER2', 'extra-label'): 'v3',
      ('ok', 'keep'): 'keep-value',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  for raw in ('HUNTER2', 'extra-label'):
    assert raw not in blob, raw
  assert '***REDACTED***' in blob
  keys = list(entry['details'].keys())
  assert ('ok', 'keep') in keys
  assert entry['details'][('ok', 'keep')] == 'keep-value'
  assert all(isinstance(k, tuple) for k in keys)


def test_audit_scrubs_interleaved_marker_parts_in_tuple_key():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      ('api', 'HUNTER2', 'key'): 'v1',
      ('api', b'HUNTER2', 'key'): 'v2',
      (b'HUNTER2', 'password'): 'v3',
      ('access', 'HUNTER2', 'key'): 'v4',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  assert 'HUNTER2' not in blob
  assert '***REDACTED***' in blob
  # Bytes elements decode to str, so str/bytes variants of
  # the same composite key scrub to the same key.
  assert len(entry['details']) == 3


def test_audit_scrubs_secret_before_marker_in_frozenset_key():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      frozenset({'api', 'HUNTER2', 'key'}): 'v1',
      frozenset({'HUNTER2', 'password'}): 'v2',
      frozenset({'ok', 'keep'}): 'keep-value',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  assert 'HUNTER2' not in blob
  assert '***REDACTED***' in blob
  keys = list(entry['details'].keys())
  assert frozenset({'ok', 'keep'}) in keys


def test_audit_scrubs_nested_composite_secret_before_marker():
  from forecast_ml_framework.governance.audit import audit_log
  details = {
      (('HUNTER2', 'password'), 'label'): 'v1',
      (frozenset({'HUNTER2', 'password'}),): 'v2',
      ('ok', 'keep'): 'keep-value',
  }
  entry = audit_log('train', 'alice', details)
  blob = str(entry)
  assert 'HUNTER2' not in blob
  assert '***REDACTED***' in blob
  assert entry['details'][('ok', 'keep')] == 'keep-value'


def test_audit_scrubs_filler_chain_beyond_32_words():
  from forecast_ml_framework.governance.audit import audit_log
  for count in (33, 50, 100):
    text = 'password ' + ('filler ' * count) + 'HUNTER2'
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'HUNTER2' not in blob, count
    assert '***REDACTED***' in blob, count


def test_audit_scrub_long_token_is_fast():
  from forecast_ml_framework.governance.audit import scrub_details
  for size in (5000, 10000):
    start = time.perf_counter()
    out = scrub_details({'k': 'x' * size})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (size, elapsed)
    assert out == {'k': 'x' * size}


def test_audit_truncates_long_text_and_notes_remainder():
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'k': 'x' * 9850 + ' password hunter2'
                         + ' ' * 200})
  blob = str(out)
  assert 'hunter2' not in blob
  assert 'TRUNCATED' in blob
  assert len(out['k']) <= 10100


def test_audit_truncation_drops_secret_beyond_cutoff():
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'k': 'x' * 10050 + ' password hunter2'})
  blob = str(out)
  assert 'hunter2' not in blob
  assert 'TRUNCATED' in blob


# --- loop10: trailing pair / AWSAccessKeyId / composite values (RED) ---

@pytest.mark.parametrize('text', [
    'password=QZ other=1',
    'password QZ other=1',
    'password=QZ other:1',
    'token=QZ other=1',
])
def test_audit_scrubs_secret_before_trailing_pair(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'QZ' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_trailing_pair_preserves_non_secret_pair():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', {'note': 'password=QZ other=1'})
  assert entry['details']['note'] == 'password=***REDACTED*** other=1'


@pytest.mark.parametrize('text', [
    'AWSAccessKeyId=hunter2',
    'AWSAccessKeyId hunter2',
    'awsaccesskeyid=hunter2',
    'AWSAccessKeyId: hunter2',
    'accessKeyId=hunter2',
    'AWSAccessKeyId=hunter2 other=1',
])
def test_audit_scrubs_aws_access_key_id_free_text(text):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_scrubs_composite_secret_in_container_values():
  from forecast_ml_framework.governance.audit import scrub_details
  details = {
      'tuple_val': ('password', 'HUNTER2'),
      'parts_val': ('api', 'HUNTER2', 'key'),
      'before_val': ('HUNTER2', 'password'),
      'list_val': ['password', 'HUNTER2'],
      'set_val': {'password', 'HUNTER2'},
      'frozen_val': frozenset({'api', 'HUNTER2', 'key'}),
      'nested_val': (('HUNTER2', 'password'), 'label'),
      'ok_val': ('ok', 'keep'),
  }
  out = scrub_details(details)
  blob = str(out)
  assert 'HUNTER2' not in blob
  assert '***REDACTED***' in blob
  assert out['ok_val'] == ('ok', 'keep')
  assert isinstance(out['tuple_val'], tuple)
  assert isinstance(out['list_val'], list)
  assert isinstance(out['set_val'], set)
  assert isinstance(out['frozen_val'], frozenset)
  # Non-marker, non-marker-part elements are redacted, markers kept.
  assert out['tuple_val'] == ('password', '***REDACTED***')
  assert out['parts_val'] == ('api', '***REDACTED***', 'key')


# --- loop11: first value token, trailing prose preserved (RED) ---

@pytest.mark.parametrize('text,expected', [
    ('password HUNTER2 end', 'password ***REDACTED*** end'),
    ('password hunter2 end', 'password ***REDACTED*** end'),
    ('password HUNTER2 the end', 'password ***REDACTED*** the end'),
    ('token tok12345 trailing words here',
     'token ***REDACTED*** trailing words here'),
    ('secret hunter2 done.', 'secret ***REDACTED*** done.'),
    ('password is hunter2 end', 'password is ***REDACTED*** end'),
    ('password definitely hunter2 end',
     'password definitely ***REDACTED*** end'),
    ('password=QZ other=1', 'password=***REDACTED*** other=1'),
    ('password QZ other=1', 'password ***REDACTED*** other=1'),
    ('password "hunter2 phrase" end',
     'password "***REDACTED***" end'),
])
def test_audit_redacts_first_value_preserves_trailing_prose(
        text, expected):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  assert entry['details']['note'] == expected, repr(text)
  for raw in ('HUNTER2', 'hunter2', 'tok12345', 'QZ'):
    assert raw not in entry['details']['note'], repr(text)


# --- loop12: empty/blank quoted spans are separators (RED) ---

@pytest.mark.parametrize('text,expected', [
    ('password "" HUNTER2', 'password "" ***REDACTED***'),
    ('token "" ABC123', 'token "" ***REDACTED***'),
    ('password "" "HUNTER2"', 'password "" "***REDACTED***"'),
    ("password '' HUNTER2", "password '' ***REDACTED***"),
    ('password "   " HUNTER2', 'password "   " ***REDACTED***'),
    ('password `` HUNTER2', 'password `` ***REDACTED***'),
    ('password \u201c\u201d HUNTER2',
     'password \u201c\u201d ***REDACTED***'),
    ('password "" "hunter2 phrase" end',
     'password "" "***REDACTED***" end'),
    ('password "" "" HUNTER2', 'password "" "" ***REDACTED***'),
    ('secret "" hunter2 end', 'secret "" ***REDACTED*** end'),
])
def test_audit_skips_empty_quote_finds_real_secret(text, expected):
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  assert entry['details']['note'] == expected, repr(text)
  for raw in ('HUNTER2', 'hunter2', 'ABC123'):
    assert raw not in entry['details']['note'], repr(text)


def test_audit_empty_quote_alone_is_not_a_value():
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': 'password ""'})
  assert entry['details']['note'] == 'password ""'
