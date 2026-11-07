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
  '''Test lineage contains model and config hash.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  lineage = build_lineage(_run_info(), _catalog())
  assert lineage['model'] == 'my-model'
  assert lineage['inputs'] == ['raw']
  assert lineage['features'] == ['f1', 'f2']
  assert lineage['outputs'] == ['predictions']
  assert lineage['config_hash'] == _expected_hash(
    {'target': 'y', 'horizon': 7}, {'lr': 0.1, 'epochs': 10})


def test_lineage_config_hash_is_stable_and_order_independent():
  '''Test lineage config hash is stable and order independent.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  first = build_lineage(_run_info(), _catalog())
  reordered = _run_info()
  reordered['variables'] = {'horizon': 7, 'target': 'y'}
  reordered['parameters'] = {'epochs': 10, 'lr': 0.1}
  second = build_lineage(reordered, _catalog())
  assert first['config_hash'] == second['config_hash']


def test_lineage_config_hash_changes_with_parameters():
  '''Test lineage config hash changes with parameters.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  changed = _run_info()
  changed['parameters'] = {'lr': 0.2, 'epochs': 10}
  assert build_lineage(changed, _catalog())['config_hash'] != (
    build_lineage(_run_info(), _catalog())['config_hash'])


def test_lineage_defaults_without_optional_fields():
  '''Test lineage defaults without optional fields.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  lineage = build_lineage({}, {})
  assert lineage['model'] is None
  assert lineage['inputs'] == []
  assert lineage['features'] == []
  assert lineage['outputs'] == []
  assert lineage['config_hash'] == _expected_hash({}, {})


# --- audit -----------------------------------------------------------------

def test_audit_log_entry_has_timestamp_actor_and_event():
  '''Test audit log entry has timestamp actor and event.'''
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
  '''Test audit log requires actor.'''
  from forecast_ml_framework.governance.audit import audit_log
  with pytest.raises(ValueError):
    audit_log('train', actor, {})


def test_audit_log_redacts_secret_like_keys():
  '''Test audit log redacts secret like keys.'''
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
  '''Test audit log does not mutate details.'''
  from forecast_ml_framework.governance.audit import audit_log
  details = {'password': 'hunter2', 'nested': {'token': 'abc'}}
  audit_log('train', 'alice', details)
  assert details == {'password': 'hunter2', 'nested': {'token': 'abc'}}


def test_audit_log_redacts_camelcase_and_expanded_markers():
  '''Test audit log redacts camelcase and expanded markers.'''
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
  '''Test audit log scrubs event and actor secret patterns.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train password=hunter2', 'alice token=tok-123', {})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert 'tok-123' not in blob
  assert '***REDACTED***' in entry['event']
  assert '***REDACTED***' in entry['actor']


def test_audit_log_scrubs_string_details():
  '''Test audit log scrubs string details.'''
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
  '''Test retention splits keep and delete.'''
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2026, 11, 5, tzinfo=timezone.utc)
  result = check_retention(
    _artifacts(now), {'retention_days': 30}, now=now)
  assert [a['name'] for a in result['keep']] == ['fresh', 'boundary']
  assert [a['name'] for a in result['delete']] == ['old']


def test_retention_accepts_retention_days_keyword():
  '''Test retention accepts retention days keyword.'''
  from forecast_ml_framework.governance.retention import check_retention
  now = datetime(2026, 11, 5, tzinfo=timezone.utc)
  result = check_retention(_artifacts(now), retention_days=30, now=now)
  assert len(result['keep']) == 2
  assert len(result['delete']) == 1


def test_retention_age_days_field():
  '''Test retention age days field.'''
  from forecast_ml_framework.governance.retention import check_retention
  result = check_retention(
    [{'name': 'a', 'age_days': 5}, {'name': 'b', 'age_days': 90}],
    {'retention_days': 30})
  assert [a['name'] for a in result['keep']] == ['a']
  assert [a['name'] for a in result['delete']] == ['b']


def test_retention_without_age_is_kept():
  '''Test retention without age is kept.'''
  from forecast_ml_framework.governance.retention import check_retention
  result = check_retention(
    [{'name': 'unknown'}], {'retention_days': 30})
  assert [a['name'] for a in result['keep']] == ['unknown']
  assert result['delete'] == []


def test_audit_scrubs_quoted_json_secrets():
  '''Test audit scrubs quoted json secrets.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', '{"password": "hunter2", "ok": "keep"}')
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_url_credentials():
  '''Test audit scrubs url credentials.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'url': 'https://user:secretpass@example.com/db'})
  blob = json.dumps(entry)
  assert 'secretpass' not in blob
  assert 'user' in blob


def test_audit_scrubs_uppercase_pass_assignment():
  '''Test audit scrubs uppercase pass assignment.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train PASS=hunter2', 'alice', {})
  assert 'hunter2' not in json.dumps(entry)


def test_audit_redacts_pass_key():
  '''Test audit redacts pass key.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'pass': 'hunter2', 'ok': 1})
  assert entry['details']['pass'] == '***REDACTED***'
  assert entry['details']['ok'] == 1


def test_audit_scrubs_sets_and_bytes():
  '''Test audit scrubs sets and bytes.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details(
      {'tags': {'password=hunter2'}, 'raw': b'password=hunter2'})
  blob = json.dumps(out, default=str)
  assert 'hunter2' not in blob
  assert '***REDACTED***' in blob
  frozen = scrub_details(frozenset({'token=tok-123'}))
  assert 'tok-123' not in json.dumps(list(frozen))


def test_audit_scrubs_url_password_containing_at():
  '''Test audit scrubs url password containing at.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'url': 'postgres://user:p@ssw0rd@host/db'})
  blob = json.dumps(entry)
  assert 'ssw0rd' not in blob
  assert '***REDACTED***' in blob


def test_pyproject_dependencies_cover_used_and_trim_unused():
  '''Test pyproject dependencies cover used and trim unused.'''
  import re
  import tomllib
  from pathlib import Path
  raw = Path(__file__).resolve().parents[1] / 'pyproject.toml'
  data = tomllib.loads(raw.read_text())
  deps = data['project']['dependencies']
  dep_names = {
    re.split(r'[<>=!~ ;\[]', dep, maxsplit=1)[0]
    for dep in deps
  }
  for needed in ('kedro-datasets', 'scipy', 'pyarrow'):
    assert needed in dep_names
  for unused in ('polars', 'seaborn', 'dvc', 'pyyaml'):
    assert unused not in dep_names
  dev = str(data.get('dependency-groups', {}).get('dev', []))
  assert 'pyyaml' in dev
  extras = str(data['project']['optional-dependencies'])
  for unused in ('polars', 'seaborn', 'dvc'):
    assert unused in extras


# --- Coverage: lineage extraction branches ---


def test_lineage_get_from_object_run_info():
  '''Test lineage get from object run info.'''
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
  '''Test lineage get alias fields and config nested.'''
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
  '''Test lineage catalog names tagged entries.'''
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
  '''Test retention resolve days variants.'''
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
  '''Test retention clock and now injection.'''
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
  '''Test retention age from datetime and mixed tz.'''
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
  '''Test retention dict input returns names.'''
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
  '''Test audit scrubs bare aws access key.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train AKIAIOSFODNN7EXAMPLE', 'alice',
      {'note': 'key is AKIAIOSFODNN7EXAMPLE here'})
  blob = json.dumps(entry)
  assert 'AKIAIOSFODNN7EXAMPLE' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_bare_openai_sk_key():
  '''Test audit scrubs bare openai sk key.'''
  from forecast_ml_framework.governance.audit import audit_log
  for raw in ('sk-1234567890abcdefghij', 'sk-proj-1234567890abcdefghij'):
    entry = audit_log('train', 'alice', {'note': raw})
    assert raw not in json.dumps(entry)
    assert '***REDACTED***' in json.dumps(entry)


def test_audit_scrubs_pem_private_key_block():
  '''Test audit scrubs pem private key block.'''
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
  '''Test audit scrubs space separated secret.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password hunter2', 'secret hunter2', 'api_key hunter2'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_bare_secrets_in_sets():
  '''Test audit scrubs bare secrets in sets.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details(
      {'tags': {'AKIAIOSFODNN7EXAMPLE', 'sk-1234567890abcdefghij'}})
  blob = json.dumps(out, default=str)
  assert 'AKIAIOSFODNN7EXAMPLE' not in blob
  assert 'sk-1234567890abcdefghij' not in blob


def test_audit_scrubs_bare_aws_asia_session_key():
  '''Test audit scrubs bare aws asia session key.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train ASIAIOSFODNN7EXAMPLE', 'alice',
      {'note': 'key is ASIAIOSFODNN7EXAMPLE here'})
  blob = json.dumps(entry)
  assert 'ASIAIOSFODNN7EXAMPLE' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_space_secret_with_copula():
  '''Test audit scrubs space secret with copula.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password is hunter2', 'password was hunter2',
               'secret is hunter2', 'token was tok-123'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert 'tok-123' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_api_key_space_and_hyphen_variants():
  '''Test audit scrubs api key space and hyphen variants.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('api key hunter2', 'api-key hunter2',
               'api key is hunter2', 'api-key was hunter2',
               'access key hunter2', 'private key hunter2'):
    entry = audit_log(f'train {text}', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_space_secret_quoted_and_separators():
  '''Test audit scrubs space secret quoted and separators.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password "hunter2"', 'password \'hunter2\'',
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
  '''Test audit scrubs quoted multiword secret.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password "hunter2 secret phrase"',
               'password \'hunter2 secret phrase\'',
               'password="hunter2 secret phrase"',
               'password is "hunter2 secret phrase"',
               '{"password": "hunter2 secret phrase"}'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    for raw in ('hunter2', 'secret', 'phrase'):
      assert raw not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_quoted_comma_semicolon_secret():
  '''Test audit scrubs quoted comma semicolon secret.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('token: "abc,def"', 'token: \'abc,def\'',
               'token: "abc;def"', 'token="abc,def;ghi"'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    for raw in ('abc', 'def', 'ghi'):
      assert raw not in blob, text
    assert '***REDACTED***' in blob, text


def test_audit_scrubs_api_key_multi_space_and_tab():
  '''Test audit scrubs api key multi space and tab.'''
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
    'password=\'hunter2',
    'password="hunter2""',
    'password=\'hunter2\'\'s\'',
    'password="hunter2\\"x"',
    'password definitely hunter2',
    'pass word hunter2',
])
def test_audit_scrubs_loop5_quote_adverb_spaced_pocs(text):
  '''Test audit scrubs loop5 quote adverb spaced pocs.'''
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
  '''Test audit scrubs loop5 filler and unterminated variants.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_leaves_normal_text_unchanged():
  '''Test audit leaves normal text unchanged.'''
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
  '''Test audit scrubs loop6 filler and separator pocs.'''
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
  '''Test audit scrubs loop7 symbolic separator pocs.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert 'tok12345' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_symbolic_separator_redaction_is_idempotent():
  '''Test audit symbolic separator redaction is idempotent.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', {'note': 'password >> hunter2'})
  assert entry['details']['note'] == 'password >> ***REDACTED***'


def test_audit_scrubs_secret_bearing_dict_keys():
  '''Test audit scrubs secret bearing dict keys.'''
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
  '''Test audit scrubs tuple dict keys.'''
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
  '''Test audit scrubs frozenset dict keys.'''
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
  '''Test audit scrubs nested tuple frozenset dict key.'''
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
  '''Test audit scrubs long filler chain.'''
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
  '''Test audit scrubs secret before marker in tuple key.'''
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
  '''Test audit scrubs interleaved marker parts in tuple key.'''
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
  '''Test audit scrubs secret before marker in frozenset key.'''
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
  '''Test audit scrubs nested composite secret before marker.'''
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
  '''Test audit scrubs filler chain beyond 32 words.'''
  from forecast_ml_framework.governance.audit import audit_log
  for count in (33, 50, 100):
    text = 'password ' + ('filler ' * count) + 'HUNTER2'
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'HUNTER2' not in blob, count
    assert '***REDACTED***' in blob, count


def test_audit_scrub_long_token_is_fast():
  '''Test audit scrub long token is fast.'''
  from forecast_ml_framework.governance.audit import scrub_details
  for size in (5000, 10000):
    start = time.perf_counter()
    out = scrub_details({'k': 'x' * size})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (size, elapsed)
    assert out == {'k': 'x' * size}


def test_audit_truncates_long_text_and_notes_remainder():
  '''Test audit truncates long text and notes remainder.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'k': 'x' * 9850 + ' password hunter2'
                         + ' ' * 200})
  blob = str(out)
  assert 'hunter2' not in blob
  assert 'TRUNCATED' in blob
  assert len(out['k']) <= 10100


def test_audit_truncation_drops_secret_beyond_cutoff():
  '''Test audit truncation drops secret beyond cutoff.'''
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
  '''Test audit scrubs secret before trailing pair.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'QZ' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_trailing_pair_preserves_non_secret_pair():
  '''Test audit trailing pair preserves non secret pair.'''
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
  '''Test audit scrubs aws access key id free text.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_audit_scrubs_composite_secret_in_container_values():
  '''Test audit scrubs composite secret in container values.'''
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
  '''Test audit redacts first value preserves trailing prose.'''
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
    ('password \'\' HUNTER2', 'password \'\' ***REDACTED***'),
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
  '''Test audit skips empty quote finds real secret.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  assert entry['details']['note'] == expected, repr(text)
  for raw in ('HUNTER2', 'hunter2', 'ABC123'):
    assert raw not in entry['details']['note'], repr(text)


def test_audit_empty_quote_alone_is_not_a_value():
  '''Test audit empty quote alone is not a value.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': 'password ""'})
  assert entry['details']['note'] == 'password ""'


# --- high-fix: lineage set ordering determinism (RED) ---

def test_config_hash_set_order_independent():
  '''Test config hash set order independent.'''
  from forecast_ml_framework.governance.lineage import compute_config_hash
  assert compute_config_hash({'s': {'b', 'a', 'c'}}, {}) == (
    compute_config_hash({'s': {'c', 'a', 'b'}}, {}))
  assert compute_config_hash({'s': {'b', 'a'}}, {}) == (
    compute_config_hash({'s': ['a', 'b']}, {}))


def test_config_hash_set_of_ints_sorted_numerically():
  '''Test config hash set of ints sorted numerically.'''
  from forecast_ml_framework.governance.lineage import compute_config_hash
  assert compute_config_hash({'s': {3, 1, 10, 2}}, {}) == (
    compute_config_hash({'s': [1, 2, 3, 10]}, {}))


def test_config_hash_nested_sets_in_lists_and_dicts():
  '''Test config hash nested sets in lists and dicts.'''
  from forecast_ml_framework.governance.lineage import compute_config_hash
  left = {'a': [{'x', 'y'}], 'b': {'k': frozenset((2, 1))}}
  assert compute_config_hash(left, {'p': {5, 4}}) == (
    compute_config_hash({'a': [['x', 'y']], 'b': {'k': [1, 2]}},
                        {'p': [4, 5]}))


# --- telemetry-scrub-leaf: governance.scrub leaf (RED) ---

def test_scrub_leaf_is_stdlib_only_and_audit_reexports_it():
  '''scrub_details lives in governance.scrub (stdlib-only leaf)
  and governance.audit re-exports the same object (compat).'''
  import ast
  import sys
  from pathlib import Path

  src = (
      Path(__file__).resolve().parents[1]
      / 'src' / 'forecast_ml_framework' / 'governance'
      / 'scrub.py')
  assert src.exists()
  tree = ast.parse(src.read_text())
  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
      assert 'forecast_ml_framework' not in (node.module or '')
      assert 'telemetry' not in (node.module or '')
      assert 'governance' not in (node.module or '')
      imported.add(node.module or '')
    elif isinstance(node, ast.Import):
      for alias in node.names:
        assert 'forecast_ml_framework' not in alias.name
        assert 'telemetry' not in alias.name
        assert 'governance' not in alias.name
        imported.add(alias.name)
  assert imported <= set(sys.stdlib_module_names)

  from forecast_ml_framework.governance import audit, scrub

  assert audit.scrub_details is scrub.scrub_details
  assert audit.REDACTED == scrub.REDACTED == '***REDACTED***'
  assert scrub.scrub_details({'password': 'hunter2'}) == {
      'password': '***REDACTED***'}
  # audit_log is unchanged (still scrubs via the leaf).
  entry = audit.audit_log('train', 'alice', {'password': 'hunter2'})
  assert entry['details'] == {'password': '***REDACTED***'}


def test_audit_imports_scrub_leaf():
  '''governance.audit imports scrub_details from governance.scrub.'''
  import ast
  from pathlib import Path

  src = (
      Path(__file__).resolve().parents[1]
      / 'src' / 'forecast_ml_framework' / 'governance'
      / 'audit.py')
  tree = ast.parse(src.read_text())
  modules = [
      node.module or ''
      for node in ast.walk(tree)
      if isinstance(node, ast.ImportFrom)
  ]
  assert 'forecast_ml_framework.governance.scrub' in modules


# --- telemetry-final-fixes: homoglyph scrub (RED) ---

def test_audit_scrubs_fullwidth_homoglyph_marker():
  '''Test audit scrubs fullwidth homoglyph marker.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'note': '\uff50\uff41\uff53\uff53\uff57\uff4f\uff52\uff44=hunter2'})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert '***REDACTED***' in blob


def test_audit_scrubs_zero_width_homoglyph_marker():
  '''Test audit scrubs zero width homoglyph marker.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('pass\u200bword hunter2', 'password\u200b=hunter2',
               'pass\u200cword=hunter2'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)


def test_audit_scrubs_zero_width_in_dict_key():
  '''Test audit scrubs zero width in dict key.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice', {'pass\u200bword': 'hunter2', 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert entry['details']['ok'] == 'keep'


# --- loop2-audit-fixes: nested / glued / Cyrillic (RED) ---

def test_audit_scrubs_outer_marker_nested_secret_tuple_value():
  '''Test audit scrubs outer marker nested secret tuple value.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'k': ('password', (b'HUNTER2',))})
  assert 'HUNTER2' not in str(out)
  assert '***REDACTED***' in str(out)
  out2 = scrub_details({'k': (('HUNTER2',), 'password')})
  assert 'HUNTER2' not in str(out2)


def test_audit_scrubs_outer_marker_nested_secret_tuple_key():
  '''Test audit scrubs outer marker nested secret tuple key.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {
      ('password', (b'HUNTER2',)): 'v1',
      ('ok', 'keep'): 'keep-value'})
  assert 'HUNTER2' not in str(entry)
  assert entry['details'][('ok', 'keep')] == 'keep-value'


def test_audit_scrubs_glued_marker_value():
  '''Test audit scrubs glued marker value.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('password=secretHUNTER2', 'password=fooSecretBar',
               'token=secretHUNTER2'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'HUNTER2' not in blob, repr(text)
    assert 'fooSecretBar' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)


def test_audit_scrubs_cyrillic_homoglyph_marker():
  '''Test audit scrubs cyrillic homoglyph marker.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('p\u0430ssword=hunter2', 's\u0435cret=hunter2',
               't\u043ek\u0435n=hunter2'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)
  entry = audit_log('train', 'alice', {
      'p\u0430ssword': 'hunter2', 'ok': 'keep'})
  assert 'hunter2' not in json.dumps(entry)
  assert entry['details']['ok'] == 'keep'


# --- loop3-scrub-fixes: glued / dict-in-marker-tuple / deepcopy (RED) ---

@pytest.mark.parametrize('text,expected', [
    ('passwordZq9SENTINEL', 'password***REDACTED***'),
    ('tokenZq9SENTINEL', 'token***REDACTED***'),
    ('secretZq9SENTINEL', 'secret***REDACTED***'),
    ('apikeyZq9SENTINEL', 'apikey***REDACTED***'),
    ('passwordZq9SENTINEL end', 'password***REDACTED*** end'),
    ('note passwordZq9SENTINEL', 'note password***REDACTED***'),
])
def test_audit_scrubs_glued_no_separator_marker(text, expected):
  '''Test audit scrubs glued no separator marker.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  assert entry['details']['note'] == expected, repr(text)
  assert 'Zq9SENTINEL' not in json.dumps(entry)


def test_audit_glued_marker_extension_preserves_following_value():
  '''Test audit glued marker extension preserves following value.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': 'accessKeyId=hunter2'})
  assert 'hunter2' not in json.dumps(entry)
  assert '***REDACTED***' in json.dumps(entry)
  # CamelCase type names carrying a marker are preserved,
  # while a following value is still redacted.
  entry2 = audit_log('train', 'alice', {
      'note': 'SecretThing', 'pair': 'SecretThing=hunter2'})
  assert entry2['details']['note'] == 'SecretThing'
  assert 'hunter2' not in json.dumps(entry2)


def test_audit_scrubs_dict_nested_in_marker_tuple_value():
  '''Test audit scrubs dict nested in marker tuple value.'''
  from forecast_ml_framework.governance.audit import scrub_details
  # Loop4: a dict nested in a marker tuple is secret payload,
  # so its plain (non-marker) keys are redacted too, not
  # merely its values; scalar values are preserved.
  out = scrub_details({'k': ('password', {'a': 'Zq9SENTINEL'})})
  assert 'Zq9SENTINEL' not in str(out)
  assert out == {'k': ('password', {'***REDACTED***': '***REDACTED***'})}
  out2 = scrub_details({'k': ['password', {'a': 'Zq9SENTINEL'}]})
  assert out2 == {'k': ['password', {'***REDACTED***': '***REDACTED***'}]}
  out2b = scrub_details({'k': ['password', {'b': 2}]})
  assert out2b == {'k': ['password', {'***REDACTED***': 2}]}
  out3 = scrub_details({'k': ('password', [{'a': 'Zq9SENTINEL'}])})
  assert out3 == {'k': ('password', [{'***REDACTED***': '***REDACTED***'}])}


def test_audit_scrubs_dict_nested_in_marker_tuple_via_audit_log():
  '''Test audit scrubs dict nested in marker tuple via audit log.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {
      'k': ('password', {'a': 'Zq9SENTINEL'})})
  assert 'Zq9SENTINEL' not in json.dumps(entry)
  assert '***REDACTED***' in json.dumps(entry)


def test_scrub_does_not_deepcopy_arbitrary_objects():
  '''Test scrub does not deepcopy arbitrary objects.'''
  from forecast_ml_framework.governance.audit import scrub_details
  calls = []

  class Evil:
    '''Test double Evil.'''
    def __deepcopy__(self, memo):
      '''Return a deep copy.'''
      calls.append('deepcopy')
      return self

    def __reduce__(self):
      '''Return the reduction tuple.'''
      calls.append('reduce')
      return ('Evil', ())

    def __str__(self):
      '''Return the string representation.'''
      calls.append('str')
      return 'Zq9SENTINEL'

    __repr__ = __str__

  out = scrub_details({'k': Evil(), 'items': [Evil()]})
  assert calls == [], calls
  assert out == {'k': {'type': 'Evil'}, 'items': [{'type': 'Evil'}]}
  assert 'Zq9SENTINEL' not in str(out)


def test_scrub_file_handle_returns_type_only():
  '''Test scrub file handle returns type only.'''
  import tempfile

  from forecast_ml_framework.governance.audit import scrub_details
  with tempfile.NamedTemporaryFile(mode='w+', delete=False) as handle:
    name = type(handle).__name__
    out = scrub_details({'f': handle, 'n': 1, 'x': 1.5, 'b': True,
                         'z': None})
    assert out['f'] == {'type': name}
    assert out['n'] == 1 and out['x'] == 1.5
    assert out['b'] is True and out['z'] is None


# --- loop4-scrub-fixes: inherited dict keys / hostile keys / sets (RED) ---

def test_loop4_dict_nested_in_marker_tuple_does_not_leak_key():
  '''Test loop4 dict nested in marker tuple does not leak key.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'k': ('password', {'Zq': 'x'})})
  assert 'Zq' not in str(out)
  assert '***REDACTED***' in str(out)
  out2 = scrub_details({'k': ('password', {('Zq', 'x'): 'y'})})
  assert 'Zq' not in str(out2)


def test_loop4_hostile_key_repr_raises_no_crash_no_leak():
  '''Test loop4 hostile key repr raises no crash no leak.'''
  from forecast_ml_framework.governance.audit import scrub_details

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
      return 4242

  out = scrub_details({BadKey(): 'v', 'ok': 'keep'})
  assert 'Zq9SENTINEL' not in str(out)
  assert '<unrepresentable>' in str(out)
  assert out['ok'] == 'keep'


def test_loop4_hostile_tuple_key_repr_raises_no_crash():
  '''Test loop4 hostile tuple key repr raises no crash.'''
  from forecast_ml_framework.governance.audit import scrub_details

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
      return 4243

  out = scrub_details({(BadKey(), 'x'): 'v', ('ok', 'keep'): 1})
  assert 'Zq9SENTINEL' not in str(out)
  assert '<unrepresentable>' in str(out)
  assert out[('ok', 'keep')] == 1


def test_loop4_hostile_key_repr_secret_is_scrubbed():
  '''Test loop4 hostile key repr secret is scrubbed.'''
  from forecast_ml_framework.governance.audit import scrub_details

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
      return 4244

  out = scrub_details({EvilKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(out)
  assert '***REDACTED***' in str(out)


def test_loop4_unknown_object_in_set_stays_hashable():
  '''Test loop4 unknown object in set stays hashable.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class Obj:
    '''Test double Obj.'''
    pass

  out = scrub_details({'s': {Obj()}, 'f': frozenset({Obj()})})
  assert isinstance(out['s'], set)
  assert isinstance(out['f'], frozenset)
  for elem in out['s'] | out['f']:
    hash(elem)
    assert isinstance(elem, str)
  assert 'Obj' in str(out)


def test_loop4_hostile_object_in_set_no_crash():
  '''Test loop4 hostile object in set no crash.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class BadObj:
    '''Test double BadObj.'''
    def __repr__(self):
      '''Return the representation.'''
      raise RuntimeError('repr boom')

    def __str__(self):
      '''Return the string representation.'''
      return 'password=Zq9SENTINEL'

    def __hash__(self):
      '''Return the hash value.'''
      return 4245

  out = scrub_details({'s': {BadObj()}})
  assert isinstance(out['s'], set)
  assert 'Zq9SENTINEL' not in str(out)
  for elem in out['s']:
    hash(elem)


def test_loop4_cyrillic_homoglyph_still_blocked():
  '''Test loop4 cyrillic homoglyph still blocked.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'note': 'p\u0430ssword=hunter2',
                       'p\u0430ssword': 'hunter2'})
  assert 'hunter2' not in str(out)
  assert '***REDACTED***' in str(out)


# --- loop5-scrub-fixes: homoglyph / unknown keys / hostile (RED) ---

@pytest.mark.parametrize('text', [
    'p\u00e1ssword=hunter2',
    'p\u0251ssword=hunter2',
    'pa\u0301ssword=hunter2',
    'p\u0301assword=hunter2',
    'p\u00e0ssword=hunter2',
    'p\u00e2ssword=hunter2',
    'passwo\u044fd=hunter2',
    'p\u044fssword=hunter2',
    'P\u00c1SSWORD=hunter2',
    's\u00e9cret=hunter2',
    'passwo\u042fd=hunter2',
])
def test_loop5_homoglyph_evasion_scrubbed(text):
  '''Test loop5 homoglyph evasion scrubbed.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


@pytest.mark.parametrize('key', [
    'p\u00e1ssword',
    'p\u0251ssword',
    'pa\u0301ssword',
    'passwo\u044fd',
    'p\u044fssword',
])
def test_loop5_homoglyph_dict_key_scrubbed(key):
  '''Test loop5 homoglyph dict key scrubbed.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({key: 'hunter2', 'ok': 'keep'})
  assert 'hunter2' not in str(out), repr(key)
  assert '***REDACTED***' in str(out), repr(key)
  assert out['ok'] == 'keep'


def test_loop5_unknown_key_bare_secret_repr_not_preserved():
  '''Test loop5 unknown key bare secret repr not preserved.'''
  from forecast_ml_framework.governance.audit import scrub_details

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
      return 4246

  out = scrub_details({EvilKey(): 'v', 'ok': 'keep'})
  assert 'Zq9SENTINEL' not in str(out)
  assert 'EvilKey' in str(out)
  assert out['ok'] == 'keep'
  for k in out:
    hash(k)


def test_loop5_unknown_key_bare_secret_in_tuple_and_frozenset():
  '''Test loop5 unknown key bare secret in tuple and frozenset.'''
  from forecast_ml_framework.governance.audit import scrub_details

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
      return 4247

  out = scrub_details({(EvilKey(), 'x'): 'v', ('ok', 'keep'): 1})
  assert 'Zq9SENTINEL' not in str(out)
  assert out[('ok', 'keep')] == 1
  out2 = scrub_details({frozenset({EvilKey()}): 'v',
                        frozenset({'ok'}): 'keep'})
  assert 'Zq9SENTINEL' not in str(out2)
  assert out2[frozenset({'ok'})] == 'keep'


def test_loop5_hostile_tuple_container_no_crash_scrub():
  '''Test loop5 hostile tuple container no crash scrub.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class EvilTuple(tuple):
    '''Test double EvilTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  out = scrub_details({'k': EvilTuple((1, 2)), 'ok': 'keep'})
  assert 'iter boom' not in str(out)
  assert out['ok'] == 'keep'
  json.dumps(out, default=str)


def test_loop5_hostile_dict_container_no_crash_scrub():
  '''Test loop5 hostile dict container no crash scrub.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class EvilDict(dict):
    '''Test double EvilDict.'''
    def items(self):
      '''Items.'''
      raise RuntimeError('items boom')

    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  out = scrub_details({'k': EvilDict({'a': 1}), 'ok': 'keep'})
  assert 'boom' not in str(out)
  assert out['ok'] == 'keep'
  out2 = scrub_details(EvilDict({'a': 1}))
  assert 'boom' not in str(out2)
  json.dumps(out2, default=str)


def test_loop5_hostile_tuple_key_no_crash_scrub():
  '''Test loop5 hostile tuple key no crash scrub.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class EvilTuple(tuple):
    '''Test double EvilTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      raise RuntimeError('iter boom')

  out = scrub_details({EvilTuple(('a',)): 'v', 'ok': 'keep'})
  assert 'boom' not in str(out)
  assert out['ok'] == 'keep'


# --- loop6-scrub-fixes: Cf format-char bypass (RED) ---

@pytest.mark.parametrize('cf', [
    '\u200e', '\u200f',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
    '\u2066', '\u2067', '\u2068', '\u2069',
    '\u2061', '\u2062', '\u2063', '\u2064',
])
def test_loop6_normalize_unicode_drops_cf(cf):
  '''Test loop6 normalize unicode drops cf.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  assert _normalize_unicode('a' + cf + 'b') == 'ab'


@pytest.mark.parametrize('cf', [
    '\u200e', '\u200f',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
    '\u2066', '\u2067', '\u2068', '\u2069',
    '\u2061', '\u2062', '\u2063', '\u2064',
])
def test_loop6_cf_marker_bypass_scrubbed(cf):
  '''Test loop6 cf marker bypass scrubbed.'''
  from forecast_ml_framework.governance.audit import audit_log
  for text in ('p' + cf + 'assword hunter2',
               'pas' + cf + 'sword=hunter2'):
    entry = audit_log('train', 'alice', {'note': text})
    blob = json.dumps(entry)
    assert 'hunter2' not in blob, repr(text)
    assert '***REDACTED***' in blob, repr(text)


def test_loop6_cf_marker_in_dict_key_scrubbed():
  '''Test loop6 cf marker in dict key scrubbed.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details(
      {'p\u200eassword': 'hunter2', 'ok': 'keep'})
  assert 'hunter2' not in str(out)
  assert '***REDACTED***' in str(out)
  assert out['ok'] == 'keep'


# --- telemetry-final-fixes loop6 gates (RED) ---

@pytest.mark.parametrize('repr_text', [
    '\u00e1Zq9SENTINEL',
    '\uff41Zq9SENTINEL',
    'a\u0301Zq9SENTINEL',
    'a\u200bZq9SENTINEL',
    '\u00e1Zq9SENTINEL extra',
])
def test_final_unknown_key_normalization_bare_repr_not_preserved(
        repr_text):
  '''Test final unknown key normalization bare repr not preserved.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return repr_text

    __str__ = __repr__

    def __hash__(self):
      '''Return the hash value.'''
      return 4260

  out = scrub_details({EvilKey(): 'v', 'ok': 'keep'})
  assert 'Zq9SENTINEL' not in str(out), repr(repr_text)
  assert 'aZq9SENTINEL' not in str(out), repr(repr_text)
  assert 'EvilKey' in str(out), repr(repr_text)
  assert out['ok'] == 'keep'


def test_final_unknown_key_marker_repr_placeholder_value_redacted():
  '''Test final unknown key marker repr placeholder value redacted.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class EvilKey:
    '''Test double EvilKey.'''
    def __repr__(self):
      '''Return the representation.'''
      return 'p\u00e1ssword=Zq9SENTINEL'

    __str__ = __repr__

    def __hash__(self):
      '''Return the hash value.'''
      return 4261

  out = scrub_details({EvilKey(): 'v'})
  assert 'Zq9SENTINEL' not in str(out)
  assert 'EvilKey' in str(out)
  assert '***REDACTED***' in str(out)


@pytest.mark.parametrize('text', [
    'secret_key=hunter2',
    'secret.key=hunter2',
    'secret-key hunter2',
    'secret/key=hunter2',
    'password_hash=hunter2',
    'password.hash hunter2',
    'password-hash=hunter2',
    'password_hash hunter2',
    'token_value=hunter2',
    'token.value hunter2',
    'token-value=hunter2',
    'aws_secret_key=hunter2',
    'aws.secret.key=hunter2',
    'aws_secret_key hunter2',
    'api.key=hunter2',
    'api.key hunter2',
    'access.key=hunter2',
    'access.key hunter2',
    'pass.word=hunter2',
    'pass.word hunter2',
    'private.key=hunter2',
    'secret_hash hunter2',
    'secret_value=hunter2',
])
def test_final_compound_and_dotted_markers_scrubbed(text):
  '''Test final compound and dotted markers scrubbed.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


@pytest.mark.parametrize('text', [
    'Bearer is tok12345',
    'Bearer was tok12345',
    'Bearer very tok12345',
    'Bearer is very tok12345',
    'Bearer really is tok12345 end',
    'Bearer definitely tok12345',
    'Bearer is: tok12345',
])
def test_final_bearer_filler_scrubbed(text):
  '''Test final bearer filler scrubbed.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'tok12345' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


def test_final_infinite_tuple_container_capped():
  '''Test final infinite tuple container capped.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class InfiniteTuple(tuple):
    '''Test double InfiniteTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      count = 0
      while True:
        yield count
        count += 1

  out = scrub_details({'k': InfiniteTuple((1, 2))})
  assert len(out['k']) <= 1001
  assert out['k'][:3] == (0, 1, 2)


def test_final_infinite_tuple_key_capped():
  '''Test final infinite tuple key capped.'''
  from forecast_ml_framework.governance.audit import scrub_details

  class InfiniteTuple(tuple):
    '''Test double InfiniteTuple.'''
    def __iter__(self):
      '''Return an iterator.'''
      count = 0
      while True:
        yield count
        count += 1

  out = scrub_details({InfiniteTuple((1,)): 'v', 'ok': 'keep'})
  assert out['ok'] == 'keep'
  for key in out:
    hash(key)


# --- loop7-nfkc-split-fix: NFKC separator splits marker (RED) ---

@pytest.mark.parametrize('text', [
    'toke\u0149=Zq9SENTINEL7x',
    'crede\u0149tial=Zq9SENTINEL7x',
    'p' + '\u1e9a' + 'ssword=Zq9SENTINEL7x',
    'credenti' + '\u1e9a' + 'l=Zq9SENTINEL7x',
    '\u1e9a' + 'pi_key=Zq9SENTINEL7x',
    'cre\u249fential=Zq9SENTINEL7x',
    'p\u249cssword=Zq9SENTINEL7x',
    '\u249dearer=Zq9SENTINEL7x',
    'toke\u24a9=Zq9SENTINEL7x',
    'cre\U0001f113ential=Zq9SENTINEL7x',
    'credentia\u0140=Zq9SENTINEL7x',
    'credentia\u013f=Zq9SENTINEL7x',
    'toke\u02bcn=Zq9SENTINEL7x',
    'pa\u02bessword=Zq9SENTINEL7x',
    'toke\u02c2n=Zq9SENTINEL7x',
    'toke\u02d2n=Zq9SENTINEL7x',
])
def test_loop7_nfkc_split_marker_scrubbed_in_note(text):
  '''Test loop7 nfkc split marker scrubbed in note.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


@pytest.mark.parametrize('text', [
    'toke\u0149=Zq9SENTINEL7x',
    'crede\u0149tial=Zq9SENTINEL7x',
    'p' + '\u1e9a' + 'ssword Zq9SENTINEL7x',
    'cre\u249fential Zq9SENTINEL7x',
])
def test_loop7_nfkc_split_marker_scrubbed_via_scrub_details(text):
  '''Test loop7 nfkc split marker scrubbed via scrub details.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert '***REDACTED***' in str(out), repr(text)
  assert scrub_details(text) != text, repr(text)
  assert 'Zq9SENTINEL7x' not in str(scrub_details(text)), repr(text)


@pytest.mark.parametrize('text', [
    'toke\u0149=Zq9SENTINEL7x',
    'crede\u0149tial=Zq9SENTINEL7x',
    'p' + '\u1e9a' + 'ssword=Zq9SENTINEL7x',
])
def test_loop7_nfkc_split_marker_scrubbed_in_audit_event(text):
  '''Test loop7 nfkc split marker scrubbed in audit event.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(text, 'alice', {'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details'] == {'ok': 'keep'}


@pytest.mark.parametrize('raw,expected', [
    ('toke\u0149', 'token'),
    ('crede\u0149tial', 'credential'),
    ('p' + '\u1e9a' + 'ssword', 'password'),
    ('credenti' + '\u1e9a' + 'l', 'credential'),
    ('cre\u249fential', 'credential'),
    ('p\u249cssword', 'password'),
    ('toke\u02bcn', 'token'),
    ('toke\u02c2n', 'token'),
    ('credentia\u0140', 'credential'),
])
def test_loop7_normalize_unicode_joins_split_marker(raw, expected):
  '''Test loop7 normalize unicode joins split marker.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  assert _normalize_unicode(raw) == expected, repr(raw)


def test_loop7_normalize_preserves_spaced_parentheses():
  '''Test loop7 normalize preserves spaced parentheses.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  assert _normalize_unicode('option (a) is ok') == (
      'option (a) is ok')
  assert _normalize_unicode('hello (world) test') == (
      'hello (world) test')


# --- loop8-proven-fixes: NFKC space-mark split (RED) ---

_LOOP8_SPACE_CPS = [
  0x00A8, 0x00AF, 0x00B4, 0x00B8, 0x02D8, 0x02D9, 0x02DA,
  0x02DB, 0x02DC, 0x02DD, 0x037A, 0x0384, 0x0385, 0x1FBD,
  0x1FBF, 0x1FC0, 0x1FC1, 0x1FCD, 0x1FCE, 0x1FCF, 0x1FDD,
  0x1FDE, 0x1FDF, 0x1FED, 0x1FEE, 0x1FFD, 0x1FFE, 0x2017,
  0x203E, 0x309B, 0x309C, 0xFC5E, 0xFC5F, 0xFC60, 0xFC61,
  0xFC62, 0xFC63, 0xFE49, 0xFE4A, 0xFE4B, 0xFE4C, 0xFE70,
  0xFE72, 0xFE74, 0xFE76, 0xFE78, 0xFE7A, 0xFE7C, 0xFE7E,
  0xFFE3,
]


def test_loop8_nfkc_space_mark_joins_marker():
  '''Space+Mn chars join markers, spaced prose kept.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  markers = [
    ('p', 'assword', 'password'),
    ('t', 'oken', 'token'),
    ('c', 'redential', 'credential'),
    ('s', 'ecret', 'secret'),
    ('a', 'pi_key', 'api_key'),
  ]
  for cp in _LOOP8_SPACE_CPS:
    ch = chr(cp)
    for head, tail, word in markers:
      raw = head + ch + tail
      assert _normalize_unicode(raw) == word, (hex(cp), repr(raw))
  assert _normalize_unicode('hello world test') == (
      'hello world test')
  assert _normalize_unicode('option (a) is ok') == (
      'option (a) is ok')
  for cp in _LOOP8_SPACE_CPS:
    spaced = 'hello ' + chr(cp) + ' world'
    assert 'helloworld' not in _normalize_unicode(spaced), hex(cp)


def test_loop8_nfkc_space_mark_scrubbed():
  '''Space+Mn split markers redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  markers = [
    ('p', 'assword'), ('t', 'oken'), ('c', 'redential'),
    ('s', 'ecret'), ('a', 'pi_key'),
  ]
  for cp in _LOOP8_SPACE_CPS:
    ch = chr(cp)
    for head, tail in markers:
      text = head + ch + tail + '=Zq9SENTINEL7x'
      out = scrub_details({'note': text})
      assert 'Zq9SENTINEL7x' not in str(out), (hex(cp), repr(text))
      assert '***REDACTED***' in str(out), (hex(cp), repr(text))
      entry = audit_log('train', 'alice', {'note': text})
      assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
      key_out = scrub_details({head + ch + tail: 'Zq9SENTINEL7x'})
      assert 'Zq9SENTINEL7x' not in str(key_out), repr(text)


# --- loop8-proven-fixes: cyclic scrub (RED) ---

def test_loop8_scrub_cyclic_dict_placeholder():
  '''Cyclic dict yields <cyclic>, never RecursionError.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  cyc = {}
  cyc['self'] = cyc
  out = scrub_details(cyc)
  assert out == {'self': '<cyclic>'}
  entry = audit_log('train', 'alice', cyc)
  assert entry['details'] == {'self': '<cyclic>'}


def test_loop8_scrub_cyclic_list_placeholder():
  '''Cyclic list yields <cyclic> element, never crashes.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  items = [1]
  items.append(items)
  assert scrub_details({'k': items}) == {'k': [1, '<cyclic>']}
  assert scrub_details(items) == [1, '<cyclic>']


def test_loop8_scrub_tuple_list_cycle_placeholder():
  '''Tuple<->list cycle yields <cyclic>, never crashes.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  items = []
  tup = (items,)
  items.append(tup)
  out = scrub_details({'k': tup})
  assert '<cyclic>' in str(out), repr(out)
  assert 'Zq9SENTINEL7x' not in str(out)


def test_loop8_scrub_deep_nesting_capped():
  '''Deep nesting caps at <max-depth>, sentinel stays hidden.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  deep = {}
  current = deep
  for _ in range(50):
    nxt = {}
    current['next'] = nxt
    current = nxt
  current['password'] = 'Zq9SENTINEL7x'
  out = scrub_details(deep)
  assert '<max-depth>' in str(out), repr(out)[:200]
  assert 'Zq9SENTINEL7x' not in str(out)
  deep_list = []
  current_list = deep_list
  for _ in range(300):
    nxt_list = []
    current_list.append(nxt_list)
    current_list = nxt_list
  out_list = scrub_details({'k': deep_list})
  assert '<max-depth>' in str(out_list)


# --- loop9-proven-fixes: digraph ss/th + hostile bytes (RED) ---

@pytest.mark.parametrize('raw,expected', [
    ('pa\u00dfword', 'password'),
    ('pa\u00df', 'pass'),
    ('pa\u00dfwd', 'passwd'),
    ('pa\u00dfphrase', 'passphrase'),
    ('acce\u00df', 'access'),
    ('au\u00fe', 'auth'),
    ('au\u00feorization', 'authorization'),
])
def test_loop9_normalize_unicode_joins_digraphs(raw, expected):
  '''Digraph ß->ss and þ->th join markers.'''
  from forecast_ml_framework.governance.scrub import (
    _CONFUSABLE_TABLE_ALT,
    _normalize_unicode,
  )
  assert _normalize_unicode(raw) == expected, repr(raw)
  alt_expected = (
      expected.replace('th', 'p') if '\u00fe' in raw else expected)
  assert _normalize_unicode(raw, _CONFUSABLE_TABLE_ALT) == (
      alt_expected), repr(raw)


@pytest.mark.parametrize('raw,expected', [
    ('pa\u1e9eword', 'paSSword'),
    ('au\u00deorization', 'auTHorization'),
    ('PA\u00dfWORD', 'PAssWORD'),
    ('AU\u00deORIZATION', 'AUTHORIZATION'),
])
def test_loop9_normalize_unicode_joins_capital_digraphs(raw, expected):
  '''Capital digraphs ẞ->SS and Þ->TH join markers.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  assert _normalize_unicode(raw) == expected, repr(raw)


@pytest.mark.parametrize('text', [
    'pa\u00dfword=Zq9SENTINEL7x',
    'pa\u1e9eword=Zq9SENTINEL7x',
    'pa\u00df=Zq9SENTINEL7x',
    'pa\u00dfwd=Zq9SENTINEL7x',
    'pa\u00dfphrase=Zq9SENTINEL7x',
    'acce\u00df_key=Zq9SENTINEL7x',
    'pa\u00dfword Zq9SENTINEL7x',
    'au\u00fe=Zq9SENTINEL7x',
    'au\u00feorization=Zq9SENTINEL7x',
    'au\u00deorization=Zq9SENTINEL7x',
    'au\u00feorization Zq9SENTINEL7x',
])
def test_loop9_digraph_markers_scrubbed_in_note(text):
  '''Digraph ss/th markers redact sentinel in free text.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert '***REDACTED***' in str(out), repr(text)
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)


@pytest.mark.parametrize('key', [
    'pa\u00dfword',
    'pa\u1e9eword',
    'pa\u00dfwd',
    'pa\u00dfphrase',
    'acce\u00df_key',
    'au\u00fe',
    'au\u00feorization',
    'au\u00deorization',
])
def test_loop9_digraph_markers_scrubbed_in_dict_key(key):
  '''Digraph ss/th markers redact value under dict key.'''
  from forecast_ml_framework.governance.audit import scrub_details
  out = scrub_details({key: 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(key)
  assert '***REDACTED***' in str(out), repr(key)
  assert out['ok'] == 'keep'


class _Loop9EvilBytes(bytes):
  '''Bytes whose decode raises, carrying sentinel.'''

  def __new__(cls, *args, **kwargs):
    '''Create a new instance.'''
    return super().__new__(cls, b'hello')

  def decode(self, *args, **kwargs):
    '''Decode.'''
    raise RuntimeError('Zq9SENTINEL7x')


class _Loop9EvilBytearray(bytearray):
  '''Bytearray whose decode raises, carrying sentinel.'''

  def decode(self, *args, **kwargs):
    '''Decode.'''
    raise RuntimeError('Zq9SENTINEL7x')


class _Loop9EvilBytesPassword(bytes):
  '''Bytes decoding-hostile but underlying plain password.'''

  def __new__(cls, *args, **kwargs):
    '''Create a new instance.'''
    return super().__new__(cls, b'password')

  def decode(self, *args, **kwargs):
    '''Decode.'''
    raise RuntimeError('Zq9SENTINEL7x')


def test_loop9_hostile_bytes_bare_no_crash_no_leak():
  '''Bare hostile bytes become REDACTED, sentinel hidden.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  assert scrub_details(_Loop9EvilBytes()) == '***REDACTED***'
  assert scrub_details(
      _Loop9EvilBytearray(b'hello')) == '***REDACTED***'
  assert 'Zq9SENTINEL7x' not in str(
      scrub_details(_Loop9EvilBytes()))


def test_loop9_hostile_bytes_dict_and_list_vals_no_crash():
  '''Hostile bytes in dict/list/tuple vals are REDACTED.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({'k': _Loop9EvilBytes(), 'ok': 'keep'})
  assert out == {'k': '***REDACTED***', 'ok': 'keep'}
  out_list = scrub_details(
      {'k': [_Loop9EvilBytes(), _Loop9EvilBytearray(b'x')]})
  assert out_list == {'k': ['***REDACTED***', '***REDACTED***']}
  out_tuple = scrub_details({'k': (_Loop9EvilBytes(),)})
  assert out_tuple == {'k': ('***REDACTED***',)}
  assert 'Zq9SENTINEL7x' not in str(out_list)


def test_loop9_hostile_bytes_key_no_crash_no_leak():
  '''Hostile bytes keys use plain bytes, never hostile decode.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({_Loop9EvilBytes(): 'keep', 'ok': 1})
  assert 'Zq9SENTINEL7x' not in str(out)
  assert out['ok'] == 1
  for key in out:
    hash(key)
  secret_out = scrub_details(
      {_Loop9EvilBytesPassword(): 'hunter2', 'ok': 'keep'})
  assert 'hunter2' not in str(secret_out)
  assert 'Zq9SENTINEL7x' not in str(secret_out)
  assert secret_out['ok'] == 'keep'


def test_loop9_hostile_bytes_via_audit_log_no_crash():
  '''Hostile bytes via audit_log are REDACTED, no crash.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'k': _Loop9EvilBytes(), 'items': [_Loop9EvilBytes()]})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob
  assert '***REDACTED***' in blob
  assert entry['details']['k'] == '***REDACTED***'


# --- loop10-reviewer-fix: non-str decode return (RED) ---


class _Loop10DictDecodeBytes(bytes):
  '''Bytes whose decode returns a dict, carrying sentinel.'''

  def __new__(cls, *args, **kwargs):
    '''Create a new instance.'''
    return super().__new__(cls, b'hello')

  def decode(self, *args, **kwargs):
    '''Decode.'''
    return {'password': 'Zq9SENTINEL7x'}


class _Loop10ListDecodeBytes(bytes):
  '''Bytes whose decode returns a list, carrying sentinel.'''

  def __new__(cls, *args, **kwargs):
    '''Create a new instance.'''
    return super().__new__(cls, b'hello')

  def decode(self, *args, **kwargs):
    '''Decode.'''
    return ['Zq9SENTINEL7x']


class _Loop10DictDecodeBytearray(bytearray):
  '''Bytearray whose decode returns a dict, carrying sentinel.'''

  def decode(self, *args, **kwargs):
    '''Decode.'''
    return {'password': 'Zq9SENTINEL7x'}


def test_loop10_non_str_decode_bytes_bare_redacted():
  '''Bare non-str decode returns become REDACTED, no leak.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  assert scrub_details(_Loop10DictDecodeBytes()) == '***REDACTED***'
  assert scrub_details(_Loop10ListDecodeBytes()) == '***REDACTED***'
  assert scrub_details(
      _Loop10DictDecodeBytearray(b'hello')) == '***REDACTED***'
  assert 'Zq9SENTINEL7x' not in str(
      scrub_details(_Loop10DictDecodeBytes()))
  assert 'Zq9SENTINEL7x' not in str(
      scrub_details(_Loop10ListDecodeBytes()))


def test_loop10_non_str_decode_bytes_dict_and_list_vals_redacted():
  '''Non-str decode returns in dict/list vals are REDACTED.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details(
      {'k': _Loop10DictDecodeBytes(), 'ok': 'keep'})
  assert out == {'k': '***REDACTED***', 'ok': 'keep'}
  out_list = scrub_details({'k': [
      _Loop10DictDecodeBytes(), _Loop10ListDecodeBytes()]})
  assert out_list == {'k': ['***REDACTED***', '***REDACTED***']}
  assert 'Zq9SENTINEL7x' not in str(out_list)


def test_loop10_non_str_decode_bytes_via_audit_log_redacted():
  '''Non-str decode returns via audit_log are REDACTED.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
      'train', 'alice',
      {'k': _Loop10DictDecodeBytes(),
       'items': [_Loop10ListDecodeBytes()]})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob
  assert '***REDACTED***' in blob
  assert entry['details']['k'] == '***REDACTED***'
  assert entry['details']['items'] == ['***REDACTED***']


# --- loop10-proven-fixes P1: negative circled (RED) ---

def test_loop10_p1_negative_circled_joins_marker():
  '''Negative circled A-Z fold to Latin letters.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for offset in range(26):
    ch = chr(0x1F150 + offset)
    letter = chr(ord('A') + offset)
    assert _normalize_unicode(ch) == letter, hex(0x1F150 + offset)
  assert _normalize_unicode('\U0001F15Fassword') == 'Password'
  assert _normalize_unicode('p\U0001F150ssword') == 'pAssword'


def test_loop10_p1_negative_circled_scrubbed():
  '''Negative circled markers redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  texts = [
    '\U0001F15Fassword=Zq9SENTINEL7x',
    'p\U0001F150ssword=Zq9SENTINEL7x',
    '\U0001F15Fassword Zq9SENTINEL7x',
  ]
  for text in texts:
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    assert '***REDACTED***' in str(out), repr(text)
    entry = audit_log('train', 'alice', {'note': text})
    assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  key_out = scrub_details(
    {'\U0001F15Fassword': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop10-proven-fixes P2: negative squared (RED) ---

def test_loop10_p2_negative_squared_joins_marker():
  '''Negative squared A-Z, crossed P and abbrs fold.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for offset in range(26):
    ch = chr(0x1F170 + offset)
    letter = chr(ord('A') + offset)
    assert _normalize_unicode(ch) == letter, hex(0x1F170 + offset)
  assert _normalize_unicode('\U0001F18A') == 'P'
  assert _normalize_unicode('\U0001F18B') == 'IC'
  assert _normalize_unicode('\U0001F18C') == 'PA'
  assert _normalize_unicode('\U0001F18D') == 'SA'
  assert _normalize_unicode('\U0001F18E') == 'AB'
  assert _normalize_unicode('\U0001F18F') == 'WC'


def test_loop10_p2_negative_squared_scrubbed():
  '''Negative squared markers redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  texts = [
    '\U0001F17Fassword=Zq9SENTINEL7x',
    'p\U0001F170ssword=Zq9SENTINEL7x',
    '\U0001F18Cssword=Zq9SENTINEL7x',
    '\U0001F18Aassword=Zq9SENTINEL7x',
  ]
  for text in texts:
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    assert '***REDACTED***' in str(out), repr(text)
    entry = audit_log('train', 'alice', {'note': text})
    assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  key_out = scrub_details(
    {'\U0001F17Fassword': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop10-proven-fixes P3: squared words/small (RED) ---

_LOOP10_SQUARED_WORDS = {
  '\U0001F191': 'CL',
  '\U0001F192': 'COOL',
  '\U0001F193': 'FREE',
  '\U0001F194': 'ID',
  '\U0001F195': 'NEW',
  '\U0001F196': 'NG',
  '\U0001F197': 'OK',
  '\U0001F198': 'SOS',
  '\U0001F199': 'UP',
  '\U0001F19A': 'VS',
  '\U0001F1A5': 'd',
  '\U0001F1A6': 'HC',
  '\U0001F1A7': 'HDR',
  '\U0001F1A9': 'LOSSLESS',
  '\U0001F1AA': 'SHV',
  '\U0001F1AB': 'UHD',
  '\U0001F1AC': 'VOD',
}


def test_loop10_p3_squared_words_join_marker():
  '''Squared ASCII words and small D fold to letters.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for ch, word in _LOOP10_SQUARED_WORDS.items():
    assert _normalize_unicode(ch) == word, repr(ch)
  assert _normalize_unicode('t\U0001F197en') == 'tOKen'
  assert _normalize_unicode('pw\U0001F1A5') == 'pwd'


def test_loop10_p3_squared_words_scrubbed():
  '''Squared OK/D fragments redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  texts = [
    't\U0001F197en=Zq9SENTINEL7x',
    'pw\U0001F1A5=Zq9SENTINEL7x',
    't\U0001F197en Zq9SENTINEL7x',
  ]
  for text in texts:
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    assert '***REDACTED***' in str(out), repr(text)
    entry = audit_log('train', 'alice', {'note': text})
    assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  key_out = scrub_details(
    {'pw\U0001F1A5': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop10-proven-fixes P4: Roman late/early/reversed (RED) ---

_LOOP10_ROMAN = {
  '\u2180': 'CD',
  '\u2181': 'V',
  '\u2182': 'X',
  '\u2183': 'C',
  '\u2184': 'c',
  '\u2185': 'VI',
  '\u2186': 'L',
  '\u2187': 'L',
  '\u2188': 'C',
}


def test_loop10_p4_roman_joins_marker():
  '''Non-NFKC Roman forms fold to their letters.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for ch, word in _LOOP10_ROMAN.items():
    assert _normalize_unicode(ch) == word, repr(ch)
  assert _normalize_unicode('pri\u2185ate_key') == 'priVIate_key'
  assert _normalize_unicode('credentia\u2186') == 'credentiaL'
  assert _normalize_unicode('se\u2184ret') == 'secret'


def test_loop10_p4_roman_scrubbed():
  '''Roman VI/L/C fragments redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  texts = [
    'pri\u2185ate_key=Zq9SENTINEL7x',
    'credentia\u2186=Zq9SENTINEL7x',
    'se\u2184ret=Zq9SENTINEL7x',
    'se\u2183ret=Zq9SENTINEL7x',
  ]
  for text in texts:
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    assert '***REDACTED***' in str(out), repr(text)
    entry = audit_log('train', 'alice', {'note': text})
    assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  key_out = scrub_details(
    {'credentia\u2186': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop10-proven-fixes P5: UTS39 confusables (RED) ---

def test_loop10_p5_confusable_representatives_join_marker():
  '''UTS39 letter confusables fold to ASCII letters.'''
  from forecast_ml_framework.governance.scrub import (
    _CONFUSABLE_TABLE,
    _CONFUSABLE_TABLE_ALT,
    _normalize_unicode,
  )
  cases = [
    ('\u2c63assword', 'password'),
    ('\u2ca3assword', 'password'),
    ('t\u2c9fken', 'token'),
    ('\u13aa' + 'uth', 'Auth'),
    ('se\u1d04ret', 'secret'),
    ('t\u1d0fken', 'token'),
    ('t\u0585ken', 'token'),
  ]
  for raw, expected in cases:
    assert _normalize_unicode(raw) == expected, repr(raw)
    assert _normalize_unicode(raw, _CONFUSABLE_TABLE_ALT) == (
      expected), repr(raw)
  assert str.maketrans({'a': 'b'})
  assert _CONFUSABLE_TABLE is not None
  assert _CONFUSABLE_TABLE_ALT is not None


def test_loop10_p5_confusable_scrubbed():
  '''UTS39 confusable markers redact sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  texts = [
    '\u2c63assword=Zq9SENTINEL7x',
    't\u2c9fken=Zq9SENTINEL7x',
    '\u13aa' + 'uth=Zq9SENTINEL7x',
    'se\u1d04ret=Zq9SENTINEL7x',
    't\u1d0fken=Zq9SENTINEL7x',
    't\u0585ken=Zq9SENTINEL7x',
  ]
  for text in texts:
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    assert '***REDACTED***' in str(out), repr(text)
    entry = audit_log('train', 'alice', {'note': text})
    assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  key_out = scrub_details(
    {'\u2c63assword': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop10-proven-fixes P6: audit hostile (RED) ---

class _Loop10EvilStr:
  '''Object whose __str__ raises, carrying sentinel.'''

  def __str__(self):
    '''Return the string representation.'''
    raise RuntimeError('Zq9SENTINEL7x')

  __repr__ = __str__


class _Loop10EvilStrReturn:
  '''Object whose __str__ returns a secret pattern.'''

  def __str__(self):
    '''Return the string representation.'''
    return 'password=Zq9SENTINEL7x'

  __repr__ = __str__


class _Loop10EvilDT(datetime):
  '''Datetime whose isoformat raises, carrying sentinel.'''

  def isoformat(self, sep='T', timespec='auto'):
    '''Isoformat.'''
    raise RuntimeError('Zq9SENTINEL7x')


def test_loop10_p6_audit_evil_str_actor_event_no_crash():
  '''Hostile __str__ actor/event are type-safe, no leak.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', _Loop10EvilStr(), {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert isinstance(entry['actor'], str)
  assert entry['details'] == {'ok': 'keep'}
  entry2 = audit_log(_Loop10EvilStr(), 'alice', {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry2)
  assert isinstance(entry2['event'], str)
  entry3 = audit_log('train', _Loop10EvilStrReturn(), {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry3)
  assert '***REDACTED***' in json.dumps(entry3)
  entry4 = audit_log(_Loop10EvilStrReturn(), 'alice', {})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry4)
  assert '***REDACTED***' in json.dumps(entry4)


def test_loop10_p6_audit_bytes_event_actor_scrubbed():
  '''Bytes event/actor with sentinel are scrubbed, JSON-safe.'''
  from forecast_ml_framework.governance.audit import audit_log
  raw = b'password=Zq9SENTINEL7x'
  entry = audit_log(raw, 'alice', {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert '***REDACTED***' in json.dumps(entry)
  assert isinstance(entry['event'], str)
  entry2 = audit_log('train', raw, {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry2)
  assert '***REDACTED***' in json.dumps(entry2)
  assert isinstance(entry2['actor'], str)
  entry3 = audit_log('train', _Loop9EvilBytes(), {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry3)
  assert isinstance(entry3['actor'], str)


def test_loop10_p6_audit_hostile_timestamp_no_crash():
  '''Hostile now/clock timestamps are type-safe, no leak.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
    'train', 'alice', {'ok': 'keep'}, now=_Loop10EvilDT(2024, 1, 1))
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert isinstance(entry['timestamp'], str)
  entry2 = audit_log(
    'train', 'alice', {'ok': 'keep'},
    clock=lambda: _Loop10EvilDT(2024, 1, 1))
  assert 'Zq9SENTINEL7x' not in json.dumps(entry2)
  assert isinstance(entry2['timestamp'], str)
  entry3 = audit_log(
    'train', 'alice', {'ok': 'keep'}, now=_Loop10EvilStr())
  assert 'Zq9SENTINEL7x' not in json.dumps(entry3)
  assert isinstance(entry3['timestamp'], str)


def test_loop10_p6_audit_bytes_and_str_timestamp_scrubbed():
  '''Bytes/str now with sentinel are scrubbed, JSON-safe.'''
  from forecast_ml_framework.governance.audit import audit_log
  raw = b'password=Zq9SENTINEL7x'
  entry = audit_log('train', 'alice', {'ok': 'keep'}, now=raw)
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert '***REDACTED***' in json.dumps(entry)
  assert isinstance(entry['timestamp'], str)
  entry2 = audit_log(
    'train', 'alice', {'ok': 'keep'},
    now='password=Zq9SENTINEL7x')
  assert 'Zq9SENTINEL7x' not in json.dumps(entry2)
  assert '***REDACTED***' in json.dumps(entry2)
  entry3 = audit_log(
    'train', 'alice', {'ok': 'keep'},
    now=_Loop9EvilBytes())
  assert 'Zq9SENTINEL7x' not in json.dumps(entry3)
  assert isinstance(entry3['timestamp'], str)


class _Loop10EvilDTNonStr(datetime):
  '''Datetime whose isoformat returns a dict, with sentinel.'''

  def isoformat(self, sep='T', timespec='auto'):
    '''Isoformat.'''
    return {'note': 'password=Zq9SENTINEL7x'}


def test_loop10_p6_audit_clock_raises_falls_back():
  '''Hostile clock callable falls back, log append works.'''
  from forecast_ml_framework.governance.audit import audit_log

  def _bad_clock():
    '''Bad clock.'''
    raise RuntimeError('Zq9SENTINEL7x')

  log = []
  entry = audit_log(
    'train', 'alice', {'ok': 'keep'}, clock=_bad_clock, log=log)
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert isinstance(entry['timestamp'], str)
  assert log == [entry]
  entry2 = audit_log(
    'train', 'alice', {'ok': 'keep'},
    now=_Loop10EvilDTNonStr(2024, 1, 1))
  assert 'Zq9SENTINEL7x' not in json.dumps(entry2)
  assert isinstance(entry2['timestamp'], str)


def test_loop10_p6_audit_non_str_decode_actor_redacted():
  '''Non-str decode bytes actor becomes REDACTED, no leak.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log(
    'train', _Loop10DictDecodeBytes(), {'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry)
  assert entry['actor'] == '***REDACTED***'
  assert isinstance(entry['actor'], str)


# --- loop11 reviewer: mixed primary+alt (RED) ---

def test_loop11_mixed_primary_alt_roman_scrub_text():
  '''Primary Roman + alt Roman in one string both redact.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(
    'password=hunter2 pri\u2185ate_key=Zq9SENTINEL')
  assert 'hunter2' not in out
  assert 'Zq9SENTINEL' not in out
  assert '***REDACTED***' in out
  out2 = _scrub_text(
    'pri\u2185ate_key=Zq9SENTINEL password=hunter2')
  assert 'hunter2' not in out2
  assert 'Zq9SENTINEL' not in out2
  assert '***REDACTED***' in out2


def test_loop11_mixed_primary_alt_ya_scrub_text():
  '''Primary ya + alt ya in one string both redact.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(
    'passwo\u044fd=Zq9SENTINEL p\u044fssword=hunter2')
  assert 'hunter2' not in out
  assert 'Zq9SENTINEL' not in out
  assert '***REDACTED***' in out
  out2 = _scrub_text(
    'p\u044fssword=hunter2 passwo\u044fd=Zq9SENTINEL')
  assert 'hunter2' not in out2
  assert 'Zq9SENTINEL' not in out2
  assert '***REDACTED***' in out2


def test_loop11_mixed_primary_alt_details_audit_telemetry():
  '''Mixed skeletons redact via details, audit and error.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import (
    FrameworkError,
  )
  text = (
    'password=hunter2 pri\u2185ate_key=Zq9SENTINEL '
    'passwo\u044fd=Zq9SECOND p\u044fssword=Zq9THIRD')
  out = scrub_details({'note': text, 'ok': 'keep'})
  blob = str(out)
  assert 'hunter2' not in blob
  assert 'Zq9SENTINEL' not in blob
  assert 'Zq9SECOND' not in blob
  assert 'Zq9THIRD' not in blob
  assert out['ok'] == 'keep'
  entry = audit_log('train', 'alice', {'note': text})
  blob = json.dumps(entry)
  assert 'hunter2' not in blob
  assert 'Zq9SENTINEL' not in blob
  assert 'Zq9SECOND' not in blob
  assert 'Zq9THIRD' not in blob
  err = FrameworkError(
    'oops ' + text, stage='train',
    context={'note': text})
  assert 'hunter2' not in str(err)
  assert 'Zq9SENTINEL' not in str(err)
  assert 'Zq9SECOND' not in json.dumps(err.to_dict())
  assert 'Zq9THIRD' not in json.dumps(err.to_dict())


def test_loop11_alt_marker_part_composite_redacted():
  '''Alt-only marker parts join to a marker, secret redacts.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details(
    {'items': ('\u044fpi', 'Zq9SENTINEL', 'key'), 'ok': 'keep'})
  assert 'Zq9SENTINEL' not in str(out)
  assert out['ok'] == 'keep'
  out2 = scrub_details(
    {('\u044fpi', 'key'): 'Zq9SENTINEL', 'ok': 'keep'})
  assert 'Zq9SENTINEL' not in str(out2)
  assert out2['ok'] == 'keep'


def test_loop11_normalized_markers_deduped():
  '''Normalized markers carry no duplicates.'''
  from forecast_ml_framework.governance.scrub import (
    _NORMALIZED_MARKERS,
  )
  assert len(_NORMALIZED_MARKERS) == len(set(_NORMALIZED_MARKERS))


def test_loop11_dead_confusables_removed():
  '''NFKC-shadowed confusables are gone, alt split keeps l.'''
  from forecast_ml_framework.governance.scrub import (
    _CONFUSABLE_TABLE_ALT,
    _CONFUSABLES,
    _normalize_unicode,
  )
  assert '\u0149' not in _CONFUSABLES
  assert '\u2110' not in _CONFUSABLES
  assert '\u03f2' not in _CONFUSABLES
  assert _normalize_unicode('\u2110', _CONFUSABLE_TABLE_ALT) == 'l'
  assert _normalize_unicode('\u2110') == 'I'


# --- loop11 security F1: regional indicators (RED) ---

def test_loop11_regional_indicators_join_marker():
  '''Regional indicator letters fold to A-Z.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for offset in range(26):
    ch = chr(0x1F1E6 + offset)
    letter = chr(ord('A') + offset)
    assert _normalize_unicode(ch) == letter, repr(ch)
  assert _normalize_unicode('\U0001F1F5assword') == 'Password'
  assert _normalize_unicode('\U0001F1F9oken') == 'Token'
  assert _normalize_unicode('hello world test') == (
    'hello world test')


@pytest.mark.parametrize('text', [
  '\U0001F1F5assword=Zq9SENTINEL7x',
  '\U0001F1F9oken=Zq9SENTINEL7x',
  '\U0001F1E6pi_key=Zq9SENTINEL7x',
  'password \U0001F1F5assword Zq9SENTINEL7x',
])
def test_loop11_regional_indicators_scrubbed(text):
  '''Regional marker text redacts sentinel everywhere.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import (
    FrameworkError,
  )
  out = scrub_details({'note': text, 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert '***REDACTED***' in str(out), repr(text)
  assert out['ok'] == 'keep'
  entry = audit_log('train', 'alice', {'note': text})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  err = FrameworkError(
    'oops ' + text, stage='train',
    context={'note': text})
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(
    err.to_dict()), repr(text)
  key_out = scrub_details(
    {'\U0001F1F5assword': 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(key_out)
  assert key_out['ok'] == 'keep'


# --- loop11 adversarial G2: dict in tuple (RED) ---

@pytest.mark.parametrize('value', [
  ('api', {'key': 'Zq9SENTINEL7x'}),
  ('access', {'key': 'Zq9SENTINEL7x'}),
  ('private', {'key': 'Zq9SENTINEL7x'}),
  ({'api': 'Zq9SENTINEL7x'}, 'key'),
  ('api', [{'key': 'Zq9SENTINEL7x'}]),
  ('api', {'outer': {'key': 'Zq9SENTINEL7x'}}),
  ('api', {'key': ('nested', 'Zq9SENTINEL7x')}),
])
def test_loop11_g2_dict_in_tuple_split_marker(value):
  '''Dict in tuple under split marker redacts sentinel.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import (
    FrameworkError,
  )
  out = scrub_details({'note': value, 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(value)
  assert out['ok'] == 'keep'
  entry = audit_log('train', 'alice', {'note': value})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(value)
  err = FrameworkError(
    'oops', stage='train', context={'note': value})
  assert 'Zq9SENTINEL7x' not in json.dumps(
    err.to_dict()), repr(value)
  assert 'Zq9SENTINEL7x' not in str(err), repr(value)


# --- loop11 adversarial G3: alt I and thorn-p (RED) ---

@pytest.mark.parametrize('text', [
  'credentiaI=Zq9SENTINEL7x',
  'credentia\u0399=Zq9SENTINEL7x',
  'credentia\u0406=Zq9SENTINEL7x',
  '\u00feassword=Zq9SENTINEL7x',
  '\u00deassword=Zq9SENTINEL7x',
])
def test_loop11_g3_alt_i_and_thorn_p_scrubbed(text):
  '''Alt I->l and thorn->p markers redact sentinel.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import (
    FrameworkError,
  )
  out = scrub_details({'note': text, 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert '***REDACTED***' in str(out), repr(text)
  assert out['ok'] == 'keep'
  entry = audit_log('train', 'alice', {'note': text})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  err = FrameworkError(
    'oops ' + text, stage='train',
    context={'note': text})
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(
    err.to_dict()), repr(text)


def test_loop11_g3_alt_normalize_i_and_thorn():
  '''Alt skeleton folds I and thorn to l and p.'''
  from forecast_ml_framework.governance.scrub import (
    _CONFUSABLE_TABLE_ALT,
    _normalize_unicode,
  )
  assert _normalize_unicode(
    'credentiaI', _CONFUSABLE_TABLE_ALT) == 'credential'
  assert _normalize_unicode(
    'credentia\u0399', _CONFUSABLE_TABLE_ALT) == 'credential'
  assert _normalize_unicode(
    'credentia\u0406', _CONFUSABLE_TABLE_ALT) == 'credential'
  assert _normalize_unicode(
    '\u00feassword', _CONFUSABLE_TABLE_ALT) == 'password'
  assert _normalize_unicode('\u00feassword') == 'thassword'
  assert _normalize_unicode('au\u00fe', ) == 'auth'


# --- loop11 adversarial G4: Cn codepoints (RED) ---

_LOOP11_CN = [
  (0x1DF5A, 'a'), (0x1DF6A, 'A'), (0x16EB6, 'b'),
  (0x1DF81, 'E'), (0x11DE1, 'l'), (0x11DDA, 'l'),
  (0x16EAA, 'l'), (0x11DE0, 'O'), (0x1CEFC, 'V'),
  (0x1DF7D, 'w'),
]


def test_loop11_g4_cn_codepoints_join_marker():
  '''Cn codepoints fold to their Unicode 18 letters.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  for cp, letter in _LOOP11_CN:
    assert _normalize_unicode(chr(cp)) == letter, hex(cp)


def test_loop11_g4_cn_codepoints_scrubbed():
  '''Cn letter inside a marker redacts sentinel.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  markers = [
    'password', 'token', 'secret', 'credential', 'api_key',
    'authorization', 'bearer', 'private_key', 'access_key',
    'cookie', 'auth',
  ]
  for cp, letter in _LOOP11_CN:
    ch = chr(cp)
    for marker in markers:
      if letter.lower() in marker:
        idx = marker.index(letter.lower())
        word = marker[:idx] + ch + marker[idx + 1:]
        text = f'{word}=Zq9SENTINEL7x'
        out = scrub_details({'note': text})
        assert 'Zq9SENTINEL7x' not in str(out), repr(text)
        entry = audit_log('train', 'alice', {'note': text})
        assert 'Zq9SENTINEL7x' not in json.dumps(
            entry), repr(text)


# --- loop12 arch: Bearer base64, >5 parts, hostile dict (RED) ---

@pytest.mark.parametrize('text', [
  'Bearer, AAA+Zq9SENTINEL7x',
  'Bearer; AAA+Zq9SENTINEL7x',
  'Bearer,AAA+BBB/CCC=',
  'Bearer, AAA+BBB/CCC=',
])
def test_loop12_bearer_base64_remainder_scrubbed(text):
  '''Bearer base64 token after comma/semicolon is fully read.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert 'BBB' not in out, repr(text)
  assert 'CCC' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop12_composite_gt5_awssecretaccesskey():
  '''Unordered >5 parts naming awssecretaccesskey are bearing.'''
  from forecast_ml_framework.governance.scrub import (
    _composite_marker_bearing,
    scrub_details,
  )
  key = frozenset({
    'aws', 'secret', 'access', 'key', 'hash', 'value',
    'Zq9SENTINEL7x',
  })
  out = scrub_details({key: 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out)
  items = [
    'aws', 'hash', 'access', 'value', 'word',
    {('secret',): 'x'},
  ]
  assert _composite_marker_bearing(items, False) is True


def test_loop12_hostile_dict_items_non_pair_no_crash():
  '''Hostile dict items() with non-pairs falls back to type.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  class HostileDict(dict):
    '''Test double HostileDict.'''
    def items(self):
      '''Items.'''
      return [('k', 'v', 'extra'), ('a', 'b')]

  assert scrub_details(HostileDict()) == {
    'type': 'HostileDict'}
  assert scrub_details({'outer': HostileDict()}) == {
    'outer': {'type': 'HostileDict'}}
  out = scrub_details({'k': [HostileDict()]})
  assert 'Zq9SENTINEL7x' not in str(out)


# --- loop12 adversarial F2: UTS39 non-L single confusables (RED) ---

_LOOP12_F2 = [
  (0x30, 'o'), (0x31, 'l'), (0x7C, 'l'), (0xA1, 'i'),
  (0x5C0, 'l'), (0x661, 'l'), (0x665, 'o'), (0x667, 'v'),
  (0x6F1, 'l'), (0x6F5, 'o'), (0x6F7, 'v'), (0x7C0, 'o'),
  (0x964, 'l'), (0x966, 'o'), (0x9E6, 'o'), (0xA66, 'o'),
  (0xAE6, 'o'), (0xB66, 'o'), (0xBE6, 'o'), (0xC02, 'o'),
  (0xC66, 'o'), (0xC82, 'o'), (0xCE6, 'o'), (0xD02, 'o'),
  (0xD66, 'o'), (0xD82, 'o'), (0xE50, 'o'), (0xED0, 'o'),
  (0x1040, 'o'), (0x104A, 'l'), (0x17E0, 'o'), (0x1A80, 'o'),
  (0x1A90, 'o'), (0x1BEA, 'o'), (0x212E, 'e'), (0x21BF, 'l'),
  (0x2223, 'l'), (0x2228, 'v'), (0x222A, 'u'), (0x22A4, 't'),
  (0x22C1, 'v'), (0x22C3, 'u'), (0x22FF, 'e'), (0x2373, 'i'),
  (0x2374, 'p'), (0x237A, 'a'), (0x23FD, 'l'), (0x2502, 'l'),
  (0x2503, 'l'), (0x27D9, 't'), (0x29E2, 'w'), (0x3007, 'o'),
  (0xA830, 'l'), (0xA8CE, 'l'), (0xAA5D, 'l'), (0xFE31, 'l'),
  (0xFF10, 'o'), (0xFF11, 'l'), (0xFF5C, 'l'), (0xFFE8, 'l'),
  (0x1018B, 'd'), (0x10320, 'l'), (0x10CFA, 'l'),
  (0x11047, 'l'), (0x110C0, 'l'), (0x11141, 'l'),
  (0x111C5, 'l'), (0x11302, 'o'), (0x113D4, 'l'),
  (0x1144B, 'l'), (0x114D0, 'o'), (0x115C5, 'l'),
  (0x11641, 'l'), (0x118E0, 'o'), (0x118E6, 'w'),
  (0x118E9, 'c'), (0x118EF, 'w'), (0x118F2, 'c'),
  (0x11C41, 'l'), (0x1CCF0, 'o'), (0x1CCF1, 'l'),
  (0x1D100, 'l'), (0x1D134, 'c'), (0x1D207, 'b'),
  (0x1D20C, 'w'), (0x1D20D, 'v'), (0x1D216, 'r'),
  (0x1D22A, 'l'), (0x1D373, 't'), (0x1D377, 'l'),
  (0x1D7CE, 'o'), (0x1D7CF, 'l'), (0x1D7D8, 'o'),
  (0x1D7D9, 'l'), (0x1D7E2, 'o'), (0x1D7E3, 'l'),
  (0x1D7EC, 'o'), (0x1D7ED, 'l'), (0x1D7F6, 'o'),
  (0x1D7F7, 'l'), (0x1E140, 'o'), (0x1E141, 'l'),
  (0x1E145, 'v'), (0x1E2F0, 'o'), (0x1E8C7, 'l'),
  (0x1ED01, 'l'), (0x1F74C, 'c'), (0x1F768, 't'),
  (0x1FBF0, 'o'), (0x1FBF1, 'l'),
]


def test_loop12_f2_uts39_non_l_scrubbed():
  '''UTS39 non-L glyphs spelling a marker redact sentinel.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  markers = [
    'password', 'token', 'secret', 'credential', 'api_key',
    'authorization', 'bearer', 'private_key', 'access_key',
    'cookie', 'auth',
  ]
  for cp, letter in _LOOP12_F2:
    ch = chr(cp)
    for marker in markers:
      if letter in marker:
        idx = marker.index(letter)
        word = marker[:idx] + ch + marker[idx + 1:]
        text = f'{word}=Zq9SENTINEL7x'
        out = scrub_details({'note': text})
        assert 'Zq9SENTINEL7x' not in str(out), repr(text)
        out_key = scrub_details({word: 'Zq9SENTINEL7x'})
        assert 'Zq9SENTINEL7x' not in str(out_key), repr(word)


def test_loop12_f2_leet_and_legit_numbers():
  '''Leet t0ken is a marker; legit number text is preserved.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  assert 'Zq9SENTINEL7x' not in _scrub_text(
    't0ken=Zq9SENTINEL7x')
  assert 'Zq9SENTINEL7x' not in _scrub_text(
    'credentia1=Zq9SENTINEL7x')
  assert _scrub_text(
    'train completed successfully learning_rate 0.1') == (
    'train completed successfully learning_rate 0.1')
  assert _scrub_text('epoch 2026-01-02') == 'epoch 2026-01-02'


# --- loop12 adversarial F3: digraph confusables (RED) ---

_LOOP12_F3 = [
  (0x1147, 'oo'), (0x11EE, 'oo'), (0x3180, 'oo'),
  (0xA698, 'OO'), (0xA699, 'oo'), (0xA736, 'AU'),
  (0xA737, 'au'), (0xA74E, 'OO'), (0xA74F, 'oo'),
  (0xAB43, 'co'), (0x221E, 'oo'), (0x1DF5D, 'ie'),
  (0x1DF64, 'th'),
]


def test_loop12_f3_digraph_scrubbed():
  '''Digraph glyphs spelling oo/au/co/ie/th redact sentinel.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  cases = [
    (0x221E, 'c', 'kie', 'cookie'),
    (0xA737, '', 'thorization', 'authorization'),
    (0xAB43, '', 'okie', 'cookie'),
    (0x1DF5D, 'cook', '', 'cookie'),
    (0x1DF64, 'au', 'orization', 'authorization'),
  ]
  for cp, pre, post, marker in cases:
    word = pre + chr(cp) + post
    text = f'{word}=Zq9SENTINEL7x'
    out = scrub_details({'note': text})
    assert 'Zq9SENTINEL7x' not in str(out), repr(text)
    out_key = scrub_details({word: 'Zq9SENTINEL7x'})
    assert 'Zq9SENTINEL7x' not in str(out_key), repr(word)
  for cp, digraph in _LOOP12_F3:
    from forecast_ml_framework.governance.scrub import (
      _normalize_unicode,
    )
    assert _normalize_unicode(chr(cp)) == digraph, hex(cp)


# --- loop12 adversarial F4: enclosed U+3207 (RED) ---

@pytest.mark.parametrize('word', [
  'passw\u3207rd', 't\u3207ken', 'c\u3207okie',
  'auth\u3207rization',
])
def test_loop12_f4_enclosed_3207_scrubbed(word):
  '''U+3207 splits o in a marker; sentinel is redacted.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  text = f'{word}=Zq9SENTINEL7x'
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  out_key = scrub_details({word: 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key), repr(word)


# --- loop12 adversarial F1: within-word ALT mixer (RED) ---

@pytest.mark.parametrize('word', [
  '\u044fpI_key',
  'p\u044fIv\u044fte_key',
  'c\u044fedentI\u044fI',
  '\u044fu\u00feorization',
  '\u00fe\u044fivate_key',
  'credentIaI',
  '\u044fpi_key',
  'au\u00fe',
])
def test_loop12_f1_alt_mixer_scrubbed(word):
  '''Mixed primary/alt readings in one word redact sentinel.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
    _scrub_text,
    scrub_details,
  )
  from forecast_ml_framework.telemetry.exceptions import (
    FrameworkError,
  )
  text = f'{word}=Zq9SENTINEL7x'
  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  out = scrub_details({'note': text, 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert out['ok'] == 'keep'
  entry = audit_log('train', 'alice', {'note': text})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  err = FrameworkError(
    'oops ' + text, stage='train', context={'note': text})
  assert 'Zq9SENTINEL7x' not in str(err), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict()), repr(
    text)


@pytest.mark.parametrize('word', [
  'credentIaI',
  '\u044fpI_key',
  'c\u044fedentI\u044fI',
  '\u044fu\u00feorization',
  '\u00fe\u044fivate_key',
])
def test_loop12_f1_alt_mixer_dict_key(word):
  '''Mixed-reading word as a dict key redacts its value.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({word: 'Zq9SENTINEL7x', 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(word)
  assert '***REDACTED***' in str(out), repr(word)
  assert out['ok'] == 'keep'


def test_loop12_f1_alt_mixer_composite_parts():
  '''Mixed-reading marker parts in a composite redact sentinel.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({
    'note': ('\u044fpI', 'Zq9SENTINEL7x', 'key'),
    'ok': 'keep',
  })
  assert 'Zq9SENTINEL7x' not in str(out)
  assert out['ok'] == 'keep'


# --- loop13: hybrid DoS / Bearer / parenthesized digit / hybrid | (RED) ---

def test_loop13_hybrid_build_is_truncated_to_max_text_len(monkeypatch):
  '''Hybrid skeleton is built from truncated text, not full text.'''
  from forecast_ml_framework.governance import scrub
  from forecast_ml_framework.governance.scrub import (
    _MAX_TEXT_LEN,
    _scrub_text,
  )
  seen = []
  original = scrub._build_hybrid_text

  def _spy(text):
    '''Spy.'''
    seen.append(len(text))
    return original(text)

  monkeypatch.setattr(scrub, '_build_hybrid_text', _spy)
  text = 'credentIaI=Zq9SENTINEL7x ' + ('x' * 50000)
  out = _scrub_text(text)
  assert seen, 'hybrid builder was not called'
  assert max(seen) <= _MAX_TEXT_LEN, seen
  assert 'Zq9SENTINEL7x' not in out
  assert 'TRUNCATED' in out


@pytest.mark.parametrize('marker', [
  'token', 'auth', 'secret', 'Bearer', 'password', 'api_key',
])
def test_loop13_bearer_marker_word_does_not_leak_secret(marker):
  '''Bearer followed by a marker word still redacts the secret.'''
  from forecast_ml_framework.governance.scrub import (
    _scrub_text,
    scrub_details,
  )
  text = f'Bearer {marker} Zq9SENTINEL7x'
  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  assert '***REDACTED***' in _scrub_text(text), repr(text)
  out = scrub_details({'note': text, 'ok': 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert out['ok'] == 'keep'


@pytest.mark.parametrize('marker', [
  'tokens', 'secrets', 'passwords', 'api_keys',
  'access_token', 'secret_key', 'token_id', 'auth_token',
])
def test_loop13_bearer_compound_marker_does_not_leak_secret(marker):
  '''Bearer compound/plural marker does not leak secret.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  text = f'Bearer {marker} Zq9SENTINEL7x'
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'Bearer api key Zq9SENTINEL7x',
  'Bearer is token Zq9SENTINEL7x',
  'Bearer token: Zq9SENTINEL7x',
])
def test_loop13_bearer_marker_phrase_does_not_leak_secret(text):
  '''Bearer marker phrases still redact the following secret.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop13_parenthesized_digits_fold_to_digits():
  '''Parenthesized digits U+2474-U+2487 fold to their digits.'''
  from forecast_ml_framework.governance.scrub import _normalize_unicode
  for number in range(1, 21):
    assert _normalize_unicode(chr(0x2473 + number)) == str(
      number), hex(0x2473 + number)


def test_loop13_parenthesized_digit_marker_scrubbed():
  '''credentia + parenthesized 1 is credential in alt reading.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  text = 'credentia\u2474=Zq9SENTINEL7x'
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop13_hybrid_pipe_joins_marker():
  '''Alt-alnum pipe tokenises with its token in hybrid text.'''
  from forecast_ml_framework.governance.scrub import (
    _build_hybrid_text,
    _scrub_text,
  )
  assert _build_hybrid_text('credentI\u044f|') == 'credential'
  text = 'credentI\u044f|=Zq9SENTINEL7x'
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)
  assert _scrub_text('a|b') == 'a|b'


# --- loop14: lazy governance __init__ (RED) ---

def _loop14_run_isolated(code):
  '''Run ``code`` in a fresh interpreter with PYTHONPATH=src.'''
  import os
  import subprocess
  import sys
  from pathlib import Path

  root = Path(__file__).resolve().parents[1]
  env = dict(os.environ)
  env['PYTHONPATH'] = (
    str(root / 'src') + os.pathsep + env.get('PYTHONPATH', ''))
  proc = subprocess.run(
    [sys.executable, '-c', code],
    capture_output=True, text=True, env=env, timeout=60,
    check=False)
  assert proc.returncode == 0, proc.stderr
  return proc


def test_loop14_scrub_import_does_not_load_audit():
  '''import governance.scrub must not load audit/lineage.'''
  code = '\n'.join([
    'import sys',
    'import forecast_ml_framework.governance.scrub as s',
    'assert \'forecast_ml_framework.governance.audit\' not in sys.modules',
    'assert \'forecast_ml_framework.governance.lineage\' not in sys.modules',
    'assert \'forecast_ml_framework.governance.retention\' not in sys.modules',
    'assert s.scrub_details({\'password\': \'hunter2\'}) == '
    '{\'password\': \'***REDACTED***\'}',
  ])
  _loop14_run_isolated(code)


def test_loop14_governance_lazy_init_preserves_public_api():
  '''Lazy __init__ still exposes the full public API.'''
  import forecast_ml_framework.governance as gov
  from forecast_ml_framework.governance import (
    REDACTED,
    audit_log,
    build_lineage,
    check_retention,
    compute_config_hash,
    scrub_details,
  )

  assert gov.__all__ == [
    'REDACTED', 'audit_log', 'build_lineage', 'check_retention',
    'compute_config_hash', 'scrub_details']
  assert REDACTED == '***REDACTED***'
  assert scrub_details({'password': 'hunter2'}) == {
    'password': '***REDACTED***'}
  assert build_lineage({})['model'] is None
  assert compute_config_hash({}, {})
  assert check_retention(
    [{'age_days': 1}], params=10)['keep'] == [{'age_days': 1}]
  entry = audit_log('train', 'alice', {'password': 'hunter2'})
  assert entry['details'] == {'password': '***REDACTED***'}
  assert 'audit_log' in dir(gov)
  assert 'scrub_details' in dir(gov)
  gov.__dict__.pop('audit', None)
  assert gov.audit.audit_log is not None
  with pytest.raises(AttributeError):
    _missing = gov.no_such_thing_loop14


def test_loop14_governance_submodule_import_still_works():
  '''from governance import audit, scrub still resolves.'''
  from forecast_ml_framework.governance import audit, scrub

  assert audit.audit_log is not None
  assert scrub.scrub_details is not None
  assert audit.scrub_details is scrub.scrub_details


# --- loop14 adversarial: Bearer quoted/bracket/plural, enclosed
# digits, glued ASCII paren digit (RED) ---

@pytest.mark.parametrize('text', [
  'Bearer "token" Zq9SENTINEL7x',
  'Bearer \'password\' Zq9SENTINEL7x',
])
def test_loop14_adversarial_bearer_quoted_marker(text):
  '''Quoted marker after Bearer does not leak secret.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'Bearer (token) Zq9SENTINEL7x',
  'Bearer [token] Zq9SENTINEL7x',
  'Bearer {token} Zq9SENTINEL7x',
  'Bearer token, Zq9SENTINEL7x',
  'Bearer token; Zq9SENTINEL7x',
  'Bearer token) Zq9SENTINEL7x',
  'Bearer token] Zq9SENTINEL7x',
  'Bearer token} Zq9SENTINEL7x',
])
def test_loop14_adversarial_bearer_bracket_terminator(text):
  '''Bracketed/terminated marker after Bearer: no leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'Bearer api keys Zq9SENTINEL7x',
  'Bearer pass words Zq9SENTINEL7x',
  'Bearer aws access keys Zq9SENTINEL7x',
  'Bearer aws access key ids Zq9SENTINEL7x',
])
def test_loop14_adversarial_bearer_plural_marker(text):
  '''Plural marker phrase after Bearer: no leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('cp_digit', [
  (0x24F5, 1), (0x24F6, 2), (0x24F7, 3), (0x24F8, 4),
  (0x24F9, 5), (0x24FA, 6), (0x24FB, 7), (0x24FC, 8),
  (0x24FD, 9), (0x24FF, 0),
  (0x2776, 1), (0x2777, 2), (0x2778, 3), (0x2779, 4),
  (0x277A, 5), (0x277B, 6), (0x277C, 7), (0x277D, 8),
  (0x277E, 9),
  (0x2780, 1), (0x2781, 2), (0x2782, 3), (0x2783, 4),
  (0x2784, 5), (0x2785, 6), (0x2786, 7), (0x2787, 8),
  (0x2788, 9),
  (0x278A, 1), (0x278B, 2), (0x278C, 3), (0x278D, 4),
  (0x278E, 5), (0x278F, 6), (0x2790, 7), (0x2791, 8),
  (0x2792, 9),
  (0x1F10B, 0), (0x1F10C, 0),
])
def test_loop14_adversarial_enclosed_digits_fold(cp_digit):
  '''All 39 enclosed digits fold to their digit.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  cp, digit = cp_digit
  assert _normalize_unicode(chr(cp)) == str(digit), hex(cp)


@pytest.mark.parametrize('text', [
  'credentia\u24f5=Zq9SENTINEL7x',
  'credentia\u2776=Zq9SENTINEL7x',
  'credentia\u2780=Zq9SENTINEL7x',
  'credentia\u278a=Zq9SENTINEL7x',
  't\u24ffken=Zq9SENTINEL7x',
  't\U0001f10bken=Zq9SENTINEL7x',
  't\U0001f10cken=Zq9SENTINEL7x',
])
def test_loop14_adversarial_enclosed_digit_marker(text):
  '''Enclosed 1/0 spelling a marker: no leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop14_adversarial_enclosed_digit_dictkey():
  '''Enclosed-digit marker as dict key: value redacted.'''
  from forecast_ml_framework.governance.scrub import scrub_details
  out = scrub_details({'credentia\u2776': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out), repr(out)
  assert '***REDACTED***' in str(out), repr(out)


@pytest.mark.parametrize('text', [
  'credentia(1)=Zq9SENTINEL7x',
  't(0)ken=Zq9SENTINEL7x',
  'credent(1)al=Zq9SENTINEL7x',
])
def test_loop14_adversarial_glued_paren_digit(text):
  '''Glued ASCII (digit) inside marker: no leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop14_adversarial_glued_paren_preserves_spaced():
  '''Spaced (a)/(1)/(world) are preserved, not folded.'''
  from forecast_ml_framework.governance.scrub import (
    _normalize_unicode,
  )
  assert _normalize_unicode('option (a) is ok') == (
    'option (a) is ok')
  assert _normalize_unicode('option (1) is ok') == (
    'option (1) is ok')
  assert _normalize_unicode('hello (world) test') == (
    'hello (world) test')


# --- loop15 B1: intra-marker separator any non-alnum (RED) ---

@pytest.mark.parametrize('text', [
  'api|key=Zq9SENTINEL7x',
  'api*key=Zq9SENTINEL7x',
  'access*key=Zq9SENTINEL7x',
  'private*key=Zq9SENTINEL7x',
  'pass*word=Zq9SENTINEL7x',
  'api\u00b7key=Zq9SENTINEL7x',
  'pass\u2212word=Zq9SENTINEL7x',
])
def test_loop15_b1_listed_separators_scrubbed(text):
  '''Loop15 B1 PoC: listed intra-marker separators leak nothing.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'api|key=Zq9SENTINEL7x',
  'api*key=Zq9SENTINEL7x',
  'access*key=Zq9SENTINEL7x',
  'private*key=Zq9SENTINEL7x',
  'pass*word=Zq9SENTINEL7x',
  'api\u00b7key=Zq9SENTINEL7x',
  'pass\u2212word=Zq9SENTINEL7x',
])
def test_loop15_b1_audit_matches_dict_key(text):
  '''Loop15 B1: audit free-text scrub matches dict-key block.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
    _is_secret_key,
  )
  key = text.split('=')[0]
  assert _is_secret_key(key), repr(key)
  entry = audit_log('train', 'alice', {'note': text, 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details']['ok'] == 'keep'


@pytest.mark.parametrize('sep', [
  '|', '*', '~', '!', '?', '@', '#', '$', '%', '^', '&', '+',
  '<', '>', ':', ';', ',', '(', ')', '[', ']', '{', '}', '"',
  '\'', '`', '=', '/', '\\', '.', '-', '_', ' ', '\t',
  '\u00b7', '\u2212', '\u2014', '\u2013', '\u2022', '\u00d7',
  '\u00f7', '\u2192', '\u2026', '\u2044',
])
def test_loop15_b1_any_nonalnum_separator_scrubbed(sep):
  '''Any non-alnum intra-marker separator matches _normalize_key.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  for text in (
    f'api{sep}key=Zq9SENTINEL7x',
    f'pass{sep}word=Zq9SENTINEL7x',
    f'access{sep}key=Zq9SENTINEL7x',
    f'private{sep}key=Zq9SENTINEL7x',
    f'aws{sep}access{sep}key{sep}id=Zq9SENTINEL7x',
    f'secret{sep}access{sep}key=Zq9SENTINEL7x',
  ):
    out = _scrub_text(text)
    assert 'Zq9SENTINEL7x' not in out, repr(text)
    assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'Bearer api|key Zq9SENTINEL7x',
  'Bearer api*key Zq9SENTINEL7x',
  'Bearer access*key Zq9SENTINEL7x',
  'Bearer private*key Zq9SENTINEL7x',
  'Bearer pass*word Zq9SENTINEL7x',
  'Bearer api\u00b7key Zq9SENTINEL7x',
  'Bearer pass\u2212word Zq9SENTINEL7x',
])
def test_loop15_b1_bearer_lookahead_widened(text):
  '''Bearer lookahead uses the widened intra-marker separator.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'api|*key=Zq9SENTINEL7x',
  'api|\u00b7key Zq9SENTINEL7x',
  'pass*\u2212word=Zq9SENTINEL7x',
])
def test_loop15_b1_mixed_separator_run_scrubbed(text):
  '''Mixed non-alnum separator runs inside a marker scrub.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_loop15_b1_long_separator_run_is_fast():
  '''A long separator run does not cause backtracking blowup.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  text = 'api' + ('|' * 5000) + 'key=Zq9SENTINEL7x'
  start = time.perf_counter()
  out = _scrub_text(text)
  elapsed = time.perf_counter() - start
  assert elapsed < 1.0, elapsed
  assert 'Zq9SENTINEL7x' not in out
  assert '***REDACTED***' in out


def test_loop15_b1_non_marker_separator_text_preserved():
  '''Non-marker text with separators is preserved.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  for text in ('a|b', 'hello*world', 'price|quantity 5',
               'api|description is fine'):
    assert _scrub_text(text) == text, repr(text)


@pytest.mark.parametrize('text', [
  'Bearer aws*access*key Zq9SENTINEL7x',
  'Bearer aws|access|key Zq9SENTINEL7x',
  'Bearer aws\u00b7access\u00b7key Zq9SENTINEL7x',
  'Bearer aws\u2212access\u2212key Zq9SENTINEL7x',
  'Bearer aws*access*keys Zq9SENTINEL7x',
  'Bearer aws*access*key*ids Zq9SENTINEL7x',
  'Bearer api*keys Zq9SENTINEL7x',
  'Bearer api|keys Zq9SENTINEL7x',
  'Bearer pass*words Zq9SENTINEL7x',
  'Bearer pass|words Zq9SENTINEL7x',
])
def test_loop15_b1_bearer_aws_and_plural_phrase(text):
  '''Bearer aws/plural marker phrases with new separators scrub.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
  'password api*keys Zq9SENTINEL7x',
  'password aws*access*key Zq9SENTINEL7x',
  'token pass*words Zq9SENTINEL7x',
])
def test_loop15_b1_marker_phrase_after_marker_scrubbed(text):
  '''Glued plural/aws marker phrase after a marker is skipped.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


# --- loop16: filler-chain symbolic glue (RED) ---

@pytest.mark.parametrize('text', [
  'password very*really Zq9SENTINEL7x',
  'password very|really Zq9SENTINEL7x',
  'password very~really Zq9SENTINEL7x',
  'token very*really Zq9SENTINEL7x',
])
def test_loop16_filler_chain_symbolic_glue_scrubbed(text):
  '''Filler parts glued by a symbol are filler, not the value.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text, 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details']['ok'] == 'keep'


@pytest.mark.parametrize('text', [
  'password is=was Zq9SENTINEL7x',
  'password is:was Zq9SENTINEL7x',
  'password is|was Zq9SENTINEL7x',
])
def test_loop16_filler_pair_separators_scrubbed(text):
  '''Filler+sep+filler is filler, not a new key=value pair.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text, 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details']['ok'] == 'keep'


@pytest.mark.parametrize('text', [
  'Bearer very*really Zq9SENTINEL7x',
  'Bearer very|really Zq9SENTINEL7x',
])
def test_loop16_bearer_filler_chain_symbolic_glue_scrubbed(text):
  '''Bearer filler chain glued by a symbol does not leak.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text, 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details']['ok'] == 'keep'


@pytest.mark.parametrize('text', [
  'password "very*really" Zq9SENTINEL7x',
  'password \'very*really\' Zq9SENTINEL7x',
  'password very*really "Zq9SENTINEL7x"',
  'password very*really \'Zq9SENTINEL7x\'',
])
def test_loop16_quoted_filler_chain_symbolic_glue_scrubbed(text):
  '''Quoted filler chain glued by a symbol is filler, not value.'''
  from forecast_ml_framework.governance.audit import audit_log
  entry = audit_log('train', 'alice', {'note': text, 'ok': 'keep'})
  blob = json.dumps(entry)
  assert 'Zq9SENTINEL7x' not in blob, repr(text)
  assert '***REDACTED***' in blob, repr(text)
  assert entry['details']['ok'] == 'keep'



# --- loop17: proven fixes P1-P4 (RED) ---

@pytest.mark.parametrize('type_name', [
    'password=Zq9SENTINEL7x',
    'passwordZq9SENTINEL7x',
    'token=Zq9SENTINEL7x',
])
def test_loop17_type_name_marker_pattern_scrubbed(type_name):
  '''P1: dynamic type name carrying marker pattern is scrubbed.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  obj = type(type_name, (), {})()
  out = scrub_details({'note': obj})
  assert 'Zq9SENTINEL7x' not in str(out), repr(type_name)
  assert '***REDACTED***' in str(out), repr(type_name)
  out_key = scrub_details({obj: 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out_key), repr(type_name)


def test_loop17_type_name_bare_sentinel_scrubbed():
  '''P1: bare sentinel as a dynamic type name does not leak.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  obj = type('Zq9SENTINEL7x', (), {})()
  out = scrub_details({'note': obj})
  assert 'Zq9SENTINEL7x' not in str(out)
  out_key = scrub_details({obj: 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out_key)


@pytest.mark.parametrize('text', [
    'api very key=Zq9SENTINEL7x',
    'access very key=Zq9SENTINEL7x',
    'private very key=Zq9SENTINEL7x',
    'api is key=Zq9SENTINEL7x',
    'access very key very id=Zq9SENTINEL7x',
    'api very key Zq9SENTINEL7x',
    'api very key: Zq9SENTINEL7x',
])
def test_loop17_intra_marker_filler_scrubbed(text):
  '''P2: filler inside api/access/private compound does not leak.'''
  from forecast_ml_framework.governance.scrub import (
        _is_secret_key,
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert _is_secret_key('api very key') is True
  assert _is_secret_key('access very key') is True
  out_key = scrub_details({'api very key': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key)
  out_set = scrub_details(
        {'k': {'api very key', 'Zq9SENTINEL7x'}})
  assert 'Zq9SENTINEL7x' not in str(out_set)


@pytest.mark.parametrize('text', [
    'password pretty Zq9SENTINEL7x',
    'password is quite Zq9SENTINEL7x',
    'password is pretty much Zq9SENTINEL7x',
    'token is so Zq9SENTINEL7x',
    'secret was too Zq9SENTINEL7x',
    'password somewhat Zq9SENTINEL7x',
    'password extremely Zq9SENTINEL7x',
    'password rather Zq9SENTINEL7x',
    'password super Zq9SENTINEL7x',
    'password much Zq9SENTINEL7x',
    'Bearer pretty Zq9SENTINEL7x',
    'Bearer is quite Zq9SENTINEL7x',
])
def test_loop17_filler_vocab_scrubbed(text):
  '''P3: degree-adverb filler vocab does not leak.'''
  from forecast_ml_framework.governance.scrub import (
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)


@pytest.mark.parametrize('text', [
    'Bearer very)really Zq9SENTINEL7x',
    'Bearer very) Zq9SENTINEL7x',
    'Bearer is] Zq9SENTINEL7x',
    'Bearer very}really Zq9SENTINEL7x',
    'Bearer very] Zq9SENTINEL7x',
    'Bearer is} Zq9SENTINEL7x',
])
def test_loop17_bearer_bracket_filler_scrubbed(text):
  '''P4: Bearer filler plus closing bracket does not leak.'''
  from forecast_ml_framework.governance.scrub import (
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)


# --- loop17-arch: H1/H2/H3/F1/F2 (RED) ---

@pytest.mark.parametrize('text', [
    'password tremendously Zq9SENTINEL7x',
    'password literally Zq9SENTINEL7x',
    'password honestly Zq9SENTINEL7x',
    'password literaly Zq9SENTINEL7x',
    'password absurdly Zq9SENTINEL7x',
    'api tremendously key=Zq9SENTINEL7x',
    'api literally key=Zq9SENTINEL7x',
    'api honestly key Zq9SENTINEL7x',
    'api absurdly key=Zq9SENTINEL7x',
    'access tremendously key=Zq9SENTINEL7x',
    'password is tremendously Zq9SENTINEL7x',
])
def test_loop17_arch_h1_unknown_filler_scrubbed(text):
  '''H1: unseen degree adverbs do not leak secret.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
        _is_secret_key,
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  assert '***REDACTED***' in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert '***REDACTED***' in str(out), repr(text)
  entry = audit_log('train', 'alice', {'note': text})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  assert _is_secret_key('api tremendously key') is True
  assert _is_secret_key('api absurdly key') is True
  out_key = scrub_details(
        {'api tremendously key': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key)


@pytest.mark.parametrize('text', [
    'apiverykey=Zq9SENTINEL7x',
    'accessverykey=Zq9SENTINEL7x',
    'privateverykey=Zq9SENTINEL7x',
    'apiverykey Zq9SENTINEL7x',
    'apireallykey=Zq9SENTINEL7x',
    'apiiskey=Zq9SENTINEL7x',
])
def test_loop17_arch_h2_glued_intra_filler_scrubbed(text):
  '''H2: glued filler inside compound marker leaks not.'''
  from forecast_ml_framework.governance.scrub import (
        _is_secret_key,
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  assert '***REDACTED***' in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert _is_secret_key('apiverykey') is True
  assert _is_secret_key('accessverykey') is True
  out_key = scrub_details({'apiverykey': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key)
  assert '***REDACTED***' in str(out_key)


@pytest.mark.parametrize('type_name', [
    'zqsentinelvalue',
    'zq9sentinelvalue',
    '\u00e1Zq9SENTINEL7x',
    '\uff41Zq9SENTINEL7x',
    'a\u0301Zq9SENTINEL7x',
])
def test_loop17_arch_h3_f1_type_name_scrubbed(type_name):
  '''H3/F1: lowercase/normalized sentinel type names leak not.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
        _scrub_type_name,
        scrub_details,
    )

  obj = type(type_name, (), {})()
  assert 'Zq9SENTINEL7x' not in _scrub_type_name(type_name)
  assert 'zqsentinelvalue' not in _scrub_type_name(type_name)
  out = scrub_details({'note': obj})
  assert 'Zq9SENTINEL7x' not in str(out), repr(type_name)
  assert 'zqsentinelvalue' not in str(out), repr(type_name)
  assert '***REDACTED***' in str(out), repr(type_name)
  out_key = scrub_details({obj: 'keep'})
  assert 'Zq9SENTINEL7x' not in str(out_key), repr(type_name)
  assert 'zqsentinelvalue' not in str(out_key), repr(type_name)
  entry = audit_log('train', 'alice', {'note': obj})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(type_name)
  assert 'zqsentinelvalue' not in json.dumps(entry), repr(type_name)


def test_loop17_arch_h3_camelcase_type_name_preserved():
  '''H3: plain CamelCase and private CamelCase preserved.'''
  from forecast_ml_framework.governance.scrub import (
        _scrub_type_name,
        scrub_details,
    )

  assert _scrub_type_name('SecretThing') == 'SecretThing'
  assert _scrub_type_name('Evil') == 'Evil'
  assert _scrub_type_name('_Loop8EvilKeyPw') == '_Loop8EvilKeyPw'
  obj = type('SecretThing', (), {})()
  assert scrub_details({'note': obj}) == {
        'note': {'type': 'SecretThing'}}


@pytest.mark.parametrize('text', [
    'api well-known key=Zq9SENTINEL7x',
    'api wellknown key=Zq9SENTINEL7x',
    'access well-known key=Zq9SENTINEL7x',
    'access wellknown key Zq9SENTINEL7x',
    'private well-known key=Zq9SENTINEL7x',
    'private wellknown key=Zq9SENTINEL7x',
    'api filler key=Zq9SENTINEL7x',
    'api one key=Zq9SENTINEL7x',
])
def test_loop17_arch_f2_wellknown_intra_filler_scrubbed(text):
  '''F2: well-known/wellknown and filler audit do not leak.'''
  from forecast_ml_framework.governance.scrub import (
        _is_secret_key,
        _scrub_text,
        scrub_details,
    )

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  out = scrub_details({'note': text})
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert _is_secret_key('api well-known key') is True
  assert _is_secret_key('api wellknown key') is True
  out_key = scrub_details(
        {'api well-known key': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key)
  out_key2 = scrub_details(
        {'api wellknown key': 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out_key2)


# --- loop18: P2/P3 (RED) ---

@pytest.mark.parametrize('text', [
    'pass phrase=Zq9SENTINEL7x',
    'pass phrase Zq9SENTINEL7x',
    'pass code=Zq9SENTINEL7x',
    'pass code Zq9SENTINEL7x',
    'pass-phrase=Zq9SENTINEL7x',
    'pass-code Zq9SENTINEL7x',
    'passphrase=Zq9SENTINEL7x',
])
def test_loop18_pass_phrase_code_scrubbed(text):
  '''P2: pass phrase/code free text does not leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)
  assert '***REDACTED***' in _scrub_text(text), repr(text)


@pytest.mark.parametrize('text', [
    'password highly Zq9SENTINEL7x',
    'password incredibly Zq9SENTINEL7x',
    'password fairly Zq9SENTINEL7x',
    'password particularly Zq9SENTINEL7x',
    'password especially Zq9SENTINEL7x',
    'api highly key=Zq9SENTINEL7x',
    'api incredibly key=Zq9SENTINEL7x',
])
def test_loop18_degree_adverbs_scrubbed(text):
  '''P3: degree adverbs highly/incredibly/... do not leak.'''
  from forecast_ml_framework.governance.scrub import _scrub_text

  assert 'Zq9SENTINEL7x' not in _scrub_text(text), repr(text)


def test_loop17_arch_public_aliases_and_docs():
  '''Public scrub aliases exist; audit doc points to scrub.'''
  from forecast_ml_framework.governance import audit, scrub

  assert scrub.safe_str is scrub._safe_str
  assert scrub.safe_repr is scrub._safe_repr
  assert scrub.safe_list is scrub._safe_list
  assert scrub.scrub_text is scrub._scrub_text
  assert scrub.scrub_type_name is scrub._scrub_type_name
  assert scrub.unknown_key is scrub._unknown_key
  assert scrub.is_secret_key is scrub._is_secret_key
  assert scrub.decode_bytes_scrubbed is (
        scrub._decode_bytes_scrubbed)
  assert scrub.composite_marker_bearing is (
        scrub._composite_marker_bearing)
  assert scrub.marker_part is scrub._marker_part
  assert scrub.is_marker_element is scrub._is_marker_element
  assert scrub.MAX_ITEMS == scrub._MAX_ITEMS
  assert scrub.MAX_COMPOSITE_DEPTH == scrub._MAX_COMPOSITE_DEPTH
  doc = (audit.__doc__ or '').lower()
  assert 'scrub' in doc
  assert len(doc) < 500, len(doc)


# --- loop19: marker regex must not scan the value (RED) ---

def test_loop19_marker_long_token_is_fast():
  '''Marker plus a long alnum token is linear, not backtracking.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  for marker in ('password', 'secret', 'token', 'api key'):
    text = marker + ' ' + ('a' * 5000)
    start = time.perf_counter()
    out = _scrub_text(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (marker, elapsed)
    assert 'a' * 100 not in out, marker
    assert '***REDACTED***' in out, marker


def test_loop19_marker_prefix_long_token_is_fast():
  '''A marker prefix plus a long token is linear, not backtracking.'''
  from forecast_ml_framework.governance.scrub import _scrub_text
  for text in ('api ' + ('a' * 5000), 'api' + ('a' * 5000)):
    start = time.perf_counter()
    out = _scrub_text(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (text[:10], elapsed)
    assert out == text, text[:10]


# --- loop21: short glued suffix and chained glued filler (RED) ---

@pytest.mark.parametrize('text', [
    'passwordX Zq9SENTINEL7x',
    'passwordZ Zq9SENTINEL7x',
    'password1 Zq9SENTINEL7x',
    'password9 Zq9SENTINEL7x',
    'tokenX Zq9SENTINEL7x',
    'token1 Zq9SENTINEL7x',
    'secretX Zq9SENTINEL7x',
    'passwordHu Zq9SENTINEL7x',
    'passwordTo Zq9SENTINEL7x',
    'passwordAB Zq9SENTINEL7x',
    'passwordXY Zq9SENTINEL7x',
    'passwordab Zq9SENTINEL7x',
    'passworda1 Zq9SENTINEL7x',
    'passwordX1 Zq9SENTINEL7x',
    'passwordab1 Zq9SENTINEL7x',
    'api keyX Zq9SENTINEL7x',
    'apikeyX Zq9SENTINEL7x',
    'api-keyX Zq9SENTINEL7x',
    'accesskeyX Zq9SENTINEL7x',
    'privatekeyX Zq9SENTINEL7x',
    'passwordsX Zq9SENTINEL7x',
    'passwordX: Zq9SENTINEL7x',
    'passwordX, Zq9SENTINEL7x',
    'passwordX "Zq9SENTINEL7x"',
])
def test_loop21_f1_short_glued_suffix_does_not_starve(text):
  '''F1: a short glued suffix is a marker extension, not value.'''
  from forecast_ml_framework.governance.scrub import _scrub_text

  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
    'passwordX=Zq9SENTINEL7x',
    'passwordId Zq9SENTINEL7x',
    'passwordId=Zq9SENTINEL7x',
    'passwordHunter Zq9SENTINEL7x',
    'passworda Zq9SENTINEL7x',
    'passwords Zq9SENTINEL7x',
    'passwordZq9SENTINEL7x',
])
def test_loop21_f1_glued_controls_still_redact(text):
  '''F1: glued ``=`` value and known extensions still redact.'''
  from forecast_ml_framework.governance.scrub import _scrub_text

  out = _scrub_text(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('key', [
    'apiveryverykey',
    'apireallyreallykey',
    'apiisverykey',
    'apiiskey',
    'accessveryverykey',
    'privateveryverykey',
    'secretveryverykey',
])
def test_loop21_f2_chained_glued_filler_key(key):
  '''F2: 2+ chained glued fillers between marker parts redact.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )
  from forecast_ml_framework.telemetry.exceptions import FrameworkError

  assert _is_secret_key(key) is True, key
  out = scrub_details({key: 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in str(out), key
  assert '***REDACTED***' in str(out), key
  entry = audit_log('train', 'alice', {key: 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), key
  err = FrameworkError('m', context={key: 'Zq9SENTINEL7x'})
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict()), key


def test_loop21_f2_long_chained_filler_key_is_fast():
  '''F2: an 8192-char chained filler key is linear, not quadratic.'''
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )

  key = 'api' + ('very' * 2046) + 'key'
  assert len(key) == 8190
  start = time.perf_counter()
  assert _is_secret_key(key) is True
  out = scrub_details({key: 'Zq9SENTINEL7x'})
  elapsed = time.perf_counter() - start
  assert elapsed < 1.0, elapsed
  assert 'Zq9SENTINEL7x' not in str(out)


# --- cycle1 W1: adversarial F1/F2/F3/F5 (RED) ---


@pytest.mark.parametrize('text', [
    'Cookie: session=HUNTER2ZQ9SENTINEL',
    'password: user=HUNTER2ZQ9SENTINEL',
    'token: data=HUNTER2ZQ9SENTINEL',
    'Authorization: user=HUNTER2ZQ9SENTINEL',
    'api_key: foo=HUNTER2ZQ9SENTINEL',
    'password foo=HUNTER2ZQ9SENTINEL',
])
def test_w1_f1_pair_deferral_non_secret_key_redacts_token(text):
  '''F1: non-secret pair key after a marker redacts the token.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text,expected', [
    (
        'password token=HUNTER2ZQ9SENTINEL',
        'password token=***REDACTED***',
    ),
    (
        'password api_key=HUNTER2ZQ9SENTINEL',
        'password api_key=***REDACTED***',
    ),
    (
        'token password=HUNTER2ZQ9SENTINEL',
        'token password=***REDACTED***',
    ),
])
def test_w1_f1_secret_pair_key_still_defers(text, expected):
  '''F1: secret-bearing pair key defers to its own marker.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  assert scrub_text(text) == expected, repr(text)


def test_w1_f1_trailing_non_secret_pair_preserved():
  '''F1: password=QZ other=1 keeps other=1.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  assert scrub_text('password=QZ other=1') == (
      'password=***REDACTED*** other=1')


@pytest.mark.parametrize('key', [
    'encryption_key',
    'master_key',
    'signing_key',
    'aws_key',
    'gcp_key',
    'private_key',
    'otp',
    'pin_code',
    'pin_number',
    'pincode',
])
def test_w1_f2_new_compound_dict_keys_redacted(key):
  '''F2: new secret compounds redact dict values.'''
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )

  assert _is_secret_key(key) is True, key
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out), key
  assert '***REDACTED***' in str(out), key


@pytest.mark.parametrize('text', [
    'encryption_key=HUNTER2ZQ9SENTINEL',
    'master_key=HUNTER2ZQ9SENTINEL',
    'signing_key=HUNTER2ZQ9SENTINEL',
    'aws_key=HUNTER2ZQ9SENTINEL',
    'gcp_key=HUNTER2ZQ9SENTINEL',
    'encryption key HUNTER2ZQ9SENTINEL',
    'master-key HUNTER2ZQ9SENTINEL',
    'otp=HUNTER2ZQ9SENTINEL',
    'otp HUNTER2ZQ9SENTINEL',
    'pin_code=HUNTER2ZQ9SENTINEL',
    'pin_number=HUNTER2ZQ9SENTINEL',
    'pin code HUNTER2ZQ9SENTINEL',
])
def test_w1_f2_new_compound_free_text_redacted(text):
  '''F2: new secret compounds redact free-text values.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


def test_w1_f2_bare_key_hash_not_substring_markers():
  '''F2: bare key/hash are not substring markers.'''
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
      scrub_text,
  )

  assert _is_secret_key('key_cols_list') is False
  assert _is_secret_key('hash_col') is False
  assert _is_secret_key('mapping') is False
  assert _is_secret_key('monkey') is False
  details = {
      'key_cols_list': ['a', 'b'],
      'hash_col': 'keep',
      'mapping': 'keep',
      'monkey': 'keep',
  }
  assert scrub_details(details) == details
  text = 'key_cols_list hash_col mapping monkey'
  assert scrub_text(text) == text


@pytest.mark.parametrize('text', [
    'p4ssword=HUNTER2ZQ9SENTINEL',
    'p455word=HUNTER2ZQ9SENTINEL',
    'p@ssword=HUNTER2ZQ9SENTINEL',
    'p@ssw0rd=HUNTER2ZQ9SENTINEL',
    '$ecret=HUNTER2ZQ9SENTINEL',
    '5ecret=HUNTER2ZQ9SENTINEL',
    's3cret=HUNTER2ZQ9SENTINEL',
    'p4ssw0rd=HUNTER2ZQ9SENTINEL',
])
def test_w1_f3_ascii_leet_markers_redacted(text):
  '''F3: ASCII leet markers redact free-text values.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('key', [
    'p4ssword',
    'p@ssword',
    '$ecret',
    '5ecret',
])
def test_w1_f3_leet_dict_keys_redacted(key):
  '''F3: ASCII leet markers redact dict values.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out), key
  assert '***REDACTED***' in str(out), key


def test_w1_f3_digit_bearing_non_markers_preserved():
  '''F3: legit digit text and keys are not leet markers.'''
  from forecast_ml_framework.governance.scrub import (
      scrub_details,
      scrub_text,
  )

  assert scrub_text('epoch 2026-01-02') == 'epoch 2026-01-02'
  text = 'train completed successfully learning_rate 0.1'
  assert scrub_text(text) == text
  assert scrub_text('room 101 value 42') == 'room 101 value 42'
  details = {
      '2026': 'keep',
      'a4': 'keep',
      'h4': 'keep',
      'key_cols_list': ['x'],
  }
  assert scrub_details(details) == details


def test_w1_f5_large_plain_text_is_fast():
  '''F5: 2MB plain text is truncated before work, fast.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  start = time.perf_counter()
  out = scrub_text('a' * 2000000)
  elapsed = time.perf_counter() - start
  assert elapsed < 0.1, elapsed
  assert len(out) <= 10100
  assert 'TRUNCATED' in out


def test_w1_f5_marker_dense_text_is_fast():
  '''F5: marker-dense 11KB text is near-linear, fast.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  text = 'password x ' * 1000
  assert len(text) == 11000
  start = time.perf_counter()
  out = scrub_text(text)
  elapsed = time.perf_counter() - start
  assert elapsed < 0.1, elapsed
  assert '***REDACTED***' in out


def test_w1_audit_docstring_points_to_scrub():
  '''F11: audit docstring is a pointer to the scrub leaf.'''
  import forecast_ml_framework.governance.audit as audit

  assert audit.__doc__ is not None
  assert 'scrub' in audit.__doc__
  assert len(audit.__doc__) < 500, len(audit.__doc__)


@pytest.mark.parametrize('text', [
    'Bearer encryption_key HUNTER2ZQ9SENTINEL',
    'Bearer master_key HUNTER2ZQ9SENTINEL',
    'Bearer pin_code HUNTER2ZQ9SENTINEL',
    'Bearer aws_key HUNTER2ZQ9SENTINEL',
    'password encryption_key HUNTER2ZQ9SENTINEL',
])
def test_w1_f2_bearer_compound_secret_redacted(text):
  '''F2: Bearer plus a new compound still redacts secret.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


# --- cycle7 Q2: F-SCRUB-LONGKEY (RED) ---


def test_q2_longkey_8199_marker_is_secret_key():
  '''Q2: marker past the 8192 key cap must not hide.'''
  from forecast_ml_framework.governance.scrub import _is_secret_key

  key = 'a' * 8190 + '_password'
  assert len(key) == 8199
  assert _is_secret_key(key) is True


def test_q2_longkey_8199_scrub_details_redacts():
  '''Q2: sentinel under 8199-char marker key is redacted.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  key = 'a' * 8190 + '_password'
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out)
  assert '***REDACTED***' in str(out)


def test_q2_longkey_8199_audit_and_error_redact():
  '''Q2: audit_log and FrameworkError sinks redact too.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.telemetry.exceptions import FrameworkError

  key = 'a' * 8190 + '_password'
  entry = audit_log('train', 'alice', {key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in json.dumps(entry)
  err = FrameworkError('m', context={key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in json.dumps(err.to_dict())


def test_q2_longkey_8009_control_redacts():
  '''Q2 control: marker inside the cap still redacts.'''
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )

  key = 'a' * 8000 + '_password'
  assert len(key) == 8009
  assert _is_secret_key(key) is True
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out)
  assert '***REDACTED***' in str(out)


def test_q2_longkey_bytes_marker_past_cap_redacts():
  '''Q2: bytes key with marker past the cap redacts.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  key = b'a' * 8190 + b'_password'
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out)
  assert '***REDACTED***' in str(out)


def test_q2_longkey_marker_in_middle_redacts():
  '''Q2: plain marker in the middle of a huge key redacts.'''
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )

  key = 'a' * 50000 + '_password' + 'a' * 50000
  assert _is_secret_key(key) is True
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out)


def test_q2_giant_100k_key_is_bounded_and_fast():
  '''Q2: 100k-char key stays bounded work under the cap.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  key = 'a' * 99991 + '_password'
  assert len(key) == 100000
  start = time.perf_counter()
  out = scrub_details({key: 'HUNTER2ZQ9SENTINEL'})
  elapsed = time.perf_counter() - start
  assert elapsed < 1.0, elapsed
  assert 'HUNTER2ZQ9SENTINEL' not in str(out)
  assert max(len(str(k)) for k in out) <= 8192


# --- cycle8 Q5: C8-01 composite middle-exotic sibling (RED) ---

_Q5_SENTINEL = 'HUNTER2ZQ9SENTINEL'
_Q5_MID_MARKERS = (
  'api very key',
  'p4ssword',
  'credentIaI',
  'ｐａｓｓｗｏｒｄ',
  'api_very_key',
)


def _q5_over_cap_mid(mid_marker):
  '''Build an over-cap middle-exotic marker string.'''
  return 'a' * 10000 + mid_marker + 'a' * 10000


@pytest.mark.parametrize('mid_marker', _Q5_MID_MARKERS)
def test_q5_tuple_middle_exotic_sentinel_gone_all_sinks(mid_marker):
  '''Q5: tuple middle-exotic sibling sentinel gone, all sinks.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import (
      _is_secret_key,
      scrub_details,
  )
  from forecast_ml_framework.telemetry.exceptions import FrameworkError

  mid = _q5_over_cap_mid(mid_marker)
  # Detector stays non-blanket-fail-closed: middle-exotic alone
  # is not a secret key, yet sibling finalization must redact.
  assert _is_secret_key(mid) is False
  key = (mid, _Q5_SENTINEL)
  out = scrub_details({key: 'hello'})
  assert _Q5_SENTINEL not in str(out)
  assert '***REDACTED***' in str(out)
  entry = audit_log('train', 'alice', {key: 'hello'})
  assert _Q5_SENTINEL not in str(entry)
  err = FrameworkError('m', context={key: 'hello'})
  assert _Q5_SENTINEL not in str(err.to_dict())


@pytest.mark.parametrize('mid_marker', _Q5_MID_MARKERS)
def test_q5_frozenset_middle_exotic_sentinel_gone_all_sinks(mid_marker):
  '''Q5: frozenset middle-exotic sibling sentinel gone, all sinks.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import FrameworkError

  mid = _q5_over_cap_mid(mid_marker)
  key = frozenset((mid, _Q5_SENTINEL))
  out = scrub_details({key: 'hello'})
  assert _Q5_SENTINEL not in str(out)
  assert '***REDACTED***' in str(out)
  entry = audit_log('train', 'alice', {key: 'hello'})
  assert _Q5_SENTINEL not in str(entry)
  err = FrameworkError('m', context={key: 'hello'})
  assert _Q5_SENTINEL not in str(err.to_dict())


def test_q5_nested_tuple_over_cap_sibling_redacted():
  '''Q5: over-cap leaf nested in inner tuple redacts outer sibling.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  mid = _q5_over_cap_mid('api very key')
  out = scrub_details({((mid,), _Q5_SENTINEL): 'hello'})
  assert _Q5_SENTINEL not in str(out)
  out2 = scrub_details({((mid, _Q5_SENTINEL), 'label'): 'hello'})
  assert _Q5_SENTINEL not in str(out2)


def test_q5_over_cap_innocent_tuple_sibling_redacted():
  '''Q5: over-cap leaf is unverifiable, so its sibling redacts.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  out = scrub_details({(('a' * 10000, 'keepme-sibling')): 'hello'})
  assert 'keepme-sibling' not in str(out)
  assert '***REDACTED***' in str(out)


def test_q5_bytes_middle_exotic_tuple_sentinel_redacted():
  '''Q5: bytes over-cap middle-exotic tuple sibling redacts.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details

  mid = b'a' * 10000 + b'api very key' + b'a' * 10000
  key = (mid, _Q5_SENTINEL)
  out = scrub_details({key: 'hello'})
  assert _Q5_SENTINEL not in str(out)
  entry = audit_log('train', 'alice', {key: 'hello'})
  assert _Q5_SENTINEL not in str(entry)


def test_q5_container_value_over_cap_sibling_redacted():
  '''Q5: container value over-cap sibling redacts, same class.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  mid = _q5_over_cap_mid('api very key')
  out = scrub_details({'k': (mid, _Q5_SENTINEL)})
  assert _Q5_SENTINEL not in str(out)
  out2 = scrub_details({'k': [mid, _Q5_SENTINEL]})
  assert _Q5_SENTINEL not in str(out2)


def test_q5_fp_guards_preserved():
  '''Q5: short FP guards and innocent composites are preserved.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  out = scrub_details({
      'key_cols_list': 'hello',
      'hash_col': 'world',
      'epoch': 123,
      'monkey': 'banana',
      ('ok', 'keep'): 'v',
  })
  assert out['key_cols_list'] == 'hello'
  assert out['hash_col'] == 'world'
  assert out['epoch'] == 123
  assert out['monkey'] == 'banana'
  assert out[('ok', 'keep')] == 'v'


def test_q5_short_and_plain_mid_controls_still_work():
  '''Q5: short marker tuple and plain-mid tuple still redact.'''
  from forecast_ml_framework.governance.scrub import scrub_details

  out = scrub_details({(('api very key', _Q5_SENTINEL)): 'hello'})
  assert _Q5_SENTINEL not in str(out)
  plain_mid = 'a' * 10000 + '_password' + 'a' * 10000
  out2 = scrub_details({((plain_mid, _Q5_SENTINEL)): 'hello'})
  assert _Q5_SENTINEL not in str(out2)


# --- cycle13 S14: HTTP auth schemes Basic/Digest/Negotiate (RED) ---


@pytest.mark.parametrize('text', [
    'Authorization: Basic HUNTER2ZQ9SENTINEL',
    'Authorization: Digest HUNTER2ZQ9SENTINEL',
    'Authorization: Negotiate HUNTER2ZQ9SENTINEL',
    'Authorization: basic HUNTER2ZQ9SENTINEL',
    'Authorization: DIGEST HUNTER2ZQ9SENTINEL',
    'Authorization: negotiate HUNTER2ZQ9SENTINEL',
    'Authorization: Basic hunter2',
    'Authorization: Basic HUNTER2ZQ9SENTINEL end',
    'Authorization: is Basic HUNTER2ZQ9SENTINEL',
    'Authorization: Basic token HUNTER2ZQ9SENTINEL',
])
def test_s14_authorization_scheme_scrub_text(text):
  '''S14: Authorization scheme credential is redacted in text.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert 'hunter2' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)
  assert 'Authorization:' in out, repr(text)


@pytest.mark.parametrize('text', [
    'Authorization: Basic HUNTER2ZQ9SENTINEL',
    'Authorization: Digest HUNTER2ZQ9SENTINEL',
    'Authorization: Negotiate HUNTER2ZQ9SENTINEL',
    'Basic HUNTER2ZQ9SENTINEL',
    'Digest Zq9SENTINEL7x',
    'Negotiate AAA+BBB/CCC=',
])
def test_s14_scheme_sentinel_gone_all_sinks(text):
  '''S14: scheme sentinel gone via audit and to_dict sinks.'''
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.scrub import scrub_details
  from forecast_ml_framework.telemetry.exceptions import FrameworkError

  entry = audit_log('train', 'alice', {'note': text})
  assert 'HUNTER2ZQ9SENTINEL' not in json.dumps(entry), repr(text)
  assert 'Zq9SENTINEL7x' not in json.dumps(entry), repr(text)
  assert 'AAA+BBB/CCC=' not in json.dumps(entry), repr(text)
  out = scrub_details({'headers': text, 'ok': 'keep'})
  assert 'HUNTER2ZQ9SENTINEL' not in str(out), repr(text)
  assert 'Zq9SENTINEL7x' not in str(out), repr(text)
  assert out['ok'] == 'keep'
  err = FrameworkError('m', context={'note': text})
  assert 'HUNTER2ZQ9SENTINEL' not in json.dumps(err.to_dict()), (
      repr(text))
  assert 'Zq9SENTINEL7x' not in json.dumps(err.to_dict()), repr(text)


@pytest.mark.parametrize('text', [
    'Basic HUNTER2ZQ9SENTINEL',
    'Digest HUNTER2ZQ9SENTINEL',
    'Negotiate HUNTER2ZQ9SENTINEL',
    'basic HUNTER2ZQ9SENTINEL',
    'BASIC Zq9SENTINEL7x',
    'Basic is HUNTER2ZQ9SENTINEL',
    'Basic AAA+BBB/CCC=',
])
def test_s14_bare_scheme_base64_token_scrubbed(text):
  '''S14: bare scheme plus base64-ish token is redacted.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert 'HUNTER2ZQ9SENTINEL' not in out, repr(text)
  assert 'Zq9SENTINEL7x' not in out, repr(text)
  assert 'AAA+BBB/CCC=' not in out, repr(text)
  assert '***REDACTED***' in out, repr(text)


@pytest.mark.parametrize('text', [
    'This is basic math',
    'This is basic math for beginners',
    'basic math',
    'Basic research methods are useful',
    'Basic Principles of accounting',
    'Digest the report carefully',
    'Negotiate a deal with the vendor',
    'This digest is easy to read',
])
def test_s14_scheme_fp_guards_preserved(text):
  '''S14: ordinary scheme words without a credential stay.'''
  from forecast_ml_framework.governance.scrub import scrub_text

  out = scrub_text(text)
  assert out == text, repr(text)
  assert '***REDACTED***' not in out, repr(text)


