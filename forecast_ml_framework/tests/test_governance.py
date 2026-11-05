#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.governance (phase 3).

Lineage (model + stable config hash), audit (actor required,
secret redaction, injectable clock), and retention (keep/delete
split by age with injected now). Pure stdlib; runs in every env.
'''

import hashlib
import json
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
