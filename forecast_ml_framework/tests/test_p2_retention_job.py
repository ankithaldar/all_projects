#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''P2-S1 tests: retention job, erasure channel, secrets wiring.

TDD RED first: ``governance.retention_job`` does not exist
yet, so the imports inside each test fail for the right
reason (feature missing) until GREEN lands. Real stores
are used for sweep and erasure; mocks are not used.
'''

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
CLEAR = 'SECRET-SUBJECT-S1-9f3a-ENTITY'
ACTOR = 'SECRET-ACTOR-S1-77bc'
OTHER = 'other-entity-s1-1'
_ROOT = Path(__file__).resolve().parents[1]
_PLATFORM = _ROOT / 'conf' / 'base' / 'platform.yml'
_CRONJOB = _ROOT / 'deploy' / 'k8s' / 'retention-cronjob.yaml'
_JOB_SRC = (_ROOT / 'src' / 'forecast_ml_framework' / 'governance'
            / 'retention_job.py')


def _write_params(tmp_path, body):
  '''Write a governance params file and return its path.'''
  path = tmp_path / 'platform.yml'
  path.write_text(body, encoding='utf-8')
  return str(path)


def _real_stores(tmp_path):
  '''Build real offline and online stores under tmp_path.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
    LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
    OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  offline = OfflineFeatureStore(LocalArtifactStore(str(tmp_path)))
  offline.materialize(
      pd.DataFrame({'entity_id': [CLEAR, OTHER], 'f1': [1.0, 2.0]}),
      as_of='2020-01-01', pii_classes=['none'])
  offline.materialize(
      pd.DataFrame({'entity_id': [OTHER], 'f1': [3.0]}),
      as_of='2026-11-01', pii_classes=['none'])
  online = OnlineFeatureStore()
  online.set_features(CLEAR, {'f1': 1.0}, as_of='2020-01-01',
                      pii_classes=['none'])
  online.set_features(OTHER, {'f1': 2.0}, as_of='2026-11-01',
                      pii_classes=['none'])
  return offline, online


def test_load_governance_params_from_platform_yml():
  '''Loader reads governance_params retention days.'''
  from forecast_ml_framework.governance.retention_job import (
    load_governance_params)
  params = load_governance_params(str(_PLATFORM))
  assert isinstance(params['retention_days'], int)
  assert params['retention_days'] == 365
  raw = yaml.safe_load(_PLATFORM.read_text(encoding='utf-8'))
  assert raw['governance_params']['retention_days'] == 365


def test_load_governance_params_rejects_bad_files(tmp_path):
  '''Loader rejects missing, malformed, and non-int params.'''
  from forecast_ml_framework.governance.retention_job import (
    load_governance_params)
  with pytest.raises(ValueError):
    load_governance_params(str(tmp_path / 'missing.yml'))
  with pytest.raises(ValueError):
    load_governance_params(
        _write_params(tmp_path, 'other:\n  retention_days: 30\n'))
  with pytest.raises(ValueError):
    load_governance_params(
        _write_params(tmp_path, 'governance_params:\n  other: 30\n'))
  with pytest.raises(ValueError):
    load_governance_params(_write_params(
        tmp_path, 'governance_params:\n  retention_days: thirty\n'))
  with pytest.raises(ValueError):
    load_governance_params(_write_params(
        tmp_path, 'governance_params:\n  retention_days: 30.5\n'))
  with pytest.raises(ValueError):
    load_governance_params(_write_params(
        tmp_path, 'governance_params:\n  retention_days: true\n'))
  with pytest.raises(ValueError):
    load_governance_params(_write_params(
        tmp_path, 'governance_params:\n  retention_days: -1\n'))
  with pytest.raises(ValueError):
    load_governance_params(
        _write_params(tmp_path, 'governance_params: [30]\n'))


def test_build_stores_returns_real_stores(tmp_path):
  '''Build stores returns real offline and online stores.'''
  from forecast_ml_framework.governance.retention_job import build_stores
  from forecast_ml_framework.feature_store.offline import (
    OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  stores = build_stores({'retention_days': 30,
                         'artifact_base_path': str(tmp_path)})
  assert len(stores) == 2
  assert isinstance(stores[0], OfflineFeatureStore)
  assert isinstance(stores[1], OnlineFeatureStore)


def test_build_stores_unknown_store_raises(tmp_path):
  '''Build stores raises on an unknown store type.'''
  from forecast_ml_framework.governance.retention_job import build_stores
  with pytest.raises(ValueError):
    build_stores({'retention_days': 30,
                  'stores': [{'type': 'bogus'}]})
  with pytest.raises(ValueError):
    build_stores({'retention_days': 30, 'store_type': 'bogus'})
  with pytest.raises(ValueError):
    build_stores({'retention_days': 30, 'stores': 'offline'})
  with pytest.raises(ValueError):
    build_stores({'retention_days': 30, 'stores': [42]})
  with pytest.raises(ValueError):
    build_stores('not-a-dict')


def test_build_stores_store_type_variants(tmp_path):
  '''Build stores honors offline, online, and both specs.'''
  from forecast_ml_framework.governance.retention_job import build_stores
  from forecast_ml_framework.feature_store.offline import (
    OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  base = {'retention_days': 30, 'artifact_base_path': str(tmp_path)}
  only_offline = build_stores({**base, 'store_type': 'offline'})
  assert len(only_offline) == 1
  assert isinstance(only_offline[0], OfflineFeatureStore)
  only_online = build_stores({**base, 'store_type': 'online'})
  assert len(only_online) == 1
  assert isinstance(only_online[0], OnlineFeatureStore)
  both = build_stores({**base, 'store_type': 'both'})
  assert len(both) == 2
  named = build_stores({**base, 'stores': ['online', 'offline']})
  assert isinstance(named[0], OnlineFeatureStore)
  assert isinstance(named[1], OfflineFeatureStore)


def test_load_governance_params_invalid_yaml(tmp_path):
  '''Loader rejects invalid YAML and non-mapping files.'''
  from forecast_ml_framework.governance.retention_job import (
    load_governance_params)
  with pytest.raises(ValueError):
    load_governance_params(
        _write_params(tmp_path, 'governance_params: [unclosed\n'))
  with pytest.raises(ValueError):
    load_governance_params(_write_params(tmp_path, 'just a string\n'))


def test_run_erasure_rejects_blank_inputs():
  '''Erasure wrapper rejects blank subject and actor.'''
  from forecast_ml_framework.governance.retention_job import run_erasure
  with pytest.raises(ValueError):
    run_erasure(subject_id='  ', stores=[], artifacts={},
                actor=ACTOR)
  with pytest.raises(ValueError):
    run_erasure(subject_id=CLEAR, stores=[], artifacts={},
                actor='')


def test_run_retention_sweep_real_stores(tmp_path, caplog):
  '''Sweep deletes expired rows and audits hashed actor.'''
  from forecast_ml_framework.governance.retention_job import (
    run_retention_sweep)
  from forecast_ml_framework.governance.scrub import subject_hash
  offline, online = _real_stores(tmp_path)
  log = []
  with caplog.at_level(logging.WARNING):
    result = run_retention_sweep(
        stores=[offline, online],
        params={'retention_days': 365}, actor=ACTOR, now=NOW, log=log)
  assert result['swept'] == 2
  assert result['kept'] == 2
  assert len(result['audit_entries']) == 2
  assert log == result['audit_entries']
  assert offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) == {}
  assert not offline.get_features(
      [OTHER], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [OTHER], as_of='2026-11-08', strict_pit=True) != {}
  assert ACTOR not in str(result)
  assert ACTOR not in str(log)
  assert CLEAR not in str(result)
  assert CLEAR not in caplog.text
  assert result['audit_entries'][0]['actor'] == subject_hash(ACTOR)


def test_run_retention_sweep_dry_run_deletes_nothing(tmp_path):
  '''Dry run plans only and deletes nothing.'''
  from forecast_ml_framework.governance.retention_job import (
    run_retention_sweep)
  offline, online = _real_stores(tmp_path)
  result = run_retention_sweep(
      stores=[offline, online],
      params={'retention_days': 365}, actor=ACTOR, now=NOW,
      dry_run=True)
  assert result['swept'] == 0
  assert not offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) != {}


def test_run_retention_sweep_unknown_store_aborts():
  '''Unknown store aborts the sweep with a partial audit.'''
  from forecast_ml_framework.governance.retention_job import (
    run_retention_sweep)
  log = []
  with pytest.raises(ValueError):
    run_retention_sweep(
        stores=[object()], params={'retention_days': 30},
        actor=ACTOR, now=NOW, log=log)
  assert log and log[0]['details']['partial'] is True
  assert ACTOR not in str(log)
  with pytest.raises(ValueError):
    run_retention_sweep(
        stores=[], params={'retention_days': 30}, actor='')


def test_run_erasure_real_stores_marker_is_hash(tmp_path, caplog):
  '''Erasure deletes rows and marker equals the subject hash.'''
  from forecast_ml_framework.governance.retention_job import run_erasure
  from forecast_ml_framework.governance.scrub import subject_hash
  offline, online = _real_stores(tmp_path)
  registry = {}
  envelope = {}
  artifacts = {'fitted_registry': registry, 'best_model': envelope}
  log = []
  with caplog.at_level(logging.WARNING):
    report = run_erasure(
        subject_id=CLEAR, stores=[offline, online],
        artifacts=artifacts, actor=ACTOR, log=log,
        at='2026-11-08T00:00:00+00:00')
  digest = subject_hash(CLEAR)
  assert report['subject_hash'] == digest
  assert registry['__meta__']['erasure']['subject_hash'] == digest
  assert envelope['erasure']['subject_hash'] == digest
  assert offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) == {}
  assert CLEAR not in str(report)
  assert ACTOR not in str(report)
  assert CLEAR not in str(log)
  assert CLEAR not in caplog.text


def test_main_erase_requires_env_subject_id(tmp_path, monkeypatch,
                                            capsys):
  '''Erase without the env subject id exits 2, no mutation.'''
  from forecast_ml_framework.governance import retention_job
  offline, online = _real_stores(tmp_path)
  monkeypatch.setattr(retention_job, 'build_stores',
                      lambda params: [offline, online])
  monkeypatch.delenv('FORECAST_ERASE_SUBJECT_ID', raising=False)
  params_path = _write_params(
      tmp_path, 'governance_params:\n  retention_days: 365\n')
  code = retention_job.main(
      ['--erase', '--governance-params', params_path])
  assert code == 2
  assert CLEAR not in capsys.readouterr().out
  assert not offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty


def test_main_erase_uses_env_subject_id(tmp_path, monkeypatch,
                                        capsys):
  '''Erase reads the clear id from env only, prints hash.'''
  from forecast_ml_framework.governance import retention_job
  from forecast_ml_framework.governance.scrub import subject_hash
  offline, online = _real_stores(tmp_path)
  monkeypatch.setattr(retention_job, 'build_stores',
                      lambda params: [offline, online])
  monkeypatch.setenv('FORECAST_ERASE_SUBJECT_ID', CLEAR)
  monkeypatch.setenv('FORECAST_ACTOR', ACTOR)
  params_path = _write_params(
      tmp_path, 'governance_params:\n  retention_days: 365\n')
  code = retention_job.main(
      ['--erase-subject', '--governance-params', params_path])
  out = capsys.readouterr().out
  assert code == 0
  assert subject_hash(CLEAR) in out
  assert CLEAR not in out
  assert ACTOR not in out
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) == {}


def test_main_erase_dry_run_combo_rejected(tmp_path, monkeypatch):
  '''Erase combined with dry run exits 2 without mutation.'''
  from forecast_ml_framework.governance import retention_job
  offline, online = _real_stores(tmp_path)
  monkeypatch.setattr(retention_job, 'build_stores',
                      lambda params: [offline, online])
  monkeypatch.setenv('FORECAST_ERASE_SUBJECT_ID', CLEAR)
  params_path = _write_params(
      tmp_path, 'governance_params:\n  retention_days: 365\n')
  code = retention_job.main(
      ['--erase', '--dry-run', '--governance-params', params_path])
  assert code == 2
  assert not offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty


def test_main_retention_dry_run_and_bad_params(tmp_path, capsys):
  '''CLI retention dry run exits 0; bad params exit 2.'''
  from forecast_ml_framework.governance import retention_job
  offline, online = _real_stores(tmp_path)
  original = retention_job.build_stores
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setattr(retention_job, 'build_stores',
                      lambda params: [offline, online])
  try:
    params_path = _write_params(
        tmp_path, 'governance_params:\n  retention_days: 365\n')
    code = retention_job.main(
        ['--retention', '--dry-run', '--governance-params',
         params_path])
    assert code == 0
    assert CLEAR not in capsys.readouterr().out
    assert not offline.get_features(
        [CLEAR], as_of='2026-11-08', strict_pit=True).empty
    code = retention_job.main(
        ['--retention', '--governance-params',
         str(tmp_path / 'missing.yml')])
    assert code == 2
  finally:
    monkeypatch.undo()


def test_retention_job_source_guards():
  '''Source has env-only erasure and no forbidden wiring.'''
  text = _JOB_SRC.read_text(encoding='utf-8')
  assert text.startswith('#!/usr/bin/env python')
  assert 'D-ERASURE-CLI-1' in text
  assert 'FORECAST_ERASE_SUBJECT_ID' in text
  assert 'subject_hash' in text
  assert 'hashlib' not in text
  assert 'def subject_hash' not in text
  assert '--erase-subject-' + 'hash' not in text
  assert 'rot' + 'ate' not in text.lower()
  assert '--governance-params' in text
  for line in text.splitlines():
    assert len(line) <= 80, line


def test_platform_yml_shape():
  '''Platform file carries only governance retention days.'''
  raw = yaml.safe_load(_PLATFORM.read_text(encoding='utf-8'))
  assert set(raw) == {'governance_params'}
  assert set(raw['governance_params']) == {'retention_days'}


def test_cronjob_manifest():
  '''CronJob runs the retention CLI daily as forecast-ml.'''
  raw = yaml.safe_load(_CRONJOB.read_text(encoding='utf-8'))
  assert raw['kind'] == 'CronJob'
  assert raw['spec']['schedule'] == '0 3 * * *'
  template = raw['spec']['jobTemplate']['spec']['template']
  assert template['spec']['serviceAccountName'] == 'forecast-ml'
  container = template['spec']['containers'][0]
  joined = ' '.join(str(part) for part in container['command'])
  assert 'retention_job' in joined
  assert '--governance-params' in joined
  text = _CRONJOB.read_text(encoding='utf-8')
  assert 'value:' not in text or 'FORECAST_ACTOR' in text
  assert 'FORECAST_ERASE_SUBJECT_ID' not in text
  assert '--erase' not in joined


def test_secrets_tf_references_only():
  '''Secrets wiring is references only, shell per cloud.'''
  gcp = (_ROOT / 'infra' / 'gcp' / 'secrets.tf').read_text(
      encoding='utf-8')
  aws = (_ROOT / 'infra' / 'aws' / 'secrets.tf').read_text(
      encoding='utf-8')
  azure = (_ROOT / 'infra' / 'azure' / 'secrets.tf').read_text(
      encoding='utf-8')
  assert 'google_secret_manager_secret' in gcp
  assert 'google_secret_manager_secret_iam_member' in gcp
  assert 'aws_secretsmanager_secret' in aws
  assert 'aws_secretsmanager_secret_' + 'version' not in aws
  assert 'azurerm_key_vault_secret' in azure
  assert 'data ' in azure
  for text in (gcp, aws, azure):
    assert 'REPLACE' + '-ME' not in text
    assert 'secret_' + 'version' not in text
    assert 'residency' not in text
    assert 'rot' + 'ate' not in text
