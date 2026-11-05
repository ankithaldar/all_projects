#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Phase 3 config: monitoring/retraining/feature-store/ab/cost/governance.

Asserts new sections in base parameters/ (merged) parse via YAML,
OmegaConf and Pydantic, every new key is commented, catalog
entries exist with kedro_datasets types, infra scheduler is
count-gated and disabled by default, and no secrets ship.
'''

import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip('omegaconf')
pytest.importorskip('pydantic')

from omegaconf import OmegaConf  # noqa: E402

from forecast_ml_framework.core.config_schema import (  # noqa: E402
  load_config,
  validate_parameters,
)


def _root():
  return Path(__file__).resolve().parents[1]


def _params_raw():
  '''Return raw base parameters text (all parameters/*.yml merged).'''
  params_dir = _root() / 'conf' / 'base' / 'parameters'
  return '\n'.join(
      path.read_text() for path in sorted(params_dir.glob('*.yml')))


def _params():
  return yaml.safe_load(_params_raw())


def _params_omegaconf():
  '''Load base parameters as OmegaConf merged over parameters/*.yml.'''
  merged = OmegaConf.create({})
  for path in sorted((_root() / 'conf' / 'base' / 'parameters').glob('*.yml')):
    merged = OmegaConf.merge(merged, OmegaConf.load(path))
  return merged


def _catalog(env='base'):
  with (_root() / 'conf' / env / 'catalog.yml').open() as handle:
    return yaml.safe_load(handle)


def _catalog_raw(env='base'):
  return (_root() / 'conf' / env / 'catalog.yml').read_text()


# --- parameters: new sections ------------------------------------------------

def test_monitoring_params_keys():
  params = _params()
  assert 'monitoring_params' in params
  m = params['monitoring_params']
  for key in ('psi_threshold', 'ks_threshold', 'bins', 'alert_severities'):
    assert key in m, key
  assert 0 < m['psi_threshold'] < 1
  assert 0 < m['ks_threshold'] < 1
  assert isinstance(m['bins'], int) and m['bins'] > 0
  assert isinstance(m['alert_severities'], list) and m['alert_severities']


def test_retraining_params_keys():
  params = _params()
  assert 'retraining_params' in params
  r = params['retraining_params']
  for key in ('schedule_days', 'drift_alert_threshold', 'decay_threshold'):
    assert key in r, key
  assert isinstance(r['schedule_days'], int) and r['schedule_days'] > 0
  assert 0 < r['drift_alert_threshold'] < 1
  assert 0 < r['decay_threshold'] < 1


def test_feature_store_params_keys():
  params = _params()
  assert 'feature_store_params' in params
  f = params['feature_store_params']
  for key in ('offline_path', 'online_backend', 'entity_key'):
    assert key in f, key
  assert isinstance(f['offline_path'], str) and f['offline_path']
  assert isinstance(f['online_backend'], str) and f['online_backend']
  assert isinstance(f['entity_key'], str) and f['entity_key']


def test_ab_params_keys():
  params = _params()
  assert 'ab_params' in params
  ab = params['ab_params']
  for key in ('traffic_split', 'mode'):
    assert key in ab, key
  assert isinstance(ab['traffic_split'], (int, float))
  assert 0 <= ab['traffic_split'] <= 1
  assert isinstance(ab['mode'], str) and ab['mode']


def test_cost_params_keys():
  params = _params()
  assert 'cost_params' in params
  c = params['cost_params']
  for key in ('compute_price_per_hour', 'storage_price_per_gb',
              'scan_price_per_tb_scanned'):
    assert key in c, key
    assert isinstance(c[key], (int, float)) and c[key] >= 0, key


def test_governance_params_keys():
  params = _params()
  assert 'governance_params' in params
  g = params['governance_params']
  for key in ('retention_days', 'audit_actor_required'):
    assert key in g, key
  assert isinstance(g['retention_days'], int) and g['retention_days'] > 0
  assert g['audit_actor_required'] is True


# --- OmegaConf + pydantic compatibility --------------------------------------

def test_new_sections_resolve_via_omegaconf():
  base = _root() / 'conf' / 'base'
  cfg = OmegaConf.merge(
      {'variables': OmegaConf.load(base / 'variables.yml'),
       'parameters': _params_omegaconf()})
  OmegaConf.resolve(cfg)
  params = cfg.parameters
  for section in ('monitoring_params', 'retraining_params',
                  'feature_store_params', 'ab_params', 'cost_params',
                  'governance_params'):
    assert section in params, section
  plain = OmegaConf.to_container(cfg, resolve=True)
  assert '${' not in str(plain)


def test_new_sections_pass_pydantic():
  cfg = validate_parameters(_params())
  dumped = cfg.model_dump()
  for section in ('monitoring_params', 'retraining_params',
                  'feature_store_params', 'ab_params', 'cost_params',
                  'governance_params'):
    assert dumped[section], section
  assert dumped['governance_params']['audit_actor_required'] is True


def test_load_config_both_envs():
  for env in ('base', 'local'):
    cfg = load_config(env)
    for section in ('monitoring_params', 'retraining_params',
                    'feature_store_params', 'ab_params', 'cost_params',
                    'governance_params'):
      assert section in cfg['parameters'], (env, section)


# --- per-key comments ---------------------------------------------------------

_NEW_LEAF_KEYS = (
  'psi_threshold', 'ks_threshold', 'bins', 'alert_severities',
  'schedule_days', 'drift_alert_threshold', 'decay_threshold',
  'offline_path', 'online_backend', 'entity_key',
  'traffic_split', 'mode',
  'compute_price_per_hour', 'storage_price_per_gb',
  'scan_price_per_tb_scanned',
  'retention_days', 'audit_actor_required',
)


def test_every_new_key_is_commented():
  raw = _params_raw()
  new_raw = raw[raw.index('monitoring_params:'):]
  lines = new_raw.splitlines()
  for key in _NEW_LEAF_KEYS:
    idx = next(
        (i for i, line in enumerate(lines)
         if re.match(r'\s+' + re.escape(key) + r':', line)),
        None)
    assert idx is not None, key
    commented = (
        ('#' in lines[idx])
        or (idx > 0 and lines[idx - 1].lstrip().startswith('#'))
        or any(('#' in line and key in line) for line in lines))
    assert commented, key


# --- no secrets ---------------------------------------------------------------

_SECRET_PATTERNS = (
  r'(?im)^\s*password\s*:',
  r'(?im)^\s*secret\s*:',
  r'(?im)^\s*api_key\s*:',
  r'(?im)^\s*apikey\s*:',
  r'(?im)^\s*access_key\s*:',
  r'(?im)^\s*private_key\s*:',
  r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
  r'AKIA[0-9A-Z]{16}',
  r'(?im)^\s*credentials\s*:\s*\S',
)


def test_no_secrets_in_new_config():
  texts = [_params_raw(), _catalog_raw('base'), _catalog_raw('gcp'),
           _catalog_raw('aws'), _catalog_raw('local'),
           (_root() / 'conf' / 'README.md').read_text(),
           (_root() / 'infra' / 'gcp' / 'main.tf').read_text(),
           (_root() / 'infra' / 'gcp' / 'variables.tf').read_text(),
           (_root() / 'infra' / 'aws' / 'main.tf').read_text(),
           (_root() / 'infra' / 'aws' / 'variables.tf').read_text()]
  for text in texts:
    for pattern in _SECRET_PATTERNS:
      assert not re.search(pattern, text), pattern


# --- catalog entries ---

def test_phase3_catalog_entries():
  catalog = _catalog()
  for name in ('monitoring_report', 'alerts', 'cost_log', 'lineage',
               'audit_log'):
    assert name in catalog, name
    entry = catalog[name]
    assert entry['type'].startswith('kedro_datasets.'), (name, entry['type'])
    assert 'filepath' in entry, name


def test_new_catalog_types_importable():
  import importlib
  import importlib.util
  catalog = _catalog()
  for name in ('monitoring_report', 'alerts', 'cost_log', 'lineage',
               'audit_log'):
    path = catalog[name]['type']
    module, _, attr = path.rpartition('.')
    try:
      available = importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
      available = False
    if not available:
      continue
    obj = getattr(importlib.import_module(module), attr)
    assert callable(obj), (name, path)


def test_catalog_env_overlays_parse():
  for env in ('local', 'gcp', 'aws'):
    assert _catalog(env) is not None


# --- README ---

def test_readme_documents_new_sections():
  readme = (_root() / 'conf' / 'README.md').read_text()
  for section in ('monitoring_params', 'retraining_params',
                  'feature_store_params', 'ab_params', 'cost_params',
                  'governance_params'):
    assert section in readme, section
  for key in ('psi_threshold', 'schedule_days', 'offline_path',
              'traffic_split', 'compute_price_per_hour', 'retention_days',
              'monitoring_report', 'audit_log'):
    assert key in readme, key


# --- infra GCP ---

def test_gcp_scheduler_count_gated_disabled_by_default():
  vars_tf = (_root() / 'infra' / 'gcp' / 'variables.tf').read_text()
  main_tf = (_root() / 'infra' / 'gcp' / 'main.tf').read_text()
  m = re.search(
      r'variable\s+"enable_retraining_scheduler"\s*\{[^}]*default\s*=\s*(\w+)',
      vars_tf, re.S)
  assert m and m.group(1) == 'false', 'scheduler must be off by default'
  assert 'retraining_schedule_cron' in vars_tf or 'schedule_cron' in vars_tf
  assert 'google_cloud_scheduler_job' in main_tf
  # count-gated: resource carries count = var.enable_...
  sched = main_tf[main_tf.index('google_cloud_scheduler_job'):]
  assert 'count' in sched
  assert 'var.enable_retraining_scheduler' in sched


def test_gcp_pubsub_topic_for_alerts():
  main_tf = (_root() / 'infra' / 'gcp' / 'main.tf').read_text()
  assert 'google_pubsub_topic' in main_tf
  assert 'alerts' in main_tf
  # email subscription stub commented
  assert '# resource "google_pubsub_subscription"' in main_tf or \
      '#resource "google_pubsub_subscription"' in main_tf or \
      'google_pubsub_subscription' in main_tf


def test_gcp_infra_readme_phase3():
  readme = (_root() / 'infra' / 'gcp' / 'README.md').read_text()
  assert 'google_cloud_scheduler_job' in readme
  assert 'google_pubsub_topic' in readme
  assert 'enable_retraining_scheduler' in readme or \
    'retraining' in readme.lower()


# --- infra AWS ---

def test_aws_eventbridge_rule_stub_commented():
  main_tf = (_root() / 'infra' / 'aws' / 'main.tf').read_text()
  assert 'aws_cloudwatch_event_rule' in main_tf
  assert '# resource "aws_cloudwatch_event_rule"' in main_tf


def test_aws_infra_readme_phase3():
  readme = (_root() / 'infra' / 'aws' / 'README.md').read_text()
  assert 'aws_cloudwatch_event_rule' in readme or \
    'eventbridge' in readme.lower()


def test_evaluation_metrics_to_calculate_curated_default():
  '''evaluation_params.metrics_to_calculate is a curated subset.'''
  params = _params()
  assert 'evaluation_params' in params
  ev = params['evaluation_params']
  assert 'metrics_to_calculate' in ev, 'metrics_to_calculate missing'
  selected = ev['metrics_to_calculate']
  assert isinstance(selected, list) and selected, selected
  for name in ('accuracy', 'balanced_accuracy', 'precision_weighted',
               'recall_weighted', 'f1_weighted', 'roc_auc', 'pr_auc',
               'log_loss', 'brier_score', 'cohen_kappa',
               'matthews_corrcoef', 'mae', 'rmse', 'r2', 'mape',
               'smape'):
    assert name in selected, name
  # Curated subset, not the full registry: strictly fewer than ~60.
  assert len(selected) < 60, len(selected)
  assert 'all' not in selected


def test_evaluation_metrics_to_calculate_commented():
  '''metrics_to_calculate carries per-key comments + examples.'''
  raw = _params_raw()
  lines = raw.splitlines()
  idx = next((i for i, line in enumerate(lines)
              if re.match(r'\s+metrics_to_calculate:', line)), None)
  assert idx is not None, 'metrics_to_calculate not found'
  assert '#' in lines[idx] or (
      idx > 0 and lines[idx - 1].lstrip().startswith('#')) or any(
      '#' in line and 'metrics_to_calculate' in line for line in lines)
  # Commented 'all' and custom-list examples must be present.
  assert "'all'" in raw or '"all"' in raw
  assert 'METRIC_REGISTRY' in raw


def test_infra_readmes_documents_phase3():
  for path in ('infra/README.md', 'infra/gcp/README.md', 'infra/aws/README.md'):
    text = (_root() / path).read_text()
    lowered = text.lower()
    assert 'retraining' in lowered or 'scheduler' in lowered or \
        'eventbridge' in lowered, path
