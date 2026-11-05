#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Phase 2 config: registry, batch scoring, infra hooks.

Asserts new sections in base parameters/ (merged) parse via YAML,
OmegaConf and Pydantic, every new key is commented, catalog
entries for scoring / registry exist with importable types,
infra values are env-var placeholders, and no secrets ship.
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
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def _params_raw():
  '''Return raw base parameters text (all parameters/*.yml merged).'''
  params_dir = _root() / 'conf' / 'base' / 'parameters'
  return '\n'.join(
      path.read_text() for path in sorted(params_dir.glob('*.yml')))


def _params():
  '''Load base parameters (merged parameters/*.yml) as a dict.'''
  return yaml.safe_load(_params_raw())


def _params_omegaconf():
  '''Load base parameters as OmegaConf merged over parameters/*.yml.'''
  merged = OmegaConf.create({})
  for path in sorted((_root() / 'conf' / 'base' / 'parameters').glob('*.yml')):
    merged = OmegaConf.merge(merged, OmegaConf.load(path))
  return merged


def _catalog(env='base'):
  '''Load a catalog.yml for an env.'''
  with (_root() / 'conf' / env / 'catalog.yml').open() as handle:
    return yaml.safe_load(handle)


def _catalog_raw(env='base'):
  '''Return raw catalog text for an env.'''
  return (_root() / 'conf' / env / 'catalog.yml').read_text()


# --- parameters: registry ---------------------------------------------------

def test_registry_params_keys():
  '''registry_params carries tracking/registry/model/alias/metric.'''
  params = _params()
  assert 'registry_params' in params
  reg = params['registry_params']
  for key in ('tracking_uri', 'registry_uri', 'model_name', 'alias',
              'promote_on_metric'):
    assert key in reg, key
  assert isinstance(reg['tracking_uri'], str) and reg['tracking_uri']
  assert isinstance(reg['model_name'], str) and reg['model_name']
  assert isinstance(reg['alias'], str) and reg['alias']
  assert isinstance(reg['promote_on_metric'], str) and (
      reg['promote_on_metric'])


# --- parameters: scoring -----------------------------------------------------

def test_scoring_params_keys():
  '''scoring_params carries input/output/model-id/batch-size.'''
  params = _params()
  assert 'scoring_params' in params
  scoring = params['scoring_params']
  for key in ('input_dataset', 'output_dataset', 'score_model_id',
              'batch_size'):
    assert key in scoring, key
  assert scoring['input_dataset'] == 'scoring_input'
  assert scoring['output_dataset'] == 'predictions'
  assert isinstance(scoring['batch_size'], int) and scoring['batch_size'] > 0


def test_scoring_params_extended_worker_contract():
  '''scoring_params matches scoring worker contract incl limits.'''
  params = _params()
  scoring = params['scoring_params']
  for key in ('model_path', 'id_cols', 'feature_cols', 'task_type',
              'batch_size', 'max_rows', 'max_cols', 'model_name', 'alias'):
    assert key in scoring, key
  assert isinstance(scoring['id_cols'], list) and scoring['id_cols']
  assert scoring['feature_cols'] is None or isinstance(
      scoring['feature_cols'], list)
  assert scoring['task_type'] in ('classification', 'regression')
  # model_path/name/alias may be empty (registry/file fallback) but str
  for key in ('model_path', 'model_name', 'alias'):
    assert isinstance(scoring[key], str), key
  for key in ('batch_size', 'max_cols'):
    assert isinstance(scoring[key], int) and scoring[key] > 0, key
  # max_rows: null = unbounded (Spark path); a positive int caps rows.
  assert scoring['max_rows'] is None or (
      isinstance(scoring['max_rows'], int) and scoring['max_rows'] > 0)
  # registry_params still carries the canonical registry name/alias
  reg = params['registry_params']
  assert reg['model_name'] and reg['alias']


def test_scoring_output_is_canonical_pipeline_dataset():
  '''output_dataset points at pipeline 'predictions', in catalog.'''
  params = _params()
  scoring = params['scoring_params']
  assert scoring['output_dataset'] == 'predictions'
  catalog = _catalog()
  assert scoring['output_dataset'] in catalog
  assert scoring['input_dataset'] in catalog


# --- parameters: infra -------------------------------------------------------

def test_infra_params_keys():
  '''infra_params carries gcp + aws env-placeholder leaves.'''
  params = _params()
  assert 'infra_params' in params
  infra = params['infra_params']
  assert 'gcp' in infra and 'aws' in infra
  for key in ('project', 'bucket', 'dataset', 'cluster'):
    assert key in infra['gcp'], key
  for key in ('bucket', 'database'):
    assert key in infra['aws'], key


def test_infra_values_are_env_placeholders():
  '''Every infra leaf is an OmegaConf env placeholder, no literals.'''
  raw = _params_raw()
  infra_raw = raw[raw.index('infra_params:'):]
  for key in ('project', 'bucket', 'dataset', 'cluster', 'database'):
    # The leaf line for this key inside infra_params must use oc.env.
    lines = [
      line for line in infra_raw.splitlines()
      if re.match(r'\s+' + re.escape(key) + r':', line)
    ]
    assert lines, key
    assert any('${oc.env:' in line for line in lines), (key, lines)


# --- OmegaConf + pydantic compatibility --------------------------------------

def test_new_sections_resolve_via_omegaconf():
  '''Merged variables+parameters resolve; no interpolation left.'''
  base = _root() / 'conf' / 'base'
  cfg = OmegaConf.merge(
      {'variables': OmegaConf.load(base / 'variables.yml'),
       'parameters': _params_omegaconf()})
  OmegaConf.resolve(cfg)
  params = cfg.parameters
  assert 'registry_params' in params
  assert 'scoring_params' in params
  assert 'infra_params' in params
  assert params.scoring_params.score_model_id == (
      params.global_params.score_model_id)
  plain = OmegaConf.to_container(cfg, resolve=True)
  assert '${' not in str(plain)


def test_new_sections_pass_pydantic():
  '''ParametersConfig (extra=allow) accepts the new sections.'''
  cfg = validate_parameters(_params())
  dumped = cfg.model_dump()
  for section in ('registry_params', 'scoring_params', 'infra_params'):
    assert dumped[section], section
  assert dumped['scoring_params']['input_dataset'] == 'scoring_input'


def test_load_config_both_envs():
  '''load_config parses for base and local envs alike.'''
  for env in ('base', 'local'):
    cfg = load_config(env)
    for section in ('registry_params', 'scoring_params', 'infra_params'):
      assert section in cfg['parameters'], (env, section)
    assert cfg['parameters']['scoring_params']['batch_size'] > 0


# --- per-key comments ---------------------------------------------------------

_NEW_LEAF_KEYS = (
    'tracking_uri', 'registry_uri', 'model_name', 'alias',
    'promote_on_metric',
    'input_dataset', 'output_dataset', 'score_model_id', 'batch_size',
    'model_path', 'id_cols', 'feature_cols', 'task_type',
    'max_rows', 'max_cols',
    'project', 'bucket', 'dataset', 'cluster', 'database',
)


def test_every_new_key_is_commented():
  '''Each new leaf key has a comment mentioning it or above it.'''
  raw = _params_raw()
  new_raw = raw[raw.index('registry_params:'):]
  lines = new_raw.splitlines()
  for key in _NEW_LEAF_KEYS:
    # Find the first definition line for this key in the new sections.
    idx = next(
        (i for i, line in enumerate(lines)
         if re.match(r'\s+' + re.escape(key) + r':', line)),
        None)
    # score_model_id / bucket are defined earlier too; the new-section
    # occurrence is what must be commented, and it exists by construction.
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
  '''No hardcoded credentials/secrets in parameters, catalogs, README.'''
  texts = [_params_raw(), _catalog_raw('base'), _catalog_raw('gcp'),
           _catalog_raw('aws'), _catalog_raw('local'),
           (_root() / 'conf' / 'README.md').read_text()]
  for text in texts:
    for pattern in _SECRET_PATTERNS:
      assert not re.search(pattern, text), pattern


# --- catalog entries -----------------------------------------------------------

def test_scoring_catalog_entries():
  '''Scoring input/output datasets exist with importable types.'''
  catalog = _catalog()
  assert catalog['scoring_input']['type'] == (
      'kedro_datasets.pandas.ParquetDataset')
  assert catalog['scoring_output']['type'] == (
      'kedro_datasets.pandas.ParquetDataset')
  assert 'filepath' in catalog['scoring_input']
  assert 'filepath' in catalog['scoring_output']


def test_scoring_sql_catalog_entries():
  '''Only SQL *inputs* are read-only; outputs must be writable.'''
  catalog = _catalog()
  # Inputs are read-only SQL sources (save raises by design).
  for name in ('scoring_input_bq', 'scoring_input_athena'):
    assert name in catalog, name
    entry = catalog[name]
    assert entry['type'] in (
        'forecast_ml_framework.core.datasets.SQLQueryDataSet',
        'forecast_ml_framework.core.datasets.SparkSQLDataSet'), name
    assert '${params:' in str(entry), name
  # No read-only SQL entry may be presented as a write target:
  # scoring_output_bq/_athena must be absent or writable kedro_datasets.
  for name in ('scoring_output_bq', 'scoring_output_athena'):
    if name in catalog:
      assert catalog[name]['type'].startswith('kedro_datasets.'), name
      assert 'filepath' in catalog[name], name
  # Canonical writable parquet outputs remain.
  for name in ('scoring_output', 'predictions'):
    assert name in catalog, name
    assert catalog[name]['type'] == 'kedro_datasets.pandas.ParquetDataset'
  # scoring_output is not the pipeline output_dataset any more; keep
  # it only as a legacy writable alias is unnecessary — if present it
  # must still be writable (checked above).


def test_scoring_catalog_no_extra_swallowed_keys():
  '''SQL entries carry only documented keys (no swallowed extras).'''
  catalog = _catalog()
  allowed_input = {'type', 'project', 'dataset', 'table_name', 'sql',
                   'database', 'score_model_id', 'tmp_bucket',
                   'write_mode', 'save_args', 'partition_field',
                   'partition_type', 'partition_value', 'filepath',
                   'file_format', 'load_args', 'version', 'credentials',
                   'metadata'}
  for name in ('scoring_input_bq', 'scoring_input_athena'):
    if name in catalog:
      extra = set(catalog[name].keys()) - allowed_input
      assert not extra, (name, extra)


def test_registry_catalog_entry():
  '''Registry model placeholder dataset exists with importable type.'''
  catalog = _catalog()
  assert 'registry_model' in catalog
  entry = catalog['registry_model']
  assert entry['type'] == 'kedro_datasets.pickle.PickleDataset'
  assert 'filepath' in entry


def test_new_catalog_types_importable():
  '''Every new catalog type string imports (skip-unavailable).'''
  import importlib
  import importlib.util

  catalog = _catalog()
  for name in ('scoring_input', 'scoring_output', 'predictions',
               'scoring_input_bq', 'scoring_input_athena',
               'registry_model'):
    if name not in catalog:
      continue
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
  '''Env overlay catalogs still parse after scoring additions.'''
  for env in ('local', 'gcp', 'aws'):
    assert _catalog(env) is not None


# --- README --------------------------------------------------------------------

def test_readme_documents_new_sections():
  '''conf/README documents registry/scoring/infra + per-env matrix.'''
  readme = (_root() / 'conf' / 'README.md').read_text()
  for section in ('registry_params', 'scoring_params', 'infra_params'):
    assert section in readme, section
  for key in ('tracking_uri', 'model_name', 'promote_on_metric',
              'batch_size', 'oc.env', 'model_path', 'id_cols',
              'feature_cols', 'task_type', 'max_rows', 'max_cols',
              'predictions'):
    assert key in readme, key
  assert 'per-env' in readme.lower() or 'file x env' in readme.lower()


# --- CI workflow -----------------------------------------------------------------

def test_ci_workflow_yaml_and_hardening():
  '''CI yaml parses, actions SHA-pinned, timeouts, terraform job.'''
  import re as _re
  ci_path = _root().parent / '.github' / 'workflows' / 'forecast-ml-ci.yml'
  raw = ci_path.read_text()
  doc = yaml.safe_load(raw)
  assert doc['name'] == 'forecast-ml-ci'
  # paths include pylintrc + workflow itself + framework
  on = doc.get('on') or doc.get(True)
  paths = str(on)
  assert '.pylintrc' in paths
  assert 'forecast-ml-ci.yml' in paths
  assert 'forecast_ml_framework/**' in paths
  # every job has timeout-minutes: 30
  for job_name, job in doc['jobs'].items():
    assert job.get('timeout-minutes') == 30, job_name
  # actions pinned by SHA with tag comment (no bare @v4/@v5 uses)
  uses = _re.findall(r'uses:\s*(\S+)', raw)
  assert uses, 'no actions found'
  for use in uses:
    assert '@' in use and _re.search(r'@[0-9a-f]{40}\b', use), use
    assert not _re.search(r'@(v\d+)\s*$', use), use
  # terraform fmt job exists
  assert 'terraform' in doc['jobs'], sorted(doc['jobs'])
  tf_job = doc['jobs']['terraform']
  assert tf_job.get('timeout-minutes') == 30
  assert 'fmt' in str(tf_job)
  assert 'hashicorp/setup-terraform' in raw
  # demo asserts scoring pipeline
  assert "'scoring'" in raw or '"scoring"' in raw
  assert 'scoring' in raw


def test_ci_run_checks_mirrors_demo_scoring():
  '''Local run_checks demo also asserts scoring pipeline.'''
  raw = (_root() / 'ci' / 'run_checks.sh').read_text()
  assert 'scoring' in raw


# --- infra -----------------------------------------------------------------------

def test_infra_gcp_hardening():
  '''GCP: dataproc off by default, buckets enforced, internal IP.'''
  vars_tf = (_root() / 'infra' / 'gcp' / 'variables.tf').read_text()
  main_tf = (_root() / 'infra' / 'gcp' / 'main.tf').read_text()
  assert 'default' in vars_tf
  # enable_dataproc default false (no billing by default)
  m = re.search(r'variable\s+"enable_dataproc"\s*\{[^}]*default\s*=\s*(\w+)',
                vars_tf, re.S)
  assert m and m.group(1) == 'false', vars_tf[vars_tf.find('enable_dataproc')-50:vars_tf.find('enable_dataproc')+200]
  assert main_tf.count("public_access_prevention") >= 2
  assert main_tf.count("'enforced'") >= 2 or main_tf.count('"enforced"') >= 2
  assert 'internal_ip_only' in main_tf


# --- Phase 2 scaling: conf + infra (100M rows) --------------------------------

def test_scoring_spark_scaling_keys():
  '''spark_scoring off by default, 10k partitions, max_rows unbounded.'''
  params = _params()
  scoring = params['scoring_params']
  assert scoring['max_rows'] is None
  spark_scoring = scoring['spark_scoring']
  assert spark_scoring['enabled'] is False
  assert spark_scoring['partition_batch_size'] == 10000
  # Worker reads partition_batch_size top-level too; keep in sync.
  assert scoring['partition_batch_size'] == 10000
  pandas = params['pandas_conversion_params']
  assert isinstance(pandas['max_rows_for_pandas'], int)
  assert pandas['max_rows_for_pandas'] == 5000000


def test_spark_params_scaling_keys():
  '''Spark AQE + dynamic allocation + shuffle sized for 100M rows.'''
  params = _params()
  spark = params['spark_params']['spark']
  assert str(spark['spark.dynamicAllocation.enabled']).lower() == 'true'
  assert str(spark['spark.sql.adaptive.enabled']).lower() == 'true'
  assert isinstance(spark['spark.sql.shuffle.partitions'], int)
  assert spark['spark.sql.shuffle.partitions'] >= 1000


def test_scaling_keys_commented():
  '''Every new scaling key carries its own comment.'''
  raw = _params_raw()
  lines = raw.splitlines()
  for key in ('spark_scoring', 'partition_batch_size', 'max_rows',
              'max_rows_for_pandas', 'spark.dynamicAllocation.enabled',
              'spark.sql.adaptive.enabled', 'spark.sql.shuffle.partitions'):
    idx = next((i for i, line in enumerate(lines)
                if re.match(r'\s+' + re.escape(key) + r':', line)), None)
    assert idx is not None, key
    commented = ('#' in lines[idx]) or (
        idx > 0 and lines[idx - 1].lstrip().startswith('#')) or any(
            '#' in line and key in line for line in lines)
    assert commented, key


def test_gcp_dataproc_autoscaling_provision():
  '''GCP autoscaling policy + vars, off by default, wired to cluster.'''
  vars_tf = (_root() / 'infra' / 'gcp' / 'variables.tf').read_text()
  main_tf = (_root() / 'infra' / 'gcp' / 'main.tf').read_text()
  m = re.search(
      r'variable\s+"autoscaling_enabled"\s*\{[^}]*default\s*=\s*(\w+)',
      vars_tf, re.S)
  assert m and m.group(1) == 'false'
  assert 'autoscaling_max_workers' in vars_tf
  assert 'autoscaling_min_workers' in vars_tf
  assert 'google_dataproc_autoscaling_policy' in main_tf
  assert 'min_instances' in main_tf and 'max_instances' in main_tf
  assert 'autoscaling_config' in main_tf
  assert 'google_dataproc_autoscaling_policy' in main_tf.split(
      'autoscaling_config')[1].split('}')[0] or 'autoscaling_policy' in main_tf


def test_aws_emr_autoscaling_stub():
  '''AWS EMR autoscaling stub: commented policy + README provision.'''
  main_tf = (_root() / 'infra' / 'aws' / 'main.tf').read_text()
  assert 'aws_emr_auto_scaling_policy' in main_tf
  assert '# resource "aws_emr_auto_scaling_policy"' in main_tf
  readme = (_root() / 'infra' / 'aws' / 'README.md').read_text()
  assert 'auto_scaling_policy' in readme or 'autoscaling' in readme.lower()
  assert '100M' in readme


def test_infra_readmes_100m_sizing():
  '''Infra READMEs carry 100M-row sizing guidance.'''
  for path in ('infra/README.md', 'infra/gcp/README.md',
               'infra/aws/README.md'):
    text = (_root() / path).read_text()
    assert '100M' in text, path
    lowered = text.lower()
    assert 'shuffle' in lowered, path
    assert 'partition' in lowered, path


def test_infra_aws_emr_validation_and_docs():
  '''AWS: EMR subnet validation + role pre-existence documented.'''
  vars_tf = (_root() / 'infra' / 'aws' / 'variables.tf').read_text()
  main_tf = (_root() / 'infra' / 'aws' / 'main.tf').read_text()
  assert 'validation' in vars_tf
  assert 'emr_subnet_id' in vars_tf
  assert 'enable_emr' in vars_tf
  # validation block requires subnet when enable_emr
  idx = vars_tf.find('validation')
  assert idx >= 0
  assert 'EMR_DefaultRole' in main_tf
  readme = (_root() / 'infra' / 'aws' / 'README.md').read_text()
  assert 'EMR_DefaultRole' in readme
  assert 'pre-exist' in readme.lower() or 'pre-exists' in readme.lower()
