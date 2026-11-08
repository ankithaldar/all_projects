#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P1-E: ONE catalog grouping, banner map, PII validation.

TDD RED->GREEN: catalog_banner_map / load_catalog /
validate_catalog_pii do not exist yet, so tests fail with
ImportError/AttributeError until lineage.py GREEN lands.
Column lineage is fully implemented; its behavior
is covered in test_column_lineage.py, not here.
'''

from pathlib import Path

import pytest


def _root():
  return Path(__file__).resolve().parents[1]


def _base_dir():
  return str(_root() / 'conf' / 'base')


def _catalog_path():
  return _root() / 'conf' / 'base' / 'catalog.yml'


def _v1_names():
  import yaml
  catalog = yaml.safe_load(_catalog_path().read_text())
  return {name for name in catalog if not name.startswith('_')}


def test_banner_map_covers_all_v1_names():
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  mapping = catalog_banner_map(_base_dir())
  assert mapping
  assert set(mapping) == _v1_names()
  for name, section in mapping.items():
    assert isinstance(section, str) and section, name
  # Deterministic return value (load-observable behavior).
  assert catalog_banner_map(_base_dir()) == mapping


def test_banner_map_accepts_file_and_root_forms():
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  via_dir = catalog_banner_map(_base_dir())
  via_file = catalog_banner_map(str(_catalog_path()))
  via_root = catalog_banner_map(str(_root()))
  assert via_file == via_dir
  assert via_root == via_dir


def test_load_catalog_returns_datacatalog_without_onnx():
  from forecast_ml_framework.governance.lineage import load_catalog
  catalog = load_catalog(_base_dir())
  assert 'best_model_onnx' not in catalog
  for name in ('raw_data', 'best_model', 'fitted_registry'):
    assert name in catalog, name


def test_no_best_model_onnx_in_catalog_config():
  import yaml
  catalog = yaml.safe_load(_catalog_path().read_text())
  assert 'best_model_onnx' not in catalog
  assert 'best_model' in catalog


def test_no_fragment_catalog_directory():
  assert not (_root() / 'conf' / 'base' / 'catalog').exists()
  assert not list(
    (_root() / 'conf' / 'base').glob('catalog*/*.yml'))


def test_banner_map_duplicate_key_names_lines(tmp_path):
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  path = tmp_path / 'catalog.yml'
  path.write_text(
    '# --- section-a ---\n'
    'foo:\n'
    '  type: kedro.io.MemoryDataset\n'
    '# --- section-b ---\n'
    'bar:\n'
    '  type: kedro.io.MemoryDataset\n'
    'foo:\n'
    '  type: kedro.io.MemoryDataset\n',
    encoding='utf-8')
  with pytest.raises(ValueError) as excinfo:
    catalog_banner_map(str(tmp_path))
  message = str(excinfo.value)
  assert 'duplicate' in message and "'foo'" in message
  assert ':2' in message and ':7' in message


def test_banner_map_missing_banner_raises(tmp_path):
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  (tmp_path / 'catalog.yml').write_text(
    'foo:\n  type: kedro.io.MemoryDataset\n',
    encoding='utf-8')
  with pytest.raises(ValueError):
    catalog_banner_map(str(tmp_path))


def test_validate_catalog_pii_unknown_raises():
  from forecast_ml_framework.governance.lineage import (
    validate_catalog_pii)
  config = {'entry': {'metadata': {'pii_classes': ['bogus']}}}
  with pytest.raises(ValueError):
    validate_catalog_pii(config)


def test_validate_catalog_pii_none_and_missing_pass():
  from forecast_ml_framework.governance.lineage import (
    validate_catalog_pii)
  validate_catalog_pii(
    {'entry': {'metadata': {'pii_classes': ['none']}}})
  validate_catalog_pii({'entry': {'type': 'x'}})
  validate_catalog_pii({'entry': {'metadata': {}}})
  # Real catalog (['none'] on raw_data, rest unclassified) passes.
  import yaml
  validate_catalog_pii(yaml.safe_load(_catalog_path().read_text()))


def test_validate_catalog_pii_mixed_none_raises():
  from forecast_ml_framework.governance.lineage import (
    validate_catalog_pii)
  config = {
    'entry': {'metadata':
              {'pii_classes': ['none', 'email']}}}
  with pytest.raises(ValueError):
    validate_catalog_pii(config)


def test_load_catalog_rejects_unknown_pii(tmp_path):
  from forecast_ml_framework.governance.lineage import load_catalog
  (tmp_path / 'catalog.yml').write_text(
    '# --- section-a ---\n'
    'foo:\n'
    '  type: kedro.io.MemoryDataset\n'
    '  metadata:\n'
    "    pii_classes: ['bogus']\n",
    encoding='utf-8')
  with pytest.raises(ValueError):
    load_catalog(str(tmp_path))


def test_catalog_path_missing_raises(tmp_path):
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  with pytest.raises(FileNotFoundError):
    catalog_banner_map(str(tmp_path))


def test_run_lineage_regression():
  from forecast_ml_framework.governance.lineage import (
    build_lineage, compute_config_hash)
  lineage = build_lineage({}, {})
  assert lineage['inputs'] == []
  assert lineage['outputs'] == []
  assert lineage['model'] is None
  assert len(lineage['config_hash']) == 64
  assert compute_config_hash({'s': {'b', 'a'}}, {}) == (
    compute_config_hash({'s': ['a', 'b']}, {}))
  out = build_lineage(
    {'inputs': ['raw_data'], 'model': 'best_model',
     'outputs': ['predictions'],
     'variables': {'k': 1}, 'parameters': {'p': 2}})
  assert out['inputs'] == ['raw_data']
  assert out['model'] == 'best_model'


def test_run_lineage_object_and_catalog_fallback():
  import types
  from forecast_ml_framework.governance.lineage import (
    _catalog_names, build_lineage)
  obj = types.SimpleNamespace(
    inputs=['a'], features=['f'], model='m', outputs=['b'])
  out = build_lineage(obj)
  assert out['inputs'] == ['a'] and out['model'] == 'm'
  # Catalog fallback when run_info omits inputs/outputs.
  entries = [{'role': 'input', 'name': 'raw_data'},
             {'role': 'output', 'name': 'predictions'},
             {'kind': 'input', 'dataset': 'other'},
             'ignored', {'name': 'untagged'}]
  assert _catalog_names(entries, 'input') == ['raw_data', 'other']
  assert _catalog_names({'x': entries[0]}, 'input') == ['raw_data']
  assert build_lineage({}, entries)['inputs'] == [
    'raw_data', 'other']
  assert build_lineage({}, entries)['outputs'] == ['predictions']
  # Non-dict config falls back to empty mapping.
  out2 = build_lineage(
    {'config': 'oops', 'inputs': ['x'], 'model': 'm',
     'outputs': ['y']})
  assert out2['config_hash']
  # Missing object attrs return defaults.
  assert build_lineage(types.SimpleNamespace())['model'] is None


def test_compute_config_hash_mixed_set_fallback():
  from forecast_ml_framework.governance.lineage import (
    compute_config_hash)
  assert compute_config_hash({'s': {1, 'a'}}, {}) == (
    compute_config_hash({'s': {'a', 1}}, {}))
  assert compute_config_hash({'t': (1, 2)}, {}) == (
    compute_config_hash({'t': [1, 2]}, {}))


def test_validate_catalog_pii_edge_branches():
  from forecast_ml_framework.governance.lineage import (
    validate_catalog_pii)
  with pytest.raises(ValueError):
    validate_catalog_pii(['not', 'a', 'map'])
  validate_catalog_pii({'entry': 'not-a-dict'})
  with pytest.raises(ValueError):
    validate_catalog_pii({'entry': {'metadata': 'nope'}})
  validate_catalog_pii(
    {'entry': {'metadata': {'pii_classes': None}}})


def test_banner_map_empty_file_raises(tmp_path):
  from forecast_ml_framework.governance.lineage import (
    catalog_banner_map)
  (tmp_path / 'catalog.yml').write_text('', encoding='utf-8')
  with pytest.raises(ValueError):
    catalog_banner_map(str(tmp_path))


def test_load_catalog_non_mapping_raises(tmp_path):
  from forecast_ml_framework.governance.lineage import load_catalog
  (tmp_path / 'catalog.yml').write_text('- a\n- b\n',
                                        encoding='utf-8')
  with pytest.raises(ValueError):
    load_catalog(str(tmp_path))
