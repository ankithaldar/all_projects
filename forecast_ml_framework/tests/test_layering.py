#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Layering + cycle-3 arch residual tests (X6, RED first, TDD).

Covers arch R1 (core.handle canonical resolver + tile
re-exports, no cross-pipeline top-level imports), ML R3
(OOF n=1), R4 (hatch force-include conf), R5 (ks single
source), R6 (split_meta chain), R7 (reg stratify conf).
'''

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def _src(*parts):
  '''Return src file path under the project.'''
  return _root().joinpath('src', 'forecast_ml_framework', *parts)


def _top_level_imports(path):
  '''Return top-level imported module names from a file.'''
  tree = ast.parse(Path(path).read_text())
  modules = []
  for node in tree.body:
    if isinstance(node, ast.Import):
      modules.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        modules.append(node.module)
  return modules


# --- R1: canonical resolver in core.handle -------------------------------


def test_core_handle_canonical_resolver_exists():
  '''R1: core.handle.resolve_handle_strict is canonical (no _).'''
  from forecast_ml_framework.core import handle
  assert hasattr(handle, 'resolve_handle_strict')
  from sklearn.ensemble import RandomForestClassifier
  assert handle.resolve_handle_strict(
      'sklearn.ensemble.RandomForestClassifier'
  ) is RandomForestClassifier
  with pytest.raises(ValueError):
    handle.resolve_handle_strict('os.system')


def test_scoring_reexports_core_resolver():
  '''R1: scoring keeps compat names pointing at core.handle.'''
  from forecast_ml_framework.core import handle
  from forecast_ml_framework.pipelines.scoring import nodes
  assert nodes._resolve_handle_strict is handle.resolve_handle_strict
  from sklearn.ensemble import RandomForestClassifier
  assert nodes.resolve_handle(
      'sklearn.ensemble.RandomForestClassifier'
  ) is RandomForestClassifier
  with pytest.raises(ValueError):
    nodes.resolve_handle('os.system')


def test_data_prep_imports_resolver_from_core_not_scoring():
  '''R1: data_prep/nodes imports resolver from core.handle.'''
  modules = _top_level_imports(_src('pipelines', 'data_prep', 'nodes.py'))
  assert 'forecast_ml_framework.core.handle' in modules
  for module in modules:
    assert 'pipelines.scoring' not in module, module
    assert 'pipelines.modeling' not in module, module


def test_layering_no_cross_pipeline_top_level_imports():
  '''R1: data_prep/scoring/modeling-pipeline do not top-level-
  import each other (scoring->fitted documented exception).'''
  trio = {
      'data_prep': _src('pipelines', 'data_prep', 'nodes.py'),
      'scoring': _src('pipelines', 'scoring', 'nodes.py'),
      'modeling_pipeline': _src('pipelines', 'modeling', 'pipeline.py'),
  }
  prefixes = {
      'data_prep': 'forecast_ml_framework.pipelines.data_prep',
      'scoring': 'forecast_ml_framework.pipelines.scoring',
      'modeling': 'forecast_ml_framework.pipelines.modeling',
  }
  for owner, path in trio.items():
    for module in _top_level_imports(path):
      for other, prefix in prefixes.items():
        if module.startswith(prefix) and not module.startswith(
            prefixes['modeling'] if owner == 'modeling_pipeline'
            else prefixes[owner]):
          raise AssertionError(
              '%s top-level-imports %s' % (owner, module))
  # Documented exception: scoring/pipeline -> data_prep.fitted is
  # one-way (the fitted cycle is broken by the resolver move);
  # modeling/pipeline -> interpretability is outside the trio.
  scoring_pipe = _top_level_imports(
      _src('pipelines', 'scoring', 'pipeline.py'))
  assert 'forecast_ml_framework.pipelines.data_prep.fitted' in \
      scoring_pipe


def test_import_graph_data_prep_does_not_load_scoring():
  '''R1: importing data_prep.nodes must not load scoring.nodes.'''
  code = (
      'import sys\n'
      'import forecast_ml_framework.pipelines.data_prep.nodes\n'
      'loaded = [m for m in sys.modules'
      ' if m.startswith("forecast_ml_framework.pipelines.scoring")]\n'
      'assert not loaded, loaded\n'
      'print("ok")\n')
  root = _root()
  import os
  proc = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True, text=True, timeout=120,
      env={**dict(os.environ),
           'PYTHONPATH': str(root / 'src') + os.pathsep
           + os.environ.get('PYTHONPATH', '')},
      check=False)
  assert proc.returncode == 0, proc.stderr
  assert 'ok' in proc.stdout


def test_core_handle_reexports_ntile_helpers():
  '''R1: core.handle re-exports ntile helpers (ntile.py unmoved).'''
  handle = importlib.import_module('forecast_ml_framework.core.handle')
  ntile = importlib.import_module(
      'forecast_ml_framework.pipelines.modeling.ntile')
  assert handle.assign_ntile is ntile.assign_ntile
  assert handle.normalize_task_kind is ntile.normalize_task_kind
  labels = handle.assign_ntile([0.1, 0.2, 0.3, 0.4], 'decile', 2)
  assert list(labels) == [1, 1, 2, 2]


def test_core_handle_tile_exports_trimmed_to_used():
  '''N1: core.handle exports only consumed tile helpers (YAGNI).'''
  handle = importlib.import_module('forecast_ml_framework.core.handle')
  assert handle.__all__ == [
      'ALLOWED_MODEL_MODULES',
      'assign_ntile',
      'normalize_task_kind',
      'resolve_handle',
      'resolve_handle_strict',
  ]
  assert handle._TILE_NAMES == frozenset({
      'assign_ntile',
      'normalize_task_kind',
  })
  assert 'parse_eval_metric' not in handle.__all__
  assert 'select_ntile_row' not in handle.__all__
  assert '_task_kind' not in handle._TILE_NAMES
  with pytest.raises(AttributeError):
    handle.parse_eval_metric
  with pytest.raises(AttributeError):
    handle.select_ntile_row
  with pytest.raises(AttributeError):
    handle._task_kind


# --- ML R3: OOF n=1 ------------------------------------------------------


def test_oof_n1_returns_global_not_own_label():
  '''ML R3: n=1 OOF encoding is the global prior, not own label.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
      _oof_encode_series,
  )
  out = _oof_encode_series(['b'], [1], 0.0)
  assert list(out) == [0.0]


def test_oof_single_row_category_in_larger_frame():
  '''ML R3: single-row category still gets others' global mean.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
      _oof_encode_series,
  )
  out = _oof_encode_series(['a', 'b', 'a'], [1, 0, 1], 0.0)
  # Row b (own label 0) encodes from others (mean 1.0), never 0.0.
  assert list(out) == [1.0, 1.0, 1.0]


# --- R5: ks single source ------------------------------------------------


def test_ks_warning_single_source_005():
  '''R5: schema + conf + node ks warning tiers all agree at 0.05.'''
  from forecast_ml_framework.core.config_schema import MonitoringParams
  from forecast_ml_framework.pipelines.monitoring.nodes import (
      DEFAULT_THRESHOLDS,
  )
  assert MonitoringParams().thresholds['ks']['warning'] == 0.05
  assert DEFAULT_THRESHOLDS['ks']['warning'] == 0.05
  platform = yaml.safe_load(
      (_root() / 'conf/base/parameters/platform.yml').read_text())
  assert platform['monitoring_params']['thresholds']['ks'][
      'warning'] == 0.05


# --- R6: split_meta chain ------------------------------------------------


def test_split_raw_with_meta_reports_actual_method():
  '''R6: data_prep emits split_meta_raw with honest meta.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.nodes import (
      split_raw_data_with_meta,
  )
  df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                     'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                     'target': [0, 1, 0, 1, 0, 1]})
  params = {'target_col': 'target', 'task_type': 'classification',
            'test_size': 0.5, 'random_state': 42,
            'sampling_params': {'method': 'random', 'test_size': 0.5}}
  train_raw, test_raw, meta = split_raw_data_with_meta(
      df, params, {'target_col': 'target'})
  assert len(train_raw) + len(test_raw) == 6
  assert meta['sampling_method'] == 'random'
  assert meta['requested_method'] == 'random'
  assert meta['shuffle'] is True
  assert meta['time_col'] is None
  assert meta['n_train'] + meta['n_test'] == 6


def test_presplit_merges_raw_split_meta():
  '''R6: modeling presplit meta merges data_prep raw meta.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.modeling.pipeline import (
      split_presplit_data,
  )
  raw_meta = {'sampling_method': 'time', 'requested_method': 'time',
              'shuffle': False, 'time_col': 'ts', 'test_size': 0.5,
              'random_state': 42, 'n_train': 3, 'n_test': 3}
  x_tr, x_te, y_tr, y_te, meta = split_presplit_data(
      pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]}),
      pd.DataFrame({'f1': [3.0], 'target': [1]}),
      {'target_col': 'target', 'task_type': 'classification'},
      raw_meta)
  assert meta['sampling_method'] == 'presplit'
  assert meta['raw_sampling_method'] == 'time'
  assert meta['raw_requested_method'] == 'time'
  assert meta['raw_shuffle'] is False
  assert meta['raw_time_col'] == 'ts'
  assert list(x_tr['f1']) == [1.0, 2.0]
  assert list(x_te['f1']) == [3.0]


def test_split_meta_raw_catalog_and_modeling_wiring():
  '''R6: catalog has split_meta_raw; modeling node consumes it.'''
  catalog = yaml.safe_load(
      (_root() / 'conf/base/catalog.yml').read_text())
  assert 'split_meta_raw' in catalog
  assert 'type' in catalog['split_meta_raw']
  from forecast_ml_framework.pipelines.modeling.pipeline import (
      create_pipeline,
  )
  by_name = {n.name: n for n in create_pipeline().nodes}
  split = by_name['split_data_node']
  assert 'split_meta_raw' in split.inputs
  assert 'train_data_selected' in split.inputs
  assert 'test_data_selected' in split.inputs


# --- R7: regression stratify conf ----------------------------------------


def test_modeling_reg_params_no_stratify_col():
  '''R7: modeling_reg_params drops stratify_col (continuous target).'''
  modeling = yaml.safe_load(
      (_root() / 'conf/base/parameters/modeling.yml').read_text())
  reg_sampling = modeling['modeling_reg_params']['sampling_params']
  assert 'stratify_col' not in reg_sampling
  # Classification keeps stratify_col on the discrete target.
  cls_sampling = modeling['modeling_params']['sampling_params']
  assert cls_sampling['stratify_col'] == 'target'


# --- R4: packaged conf ---------------------------------------------------


def test_wheel_force_includes_conf():
  '''R4: pyproject force-includes conf into the wheel package.'''
  import tomllib
  with (_root() / 'pyproject.toml').open('rb') as fh:
    pyproject = tomllib.load(fh)
  wheel = pyproject['tool']['hatch']['build']['targets']['wheel']
  force_include = wheel.get('force-include', {})
  assert force_include.get('conf') == 'forecast_ml_framework/conf'
  assert (_root() / 'conf/base/catalog.yml').is_file()
  assert (_root() / 'conf/base/parameters/modeling.yml').is_file()


# --- N6 O3: predict name divergence documented ---------------------------


def test_predict_name_divergence_documented_in_both_docstrings():
  '''O3: both docstrings state the predict divergence explicitly.

  ``forecast-ml predict`` runs scoring while
  ``kedro run --pipeline predict`` runs modeling; aligning
  them would break pinned test_cli behaviour, so the
  divergence must be documented, not removed (Lead D8).
  '''
  import forecast_ml_framework.cli
  import forecast_ml_framework.pipeline_registry
  for module in (forecast_ml_framework.cli,
                 forecast_ml_framework.pipeline_registry):
    doc = ' '.join((module.__doc__ or '').split())
    assert 'forecast-ml predict' in doc, module.__name__
    assert 'kedro run --pipeline predict' in doc, module.__name__
    assert 'scoring' in doc, module.__name__
    assert 'modeling' in doc, module.__name__


# --- N6 O2: wheel namelist (light, no real wheel build) ------------------


def test_wheel_namelist_packaged_conf_fake_tree(tmp_path):
  '''O2: force-include layout puts catalog.yml in wheel namelist.

  Light fake-wheel check (zipfile over a tmp tree, no real
  ``uv build``): the namelist entry the CI / run_checks
  ``uv build --wheel`` step asserts on must be derivable
  from the pyproject force-include mapping.
  '''
  import tomllib
  import zipfile
  with (_root() / 'pyproject.toml').open('rb') as fh:
    pyproject = tomllib.load(fh)
  wheel = pyproject['tool']['hatch']['build']['targets']['wheel']
  force_include = wheel.get('force-include', {})
  packaged = force_include.get('conf')
  assert packaged == 'forecast_ml_framework/conf'
  expected = packaged + '/base/catalog.yml'
  assert expected == 'forecast_ml_framework/conf/base/catalog.yml'
  fake_wheel = tmp_path / 'fake-0.1.0-py3-none-any.whl'
  with zipfile.ZipFile(fake_wheel, 'w') as zf:
    zf.writestr(expected, 'datasets: {}')
    zf.writestr('forecast_ml_framework/__init__.py', '')
  with zipfile.ZipFile(fake_wheel) as zf:
    assert expected in zf.namelist()


def test_packaged_conf_resolves_via_importlib_resources(
    tmp_path, monkeypatch):
  '''O2: settings prefers packaged conf via importlib.resources.

  Fake packaged tree under tmp_path (no wheel build):
  when ``importlib.resources.files`` points at a package
  shipping ``conf/``, ``_resolve_conf_source`` returns it;
  without one it falls back to the checkout ``conf``.
  '''
  import importlib.resources
  fake_pkg = tmp_path / 'forecast_ml_framework'
  (fake_pkg / 'conf' / 'base').mkdir(parents=True)
  (fake_pkg / 'conf' / 'base' / 'catalog.yml').write_text(
      'datasets: {}', encoding='utf-8')
  monkeypatch.setattr(
      importlib.resources, 'files', lambda pkg: fake_pkg)
  from forecast_ml_framework import settings
  assert settings._resolve_conf_source() == str(fake_pkg / 'conf')
  empty_pkg = tmp_path / 'empty_pkg'
  empty_pkg.mkdir()
  monkeypatch.setattr(
      importlib.resources, 'files', lambda pkg: empty_pkg)
  fallback = settings._resolve_conf_source()
  assert fallback != str(empty_pkg / 'conf')
  assert fallback.endswith('conf')


def test_wheel_build_check_wired_in_run_checks_and_ci():
  '''O2: run_checks.sh + CI build the wheel and assert namelist.'''
  run_checks = (_root() / 'ci' / 'run_checks.sh').read_text(
      encoding='utf-8')
  assert 'uv build' in run_checks
  assert '--wheel' in run_checks
  assert 'forecast_ml_framework/conf/base/catalog.yml' in run_checks
  assert 'zipfile' in run_checks
  assert 'namelist' in run_checks
  workflow_candidates = [
      _root().parent / '.github' / 'workflows' / 'forecast-ml-ci.yml',
      _root() / '.github' / 'workflows' / 'forecast-ml-ci.yml',
  ]
  workflow_path = next((p for p in workflow_candidates if p.is_file()),
                       None)
  assert workflow_path is not None, workflow_candidates
  workflow = workflow_path.read_text(encoding='utf-8')
  assert 'uv build' in workflow
  assert '--wheel' in workflow
  assert 'forecast_ml_framework/conf/base/catalog.yml' in workflow
  assert 'zipfile' in workflow
  assert 'namelist' in workflow
