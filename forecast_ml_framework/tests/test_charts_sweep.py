#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Quality-sweep tests for W14 owned files (RED first, TDD).'''

import ast
import subprocess
import sys
from pathlib import Path


def _src_path(*parts):
  '''Return src file path under the project.'''
  root = Path(__file__).resolve().parents[1]
  return root.joinpath('src', 'forecast_ml_framework', *parts)


def test_cost_curve_stub_deleted_and_absent_from_make_charts():
  '''F5: _cost_curve stub is gone; make_charts omits cost_curve.'''
  from forecast_ml_framework.pipelines.modeling import charts
  assert not hasattr(charts, '_cost_curve'), (
      '_cost_curve stub must be deleted (F5)')
  import numpy as np
  import pandas as pd
  rng = np.random.RandomState(3)
  n = 120
  x = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n),
                    'seg': np.where(rng.rand(n) > .5, 'a', 'b')})
  y_true = (x['f1'] > 0).astype(int).values
  y_pred = y_true.copy()
  y_pred[::7] = 1 - y_pred[::7]
  proba = np.column_stack([1 - y_pred, y_pred]).astype(float)
  params = {'task_type': 'classification',
            'bucket_segment_col': 'seg'}
  figs = charts.make_charts(y_true, y_pred, proba, x, params)
  try:
    assert 'cost_curve' not in figs
    # Deleting one chart must not drop the fixture below 5.
    assert len(figs) >= 5
  finally:
    import matplotlib.pyplot as plt
    for fig in figs.values():
      plt.close(fig)


def test_scoring_ohe_and_align_helpers_exist():
  '''F6: _ohe_with_caps and _align_to_fitted extracted.'''
  from forecast_ml_framework.pipelines.scoring import nodes
  assert hasattr(nodes, '_ohe_with_caps'), (
      '_score_pandas_chunk must delegate OHE to _ohe_with_caps')
  assert hasattr(nodes, '_align_to_fitted'), (
      '_score_pandas_chunk must delegate alignment to '
      '_align_to_fitted')


def test_scoring_ohe_align_behavior_preserved():
  '''F6: extracted helpers preserve OHE and fitted alignment.'''
  import numpy as np
  import pandas as pd
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _align_to_fitted,
    _ohe_with_caps,
  )
  frame = pd.DataFrame({'num': [1.0, 2.0, 3.0],
                        'cat': ['a', 'b', 'a']})
  out = _ohe_with_caps(frame, {})
  assert 'cat_a' in out.columns and 'cat_b' in out.columns
  assert set(out['num'].tolist()) == {1.0, 2.0, 3.0}
  class _Model:
    feature_names_in_ = np.array(['cat_b', 'num', 'missing'])
  aligned = _align_to_fitted(out, _Model())
  assert list(aligned.columns) == ['cat_b', 'num', 'missing']
  assert aligned['missing'].tolist() == [0, 0, 0]


def test_scoring_public_resolve_handle_canonical():
  '''F7: scoring exposes public resolve_handle (canonical).'''
  from forecast_ml_framework.pipelines.scoring import nodes
  assert hasattr(nodes, 'resolve_handle'), (
      'scoring/nodes must define canonical resolve_handle')
  import pytest
  from sklearn.ensemble import RandomForestClassifier
  assert nodes.resolve_handle(
      'sklearn.ensemble.RandomForestClassifier') is \
      RandomForestClassifier
  with pytest.raises(ValueError):
    nodes.resolve_handle('os.system')


def test_scoring_ntile_fallback_still_works():
  '''F9: _ntile_labels fallback is preserved after narrowing.'''
  import numpy as np
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _ntile_labels,
  )
  labels = _ntile_labels(np.ones(20), 10)
  assert np.asarray(labels).tolist() == [1] * 20
  assert _ntile_labels([], 10).tolist() == []


def test_spark_frame_row_count_hostile_still_none():
  '''F9: spark_factory counts tolerate hostile count()/len().'''
  from forecast_ml_framework.core import spark_factory
  class _Bad:
    def count(self):
      raise RuntimeError('hostile')
    def __len__(self):
      raise RuntimeError('hostile')
  assert spark_factory._frame_row_count(_Bad()) is None
  assert spark_factory._frame_col_count(_Bad()) is None


def test_modeling_charts_lazy_matplotlib():
  '''V8: modeling/charts defers matplotlib to chart call time.'''
  # Subprocess with matplotlib blocked: module import must succeed.
  code = (
      'import sys\n'
      'sys.modules["matplotlib"] = None\n'
      'sys.modules["matplotlib.pyplot"] = None\n'
      'import forecast_ml_framework.pipelines.modeling.charts as c\n'
      'assert hasattr(c, "make_charts")\n'
      'print("ok")\n')
  root = Path(__file__).resolve().parents[1]
  proc = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True, text=True, timeout=60,
      env={**dict(__import__('os').environ),
           'PYTHONPATH': str(root / 'src') + __import__('os').pathsep
           + __import__('os').environ.get('PYTHONPATH', '')},
      check=False)
  assert proc.returncode == 0, proc.stderr
  assert 'ok' in proc.stdout


def test_interpretability_charts_lazy_matplotlib():
  '''V8: interpretability/charts defers matplotlib.'''
  code = (
      'import sys\n'
      'sys.modules["matplotlib"] = None\n'
      'sys.modules["matplotlib.pyplot"] = None\n'
      'import forecast_ml_framework.pipelines.interpretability.charts'
      ' as c\n'
      'assert hasattr(c, "shap_beeswarm")\n'
      'print("ok")\n')
  import os
  root = Path(__file__).resolve().parents[1]
  proc = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True, text=True, timeout=60,
      env={**dict(os.environ),
           'PYTHONPATH': str(root / 'src') + os.pathsep
           + os.environ.get('PYTHONPATH', '')},
      check=False)
  assert proc.returncode == 0, proc.stderr
  assert 'ok' in proc.stdout


def test_charts_no_toplevel_matplotlib_import():
  '''V8 (static): no top-level matplotlib import in charts.'''
  for rel in ('pipelines/modeling/charts.py',
              'pipelines/interpretability/charts.py'):
    tree = ast.parse(_src_path(*rel.split('/')).read_text())
    for node in tree.body:
      if isinstance(node, ast.Import):
        for alias in node.names:
          assert 'matplotlib' not in alias.name, (
              f'top-level matplotlib import in {rel}')
      if isinstance(node, ast.ImportFrom):
        assert node.module is None or 'matplotlib' not in node.module


def test_registry_warns_on_omitted_pipeline(monkeypatch):
  '''V9: register_pipelines warns when a pipeline is omitted.'''
  import builtins
  import pytest
  import forecast_ml_framework.pipeline_registry as registry
  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == 'forecast_ml_framework.pipelines.modeling':
      raise ModuleNotFoundError(
          "No module named 'forecast_ml_framework.pipelines.modeling'",
          name=name)
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  with pytest.warns(UserWarning, match='omitted'):
    pipelines = registry.register_pipelines()
  assert 'modeling' not in pipelines
  assert 'data' in pipelines


def test_registry_tolerates_missing_kedro(monkeypatch):
  '''V9: missing kedro inside a pipeline returns None + warns.'''
  import builtins
  import pytest
  import forecast_ml_framework.pipeline_registry as registry
  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == 'forecast_ml_framework.pipelines.modeling':
      raise ModuleNotFoundError(
          "No module named 'kedro'", name='kedro')
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  with pytest.warns(UserWarning):
    pipelines = registry.register_pipelines()
  assert 'modeling' not in pipelines
  assert 'data' in pipelines


def test_settings_conf_source_resolver_exists():
  '''V10: settings resolves CONF_SOURCE via a resolver helper.'''
  from forecast_ml_framework import settings
  assert hasattr(settings, '_resolve_conf_source'), (
      'settings must use an importlib.resources-capable resolver')
  from pathlib import Path
  assert Path(settings.CONF_SOURCE).is_dir()
  assert Path(settings._resolve_conf_source()) == Path(
      settings.CONF_SOURCE)


def test_cli_aliases_documented():
  '''V12: cli documents all 12 registry names.'''
  src = _src_path('cli.py').read_text()
  for name in ('data', 'preprocessing', 'feature_selection',
               'modeling', 'evaluation', 'predict', 'scoring',
               'eda', 'monitoring', 'retraining',
               'cost_tracking', 'interpretability'):
    assert name in src, f'CLI alias {name} undocumented'
  assert 'data_prep' in src or 'pipelines.data_prep' in src or \
      'data_prep' in _src_path('cli.py').read_text()


def test_main_forwards_argv_to_kedro_fallback(monkeypatch):
  '''V12: __main__ forwards argv on the kedro path.'''
  src = _src_path('__main__.py').read_text()
  # The kedro fallback must forward the caller's argv
  # (here via ``sys.argv = [sys.argv[0], *argv]``) instead
  # of always running bare kedro_main() with sys.argv
  # untouched; the cli path must receive caller argv too.
  assert '*argv' in src, '__main__ kedro fallback must forward argv'
  assert 'cli_main(argv)' in src


def test_settings_main_docstrings_current():
  '''Docs-26: settings/__main__ docstrings match code truth.'''
  settings_src = _src_path('settings.py').read_text()
  main_src = _src_path('__main__.py').read_text()
  assert 'Hooks are empty by design' not in settings_src
  assert 'sole console script' not in main_src


def test_scoring_docstrings_single_quotes():
  '''D5: scoring/nodes uses \'\'\' docstrings only.'''
  src = _src_path('pipelines/scoring/nodes.py').read_text()
  assert '"""' not in src, 'scoring/nodes must use single quotes'


def test_positive_class_proba_multiclass_warns_and_returns_max():
  '''M4 (ML-19): (n,>2) warns (binary-only) and uses row max.'''
  import warnings
  import numpy as np
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _positive_class_proba,
  )
  arr = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = _positive_class_proba(arr)
  assert any(
      issubclass(w.category, UserWarning) for w in caught), (
      'multiclass proba must warn (binary-only)')
  assert np.allclose(out, arr.max(axis=1)), (
      'multiclass proba must fall back to row-wise max')


def test_positive_class_proba_single_column_warns():
  '''M4 (ML-19): (n,1) warns (requires (n,2) or 1-D).'''
  import warnings
  import numpy as np
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _positive_class_proba,
  )
  arr = np.array([[0.2], [0.5], [0.9]])
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = _positive_class_proba(arr)
  assert any(
      issubclass(w.category, UserWarning) for w in caught), (
      '(n,1) proba must warn (requires (n,2) or 1-D)')
  assert np.allclose(out, np.array([0.2, 0.5, 0.9]))


def test_positive_class_proba_binary_returns_col1():
  '''M4 (ML-19): (n,2) returns column 1 (positive class).'''
  import numpy as np
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _positive_class_proba,
  )
  arr = np.array([[0.9, 0.1], [0.3, 0.7]])
  assert np.allclose(_positive_class_proba(arr), arr[:, 1])


def test_high_score_ind_ge_consistent_at_threshold():
  '''M4 (ML-19): proba == threshold gives pred 1 and Y.'''
  import numpy as np
  import pandas as pd
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _score_pandas_chunk,
  )
  class _FakeClassifier:
    '''Fake classifier spanning 0..1 including 0.5.'''
    def __init__(self, scores):
      self._scores = np.asarray(scores, dtype=float)
    def predict(self, x):
      return (self._scores[:len(x)] >= 0.5).astype(int)
    def predict_proba(self, x):
      scores = self._scores[:len(x)]
      return np.column_stack([1 - scores, scores])
  scores = np.linspace(0, 1, 101)
  params = {'task_type': 'classification', 'model_threshold': 0.5}
  rng = np.random.default_rng(2)
  frame = pd.DataFrame({'id': np.arange(101), 'f1': rng.random(101)})
  out = _score_pandas_chunk(frame, _FakeClassifier(scores), params,
                            ['id'], 1000)
  assert out['proba'].iloc[50] == 0.5
  assert out['score_decile'].iloc[50] >= 5
  assert out['prediction'].iloc[50] == 1
  assert out['high_score_ind'].iloc[50] == 'Y', (
      'proba == threshold must give high_score_ind Y (>=)')
